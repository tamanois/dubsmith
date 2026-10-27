# Performance and hardware

How the pipeline chooses a device and precision, how to use several GPUs, and
what happens on a long video when memory is the constraint.

## Devices and precision

```bash
python dub.py info                     # what would this machine use, and why?
python dub.py --video talk.mp4 --target fr --device cpu    # force CPU
python dub.py --video talk.mp4 --target fr --dtype fp16    # force a precision
```

`--device` accepts `auto` (the default), `cpu`, `cuda`, `cuda:N` for a specific
GPU, and `mps` or `xpu` on Apple and Intel hardware. `--dtype` accepts `auto`
(the default), `bf16`, `fp16` or `fp32`.

The rules, and the reasoning behind them:

| Situation | What happens |
|---|---|
| CUDA or ROCm GPU present | `auto` uses it. `cuda:1` is honoured exactly, so a second GPU can be pinned |
| No GPU | `auto` uses the CPU. Everything works; it is just much slower |
| Apple Silicon | `auto` uses the CPU, because faster-whisper runs on CTranslate2, which has no Metal backend. `--device mps` still puts the *other* stages on the GPU, and ASR quietly stays on the CPU |
| Ampere or newer NVIDIA (sm_80+) | bfloat16 |
| Turing or Volta (sm_70–sm_75) | float16 — bfloat16 is unsupported on these cards, and choosing it was a real failure on the second-hand GPUs people actually own |
| Pascal or older (sm_61 and below) | float32 — emulated float16 is lossy |
| CPU | float32 |
| ROCm | float16, or bfloat16 on CDNA2 and newer |
| Intel XPU | bfloat16 |

### Using more than one card

```bash
python dub.py --video talk.mp4 --target fr --devices cuda:0,cuda:1
python dub.py --video talk.mp4 --target fr --devices cuda:1,cuda:0  # cuda:1 leads
```

`--devices` spreads the two stages that dominate the wall clock - the blend and the
synthesis - over several cards. Each device gets its own worker process with its own
copy of the checkpoint, and each takes work as soon as it is free, so the split is by
*work*, not by model: nothing is sharded inside a model, because a clone prompt
belongs to the device that built it.

| | |
|---|---|
| Spread over the cards | The blend (one task per voice) and the synthesis (one task per voice, or per piece of a voice's lines when there are fewer voices than cards) |
| Stays on the first device | ASR, diarization, separation and the accent anchors - each runs once, so there is nothing to split |
| Voice | Unchanged. Every worker builds the same prompt from the same facts, and the reference is encoded once per worker rather than once per run |
| Memory per card | Unchanged. Each worker keeps its own measured ceiling, so a card sees the peak it would have seen on its own |
| Wall clock | Measured on two RTX 3060s. A controlled A/B - the same five synthesis tasks, same texts, reference and model, only the worker count changed - went from **19.4 s on one card to 14.9 s on two**. The gain is bounded by the checkpoint load (~7-8 s per worker, paid once per job), which dominates a job this short; on the 11-minute job, where synthesis measured 616 s, the per-task work is the whole cost and the split has the room it needs |
| Cost | One checkpoint load per worker per job (workers are kept alive across the blend and the synthesis), and one full re-synthesis when you switch to a list, because per-worker batching differs from a single worker's |

The first device in the list carries the stages that cannot be split, so
`--devices cuda:1,cuda:0` also decides which card ASR and diarization use.
`--device` is unchanged and is the whole answer when `--devices` is absent: a run
that does not ask for a fan-out does exactly what it did before. Repeating a name is
allowed (`--devices cpu,cpu`), which is how the fan-out is tested on a machine with
one processor.

A full dub was run this way (`input/change-the-world.mp4`, 22 s, `--accent-mode
preserve`): both cards took work, the peaks were 5.31 GiB and 4.93 GiB — each inside
its own ceiling and independent of the other — and `audit_qc.py` exited 0.

If you ask for a precision the device cannot run, the pipeline uses the best it
can and says so in the log rather than failing the job. `report.json` → `device`
records what was actually used, so a run on unfamiliar hardware is diagnosable
from the report alone.

**Changing the precision does not measurably change the voice.** Measured on three
sentences with one reference clip: float16 versus bfloat16 scored 0.9902 (median) on
the checkpoint's own speaker encoder, against 0.9898 and 0.9935 for two draws at the
*same* precision. The cross-precision difference is smaller than the run-to-run
sampling spread, so forcing `--dtype fp16` on a pre-Ampere card is a throughput
choice rather than a quality one.

---

## Memory on a long video

A long job is a memory problem before it is a speed problem, and the reason is
specific and measurable: **the checkpoint decodes the clone prompt's reference audio
once per line, then throws it away.** `generate_voice_clone` concatenates the
prompt's speech codes onto every line's codes and only trims the reference back off
the waveform afterwards, so a batch of eight lines cloned from a 21 s clip decodes
~170 s of audio no matter how short the lines are.

Measured on a 12 GB RTX 3060, bfloat16, cloning from this repo's own 21 s blend
clip:

| Lines in one call | Audio actually produced | Peak memory held |
|---|---|---|
| 1 | 0.2 s | 4.67 GiB |
| 2 | 0.5 s | 5.33 GiB |
| 4 | 1.3 s | 6.74 GiB |
| 8 | 3.0 s | **9.55 GiB** |

So the peak is ~0.65 GiB *per line* at that reference length, and an 11-minute video
(a job with 476 lines) sits one large batch away from the card's 11.87 GiB. That is
exactly what happened: `CUDACachingAllocator` reported allocations of **0.8–2.4 GiB**
failing with as little as **54 MB** free. The job still finished — the allocator
recovers internally, so nothing reaches the pipeline as an error — but retrying at
that edge is one unlucky batch away from a failed run.

Five things now keep a long job off that ceiling, and all are on by default:

| What | Why |
|---|---|
| A batch is budgeted in **decoded** audio: the reference is charged to every line, on top of the lines' own predicted duration | The reference dominates. Capping only the lines' duration changed nothing, because 8 × 21 s of reference is a bigger term than the lines themselves |
| A measured **ceiling** on the card (`0.7` of it): a call that fills more than that halves the size of everything after it | The peak depends on the reference length, the dtype and the card, so it is measured per call rather than assumed. It never grows back, so the run cannot oscillate around the edge |
| Cached blocks go back to the driver whenever free memory falls below 3 GB | Waiting for the error before releasing is releasing too late; this costs milliseconds |
| The allocator runs with `expandable_segments:True` (CUDA only, and only when you have not set `PYTORCH_ALLOC_CONF` yourself) | The allocator holds freed blocks at the size they were requested, so a pool can have free memory in it and still be unable to serve the next shape |
| `--tts-batch` caps the lines per call outright | The manual ceiling on top of the automatic one |

`report.json` → `tts` records what happened: `batch_size`, `max_decode_seconds`,
`reference_seconds`, `seconds_per_char`, `batches`, `peak_batch_seconds`,
`oom_retries`, `cache_releases`, `ceiling_shrinks` and `peak_reserved_bytes`. If
`oom_retries` is not zero, lower `--tts-batch`:

```bash
python dub.py --video long.mp4 --target fr --tts-batch 2
```

With `--devices` there is one set of counters per worker: the counts are summed,
`peak_reserved_bytes` is the worst card, and `per_device` keeps each card's own
numbers. A card that is smaller than the others shows up there rather than in the
peak alone.

A shorter reference is the other lever, and it is the cheap one: the reference is
charged per line, so halving it halves that term for the whole job. `--ref-text`
and `--accent-text` clips of 10–20 s are plenty; a 40 s reference doubles the cost of
every line for no benefit.

**The trade is time, and on a small card it is real.** Smaller batches mean more
model calls: on the measured 12 GB card the 11-minute job's synthesis went from 393 s
(with 66 failed allocations) to 616 s (with none), roughly +55 %. On a card with room
the ceiling never fires and batches stay as large as `--tts-batch` allows.

---

Next: [Outputs and caching](outputs-and-caching.md), or the
[full option reference](options.md).
