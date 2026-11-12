# Testing and quality

Running the suite, measuring whether the voice survived line by line, and turning
the report into pass/fail gates.

## Tests

All tests run **without** a GPU, downloaded models, or ffmpeg — heavy components
are injected as fakes and binaries are mocked. The whole suite runs in about
a minute.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

```
1466 tests (all passing)
```

Run a single area while iterating:

```bash
python -m pytest tests/test_tts.py -q          # voice cloning, accent modes
python -m pytest tests/test_pipeline.py -q     # end-to-end orchestration
python -m pytest tests/test_multi_speaker.py -q  # per-speaker dubbing
python -m pytest tests/test_diarize.py -q      # speaker clustering
python -m pytest tests/test_audit.py -q        # timbre audit + quality gates
python -m pytest tests/test_mixing.py -q       # ducking, room matching
python -m pytest -k accent -q                  # everything accent-related
python -m pytest tests/test_voices.py -q       # saved voices and cards
python -m pytest tests/test_provenance.py -q   # prosody/provenance records
python -m pytest tests/test_server.py -q       # API, job state machine, uploads

# Frontend behavior (Vitest + Testing Library), from web/
npm test
npm run build
```

## Real dubs against the local checkpoints

The suite proves the wiring with fakes. To hear the feature, dub a short excerpt
with each backend. `python dub.py models --list` shows what is present; download
any missing checkpoint by name:

```bash
python dub.py models --download tts_chatterbox
python dub.py models --download tts_fish
python dub.py models --download tts_cosyvoice3
```

Use a 15–25 s clip: every run pays for ASR, diarization and translation once, so
short input keeps the loop quick without changing the synthesis path.

```bash
ffmpeg -y -ss 00:01:00 -t 20 -i source.mp4 -vn -ac 1 -ar 24000 /tmp/clip.wav
```

Dub the same clip with each backend, in its own output folder so the runs cannot
share a cache:

```bash
python dub.py --audio /tmp/clip.wav --target fr --backend qwen       --output-dir /tmp/dub-qwen
python dub.py --audio /tmp/clip.wav --target fr --backend chatterbox --prosody-plan auto --output-dir /tmp/dub-chatterbox
python dub.py --audio /tmp/clip.wav --target fr --backend cosyvoice3 --prosody-plan auto --output-dir /tmp/dub-cosyvoice
python dub.py --audio /tmp/clip.wav --target fr --backend fish       --prosody-plan auto --output-dir /tmp/dub-fish
```

What to check, in order:

1. **The plan was produced.** Run one backend with `-v` and read the planning
   and request lines:
   ```bash
   python dub.py --audio /tmp/clip.wav --target fr --backend chatterbox \
     --prosody-plan llm --output-dir /tmp/dub-chatterbox -v 2>&1 \
     | grep -Ei "planned prosody|exaggerations="
   ```
   `Planned prosody for N line(s)` means the LLM answered; `exaggerations=[...]`
   in the engine request shows the plan reaching Chatterbox.
2. **The voice record names the backend.** `output/<stem>/04_dubbed/donor.json`
   has `"backend": "chatterbox"` and the per-speaker donor clip.
3. **It sounds acted, not read.** A/B the `dub.wav` from `/tmp/dub-qwen` against
   the expressive run; energy and pacing should change line by line.
4. **The neutral escape hatch.** `--prosody-plan off` (and the `qwen` default)
   must sound flat — that is the unchanged behaviour.
5. **Switching backends re-synthesises.** Change `--backend` and rerun; the
   fingerprint includes it, so the dub is rebuilt rather than served from cache.

`auto` is the default because translation already needs the same
OpenAI-compatible endpoint: when it is reachable, planning runs. When it is not,
`auto` logs a warning and speaks neutral, while `llm` fails the run.

### Auditing a dub's voice, line by line

`audit_timbre.py` measures whether each synthesised line kept the speaker's
timbre, using the checkpoint's own speaker encoder. It needs a GPU and the base
checkpoint, because it loads the same model the pipeline does:

```bash
python audit_timbre.py "output/<input-stem>"
python audit_timbre.py "output/<input-stem>" --device cpu   # when the GPUs are busy
```

It reports, per line, `cos(line, reference) − cos(line, accent)` where both
references are in the **target language** (the blend clip and the accent clip).
Matching the language is the point: comparing a French line against an English
donor mixes a language difference into a speaker measurement and makes every line
look like a collapse. Two anchors give the number a scale — a pure accent-voice
utterance scores about −0.023, and the reference against itself about +0.023 — so
"closer to the accent voice" is a measured position, not a threshold.

Measured on this repo's TED talk, with the same donor and accent clip:

| Run | Lines | min margin | median | Closer to the accent voice |
|---|---|---|---|---|
| `--accent-blend single-stage` | 55 | −0.0049 | +0.0156 | 5 (9.1%) |
| `two-stage` (default) | 59 | **+0.0176** | **+0.0257** | **0** |

Read it as a smoke test rather than a controlled experiment: ASR re-segmented the
two runs differently, so the text differs. Pass `--reference` and `--accent`
explicitly to measure a single-stage job against another job's blend clip.

A job with several speakers is audited **one speaker at a time**, each against
their own `reference_blend_<speaker>.wav`, because a line cloned from one voice
must not be judged against another's. Each group is its own scale — two voices
have two different "pure accent" floors — so the anchors are printed per group.
`--reference` overrides that and measures everyone against one clip, which is only
useful for a deliberate A/B.

### Checking a dub automatically

`audit_qc.py` turns the signals in `report.json` — and, optionally, the timbre
audit above — into pass/fail gates. It never blocks the pipeline: the dub is
already written, so a failure is a verdict on the delivery rather than a reason to
lose the work. Exit status is `0` when every gate passes, `1` when one fails, and
`2` when the job could not be checked at all, which makes it usable in CI.

```bash
python audit_qc.py "output/<input-stem>"              # cheap gates, no model
python audit_qc.py "output/<input-stem>" --timbre      # adds the per-line timbre gate
python audit_qc.py "output/<input-stem>" --timbre --device cpu   # measure on the CPU
python audit_qc.py "output/<input-stem>" --json        # machine-readable
```

| Gate | Checks | Default limit |
|---|---|---|
| `sync` | No line starts more than the allowance late | `shifted_seconds_max <= 0.25` (override with `--max-shift`) |
| `truncation` | No line loses a meaningful share of its words | worst line ≤ 10%, no line over 10% |
| `loudness` | The dub sits at the source's level | within 1.5 dB (skipped if unmeasurable) |
| `timbre` | No line followed the accent voice instead of the speaker's | 0% (needs `--timbre`) |

Filling the pause after a line is *not* a failure — overrunning into silence costs
nothing and is how sync is preserved — so `overrun` is reported as context rather
than gated. Every limit can be overridden (`--max-shift`, `--max-deleted-ratio`,
`--max-deleted-lines`, `--max-loudness-delta`, `--max-accent-share`). A gate that
could not run (no timbre measurement, unmeasurable source loudness) is reported as
**skipped**, never as a pass.

---

Next: [Architecture](architecture.md), or [Troubleshooting](troubleshooting.md).
