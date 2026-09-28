# AI Video Dubbing

Dub a video into another language while keeping each speaker's **own voice
**timbre**. Runs entirely on local, open models; no cloud required.

See [`PLAN.md`](PLAN.md) for the full vision, roadmap (P1–P6) and design
decisions, and [`AGENTS.md`](AGENTS.md) for the project context.

---

## What it does

| | |
|---|---|
| **Input** | One video, or one audio file |
| **Speakers** | One or several: speakers are found automatically and each is cloned from their own voice. `--diarizer none` forces a single voice |
| **Source language** | One language, auto-detected (multi-language is P5) |
| **Target language** | English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian |
| **Voice** | Cloned from the source speaker, with three accent modes (see [Accent modes](#accent-modes)) |
| **Output** | Dubbed video/audio + source & translated SRT subtitles, original music/ambience preserved |
| **Not included** | Lip sync (P4); a fully automatic native accent still needs a supplied reference clip (P2) |

---

## Hardware requirements

| | |
|---|---|
| **Disk** | ~6.5 GB of model weights (`models/` is gitignored), plus your media |
| **GPU** | Optional, but strongly recommended. ~12 GB VRAM runs the default path comfortably: the TTS checkpoint is ~3.5 GB in bfloat16 and Whisper large-v3 ~3 GB |
| **CPU only** | Works end to end and produces the same *result*. Measured on a 12-core box: a 22 s clip dubs in **2:10** (`preserve`) or **5:18** (`reference-accent`), with every quality gate passing. Synthesis is ~4.3× realtime in float32, so the real constraint is **RAM** (12.7–17.2 GB peak), not patience |
| **Python** | 3.10 or newer |

The precision follows the hardware rather than the device name — bfloat16 on
Ampere and newer, float16 on older accelerators (Turing, Volta), float32 on CPU.
Run `python dub.py info` on unfamiliar hardware before a long job: it reports the
accelerator, the device and precision the pipeline would pick, which optional
packages are installed, and which models are present.

---

## Prerequisites

### 1. System binaries

These are **not** installable from `pip` and need root. On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg rubberband-cli sox
```

| Binary | Required? | Used for | If missing |
|---|---|---|---|
| `ffmpeg` / `ffprobe` | **Required for video** | audio extraction, muxing, probing | video input fails with a clear error; **audio-only input still works** |
| `rubberband` | Recommended | best-quality time stretching | falls back to `sox`, then to a pure-Python stretcher |
| `sox` | Recommended | alternative stretcher | falls back to the Python stretcher |

The pipeline **degrades gracefully**: it never hard-fails because a *stretcher* is
missing, and audio-only jobs work with no system binaries at all. Only video
input genuinely requires `ffmpeg`.

> Check what you have:
> ```bash
> for b in ffmpeg ffprobe rubberband sox; do printf '%-11s %s\n' "$b" "$(command -v $b || echo MISSING)"; done
> ```

### 2. Python

Python **3.10+** is required (developed and tested on 3.10.12).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .              # or: pip install -r requirements.txt
```

The package installs a `dub` command, so `python dub.py --target fr` and
`dub --target fr` are the same thing. Optional features are extras:

```bash
pip install ".[background]"    # Demucs: keep the original music and ambience
pip install ".[diarization]"   # funasr: clone each speaker from their own voice
pip install ".[separation]"    # SpeechBrain: split overlapping speakers
pip install ".[full]"          # all of the above
```

Every one of those degrades gracefully when absent, so a first dub needs none of
them. On a **CPU-only machine**, install the CPU torch wheel first — the default
wheel is several gigabytes of CUDA you cannot use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install .
```

> If dependency resolution is awkward, install the two pins first — this is what
> `packages.md` refers to:
> ```bash
> pip install numpy==1.26.4 typing_extensions
> pip install -r requirements.txt
> ```

**GPU is optional.** With CUDA or ROCm the pipeline runs there; without one
everything still runs on CPU, just much slower. Speech recognition always uses
CTranslate2, which offers CUDA and CPU only — so on Apple Silicon ASR runs on the
CPU while the other stages can use Metal if you ask for it. See
[Devices and precision](#devices-and-precision).

Python packages installed by `requirements.txt`:

| Package | Role |
|---|---|
| `torch` | model runtime |
| `qwen_tts` | voice-cloning TTS |
| `faster-whisper` | ASR (bundles its own Silero VAD) |
| `pyloudnorm` | EBU R128 loudness normalisation |
| `openai` | client for the OpenAI-compatible translation endpoint |
| `librosa`, `soundfile` | audio I/O, resampling, fallback stretching |
| `pydantic` | config validation |

For development and tests:

```bash
pip install -r requirements-dev.txt
```

### 3. Optional: CUDA 12 libraries for GPU speech recognition

CTranslate2 (which faster-whisper uses) links against **CUDA 12** specifically,
while recent `torch` wheels ship **CUDA 13**. On such a machine ASR logs:

```
Library libcublas.so.12 is not found or cannot be loaded
```

The pipeline handles this by itself: it preloads vendored CUDA libraries, and if
that is not possible it **falls back to CPU** with a clear warning (identical
output, just slower). To get GPU ASR instead, install the matching runtime:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

No `LD_LIBRARY_PATH` export is required — the pipeline loads these libraries into
the process itself (see `pipeline/asr.py`). Check the log lines to see which
device was actually used.

### 4. A translation endpoint

The translation stage calls any **OpenAI-compatible** `/v1/chat/completions`
server. Point it at a local server or a cloud provider via `.env`:

```bash
cp .env.example .env
```

```dotenv
LLM_BASE_URL=http://127.0.0.1:8080/v1
LLM_API_KEY=not-needed
LLM_MODEL=local-model
```

A local `llama.cpp` server works well:

```bash
llama-server -m /path/to/model.gguf --port 8080
```

Real environment variables always take precedence over `.env`.

---

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

## Download the models

Model weights are **not** stored in this repository (`models/` is gitignored).
One command fetches everything the MVP needs:

```bash
python dub.py models --download all
```

This downloads:

| Name | Hugging Face repo | Approx. size |
|---|---|---|
| `whisper_large_v3` | `Systran/faster-whisper-large-v3` | ~3 GB |
| `tts_base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | ~3.5 GB |

Other commands:

```bash
python dub.py models --list                # show every model and whether it's present
python dub.py models --download tts_base   # one model only
python dub.py models --download everything # incl. optional models
python dub.py models --download --force all
```

Optional models (registered but not needed for the MVP):

| Name | Repo | Why you might want it |
|---|---|---|
| `tts_custom_voice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 9 premium named voices + emotion instructions |
| `tts_voice_design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | described voices; accent anchor for P2 |
| `tts_tokenizer` | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | standalone tokenizer (already bundled in `tts_base`) |

You can also copy a model directory in by hand — anything present under
`MODELS_DIR` is detected automatically. Override the location with `MODELS_DIR`
in `.env`, which every command then honours. The download command also takes
`--models-dir` directly:

```bash
python dub.py models --download all --models-dir /data/models
```

Check which models are visible (and what to do about the ones that aren't):

```bash
python dub.py models --list
```

A dubbing run fails with an actionable message naming the exact download command
if a required model is missing, so you can also just try a dub and see.

---

## Usage

### Which command do I want?

| If you want… | Use | What it changes |
|---|---|---|
| a first working dub, zero setup | `--video talk.mp4 --target fr` | detects the source language, picks the donor voice from the video, preserves the original music/ambience |
| a voice that isn't the video's speaker | `--ref-audio ref.wav --ref-text "…"` | clones `ref.wav` for both timbre and accent |
| to drop the source accent | `--accent-mode speaker-only` | keeps the timbre, lets the model choose target-language phonetics |
| a native accent in the original voice | `--accent-mode reference-accent --accent-audio nat.wav --accent-text "…"` | timbre from the video's speaker, pronunciation from `nat.wav` |
| every speaker in their own voice | *nothing* | finds the speakers and clones each one from their own voice — see [Dubbing a video with several speakers](#dubbing-a-video-with-several-speakers) |
| one voice for the whole video | `--diarizer none` | the pre-P1 behaviour: whichever speaker talks most supplies the voice |
| no music behind the voice | `--no-background` | discards the separated bed instead of ducking it |
| an absolute loudness target | `--loudness lufs --lufs -16` | forces a level instead of matching the source |
| to redo a run from scratch | `--force` | ignores the stage cache |
| to judge a run without listening | *nothing* | read `report.json` in the job folder |

Every option is described in [Options in full](#options-in-full);
`python dub.py --help` prints the same list.

### Commands, grouped by purpose

**Set up**

```bash
python3 -m venv .venv && source .venv/bin/activate   # one-time
pip install .                                        # runtime dependencies
pip install -r requirements-dev.txt                  # + pytest and the optional backends

python dub.py info                                   # what will this machine use?
python dub.py models --download all                  # ~6.5 GB of weights
python dub.py models --list                          # what is present already?
```

**Dub something**

```bash
# video -> dubbed video, source language auto-detected
python dub.py --video talk.mp4 --target fr

# audio only: no ffmpeg needed, writes dub.wav + subtitles
python dub.py --audio interview.wav --target es

# trust your own language detection instead of Whisper's
python dub.py --video talk.mp4 --source en --target de

# skip the subtitle files
python dub.py --video talk.mp4 --target ja --no-subtitles
```

With no `--ref-audio`, the reference voice is taken automatically from the
longest clean speech segment and its transcript comes from the ASR pass, so
there is nothing to prepare by hand.

**Choose the voice**

```bash
# clone a specific recording instead of the speaker in the video
python dub.py --video talk.mp4 --target fr \
  --ref-audio "voice.wav" \
  --ref-text "the exact words spoken in voice.wav"

# keep the speaker's timbre, drop their accent
python dub.py --video talk.mp4 --target fr --accent-mode speaker-only

# keep the speaker's timbre, borrow a native accent
python dub.py --video talk.mp4 --target fr \
  --accent-mode reference-accent \
  --accent-audio "native.wav" \
  --accent-text "the exact words spoken in native.wav"
```

Any format `libsndfile` reads works for reference and accent clips:
`.wav` `.flac` `.mp3` `.ogg` `.aiff`, a URL, or a `(samples, rate)` tuple.
`--ref-text` must be the **exact transcript** of `--ref-audio` because the model
conditions on both (ICL mode); the same is true of `--accent-text` for
`--accent-audio`. See [Accent modes](#accent-modes) for which mode needs which.

**Control the audio bed and level**

```bash
python dub.py --video talk.mp4 --target fr --no-background    # voice only
python dub.py --video talk.mp4 --target fr --duck-db 10       # bury the music further
python dub.py --video talk.mp4 --target fr --separator none   # treat the source as one track
python dub.py --video talk.mp4 --target fr --loudness lufs --lufs -16
```

**Control timing**

```bash
python dub.py --video talk.mp4 --target fr --min-stretch 0.75  # allow faster speech, trim less
python dub.py --video talk.mp4 --target fr --max-stretch 1.10  # cap how far lines are slowed
python dub.py --video talk.mp4 --target fr --stretcher sox     # pick a specific backend
```

**Repeatable and diagnostic runs**

```bash
python dub.py --video talk.mp4 --target fr --force     # ignore the cache
python dub.py --video talk.mp4 --target fr -v          # debug logging
python dub.py --video talk.mp4 --target fr --device cpu # rule the GPU out
```

### Fast iteration

A 5-minute video takes a few minutes to dub, which is slow to iterate on. Slice
out a short section and dub that instead — the pipeline behaves identically,
just faster:

```bash
ffmpeg -ss 30 -t 60 -i talk.mp4 -c copy clip60.mp4
python dub.py --video clip60.mp4 --target fr
```

Note that `-c copy` cuts on the nearest keyframe, so `clip60.mp4` may start a
second or two earlier than requested. That is fine for trying out options.

### Accent modes

Accent and timbre are stored separately by the checkpoint: pronunciation and
prosody ride in the reference *speech codes*, while vocal identity rides in the
*speaker embedding*. That split is what makes these three modes possible.

| `--accent-mode` | Speech codes (accent, pacing) | Speaker embedding (timbre) | Use when |
|---|---|---|---|
| `preserve` *(default)* | the reference voice, or the auto-selected donor | same clip | You want the original speaker, including their accent |
| `speaker-only` | none — the model chooses | the reference voice | The source accent is too strong and you want the model's own phonetics |
| `reference-accent` | **`--accent-audio`** | the reference voice, or the auto donor | The dub still sounds foreign and you have a native-target clip to borrow pronunciation from |

`speaker-only` is the cheap option: it needs no extra clip, but it also gives up
control over delivery, so pacing drifts with the model. `reference-accent` costs
you one clip and its transcript, and buys a delivery you can actually choose.

`reference-accent` asks the model to satisfy two signals at once: the accent
clip's speech codes and the speaker's own embedding. Those describe different
people, and any single line is free to follow either one — heard as the speaker
changing mid-video. So by default the mode **builds the blend once**
(`--accent-blend two-stage`): the accent transcript is rendered in the speaker's
voice, the take that kept their timbre is kept, and every line is then cloned from
that one clip. Its codes and its embedding come from the same audio, so there is
nothing left to disagree. `--accent-blend single-stage` restores conditioning each
line on both sources directly, which is useful for an A/B listen.

### Borrow a native accent without losing the voice

Hand over a recording of a native speaker. A news or podcast segment works
better than a slow, careful personal reading — see
[Reference pacing](#reference-pacing). The voice stays the speaker's; the
pronunciation becomes the clip's:

```bash
python dub.py --video talk.mp4 --target fr \
  --accent-mode reference-accent \
  --accent-audio "native-narration.mp3" \
  --accent-text "the exact words spoken in native-narration.mp3"
```

Both flags are required together, and `--accent-text` must be the exact
transcript of `--accent-audio` — it is what aligns the codes being borrowed.
`--ref-audio` is optional: leave it out and the timbre is auto-selected from the
video, exactly as in the other modes. Keep the accent clip short; 10–20 seconds
is plenty, and a longer clip mostly costs synthesis time.

Before dubbing anything, the mode renders your accent transcript in the speaker's
own voice and writes the result to `reference_blend.wav` inside the job folder.
That clip is what every line is then cloned from.

Building that clip is not neutral between the two voices. The accent clip arrives
as a complete example — its speech codes *and* its transcript — while the speaker
contributes a single embedding, so a single pass tends to come out closer to the
accent clip than to the speaker. The mode therefore repeats it: each pass re-reads
the previous clip and re-injects the speaker's embedding, walking the voice back
toward the speaker while the accent transcript keeps pronunciation anchored. It
stops as soon as `--blend-ratio` is reached (default `0.5`, meaning at least
half-way from the accent voice to the speaker's own) or after `--blend-iters`
passes (default 4), whichever comes first, and keeps the best pass rather than the
last one. `--blend-takes` (default 3) additionally renders several candidates in
the first pass and keeps the one closest to the speaker. The clip is cached with
the rest of the job, so a rerun reuses it instead of quietly reselecting a
different voice.

How far the blend got is measured rather than guessed. `report.json` →
`donor.blend` reports `donor_ratio`: `0` is the accent clip's voice, `1` is a
French utterance in the speaker's own voice. Both ends of that scale are in the
target language, which is what makes it meaningful — a raw similarity against the
*source-language* donor would measure the language gap as much as the voice. If a
run stops short of the target, the log says so: raise `--blend-iters`, or lower
`--blend-ratio` if you would rather keep more of the accent. The `margin` field is
a different thing — a ranking score for choosing between takes, not a verdict,
since its absolute value is not comparable across clip pairs.

The quickest check on a run is still to listen to `reference_blend.wav` on its own:
twelve seconds of the speaker explaining something in the target language answers
"whose voice is this?" directly.

#### Choosing the blend strength

`--blend-ratio` is the knob you will actually turn; `--blend-iters` is the safety
limit. Start from the defaults (`0.5`, `4`) and move in one direction at a time,
judging by the log's `donor ratio` line and the `reference_blend.wav` clip:

| You want | Do |
|---|---|
| The default: a balance that keeps the speaker clearly | leave both as-is (`--blend-ratio 0.5 --blend-iters 4`) |
| The speaker's voice stronger — the accent clip is leaking through | raise `--blend-ratio` (e.g. `0.8`) so it stops closer to the speaker |
| The speaker to dominate, accent as a hint only | `--blend-ratio 0.9 --blend-iters 6` |
| More of the borrowed accent, a bit of the speaker | lower `--blend-ratio` (e.g. `0.3`) |
| The exact previous single-pass behaviour | `--blend-iters 1` |
| Tune by hand rather than by the measured ratio | `--blend-ratio 0 --blend-iters N` — every pass runs and the least accent-like one is kept |

If the log says the run *stopped short of the target* (the ratio line came in
under `--blend-ratio`), raise `--blend-iters` or lower `--blend-ratio`. A ratio
above the target means the search succeeded and stopped early; raising
`--blend-ratio` would push further toward the speaker, at the cost of one extra
synthesis of the accent transcript per pass.

### Dubbing a video with several speakers

An interview, a debate, a two-hander: with more than one voice in the source, a
single donor cannot represent them. The pipeline finds the speakers and gives
each one their own voice. This is on by default, so a one-voice video behaves
exactly as it did before.

```bash
python dub.py --video debate.mp4 --target fr              # one voice per speaker
python dub.py --video debate.mp4 --target fr --diarizer none   # force one voice
```

The log names what it found, and `report.json` → `speakers` breaks it down:

```
Diarization (embedding): 2 speaker(s) over 12 segment(s) - SPK_0 7, SPK_1 5
Auto-selected donor for SPK_0: segment 3 (7.3s)
Auto-selected donor for SPK_1: segment 8 (5.0s)
```

**How it decides.** Each ASR segment is embedded with a speaker-verification
model and the embeddings are clustered, so every segment is assigned rather than
re-segmented — a line is never cut in half, and the labels stay aligned with the
text by construction. A cluster shorter than `--min-speaker-seconds` is folded
into its nearest neighbour rather than becoming a speaker with no usable voice to
clone from. Segments whose embedding fails inherit the previous speaker's, so one
bad two-second window cannot invent a third voice.

The default backend is ERes2NetV2, which ships inside `funasr`. It is an optional
dependency: without it every line is dubbed in one voice and the log says so
rather than failing the job. Install it with `pip install -r requirements-dev.txt`
or uncomment it in `requirements.txt`.

The threshold is the one number worth knowing about. It is a cosine *distance* on
embeddings, and it is not portable across material — on the two-person clip in
this repo the merge profile is flat to `0.379` and then jumps to `0.468`, so
everything in that gap returns the same two speakers, which is why the default
`0.40` is a defensible constant. If a video comes out over-split (one person
becoming two) raise `--speaker-threshold`; if two people are merged into one,
lower it. `02b_speakers.json` in the job folder records the labels the run used,
so you can see the damage without re-running.

**What each speaker costs.** Diarization adds one embedding pass over the speech
stem, which is cheap next to ASR. Synthesis is then grouped by speaker — one
prompt per voice rather than one per line — so a two-person video costs two
prompts and two batches instead of one of each. In
[`reference-accent`](#accent-modes) mode each voice needs its own blended
reference, because a clip whose codes and embedding both describe one speaker
cannot serve another:

```
04_dubbed/reference_blend_SPK_0.wav
04_dubbed/reference_blend_SPK_1.wav
```

Each is built by the same measured search as the single-voice case, with its own
`donor_ratio`, and `--blend-*` applies to all of them. A one-voice job keeps the
original `reference_blend.wav` name, so existing scripts and
`audit_timbre.py` are unaffected.

**Overlapping speech is detected, not guessed away.** Speakers are attributed per
segment, not per word, so a segment where two people talk over each other cannot
be given to both of them. Each segment is therefore re-embedded in 2.5 s windows
and every window is checked against the speakers the labels already establish: a
line is flagged when a *run* of its windows fits another of those speakers better
than its own, while its own speaker still owns the majority of it. Flagged lines
are listed in `02c_overlap.json`, counted in `report.json` → `overlap`, and named
in the log:

```
Overlapping speech detected in 1 of 16 segment(s): #1. Each is dubbed in the
voice of its dominant speaker, so the second voice is not separately dubbed.
```

The 2.5 s window is measured, not chosen by taste. On the two-person clip in this
repo a 1 s window splits one speaker's *own* voice by up to 1.08 cosine distance —
more than three times the 0.34 that separates the two speakers — so short windows
report noise rather than a second voice (re-clustering them flagged 14 of 16
segments). Comparing against a speaker *centroid* rather than against other
windows is the other half of the fix, because an average over many windows is
what cancels that noise; and a one-voice job is never flagged at all, since with
no second centroid there is nothing for a window to match.

The trade-off is sensitivity, and it is deliberate: an interruption shorter than
about a window is not caught, and a segment whose windows *all* match another voice
is treated as a labelling problem rather than an overlap. On this repo's clip the
detector flags 1 of 16 segments. By default a flagged line is still dubbed in one
voice, but the limitation is measured rather than silent; `--overlap separate`
splits it instead. A one-voice job skips the check entirely, so its report is
unchanged.

**Both voices are dubbed with `--overlap separate`.** A flagged segment is split
into one stream per voice by SpeechBrain's 16 kHz SepFormer, each stream is
re-transcribed (the second voice's words are not in Whisper's one-utterance
output), and each stream is matched to a speaker by comparing its embedding to the
same centroids the labelling used. Both lines are then translated and synthesised
in their own voice, and placed *simultaneously* on the timeline rather than one
being pushed behind the other — the overlap is genuine, so the dub keeps it. The
two voices are recorded in `02d_overlap_segments.json`, and the run's `report.json`
→ `overlap` gains `mode: "separate"`, `separated` and `lines`.

Separation is best-effort by construction: a segment whose streams cannot be told
apart (a likely false-positive flag), or whose second stream recovers no speech, is
left exactly as it was. `speechbrain` is an optional dependency — without it
`--overlap separate` warns and behaves as `--overlap dominant`.

**What is *not* attempted.** The number of speakers comes out of the clustering
rather than being declared, so `--speaker-threshold` is the knob to reach for when
the count looks wrong. Separation handles two overlapping voices per segment
(SepFormer is a fixed two-speaker model); a third voice on the same instant is not
recovered.

Measured on this repo's TED talk, same French lines and same donor, so the three
rows are directly comparable:

| Mode | Timbre similarity to donor | French acoustic fit (logprob, higher is better) |
|---|---|---|
| `preserve` | 0.9841 | −0.138 |
| `speaker-only` | 0.9876 | −0.081 |
| `reference-accent` | 0.9854 | **−0.057** |
| *(same speaker, different utterance)* | 0.9808 | — |

All three keep the timbre above the natural same-speaker baseline; the hybrid mode
moves the pronunciation furthest toward native French.

### Reference pacing

The model imitates the reference clip's speaking rate, so a slow, deliberate
recording lengthens every dubbed line. Longer lines need more compression, and
once the stretch limit is reached the surplus has to be trimmed. This applies to
the donor (`--ref-audio`) and the accent clip (`--accent-audio`) alike.

Measured on this repo's TED talk — same source, same voice, only the reference
pacing differs:

| Reference used as | Pace | Speech trimmed |
|---|---|---|
| auto donor (the source speaker) | ~3.5 words/s | 11.4 s |
| slow personal recording, as donor | ~2.0 words/s | **23.0 s** |
| slow personal recording, as accent clip | ~2.0 words/s | 13.4 s |
| fast narrated clip, as accent clip | ~3.6 words/s | **2.3 s** |

Two conclusions worth carrying into your own reference choices. Borrowing only an
accent is cheaper than borrowing the whole voice: the same slow clip cost 13.4 s
as an accent source but 23.0 s as the donor. And reference pace is a **trade, not
a win**: a faster reference shortens the translation so less speech needs cutting,
but it also leaves the voice finishing early — on this video the fast clip left
15.2 s of silence spread over 23 lines, in slots where the original speaker is
still talking on screen.

Those two failures are not equally bad. Silence is a pause; deleted words are
information loss. So prefer a reference that leans *slightly* fast, and let
[the length repair](#when-a-translation-is-too-long-for-its-slot) handle the lines
that still come out long — it removes the deletion without adding silence, which
no choice of reference pace can do. Measured across five clip speeds, roughly half
the lines missed their slot at *every* pace, so pacing alone cannot fix this.

Failing all that, lower `--min-stretch` to allow faster speech.

Need a transcript and never wrote one down? Transcribing the clip with the same
model the pipeline uses is accurate enough in practice — the run behind the table
above used a machine transcript of the accent clip. Just avoid paraphrasing: the
text has to match the audio it describes, so correct any obviously normalised
numbers or names by hand.

### Placing the dub in the room

Synthesised speech is close-mic'd and dry. Laid over a soundtrack it can read as a
voice *on top of* the picture rather than someone *in* it — the speaker's identity
is right, the sense of place is missing. `--room match` fixes that as a
post-synthesis step, so it cannot disturb the voice cloning. It does two things:

- the **room-tone floor**: the separated speech stem's quiet passages (genuine
  ambience, not the speaker) become a continuous bed under the dub;
- a **gentler duck** (`--room-duck-db`, 3 dB instead of 6), because the room is
  part of the voice and a deep duck pulls it away again.

```bash
python dub.py --video talk.mp4 --target fr --room match
```

**The reverb tail is opt-in and off by default.** An earlier version enabled it at
−12 dB, and a real listen rejected it: 2.4 s of undamped, full-bandwidth reverb
rings under every line and reads as an echo rather than a room. `--room-wet-db -20`
adds a tail if you want one — the impulse response is now low-passed at 4 kHz,
because a real room absorbs treble and flat noise does not.

It is **off by default** for the same reason the dry voice-over is the default: a
close voice-over is correct for narrated content, so it should stay reachable
rather than being silently replaced. Two things to know before enabling it. The
RT60 is estimated from the source's own decay, so it is a starting point — trust
your ears and tune with `--room-rt60` and `--room-wet-db`. And it needs the speech
stem, which only exists when the background is kept: with `--no-background` the dub
is delivered dry and `report.json` says `room.applied: false` rather than inventing
a space.

### Options in full
**Input and language** — exactly one of `--video` / `--audio` is required.

| Option | Default | What it achieves |
|---|---|---|
| `--video PATH` | – | Video input. Enables muxing, so you get `video_dubbed.mp4`. Needs `ffmpeg`. |
| `--audio PATH` | – | Audio-only input. Produces `dub.wav` + subtitles, needs no `ffmpeg`. |
| `--target LANG` | `fr` | Language to dub into. Accepts a code (`fr`) or a name (`French`). |
| `--source LANG` | auto | Overrides the detected source language. Useful when a clip is too short or too noisy for reliable detection. |

**Voice** — see [Accent modes](#accent-modes) for how these interact.

| Option | Default | What it achieves |
|---|---|---|
| `--ref-audio PATH` | auto | Clones this recording instead of the speaker found in the video. |
| `--ref-text TEXT` | – | Exact transcript of `--ref-audio`. Required with it in `preserve` mode; ignored in the other two modes. |
| `--accent-mode MODE` | `preserve` | Where pronunciation comes from: `preserve` (the reference voice), `speaker-only` (the model's own phonetics), `reference-accent` (the `--accent-audio` clip). |
| `--accent-audio PATH` | – | Native target-language clip to borrow pronunciation from. Requires `--accent-mode reference-accent`. |
| `--accent-text TEXT` | – | Exact transcript of `--accent-audio`. Required with it. |
| `--accent-blend MODE` | `two-stage` | `two-stage` renders the accent transcript once in the speaker's voice and clones every line from that clip, so its codes and its embedding cannot disagree. `single-stage` conditions each line on both sources directly. |
| `--blend-takes N` | `3` | How many times the one-off blend is rendered before keeping the take closest to the speaker. Only that clip is sampled this way, not the lines. |
| `--blend-iters N` | `4` | Maximum passes used to build the blend. Each pass re-reads the previous clip and re-injects the speaker's embedding, pulling the voice back toward the speaker. `1` disables the iteration. See [Choosing the blend strength](#choosing-the-blend-strength). |
| `--blend-ratio T` | `0.5` | How far toward the speaker's own voice the blend should reach, from `0` (the accent clip's voice) to `1` (a French utterance in the speaker's voice). Iteration stops there. `0` disables the search. See [Choosing the blend strength](#choosing-the-blend-strength). |

**Speakers** — see [Dubbing a video with several speakers](#dubbing-a-video-with-several-speakers).

| Option | Default | What it achieves |
|---|---|---|
| `--diarizer BACKEND` | `auto` | `auto` finds the speakers with the embedding model; `none` dubs every line in one voice, which is the pre-P1 behaviour. Without `funasr` installed, `auto` warns and behaves as `none`. |
| `--speaker-threshold D` | `0.40` | Cosine distance above which two embeddings count as different people. Raise it if one speaker is being split in two; lower it if two speakers are being merged. |
| `--min-speaker-seconds S` | `3.0` | A speaker must have at least this much speech, otherwise their lines join the nearest other voice — below the 3 s donor window there is nothing to clone from. `0` keeps every cluster. |
| `--overlap MODE` | `dominant` | What to do with a segment that carries two voices at once. `dominant` dubs the whole line in its dominant speaker's voice; `separate` splits it and dubs both voices (needs the optional `speechbrain`). See [Dubbing a video with several speakers](#dubbing-a-video-with-several-speakers). |
| `--overlap-separator BACKEND` | `auto` | `auto` uses SpeechBrain's SepFormer when installed, `sepformer` requires it, `none` disables it. Only used with `--overlap separate`. |

**Audio bed and loudness**

| Option | Default | What it achieves |
|---|---|---|
| `--no-background` | off | Discards the original music/ambience. By default it is separated out and ducked under the dub instead. |
| `--separator BACKEND` | `auto` | `auto`, `demucs` (split speech from the bed), or `none` (treat the source as a single track). |
| `--duck-db DB` | `6.0` | How far the background drops while dialogue plays. Higher = the voice sits more forward. |
| `--room MODE` | `off` | `match` places the dub in the source's own acoustic space — a reverb tail, a continuous room-tone floor, and a shallower duck. `off` keeps the dry voice-over, which is the right choice for narrated content. Needs the background to be kept (the default). |
| `--room-rt60 SECONDS` | measured | Reverberation time to use instead of measuring one from the speech stem. An escape hatch for a source whose decay cannot be estimated. |
| `--room-wet-db DB` | `0` | Reverb level relative to the dry voice. **`0` means no reverb** — the tail is opt-in, because it can only be judged by ear and the first default was rejected on a real listen as an audible echo. Try `-20` to enable a subtle tail. |
| `--room-duck-db DB` | `3.0` | How far the bed drops under dialogue when `--room match`. Shallower than `--duck-db`, because the room is part of the voice. |
| `--room-tone-db DB` | `-30.0` | Level of the room-tone floor lifted from the speech stem's pauses. |
| `--loudness POLICY` | `match-source` | `match-source` puts the dub at the original track's measured level; `lufs` forces `--lufs` instead. |
| `--lufs VALUE` | `-16` | Absolute loudness target in LUFS. Only used with `--loudness lufs`. |

**Timing** — these trade speech clarity against staying in sync.

| Option | Default | What it achieves |
|---|---|---|
| `--min-stretch RATIO` | `0.85` | Most a line may be compressed to fit its slot. Lower = faster speech, so fewer words get trimmed, but it can start to sound hurried. |
| `--max-stretch RATIO` | `1.18` | Most a line may be slowed to fill its slot. Raising this makes speech sound unnatural, so prefer fixing the translation. |
| `--stretcher BACKEND` | `auto` | `auto`, `rubberband`, `sox`, `librosa`, `none`. `none` disables stretching (lines then only overrun or trim). Falls back automatically if a binary is missing. |

**Length repair** — fixes a long translation by wording rather than by cutting audio.

| Option | Default | What it achieves |
|---|---|---|
| `--refine` / `--no-refine` | on | Rewrites the lines that would otherwise have words cut, then re-synthesises just those. Lines that already fit are untouched; a rewrite is kept only if it measurably reduces the overflow. `--no-refine` restores the previous trim-only behaviour. |
| `--refine-max-passes N` | `1` | How many times a line may be rewritten before falling back to trimming. `0` disables the repair. |
| `--refine-threshold SECONDS` | `0.05` | How far a line must overrun its slot *and* the pause after it before it is worth rewriting. Raise it to repair fewer, worse lines. |
| `--length-budget MODE` | `off` | `words` also tells the translator how many words fit each slot, measured from a target-language reference clip. `off` sends only the duration, as before. Needs `--accent-audio` or a target-language `--ref-audio`; without one it falls back to `off` automatically. |

**Models and hardware**

| Option | Default | What it achieves ||---|---|---|
| `--device DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N`, `mps`, or `xpu`. `cuda:1` pins a specific GPU. `cpu` is much slower but proves the GPU is not the problem. See [Devices and precision](#devices-and-precision) |
| `--dtype DTYPE` | `auto` | Precision for the voice-cloning checkpoint: `auto`, `bf16`, `fp16`, or `fp32`. `auto` reads the device's capability. An explicit value is honoured unless the device cannot run it |
| `--whisper-model NAME` | `large-v3` | ASR model id. A smaller one is faster and less accurate — the main lever on a CPU-only machine. |
| `--llm-model NAME` | from `.env` | Overrides `LLM_MODEL` for this run. |
| `MODELS_DIR` (env) | `models` | Where weights live. Set in `.env`; the download command also takes `--models-dir`. |

**Output and caching**

| Option | Default | What it achieves |
|---|---|---|
| `--output-dir DIR` | `output` | Where job folders are written. Each input gets its own `<stem>` folder. |
| `--no-subtitles` | off | Skips both SRT files. |
| `--force` | off | Ignores the stage cache and rebuilds everything. |
| `-v` / `--verbose` | off | Debug logging, including which device each stage actually used. |

### When a translation is too long for its slot

Translated speech is routinely longer than the source: French needs roughly 15–40%
more time than English for the same content. The stretch clamp absorbs part of
that, and a line is allowed to spill into the pause that follows it, but anything
beyond that used to be **cut** — words deleted to keep the video in sync.

The length repair avoids the cut by fixing the *text* instead:

1. After the first synthesis pass, each line's spoken length is measured against
   its slot plus the following pause.
2. Lines that would overflow are sent back to the translator together, with their
   previous wording, the source line, and a word ceiling derived from how much
   time is actually available.
3. Only those lines are re-synthesised. A rewrite is adopted **only** if it
   measurably reduces the overflow, so a line can never come out worse.

Three properties make it safe to leave on:

- **Lines that already fit are never touched** — their audio is bit-for-bit
  identical to a run with `--no-refine`, which is asserted in the test suite.
- **A failed repair is a no-op.** If the translator is unreachable, or a custom
  translator cannot rewrite, the run logs a warning and keeps the audio it has.
- **Nothing is re-shortened on a re-run.** Rewrites always start from the original
  translation, recorded in `03_script.json`, and the spoken version is kept
  separately in `03b_spoken.json` — which is also what the target subtitles use,
  so they never display words the viewer cannot hear.

Measured on the 4:47 TED talk, English → French, identical settings apart from
the flag:

| Metric | `--no-refine` | with repair |
|---|---|---|
| Speech deleted | 3.76 s | **0.0 s** |
| Worst single line deleted | **59%** | **0%** |
| Lines losing >10% of their speech | 5 | **0** |
| Lines that had to be cut | 8 | 0 |
| Output exceeding slot + pause | 4.79 s | 0.51 s |
| Silence left in slots | 18.35 s | 17.11 s |
| Round-trip intelligibility (WER) | 1.5% | 1.2% |
| Drift / loudness / background | — | unchanged |

The repair rewrote 3 of 57 lines in one extra translator call. Its cost is a
second synthesis pass over just those lines, which took the stage from 56 s to
73 s.

### Verified commands
These were all run on a 4:47 English TED talk → French, on 2× RTX 3060. The
figures come from each run's `report.json`.

| Command | Result |
|---|---|
| `--video talk.mp4 --target fr` | 11.4 s of speech trimmed; sync exact; background preserved |
| `--video talk.mp4 --target fr --ref-audio slow.wav --ref-text "…"` | 23.0 s trimmed — the slow reference lengthened every line |
| `--video talk.mp4 --target fr --accent-mode speaker-only` | timbre 0.9876, French fit −0.081 |
| `--video talk.mp4 --target fr --accent-mode reference-accent --accent-audio slow.wav --accent-text "…"` | timbre 0.9854, French fit −0.057, 13.4 s trimmed |
| `--video talk.mp4 --target fr --accent-mode reference-accent --accent-audio fast.mp3 --accent-text "…"` | **best overall**: 2.3 s trimmed, 0 drift, loudness matched, background preserved |
| `--video talk.mp4 --target fr --accent-mode reference-accent --accent-audio fast.mp3 --accent-text "…" --refine` (default) | **best overall**: 0.0 s trimmed, 3 lines rewritten, worst-line deletion 59% → 0%, WER 1.5% → 1.2% |
| `--video debate.mp4 --target fr --accent-mode reference-accent …` (two speakers, 60 s) | 2 voices found (7 and 8 lines), each with its own donor and its own blend clip; 0 of 15 lines closer to the accent voice on the per-speaker audit |

The last row is the configuration to aim for: a fast-paced native clip as the
accent source gives near-native pronunciation with almost no speech lost. Full
run takes about 2 minutes 45 seconds.

---

## Outputs

Everything for one job lands in `output/<input-stem>/`:

```
output/talk/
├── 01_audio.wav         # normalised 48 kHz mono source audio
├── 01b_speech.wav       # separated speech stem (when background is preserved)
├── 01c_background.wav   # separated music/ambience stem
├── 02_transcript.json   # ASR result: segments, words, timings, language
├── 02b_speakers.json    # segment id -> speaker, from diarization
├── 02c_overlap.json     # which segments carry a second voice (multi-speaker runs)
├── 02d_overlap_segments.json   # only with --overlap separate: the per-voice lines
├── 03_script.json       # source -> target text per segment
├── 03b_spoken.json      # lines the length repair re-wrote (only those)
├── 03c_refine.json      # what the length repair achieved
├── 04_dubbed/           # one synthesised clip per segment (+ .done marker)
│   ├── donor.json       # which voice was cloned for each speaker, and from what
│   ├── reference_blend.wav   # only in reference-accent: the clip every line is cloned from
│   ├── reference_blend_SPK_1.wav   # ...one per speaker once there are several
│   └── blend.json       # only in reference-accent: how each clip was chosen
├── 05_mix.wav           # assembled dub track, before normalisation
├── 05_room.json         # only with --room match: the decay measured and levels used
├── timing.json          # per-segment stretch ratio, placement and deficit
├── dub.wav              # final loudness-normalised dub audio
├── video_dubbed.mp4     # dubbed video (video input only)
├── subs_source.srt
├── subs_target.srt
├── report.json          # counts, loudness, stretch stats, stage timings
└── manifest.json        # stage cache
```

`report.json` is the quickest way to judge a run without listening to it:

| Field | Meaning |
|---|---|
| `segment_count` | Utterances found after VAD gating |
| `source_language` | Language Whisper detected |
| `dub_seconds` | Length of the dub track (should match the source) |
| `loudness_source_lufs` | Loudness of the original audio |
| `loudness_dub_lufs` | Loudness actually delivered |
| `loudness_target_lufs` | What the dub aimed for |
| `loudness_policy` | `match-source` or `lufs` |
| `loudness_delta_vs_source` | Positive when the dub is louder than the original |
| `loudness_gain_db` | Total gain applied to reach the target |
| `loudness_peak_limited` | `true` when the limiter engaged to avoid clipping |
| `background_preserved` | `true` when the original music/ambience was kept |
| `duck_db` | Attenuation applied under dialogue (`null` when no background). With `--room match` this is the shallower `--room-duck-db`, i.e. the depth actually used |
| `room` | Present only with `--room match`: the mode, whether it was applied, the measured `rt60_s`, whether the reverb tail is on (`reverb`), and the `wet_db` / `tone_gain_db` / `duck_db` used. Absent — not `null` — on a dry run, so previous reports read unchanged. `applied: false` means the speech stem was missing (usually `--no-background`) and the dub was delivered dry |
| `accent_mode` | `preserve`, `speaker-only`, or `reference-accent` |
| `speakers` | One entry per voice with its segment count and speech time, or `null` for a one-voice job — so single-voice reports read exactly as they did before diarization existed. The names (`SPK_0`, …) are the ones in `02b_speakers.json` |
| `overlap` | Present only on multi-speaker runs: how many segments were found to carry a second voice (`segments`, `seconds`, `speech_ratio`) and how they were handled. `mode: "dominant"` means each flagged line was dubbed in its dominant speaker's voice; `mode: "separate"` (with `--overlap separate`) means both voices were recovered and dubbed, adding `separated` and `lines` counts. Absent — not `null` — on a one-voice job |
| `donor` | Which reference voice was used: an explicit file, or the auto-selected segment (with its timestamps and transcript) |
| `donor.speaker_count` | How many voices the run found |
| `donor.primary_speaker` | The speaker with the most speech. Its own fields are also copied to the top level of `donor`, so `donor.source` / `donor.accent` / `donor.blend` keep naming the voice a single-speaker job would have used |
| `donor.speakers` | Per-voice detail: the donor segment chosen for each, how many segments they own, how long they speak, and the accent and blend records for that voice. This is the job's voice bible |
| `donor.accent` | Present only in `reference-accent` mode: the accent clip's path and whether a transcript was supplied |
| `donor.blend` | Present only in `reference-accent` mode: the strategy used, the passes and takes considered, and how the kept clip scored. `donor_ratio` places it between the accent voice (`0`) and the speaker's own voice (`1`) — the field to watch, and the one to raise `--blend-iters` for. `margin` only ranks takes against each other; a negative value is not by itself evidence of a problem |
| `clamped_segments` | Lines that hit a stretch limit (either direction) |
| `clamped_long_segments` | …of which were too long to compress into their slot |
| `clamped_short_segments` | …of which were too short to slow down enough to fill it |
| `deficit_segments` | Lines that left silence in their slot |
| `deficit_seconds_total` | Total silence left inside slots |
| `overrun_segments` | Lines that spilled into the pause after them (free, no speech lost) |
| `overrun_seconds_total` | Total speech fitted into those pauses |
| `surplus_seconds_total` | Total speech exceeding slot + pause, i.e. at risk of being cut |
| `trimmed_segments` | Lines where the residue had to be cut to keep the video in sync |
| `trimmed_seconds_total` | Seconds of speech lost to those cuts |
| `worst_line_deleted_ratio` | The largest share of a single line that was deleted |
| `lines_over_10pct_deleted` | Lines that lost more than a tenth of their speech |
| `speech_rate_wps` | Words per second the cloned voice actually delivered |
| `refine` | What the length repair did: offenders, improved, seconds reclaimed |
| `shifted_segments` | Lines pushed later to avoid overlapping another |

Two of these deserve attention:

- **`clamped_segments` above zero** means the translation was too long for its
  slot, so timing hit the stretch limit. Tighten the translation (or shorten the
  lines) rather than raising `--max-stretch`, which would sound unnatural.
- **`loudness_peak_limited: true`** means the limiter reduced a few peaks so the
target could be reached without clipping. Speech is peakier than mastered audio,
and without limiting a handful of transients would hold the whole program several
dB too quiet. The delivered loudness should still match `loudness_target_lufs`;
if it does not, the material was too dynamic to reach the target cleanly.
- **`trimmed_seconds_total`** is the one to watch. Translated speech is often
  longer than the source (French runs roughly 15–40% longer than English), and
  the stretch limit caps how far a line can be compressed. A line is first
  allowed to spill into the pause that follows it — free, since nobody is talking
  — and only the remainder is cut, because letting it run on would push every
  later line out of sync with the video. Nothing is ever shifted late: sync is
  exact. To reduce trimming, ask the translator for terser output (roadmap P3
  adds an automatic length-feedback loop), or lower `--min-stretch` to allow
  faster speech.

### Caching

Stages are cached in `<job folder>/manifest.json`, keyed by a fingerprint of
every job setting. Re-running an identical command does nothing at all — that is
what makes iterating on a long video cheap.

The fingerprint is a hash of the whole configuration, so **changing any option
re-runs the whole job**: the target language, the accent mode, the reference or
accent clip, the stretch bounds, the device. There is no partial invalidation, so
a run that only changes `--accent-mode` re-does the ASR and translation as well.
That is deliberate — it keeps a result reproducible from one command — but it
means experimenting costs a full run each time. Slice the input short
(see [Fast iteration](#fast-iteration)) when you plan to try several settings.

Because the fingerprint distinguishes settings, you can safely point several
different runs at the same `--output-dir`: each input gets its own `<stem>`
folder, and a run whose settings differ from the last one starts fresh rather
than reusing stale audio.

```bash
python dub.py --video talk.mp4 --target fr --force        # ignore the cache entirely
rm -rf output/talk                                        # or just start clean
```

---

## How it works

```
video/audio ─► 1. audio extract ─► 2. separate ─► 3. ASR + VAD ─► 4. diarize ─┐
                               (speech | bed)                               │
  8. mux + SRT ◄─ 7. mix + duck ◄─ 6. clone + synthesise ◄─ 5. translate (LLM) ◄┘
```

| # | Stage | Module | What it does |
|---|---|---|---|
| 1 | Audio | `pipeline/media.py` | ffmpeg extract (video) or direct read (audio), mono 48 kHz |
| 2 | Separate | `pipeline/separation.py` | Demucs splits speech from music/ambience; ASR then reads the clean speech stem |
| 3 | ASR | `pipeline/asr.py` | faster-whisper + Silero VAD, word timestamps, drops non-speech |
| 4 | Diarize | `pipeline/diarize.py` | embeds each segment and clusters the embeddings into speakers |
| 4b | Overlap | `pipeline/overlap_separation.py` | with `--overlap separate`, splits a flagged segment into per-speaker streams for re-transcription |
| 5 | Translate | `pipeline/translate.py` | batched, context-aware, length-budgeted translation |
| 6 | Synthesise | `pipeline/tts.py` | per-speaker donor selection + voice cloning, grouped by speaker |
| 7 | Mix | `pipeline/timing.py`, `pipeline/mixing.py` | fit each line to its slot, then lay it over the ducked bed — optionally matched to the source's room |
| 8 | Deliver | `pipeline/orchestrate.py` | loudness match, subtitles, mux, report |

Three design points worth knowing:

- **Accent comes from the donor clip, timbre from cloning.** With no
  `--ref-audio`, the source speaker is the donor, so the accent is *preserved*.
  `--accent-mode speaker-only` drops the donor's phonetics altogether;
  `--accent-mode reference-accent` replaces them with a native clip's.
- **One prompt per voice, not per line.** Diarization gives each speaker their own
  donor and, in `reference-accent` mode, their own blended reference; synthesis is
  then grouped by speaker so the prompt is rebuilt only when the voice changes.
- **Timing errors never accumulate.** Each line is anchored to its own original
  start time; a line that does not fit spills into the pause after it and any
  residue beyond that is trimmed. Lines are never pushed late, so a long
  translation cannot make the rest of the video drift out of sync.
- **The room can be borrowed without touching the voice.** Synthetic speech is
  recorded close and dry, which reads as a voice laid over the picture. With
  `--room match`, the decay (RT60) is measured from the separated speech stem —
  the only material that carries the room — and applied as a convolution over the
  finished dub, while the stem's own pauses become a continuous room-tone floor.
  It runs after synthesis, so it cannot disturb the voice cloning.

---

## Tests

All tests run **without** a GPU, downloaded models, or ffmpeg — heavy components
are injected as fakes and binaries are mocked. The whole suite runs in about
a minute.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

```
825 passed
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
```

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
| `sync` | No line pushed later than its slot | `shifted_segments == 0` |
| `truncation` | No line loses a meaningful share of its words | worst line ≤ 10%, no line over 10% |
| `loudness` | The dub sits at the source's level | within 1.5 dB (skipped if unmeasurable) |
| `timbre` | No line followed the accent voice instead of the speaker's | 0% (needs `--timbre`) |

Filling the pause after a line is *not* a failure — overrunning into silence costs
nothing and is how sync is preserved — so `overrun` is reported as context rather
than gated. Every limit can be overridden (`--max-shifted`, `--max-deleted-ratio`,
`--max-deleted-lines`, `--max-loudness-delta`, `--max-accent-share`). A gate that
could not run (no timbre measurement, unmeasurable source loudness) is reported as
**skipped**, never as a pass.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Missing model: 'whisper_large_v3' …` | Run `python dub.py models --download all` |
| `Required binary 'ffmpeg' was not found on PATH` | `sudo apt-get install -y ffmpeg` (audio-only jobs still work) |
| `… contains no audio stream, so there is nothing to transcribe` | The file is video-only. Verify with `ffprobe -v error -show_entries stream=codec_type file.mp4`; mux an audio track in first |
| `CUDA is unusable for speech recognition … libcublas.so.12` | Informational — ASR fell back to CPU. Install `nvidia-cublas-cu12 nvidia-cudnn-cu12` for GPU ASR (Prerequisites §3) |
| `No speech detected in the input` | No audible speech, or the wrong audio track. Check with `ffprobe` |
| `Translation response was not valid JSON` | Lower `LLM_CHUNK_SIZE` in `.env`, or use a model better at JSON |
| `Translation request to … failed` | The LLM server isn't reachable — check `LLM_BASE_URL` |
| `--ref-audio requires --ref-text` | Add the exact transcript, or drop both flags to auto-pick a donor |
| `--accent-audio only applies to --accent-mode reference-accent` | Add that flag, or remove `--accent-audio` |
| `--accent-mode reference-accent requires --accent-audio` | Add the native-target clip, or use another mode |
| `--accent-audio requires --accent-text` | Transcribe the accent clip; the codes cannot be aligned without it |
| `does not support voice cloning` | Pointed at a CustomVoice/VoiceDesign checkpoint; use `tts_base` |
| `Unsupported language 'xx'` | Use one of the ten supported languages |
| CUDA out of memory | `--device cpu`, or free the GPU (a large local LLM often occupies it). On a multi-GPU box, `--device cuda:1` pins the run to the other card |
| An unexpected amount of VRAM is in use | `--device cuda:1` now genuinely uses the second GPU; the log line and `report.json` → `device` both report the card that was used |
| `dtype fp16 is not usable on cpu (cpu); using fp32 instead` | Informational, not an error: half-precision on a CPU is slow and patchy, so float32 is used. Drop `--dtype` to let the capability decide |
| The GPU is not being used at all | Run `python dub.py info` — it reports the accelerator, the device it would pick, and the precision. If it says `cpu`, torch either sees no GPU or the wrong wheel is installed (see Prerequisites §2 and §3) |
| Speech recognition is on the CPU when a GPU exists | CTranslate2 links against a specific CUDA major. Run `python dub.py info` and check the `asr device` line; Prerequisites §3 has the fix |
| Dub sounds rushed or slow | Check `report.json` → `clamped_segments`; tighten the translation |
| Some words are cut off | `report.json` → `trimmed_segments`: the translation is too long for its slot. Check `worst_line_deleted_ratio` too — a small total can hide one badly damaged line. Leave the length repair on (it is on by default), shorten the translation, or lower `--min-stretch` |
| A line lost a lot of words despite a low `trimmed_seconds_total` | Look at `worst_line_deleted_ratio` and `lines_over_10pct_deleted`: damage is usually concentrated in one or two lines, not spread evenly |
| Lines sound slow or have gaps | `report.json` → `deficit_seconds_total`: the translation is shorter than its slot. See [Reference pacing](#reference-pacing), or use `--length-budget words` with a target-language reference |
| The repair did not help | Check `report.json` → `refine`. `attempted: false` means no line overflowed (nothing to fix) or the translator cannot rewrite; `improved: 0` means the rewrites did not fit either, so the old behaviour applied |
| Dub drifts out of sync | Should not happen — lines are never shifted late. If `shifted_segments > 0`, check `timing.json` for genuinely overlapping source speech |
| Dub sounds quiet | Check `loudness_delta_vs_source`; `0` means it matches the original. Force a level with `--loudness lufs --lufs -16` |
| Music/ambience is missing | Install `demucs` (`pip install demucs`); without it the background cannot be separated and is discarded |
| Music fights the voice | Raise `--duck-db` (default 6 dB) |
| Dub still has the speaker's accent | Try `--accent-mode speaker-only`, which keeps the timbre but drops the donor's phonetics. If that is not enough, use `--accent-mode reference-accent` with a native clip |
| Still sounds foreign in `reference-accent` mode | Check `--accent-text` matches the clip word for word — mismatched text misaligns the borrowed codes. A longer, more natural accent clip also helps |
| The voice changes part-way through the video | Only possible in `reference-accent` with `--accent-blend single-stage`, where every line is free to follow the accent codes or the speaker embedding. The default `two-stage` removes that choice — each voice's lines are cloned from that voice's own `reference_blend[_<speaker>].wav`, so the voice cannot change between lines. Listen to that clip on its own to confirm stage 1 picked up the speaker's timbre; if it did not, raise `--blend-takes` or try a different `--accent-audio`. To measure it rather than judge it, see [Auditing a dub's voice](#auditing-a-dubs-voice-line-by-line) |
| A voice changes *between speakers* rather than mid-line | Expected, unless the two voices are the same person: each speaker is cloned from their own donor. If the count is wrong, see the speaker-threshold rows above |
| The borrowed accent dominates — the speaker's own voice is barely there | The blend was built too close to the accent clip. Check `report.json` → `donor.blend.donor_ratio`: below `0.5` means it never got half-way to the speaker. Raise `--blend-iters`, or lower `--blend-ratio` if you want more of the accent |
| The blend reached a donor ratio of only … | Informational: the target was not met within `--blend-iters` passes, and the best pass was used anyway. Raise `--blend-iters` or lower `--blend-ratio` |
| `--accent-blend only applies to --accent-mode reference-accent` | Add that flag, or drop `--accent-blend` |
| `--blend-takes only applies to --accent-mode reference-accent` | Add that flag, or drop `--blend-takes` |
| `--blend-iters only applies to --accent-mode reference-accent` | Add that flag, or drop `--blend-iters` |
| `--blend-ratio only applies to --accent-mode reference-accent` | Add that flag, or drop `--blend-ratio` |
| Every line is dubbed in one voice, but the video has several speakers | Diarization is off or unavailable. Check the log for `Diarization (none)` — it means `--diarizer none` was set, or `funasr` is not installed. Install it (`pip install -r requirements-dev.txt`), then re-run. The detection happens before synthesis, so a cached job needs `--force` to pick up a newly installed backend |
| One person is being split into two voices | Raise `--speaker-threshold` (e.g. `0.6`); a distance-based threshold is genuinely material-dependent. `02b_speakers.json` shows how the segments were labelled. (Overlap detection is supervised by those labels but does not use this threshold.) |
| A line sounds like the wrong person, or you can hear two people on it | `report.json` → `overlap` counts segments that hold a second voice; `02c_overlap.json` lists their ids. By default such a line is dubbed in its dominant speaker's voice — run with `--overlap separate` (needs the optional `speechbrain`) to split it and dub both voices. Detection is deliberately conservative (a 2.5 s run that fits another speaker better), so short interruptions are not reported |
| Two people are merged into one voice | Lower `--speaker-threshold` (e.g. `0.3`), and check `--min-speaker-seconds` is not folding the quieter speaker away. `--diarizer none` confirms the rest of the pipeline is unaffected |
| A quiet speaker's lines are in the wrong voice | Their cluster was probably shorter than `--min-speaker-seconds` and got folded into the nearest voice. Lower it, at the cost of cloning from a shorter window |
| `funasr is not installed, so every line will be dubbed in one voice` | Informational fallback, not an error: the job still runs. Install `funasr` for multi-speaker dubbing, or pass `--diarizer none` to make the intent explicit |
| The dub is rushed, or words are missing | The reference clip's pacing is being imitated, and a slow reference lengthens every line. Use a faster-paced reference — see [Reference pacing](#reference-pacing) |
| `audit_qc.py` reports `SKIP timbre — no accent clip to measure against` | Expected for a `preserve` job: nothing was borrowed, so "did the line follow the accent voice?" has no answer. Pass `--accent <clip>` to force one. The other three gates still report, and the exit code stays 0 |
| The timbre audit fails while the GPUs are busy | Pass `--device cpu` to `audit_qc.py` or `audit_timbre.py`. The measurement is slower there but identical |

Logs are written to stderr and include a job id. Set `LOG_JSON=1` for
machine-readable logs.

---

## Project layout

```
dub.py                     CLI entry point (`dub`, `dub info`, `dub models`)
audit_timbre.py            per-line speaker-timbre audit (--device cpu works)
audit_qc.py                automated quality gate for a finished job
LICENSE                    Apache-2.0
pipeline/
├── config.py              job + settings models, device capability, languages
├── errors.py              domain exception hierarchy
├── logging_utils.py       job-scoped structured logging
├── models.py              model registry, download, path resolution
├── media.py               ffmpeg/ffprobe wrappers, loudness, audio I/O
├── subtitles.py           SRT rendering
├── asr.py                 transcription + segmentation
├── diarize.py             speaker embeddings + clustering (multi-speaker)
├── overlap_separation.py  SepFormer backend for splitting overlapping voices
├── separation.py          speech/background split (Demucs)
├── mixing.py              ducking, bed mix and room matching (pure numpy)
├── translate.py           OpenAI-compatible translation
├── tts.py                 per-speaker donor selection + voice cloning
├── timing.py              stretching backends + timeline assembly
├── audit.py               timbre measurement + quality gates
└── orchestrate.py         stages, caching, reporting
tests/                     pytest suite (825 tests, no GPU or models needed)
models/                    downloaded weights (gitignored)
output/                    job artifacts (gitignored)
clone.py, main.py          standalone TTS demos kept for reference
pyproject.toml             packaging, entry points, optional extras, pytest config
.github/workflows/ci.yml   test suite on a CPU-only runner
PLAN.md                    roadmap: MVP + P1–P6
.agents/technical-guidelines.md   engineering standards
```

---

## Notes and limitations

- **Voice cloning requires consent.** Clone only voices you have permission to
  use. Voice is biometric data under GDPR, and synthetic-dubbing disclosure is
  increasingly required by platforms. A consent gate is planned for P6.
- **Licence**: this project is Apache-2.0 (see `LICENSE`).
- **Hardware portability is deliberate, and tested without the hardware.** The
  dtype follows the GPU's compute capability rather than its name (bfloat16 is
  unsupported on Pascal and Volta, and choosing it unconditionally was the most
  likely "works for me, not for you" failure). `--device cuda:N` is honoured
  exactly, so a second GPU can be pinned. Metal and XPU are usable but never
  chosen automatically, because speech recognition runs on CTranslate2. The rules
  are covered by `tests/test_device.py` against fake hardware profiles, so a
  regression is caught on a CPU-only CI runner.
- **Model licences**: Qwen3-TTS and Whisper weights are Apache-2.0 / MIT.
  Avoid CPML-licensed models (e.g. XTTS-v2) if you need commercial rights.
- **Multi-speaker dubbing attributes speakers per segment**, not per word, so
  overlapping speech is shown as a flag on the segment. By default such a line is
dubbed in the voice of its dominant speaker and reported (`report.json` →
  `overlap`, `02c_overlap.json`); `--overlap separate` splits the two voices apart
  with SepFormer, re-transcribes each and dubs both. The speaker count comes from
  clustering rather than being declared. Both are tracked in `PLAN.md`.
- **Known MVP gaps**, all tracked in `PLAN.md`: lip sync, multi-language source
  videos, and iterative length control in the translation loop. Background
  preservation, native-accent borrowing and multi-speaker dubbing are
  implemented (see [Accent modes](#accent-modes) and
  [Dubbing a video with several speakers](#dubbing-a-video-with-several-speakers)).
- **Accent borrowing needs a supplied clip.** `reference-accent` takes
  pronunciation from `--accent-audio`; it does not yet choose or generate a
  native voice by itself. Fully unattended native accent is roadmap P2.
- **Room matching is off by default and adds no reverb unless asked.**
  `--room match` adds a room-tone floor and a gentler duck, both derived from the
  separated speech stem. A dry, close voice-over is the correct choice for narrated
  content, which is why it stays the default. The reverb tail is a separate, opt-in
  step (`--room-wet-db -20`): an earlier default of −12 dB was rejected on a real
  listen as an audible echo. `--no-background` removes the stem, so room matching
  then degrades to a dry dub and says so (`report.json` → `room.applied: false`).
- **Changing any setting re-runs the whole job** — the cache fingerprint is
  per-job, not per-stage. Fine for a CLI; worth refining before P6.
