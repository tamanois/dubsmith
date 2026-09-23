# AI Video Dubbing — MVP

Dub a **single-speaker** video into another language while keeping the speaker's
**own voice timbre**. Runs entirely on local, open models; no cloud required.

See [`PLAN.md`](PLAN.md) for the full vision, roadmap (P1–P6) and design
decisions, and [`AGENTS.md`](AGENTS.md) for the project context.

---

## MVP scope

| | |
|---|---|
| **Input** | One video, or one audio file |
| **Speakers** | Single speaker (multi-speaker is roadmap P1) |
| **Source language** | One language, auto-detected (multi-language is P5) |
| **Target language** | English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian |
| **Voice** | Cloned from the source speaker, **accent preserved** (native target accent is P2) |
| **Output** | Dubbed video/audio + source & translated SRT subtitles |
| **Not included** | Lip sync (P4), background music separation (P1) |

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
pip install -r requirements.txt
```

> If dependency resolution is awkward, install the two pins first — this is what
> `packages.md` refers to:
> ```bash
> pip install numpy==1.26.4 typing_extensions
> pip install -r requirements.txt
> ```

**GPU is optional.** With CUDA the pipeline uses `cuda` + `float16`; without it,
everything still runs on CPU (`int8` ASR, `float32` TTS) — just much slower.

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
`MODELS_DIR` is detected automatically. Override the location with
`MODELS_DIR` in `.env` or `--models-dir`.

---

## Usage

### Dub a video (auto-detect language and donor voice)

```bash
python dub.py --video talk.mp4 --target fr
```

The reference voice is taken automatically from the longest clean speech
segment, and its transcript comes from the ASR pass — so no manual setup needed.

### Dub an audio file

```bash
python dub.py --audio interview.wav --target es
```

Audio-only jobs need no `ffmpeg` and produce a dubbed WAV plus subtitles.

### Use an explicit reference voice

```bash
python dub.py --video talk.mp4 --target fr \
  --ref-audio "Me - Voice extract.wav" \
  --ref-text "Bonjour, je suis Joël, développeur cloud…"
```

`--ref-text` must be the **exact transcript** of `--ref-audio`; the model
conditions on both (ICL mode). Any format `libsndfile` reads works:
`.wav` `.flac` `.mp3` `.ogg` `.aiff`, a URL, or a `(samples, rate)` tuple.

### Common options

| Option | Default | Description |
|---|---|---|
| `--target` | `fr` | Target language (code or name) |
| `--source` | auto | Force the source language |
| `--device` | `auto` | `auto`, `cuda:0`, or `cpu` |
| `--stretcher` | `auto` | `auto`, `rubberband`, `sox`, `librosa`, `none` |
| `--min-stretch` | `0.85` | Most compression allowed to fit a slot (lower = faster speech, fewer words trimmed) |
| `--max-stretch` | `1.18` | Most slow-down allowed to fill a slot |
| `--output-dir` | `output` | Where job folders are written |
| `--no-subtitles` | off | Skip SRT generation |
| `--force` | off | Ignore cached stages and re-run everything |
| `-v` / `--verbose` | off | Debug logging |

```bash
python dub.py --help
python dub.py models --help
```

---

## Outputs

Everything for one job lands in `output/<input-stem>/`:

```
output/talk/
├── 01_audio.wav         # normalised 48 kHz mono source audio
├── 02_transcript.json   # ASR result: segments, words, timings, language
├── 03_script.json       # source -> target text per segment
├── 04_dubbed/           # one synthesised clip per segment (+ .done marker)
├── 05_mix.wav           # assembled dub track, before normalisation
├── timing.json          # per-segment stretch ratio and placement
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
| `loudness_target_lufs` | Target (default −16 LUFS) |
| `loudness_gain_db` | Gain applied to reach it |
| `loudness_peak_limited` | `true` if the target needed more gain than the peak allowed |
| `clamped_segments` | Lines that hit the ±15–20% stretch limit |
| `overrun_segments` | Lines that spilled into the pause after them (free, no speech lost) |
| `overrun_seconds_total` | Total speech fitted into those pauses |
| `trimmed_segments` | Lines where the residue had to be cut to keep the video in sync |
| `trimmed_seconds_total` | Seconds of speech lost to those cuts |
| `shifted_segments` | Lines pushed later to avoid overlapping another |

Two of these deserve attention:

- **`clamped_segments` above zero** means the translation was too long for its
  slot, so timing hit the stretch limit. Tighten the translation (or shorten the
  lines) rather than raising `--max-stretch`, which would sound unnatural.
- **`loudness_peak_limited: true`** means the dub landed quieter than the −16 LUFS
  target because reaching it would have clipped. Speech often has a high crest
  factor, so this is normal and safe — the alternative is distortion. For
  reference, a typical run delivers ~5 dB louder than the original track.
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

Stages are cached per job in `manifest.json`, keyed by a fingerprint of the job
settings. Re-running the same command does nothing; use `--force` to override.
Changing any setting (target language, donor, device, stretch bounds…) changes
the fingerprint and re-runs the affected job from scratch.

---

## How it works

```
video/audio ──► 1. audio extract ──► 2. ASR + VAD ──► 3. translate (LLM)
                                                              │
                  6. mux + SRT ◄── 5. fit to timeline ◄── 4. clone + synthesise
```

| # | Stage | Module | What it does |
|---|---|---|---|
| 1 | Audio | `pipeline/media.py` | ffmpeg extract (video) or direct read (audio), mono 48 kHz |
| 2 | ASR | `pipeline/asr.py` | faster-whisper + Silero VAD, word timestamps, drops non-speech |
| 3 | Translate | `pipeline/translate.py` | batched, context-aware, length-budgeted translation |
| 4 | Synthesise | `pipeline/tts.py` | donor selection + Qwen3-TTS voice cloning |
| 5 | Timing | `pipeline/timing.py` | per-segment stretch into the original slot, no drift |
| 6 | Deliver | `pipeline/orchestrate.py` | mix, loudness, subtitles, mux, report |

Two design points worth knowing:

- **Accent comes from the donor clip, timbre from cloning.** With no
  `--ref-audio`, the source speaker is the donor, so the accent is *preserved*.
  A native-accent dub is roadmap P2.
- **Timing errors never accumulate.** Each line is anchored to its own original
  start time; a line that does not fit spills into the pause after it and any
  residue beyond that is trimmed. Lines are never pushed late, so a long
  translation cannot make the rest of the video drift out of sync.

---

## Tests

All tests run **without** a GPU, downloaded models, or ffmpeg — heavy components
are injected as fakes and binaries are mocked.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

```
253 passed
```

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
| `does not support voice cloning` | Pointed at a CustomVoice/VoiceDesign checkpoint; use `tts_base` |
| `Unsupported language 'xx'` | Use one of the ten supported languages |
| CUDA out of memory | `--device cpu`, or free the GPU (a large local LLM often occupies it) |
| Dub sounds rushed or slow | Check `report.json` → `clamped_segments`; tighten the translation |
| Some words are cut off | `report.json` → `trimmed_segments`: the translation is too long for its slot. Shorten it, or lower `--min-stretch` |
| Dub drifts out of sync | Should not happen — lines are never shifted late. If `shifted_segments > 0`, check `timing.json` for genuinely overlapping source speech |

Logs are written to stderr and include a job id. Set `LOG_JSON=1` for
machine-readable logs.

---

## Project layout

```
dub.py                     CLI entry point
pipeline/
├── config.py              job + settings models, language normalisation
├── errors.py              domain exception hierarchy
├── logging_utils.py       job-scoped structured logging
├── models.py              model registry, download, path resolution
├── media.py               ffmpeg/ffprobe wrappers, loudness, audio I/O
├── subtitles.py           SRT rendering
├── asr.py                 transcription + segmentation
├── translate.py           OpenAI-compatible translation
├── tts.py                 donor selection + voice cloning
├── timing.py              stretching backends + timeline assembly
└── orchestrate.py         stages, caching, reporting
tests/                     pytest suite (227 tests)
models/                    downloaded weights (gitignored)
output/                    job artifacts (gitignored)
clone.py, main.py          standalone TTS demos kept for reference
PLAN.md                    roadmap: MVP + P1–P6
.agents/technical-guidelines.md   engineering standards
```

---

## Notes and limitations

- **Voice cloning requires consent.** Clone only voices you have permission to
  use. Voice is biometric data under GDPR, and synthetic-dubbing disclosure is
  increasingly required by platforms. A consent gate is planned for P6.
- **Model licences**: Qwen3-TTS and Whisper weights are Apache-2.0 / MIT.
  Avoid CPML-licensed models (e.g. XTTS-v2) if you need commercial rights.
- **Known MVP gaps**, all tracked in `PLAN.md`: multi-speaker video, background
  music preservation, native target accent, lip sync, multi-language sources,
  and iterative length control in the translation loop.
- **Changing any setting re-runs the whole job** — the cache fingerprint is
  per-job, not per-stage. Fine for a CLI; worth refining before P6.
