# Dubsmith — AI dubbing + expressive voice generation

**Keep the voice. Change the language.**

Dub a video into another language while keeping each speaker's **own voice
timbre**, or generate speech in a saved or designed voice with per-line emotion.
The models run locally; the only network call is the configured
OpenAI-compatible LLM endpoint, which can be local or cloud.

[![CI](https://github.com/tamanois/dubsmith/actions/workflows/ci.yml/badge.svg)](https://github.com/tamanois/dubsmith/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](pyproject.toml)

[Documentation](docs/README.md) · [Roadmap](PLAN.md) · [Contributing](CONTRIBUTING.md) · [Model licenses](docs/model-licenses.md) · [Engineering guidelines](docs/development/guidelines.md)

![Before and after: the same speaker, dubbed from English into French with their own voice](docs/media/demo-before-after.gif)

**Before / after** — same speaker and timing, English → French, voice cloned from
the speaker's own audio.
[Source clip](docs/media/demo-before.mp4) · [Dubbed clip](docs/media/demo-after.mp4) ·
[How it was made](docs/media/README.md)

---

## What it does

| Mode | What you get |
|---|---|
| **Video dubbing** | Detect the language, separate speakers, clone each voice, translate, synthesise, and mux a new video/audio track. Ten target languages, native-accent anchors, background/room preservation, and multi-speaker support |
| **Subtitles only** | ASR + translation + SRT, with optional soft-embedded or burned-in subtitles. No TTS model is loaded |
| **Speech generation** | `dub say` turns text into speech with a saved, designed, cloned, or preset voice. Translation is not involved |
| **Voice library** | Design from a prompt, clone from a clip, or import an existing clip with `--add`; reuse by label in `say` or a dub. Qwen voice cards cache the encode; other backends read `voice.wav` |
| **Emotion and prosody** | Dubbing: `--backend cosyvoice3`, `chatterbox`, or `fish` with an LLM-planned `--prosody-plan`. Speech: manual `--prosody`, `--emotion`, `--style-instruct`, or `--prosody-plan` |
| **Quality and provenance** | `report.json`, `prosody.json`, timestamped `provenance/`, `audit_qc.py`, and `audit_timbre.py` |
| **Not included** | Lip sync (P4) and multi-language source videos (P5) |

---

## Quick start

From a fresh clone to a first dub or a first generated line.

**Requirements**

- Python **3.10–3.12** (`numpy==1.26.4` has no 3.13 wheels)
- `ffmpeg` and `ffprobe` for video input (audio-only needs neither)
- ~10 GB of disk for the core weights; optional expressive backends add ~3–11 GB each
- A GPU is optional but strongly recommended (~12 GB VRAM for the default path)
- An **OpenAI-compatible** LLM endpoint for translation and optional prosody planning

**1. Clone and install**

```bash
git clone https://github.com/tamanois/dubsmith.git
cd dubsmith
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

Optional extras (each degrades gracefully when absent):

```bash
pip install ".[full]"     # Demucs background separation, funasr diarization, SpeechBrain overlap
pip install ".[web]"      # FastAPI + uvicorn for the local browser UI
```

**2. Point the translator at a model**

```bash
cp .env.example .env
```

```dotenv
LLM_BASE_URL=http://127.0.0.1:8080/v1
LLM_API_KEY=not-needed
LLM_MODEL=local-model
```

Any OpenAI-compatible server works, including a local `llama.cpp`:
`llama-server -m /path/to/model.gguf --port 8080`.

**3. Download the models (~10 GB, once)**

```bash
python dub.py models --download all
```

**4. Check the machine**

```bash
python dub.py devices      # every GPU, its VRAM and the exact flag to use
python dub.py info          # device, precision, optional packages, models
python dub.py models --list
```

**5. Dub**

```bash
python dub.py --video talk.mp4 --target fr
```

The result is `output/talk/video_dubbed.mp4`, with `dub.wav`, subtitles and
`report.json` beside it. A first run also generates the French accent anchor,
cached under `models/anchors/`.

**6. Generate speech without a video**

```bash
python dub.py voices --design --prompt "a warm female French voice" --lang fr --label warm-fr
python dub.py say --voice warm-fr --text "Bonjour, ceci est un test."
```

`say` needs no translation step or LLM endpoint unless you use
`--prosody-plan llm` or `--style-plan llm`.

For the full setup matrix — CPU-only installs, CUDA 12 libraries for GPU ASR,
optional backends, macOS notes, `MODELS_DIR` — see [docs/setup.md](docs/setup.md).

Prefer containers? The repository ships an optional `Dockerfile` with every
dependency (models excluded); see [docs/docker.md](docs/docker.md).

### Browser UI

The same pipeline is available in a local, single-user browser app:

```bash
pip install ".[web]"
cd web && npm install && npm run build && cd ..
dub-server          # http://127.0.0.1:8000
```

It covers dubbing, subtitles, speech generation, the voice library, models,
devices, files and settings, with live stage progress and an expandable log.
Outputs go into unique per-run folders so they never overwrite each other. See
[docs/web-ui.md](docs/web-ui.md).

---

## Main commands

**Dub a video or audio file**

```bash
python dub.py --video talk.mp4 --target fr          # dubbed video + embedded soft subtitles
python dub.py --audio interview.wav --target es     # audio only, no ffmpeg
python dub.py --video talk.mp4 --source en --target de
python dub.py --video talk.mp4 --target fr --subtitles-mode burn   # burn text into frames
```

**Dub with per-line emotion**

`qwen` is the default dubbing backend: it clones the voice but has no prosody
channel. Pick an expressive backend and let the configured LLM plan each
translated line:

```bash
python dub.py --video talk.mp4 --target fr --backend chatterbox --prosody-plan auto
python dub.py --video talk.mp4 --target fr --backend cosyvoice3 --prosody-plan llm
python dub.py --video talk.mp4 --target fr --backend fish --prosody-plan auto

# keep the speaker's timbre but take a native target accent
python dub.py --video talk.mp4 --target fr --backend fish \
  --expressive-accent blend
```

`--prosody-plan auto` is the dubbing default: it plans when the LLM endpoint is
reachable and falls back to neutral otherwise. `off` speaks neutral; `llm` fails
if the endpoint is unreachable. `customvoice` is **speech-generation only** and
is not a dubbing backend. Expressive backends clone the source-language donor by
default, so they keep its accent; `--expressive-accent blend` builds the
target-language reference with Qwen Base first and then clones that.

**Add subtitles without dubbing**

```bash
python dub.py subtitles --video talk.mp4 --target fr
python dub.py subtitles --video talk.mp4 --target fr --subtitles-mode burn
python dub.py subtitles --audio interview.wav --target es            # SRT only
```

A subtitles-only run recognises speech, translates it, writes `subs_target.srt`,
and embeds it in the video. Existing subtitle tracks in other languages are kept;
a track already in the target language is replaced. No TTS model is loaded.

**Choose the voice**

```bash
# clone a specific recording (the transcript must match the audio exactly)
python dub.py --video talk.mp4 --target fr --ref-audio me.wav --ref-text "…"

# reuse a saved voice print
python dub.py --video talk.mp4 --target fr --voice warm-fr

# keep the speaker's timbre, borrow a native accent
python dub.py --video talk.mp4 --target fr \
  --accent-mode reference-accent --accent-audio native.mp3 --accent-text "…"

# keep the source accent
python dub.py --video talk.mp4 --target fr --accent-mode preserve
```

**Create and reuse a saved voice**

```bash
python dub.py voices --design --prompt "a warm female French voice" --lang fr --label warm-fr
python dub.py voices --clone --audio me.wav --clone-text "…" --lang fr --label me
python dub.py voices --add me.wav --transcribe --label me                     # no model, no GPU
python dub.py voices --list

python dub.py say --voice warm-fr --text "Bonjour, ceci est un test."
python dub.py say --voice warm-fr --backend cosyvoice3 --text "…" --prosody "emotion=happy, pace=fast, breath"
python dub.py say --voice warm-fr --backend chatterbox --text "…" --prosody "emotion=happy, exaggeration=0.8" --lang fr
python dub.py say --backend customvoice --speaker Vivian --lang en --text "…" --prosody "emotion=excited, pace=fast"
python dub.py say --backend customvoice --speaker Ryan --lang en --text "…" \
  --style-instruct "Speak with restrained anger, then fade to exhaustion."
python dub.py say --backend customvoice --speaker Ryan --lang en --text "…" \
  --style-plan llm
python dub.py say --voice warm-fr --backend fish --text "[excited] Je n'arrive pas à y croire !"
```

`--add` registers a clip verbatim, with no model or GPU. The first synthesis
writes a per-backend card under `voices/<label>/prompts/`; `provenance/` records
how the voice was made. See [docs/voices.md](docs/voices.md) for the full
library and card behavior.

**Standalone speech generation**

`dub say` takes text and exactly one voice source: `--voice`, `--design`,
`--clone`, or `--speaker` with `--backend customvoice`. It does not translate.

| Backend | Used by | Languages | Controls | Weights license |
|---|---|---|---|---|
| `qwen` (default) | dub + say | 10 | voice clone, accent modes | Apache-2.0 |
| `customvoice` | say only | 10 | preset speaker + instruction | Apache-2.0 |
| `cosyvoice3` | dub + say | 9, incl. French | emotion, pace, volume, `[breath]`, `[laughter]` | Apache-2.0 |
| `chatterbox` | dub + say | 23, incl. French | expression intensity | MIT |
| `fish` (S2-Pro) | dub + say | ~100 | inline `[tags]` in the text | **non-commercial** |

The default path does not require Fish. See
[docs/model-licenses.md](docs/model-licenses.md) before commercial use.

CustomVoice speakers: `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric` (Chinese),
`Ryan`, `Aiden` (English), `Ono_Anna` (Japanese), `Sohee` (Korean).

`--prosody` is the structured per-line control for `say` (repeat once per line,
or pass one value to apply everywhere):

| Field | Accepted values |
|---|---|
| `emotion` | free-form; CosyVoice uses the trained `happy`/`sad`/`angry` (aliases map) |
| `pace` | `normal`, `slow`, `fast` |
| `volume` | `normal`, `soft`, `loud` |
| `exaggeration` | `0.0`–`1.0` (Chatterbox's own intensity knob) |
| `breath`, `pause`, `laughter` | bare flags |

For `say`, `--prosody-plan` defaults to `off`; `llm` asks the configured model to
plan delivery from the text. For dubbing, `--prosody-plan auto` is the default.
An explicit `--prosody`/`--emotion` overrides the plan. `--style-instruct TEXT`
(`customvoice` only) sends a full delivery instruction verbatim and wins over
`--prosody`. `--style-plan llm` (`customvoice` only) has the model write that
instruction for you, one per line; `--style-plan-scope whole` writes a single
instruction for the entire text. Above `LLM_PLAN_CHUNK_SIZE` lines (default 10,
or `--plan-chunk`) it first reads the whole text for an overview, then plans each
chunk against it. `--instruct` is an alias of `--emotion`, not of
`--style-instruct`. Per-backend limitations and the full value lists are in
[docs/voices.md](docs/voices.md).

**Multi-speaker videos**

```bash
python dub.py --video debate.mp4 --target fr            # one voice per speaker
python dub.py --video debate.mp4 --target fr --overlap separate
```

**Inspect and audit a run**

```bash
cat "output/talk/report.json"                  # the quickest way to judge a run
python audit_qc.py "output/talk"               # pass/fail gates, no model
python audit_qc.py "output/talk" --timbre      # adds the per-line timbre gate
```

**Models and the environment**

```bash
python dub.py models --list
python dub.py models --download everything   # include optional backends
python dub.py info
```

Every flag and default is in [docs/options.md](docs/options.md);
`python dub.py --help` prints the same list.

---

## Tests

The suite needs no GPU, downloaded models or ffmpeg — heavy components are faked.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

```text
1485 tests (all passing)
```

Run one area while iterating:

```bash
python -m pytest tests/test_tts.py -q
python -m pytest tests/test_pipeline.py -q
python -m pytest tests/test_voices.py -q
python -m pytest tests/test_provenance.py -q
python -m pytest -k accent -q
```

The quality tools and their gates are covered in
[docs/testing-and-quality.md](docs/testing-and-quality.md).

---

## Documentation

| Guide | What it covers |
|---|---|
| [Setup](docs/setup.md) | prerequisites, install, LLM endpoint, models, optional backends |
| [Usage](docs/usage.md) | command chooser, worked examples, fast iteration |
| [Options](docs/options.md) | every flag, default, and environment variable |
| [Voices](docs/voices.md) | voice library, cloning, cards, accent modes |
| [Voice generation](docs/voice-generation.md) | `dub say`, voice design/clone/import, expression |
| [Multi-speaker](docs/multi-speaker.md) | diarization, speaker thresholds, overlap |
| [Audio and timing](docs/audio-and-timing.md) | music/ambience, room matching, long translations |
| [Performance](docs/performance.md) | devices, precision, several GPUs, long-video memory |
| [Outputs and caching](docs/outputs-and-caching.md) | job artifacts, `report.json`, `prosody.json`, provenance |
| [Testing and quality](docs/testing-and-quality.md) | tests, timbre audit, `audit_qc.py` |
| [Model licenses](docs/model-licenses.md) | code, weights, and commercial-use boundaries |
| [Architecture](docs/architecture.md) | pipeline, module map, project layout |
| [Troubleshooting](docs/troubleshooting.md) | errors, logs, notes and limitations |
| [Contributing](CONTRIBUTING.md) | development setup, tests, DCO, docs policy |
| [Security](SECURITY.md) | private vulnerability reporting and scope |

The [docs index](docs/README.md) has the same list with reading paths.

---

## Project layout

```text
dub.py                  CLI entry point (dub, models, voices, say, subtitles, anchors, info)
audit_qc.py             automated quality gate for a finished job
audit_timbre.py         per-line speaker-timbre audit
pipeline/               ASR, translation, TTS, expressive backends, timing, mixing, orchestration
  backends.py           CosyVoice3 / Chatterbox / Fish engine construction
  provenance.py         per-line emotion and text-transformation record
tests/                  pytest suite (no GPU or models needed)
models/                 downloaded weights (gitignored)
voices/                 saved voices, cards and provenance (gitignored)
output/                 job artifacts (gitignored)
docs/                   user documentation, model licenses, development guidelines
docs/media/             before/after demo GIF, poster and clips
scripts/make_demo.sh    regenerate the demo GIF and poster with ffmpeg
PLAN.md                 vision, roadmap and design decisions
NOTICE                  Apache-2.0 attribution
CONTRIBUTING.md         DCO, development setup, docs policy
SECURITY.md             private vulnerability reporting
```

Full module map: [docs/architecture.md](docs/architecture.md).

---

## License and responsible use

- **Code**: Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
- **Docs**: CC BY 4.0 unless a file says otherwise; see
  [docs/LICENSE.md](docs/LICENSE.md).
- **Model weights are separate.** Qwen3-TTS and Whisper are Apache-2.0 / MIT;
  CosyVoice3 is Apache-2.0; Chatterbox is MIT; **Fish S2-Pro is
  non-commercial**. The default path does not use Fish. See
  [docs/model-licenses.md](docs/model-licenses.md) and
  [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
- **Consent**: clone only voices you have permission to use. Voice is biometric
  data under GDPR, and synthetic-dubbing disclosure is increasingly required.
- **Privacy**: there is no telemetry. Job artifacts contain transcripts and can
  contain reference audio; `voices/`, `input/`, `output/` and `models/` are
  gitignored. Translation and optional prosody planning send text to the
  configured OpenAI-compatible LLM endpoint.
- **Known gaps** are tracked in [PLAN.md](PLAN.md): lip sync, multi-language
  source videos, and iterative length control in the translation loop.
