# Setup

Full installation and configuration. For the copy-paste happy path, see
[Quick start](../README.md#quick-start).

## Hardware requirements

| | |
|---|---|
| **Disk** | ~10 GB of model weights (`models/` is gitignored), plus your media and the generated accent anchors |
| **GPU** | Optional, but strongly recommended. ~12 GB VRAM runs the default path comfortably: each TTS checkpoint is ~3.5 GB in bfloat16 and Whisper large-v3 ~3 GB. The two TTS checkpoints are never resident together |
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
> [`packages.md`](../packages.md) refers to:
> ```bash
> pip install numpy==1.26.4 typing_extensions
> pip install -r requirements.txt
> ```

**GPU is optional.** With CUDA or ROCm the pipeline runs there; without one
everything still runs on CPU, just much slower. Speech recognition always uses
CTranslate2, which offers CUDA and CPU only — so on Apple Silicon ASR runs on the
CPU while the other stages can use Metal if you ask for it. See
[Devices and precision](performance.md#devices-and-precision).

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
| `tts_voice_design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | ~3.5 GB |

The third checkpoint generates the **accent anchors** the default `native`
accent mode borrows pronunciation from, so it is required rather than optional.
Anchors are generated once per language and cached under `models/anchors/`.

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
| `tts_tokenizer` | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | standalone tokenizer (already bundled in `tts_base`) |

You can also copy a model directory in by hand — anything present under `MODELS_DIR`
is detected automatically. Override the location with `MODELS_DIR` in `.env`, which
every command then honours. The download command also takes `--models-dir`
directly:

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

Next: [Usage](usage.md) for the command chooser and worked examples.
