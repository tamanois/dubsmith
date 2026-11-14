# Getting started

Install it, download the models, and dub something — in the browser app or from
the terminal. For the shortest possible version, see
[Quick start](../README.md#quick-start).

Both surfaces share one install: the app is the same package plus the `web`
extra and a built frontend. Native install first; [Docker](#docker-optional) is
at the end of this page.

## Hardware requirements

| | |
|---|---|
| **Disk** | ~10 GB of model weights (`models/` is gitignored), plus your media and the generated accent anchors |
| **GPU** | Optional, but strongly recommended. ~12 GB VRAM runs the default path comfortably: each TTS checkpoint is ~3.5 GB in bfloat16 and Whisper large-v3 ~3 GB. The two TTS checkpoints are never resident together |
| **CPU only** | Works end to end and produces the same *result*. Measured on a 12-core box: a 22 s clip dubs in **2:10** (`preserve`) or **5:18** (`reference-accent`), with every quality gate passing. Synthesis is ~4.3× realtime in float32, so the real constraint is **RAM** (12.7–17.2 GB peak), not patience |
| **Python** | 3.10–3.12. `numpy==1.26.4` has no 3.13 wheels, so `pip install .` fails on 3.13 |

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

On macOS:

```bash
brew install ffmpeg rubberband sox
```

On Windows, use WSL2 and follow the Debian instructions. Native Windows is not
currently tested; `ffmpeg` must be on `PATH` either way.

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

Python **3.10–3.12** is required (developed and tested on 3.10.12; CI covers
3.10, 3.11 and 3.12).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .              # installs the package and the `dub` command
pip check
```

The package installs a `dub` command, so `python dub.py --target fr` and
`dub --target fr` are the same thing. `pip install -r requirements.txt` installs
only the dependencies: it is useful for a source checkout where you run
`python dub.py`, but it does not create the `dub` console script. Optional
features are extras:

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

> If dependency resolution is awkward, install the two pins first:
> ```bash
> pip install numpy==1.26.4 typing_extensions
> pip install -r requirements.txt
> ```

After installing, verify the CLI and the environment:

```bash
dub --version
dub devices      # which GPU(s) are visible, and the flag to select each one
dub info         # the device and precision a run would pick
dub models --list
```

If a checkpoint is gated on Hugging Face, authenticate once with `hf auth login`
(or set `HF_TOKEN`). For an offline machine, pre-download the weights and set
`HF_HUB_OFFLINE=1`.

**GPU is optional.** With CUDA or ROCm the pipeline runs there; without one
everything still runs on CPU, just much slower. `dub devices` lists every GPU
torch can see and the exact `--device` flag for each; `dub info` shows what a
run would pick. The supported families are CUDA (NVIDIA), ROCm (AMD, reported
as `cuda`), XPU (Intel) and Metal (Apple Silicon). Speech recognition always
uses CTranslate2, which offers CUDA and CPU only — so on Apple Silicon and
Intel XPU, ASR runs on the CPU while the other stages can use the GPU if you
ask for it. See [Devices and precision](performance.md#devices-and-precision).

Python packages installed by `requirements.txt`:

| Package | Role |
|---|---|
| `torch` | model runtime |
| `qwen_tts` | voice-cloning TTS |
| `faster-whisper` | ASR (bundles its own Silero VAD) |
| `huggingface-hub` | model download and cache resolution |
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
| `tts_cosyvoice3` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | cloned voice + per-line emotion/pace/volume (`--backend cosyvoice3`) |
| `tts_chatterbox` | `ResembleAI/chatterbox` | cloned voice + expression intensity (`--backend chatterbox`) |
| `tts_fish` | `fishaudio/s2-pro` | Fish Speech S2-Pro: cloned voice + inline tags (`--backend fish`); **non-commercial weights** |

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

## Start the app

The browser app is the same install plus FastAPI/uvicorn and a built frontend.
Node 20+ (CI uses 22) is needed only to build or develop it:

```bash
pip install ".[web]"
cd web && npm install && npm run build && cd ..
dub-server
# http://127.0.0.1:8000
```

Every way to start it, from the repository root:

| Command | Use it for |
|---|---|
| `dub-server` | the normal start: the API and the built UI on one port |
| `python -m server` | the same, from a checkout without the console script |
| `make serve` | the API with auto-reload — `uvicorn server.asgi:app --reload --reload-dir server --reload-dir pipeline` |
| `make web` | the frontend in development: Vite on <http://localhost:5173>, HMR, `/api` proxied to port 8000 |
| `make web-build` | write `web/dist`, which the single-process server serves at `/` |

The first three read `DUB_HOST` and `DUB_PORT`. The Vite server always listens on
5173 and proxies to 8000, so start the API first; it needs Node 22 (CI's
version) and `npm install` once in `web/`.

| Variable | Default | Meaning |
|---|---|---|
| `DUB_HOST` | `127.0.0.1` | Bind address. Keep it loopback unless you add authentication. |
| `DUB_PORT` | `8000` | HTTP port. |
| `DUB_DATA_DIR` | `data` | SQLite database, uploads and job metadata (gitignored). |
| `DUB_OUTPUT_DIR` | `output` | Root for the unique per-run output folders. |

`LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` come from the same `.env` as the
CLI; secrets are never written to the database.

Install **editable** (`pip install -e ".[web]"`) while working on the code. A
plain `pip install .` freezes a copy in `site-packages`, so the running server
ignores your changes and the app looks broken.

Open <http://127.0.0.1:8000>, upload a video on **Dub**, choose the target
language and press **Start dubbing**. Progress and the log stream live on
**Activity**, and the finished files are there too. The page-by-page tour,
presets and settings are in [Usage → The app](usage.md#the-app). If the page does
not load, see [Troubleshooting → The browser app](troubleshooting.md#the-browser-app).

---

## CosyVoice 3 (optional, expressive speech)

`dub say --backend cosyvoice3` adds per-line emotion/pace/volume instructions and
paralinguistic tokens to a cloned voice. It is optional; everything else works
without it.

CosyVoice 3 pins `transformers==4.51.3`, while Qwen TTS pins `4.57.3`. Loading
CosyVoice with the newer version produces garbled audio, so the engine runs
CosyVoice in a **child process** whose `PYTHONPATH` starts with a small
transformers-4.51.3 overlay. The main environment keeps Qwen's version.

**1. Check out CosyVoice (with Matcha-TTS) and install its inference deps**

```bash
mkdir -p third_party
git clone --depth 1 --recursive https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice

pip install \
  "diffusers==0.29.0" "conformer==0.3.2" "x-transformers==2.11.24" \
  "wetext==0.0.4" "inflect==7.3.1" "openai-whisper==20250625" \
  "lightning==2.2.4" "gdown==5.1.0" "matplotlib==3.7.5" \
  "pyarrow==18.1.0" "pyworld==0.3.4" "onnxruntime-gpu==1.18.0" \
  "hyperpyyaml==1.2.3" "hydra-core==1.3.2" "omegaconf==2.3.0" \
  "modelscope==1.20.0" "wget==3.2" "torchcodec" "setuptools<81"
```

A few of these are easy to miss:

- `hyperpyyaml`, `hydra-core`, `omegaconf`, `modelscope` and `wget` are imported
  by CosyVoice at load time but are not installed by the core project.
- `openai-whisper==20250625`, not the older pin in CosyVoice's own
  `requirements.txt`: releases up to 20231117 build with `pkg_resources`, which
  setuptools 81+ no longer ships.
- `setuptools<81` for the same reason at *runtime*: `lightning==2.2.4` imports
  `pkg_resources`. Keep the pin installed; re-running `pip install .` can
  upgrade setuptools again.
- `torchcodec` on torchaudio 2.9+ (which modern torch pulls in). CosyVoice calls
  `torchaudio.load(..., backend="soundfile")`, and recent torchaudio routes that
  through torchcodec instead. Install the matching CPU or CUDA build.

CosyVoice's own `requirements.txt` pins `torch==2.3.1`. The project accepts any
`torch>=2.1.0`, and CosyVoice runs in a child process that uses the main
environment's torch, so there is no need to downgrade it; do not install torch
2.3.1 into the Qwen environment. `onnxruntime-gpu` replaces the CPU
`onnxruntime` distribution; if its CUDA provider cannot load
(`libcublasLt.so.11`), the ONNX frontend falls back to the CPU and synthesis
still works.

**2. Build the transformers overlay**

```bash
pip install --target third_party/cosyvoice-overlay --no-deps \
  "transformers==4.51.3" "tokenizers==0.21.4"
```

**3. Download the checkpoint**

```bash
python dub.py models --download tts_cosyvoice3
```

The engine adds `third_party/cosyvoice-overlay`, `third_party/CosyVoice`,
`third_party/CosyVoice/third_party/Matcha-TTS` and the repo root to the child's
`PYTHONPATH` automatically; no `.pth` file or shell export is needed. Set
`COSYVOICE_PYTHONPATH` to prepend your own paths (for example another checkout).

CosyVoice 3 runs on CUDA or CPU and reads `CUDA_VISIBLE_DEVICES` at import, so
the engine maps `--device cuda:N` to it.

**Language coverage.** CosyVoice 3 is strongest in Chinese, English and Japanese,
and unlike the previous generation it covers French and a handful of other
European languages. See
[Voices](voices.md#expressive-backends-cosyvoice3-chatterbox-and-fish).

## Chatterbox (optional, expressive speech)

`dub say --backend chatterbox` or `dub --backend chatterbox` adds a single
expression-intensity control to a cloned voice. It is optional; the default Qwen
path works without it.

Chatterbox pins `torch==2.6` and `transformers==5.2`, which cannot coexist with
the Qwen checkpoint or the CosyVoice overlay. Keep it in its own environment at
`.venv-chatterbox` (or wherever `CHATTERBOX_PYTHON` points):

```bash
python3 -m venv .venv-chatterbox
.venv-chatterbox/bin/pip install torch==2.6.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
.venv-chatterbox/bin/pip install chatterbox-tts==0.1.7
python dub.py models --download tts_chatterbox   # weights, ~3 GB
```

On a CPU-only machine, install the CPU torch wheel instead of the CUDA index.
The backend runs in a child process; no `PYTHONPATH` export is needed.

## Fish Speech S2-Pro (optional, expressive speech)

`dub say --backend fish` or `dub --backend fish` clones a voice and reads
emotion/paralinguistic tags inline in the text. It is optional, and loads its
weights **int8 by default**, so it runs on a 12 GB card (~6 GB resident, ~11 GB
peak).

Fish pins `torch==2.8` and `transformers==5.19`, which conflict with the Qwen
checkpoint. Install it in its own environment:

```bash
python3 -m venv .venv-fish
git clone --depth 1 https://github.com/fishaudio/fish-speech.git third_party/fish-speech
```

Fish's `pyaudio` dependency is only for its microphone demo, and this pipeline
never records audio. Without PortAudio headers the plain
`.venv-fish/bin/pip install -e third_party/fish-speech` fails while building
`pyaudio`, so install the runtime dependencies (everything in Fish's
`pyproject.toml` except `pyaudio`) and then the checkout itself:

```bash
.venv-fish/bin/pip install \
  "numpy" "torch==2.8.0" "torchaudio==2.8.0" "transformers==5.19.0" \
  "hydra-core>=1.3.2" "natsort>=8.4.0" "einops>=0.7.0" \
  "librosa>=0.10.1" "kui>=1.6.0" "uvicorn>=0.30.0" "loguru>=0.6.0" \
  "pyrootutils>=1.0.4" "resampy>=0.4.3" "einx[torch]==0.2.2" \
  "pydub" "modelscope==1.17.1" "opencc-python-reimplemented==0.1.7" \
  "silero-vad" "ormsgpack" "tiktoken>=0.8.0" "pydantic==2.9.2" \
  "cachetools" "descript-audio-codec" "safetensors" \
  "tokenizers==0.23.2" "soundfile>=0.12.1"
.venv-fish/bin/pip install --no-deps -e third_party/fish-speech
.venv-fish/bin/pip install torchao==0.13.0   # int8 loading
python dub.py models --download tts_fish   # fishaudio/s2-pro
```

`pip check` will still report `fish-speech 3.0.0 requires pyaudio`; that is
expected on this path and harmless. If PortAudio **is** installed
(`sudo apt install portaudio19-dev`), replace the explicit dependency list with
`.venv-fish/bin/pip install -e third_party/fish-speech`, then run the `torchao`
line.

There is no int8 checkpoint to download: Fish publishes the model in bf16 only,
so the worker quantizes the 201 linear layers on the CPU after loading and only
then moves the model to the GPU. `--fish-quantize off` skips that and loads the
bf16 weights, which is what Fish documents as needing ~24 GB.

The backend runs `pipeline.fish_worker` with `.venv-fish/bin/python`; set
`FISH_PYTHON` to point at another environment.

---

## Docker (optional)

The native install above is the normal path. The image exists for portable,
reproducible machines and hosts where installing system packages is
inconvenient: it contains every runtime dependency, including the optional
backends, but **no model weights** — those are downloaded or mounted at runtime.

**What is in the image:** Debian bookworm with Python 3.10, `ffmpeg`/`ffprobe`,
`rubberband` and `sox`; the main environment with `.[full,gpu-asr]` (Qwen TTS,
faster-whisper, Demucs, funasr, SpeechBrain, the CUDA 12 libraries CTranslate2
needs and `torchcodec`); CosyVoice 3 in the main environment (checkout under
`third_party/CosyVoice` and the transformers 4.51 overlay); Chatterbox in
`/app/.venv-chatterbox`; Fish Speech in `/app/.venv-fish`. Nothing from
`models/`, `voices/`, `input/`, `output/` or `data/` is baked in.

```bash
docker build -t dubsmith .
```

The default image includes all three optional backends, so it is large (roughly
35–40 GB on disk) and slow to build. Slim it with the build args below.

```bash
# CPU-only: no CUDA torch wheels, still runs every path, just slower
docker build -t dubsmith:cpu \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  --build-arg CHATTERBOX_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  --build-arg FISH_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  --build-arg INSTALL_GPU_ASR=0 .

# without the three isolated expressive backends
docker build -t dubsmith:lean \
  --build-arg INSTALL_COSYVOICE=0 \
  --build-arg INSTALL_CHATTERBOX=0 \
  --build-arg INSTALL_FISH=0 .
```

| Build arg | Default | Effect |
|---|---|---|
| `TORCH_INDEX_URL` | empty (PyPI, CUDA wheel) | Pre-install the main `torch`/`torchaudio` from another index, e.g. the CPU index |
| `INSTALL_GPU_ASR` | `1` | Install `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` so CTranslate2 can use the GPU |
| `INSTALL_COSYVOICE` | `1` | Clone CosyVoice and install its runtime dependencies in the main environment |
| `INSTALL_CHATTERBOX` | `1` | Create `/app/.venv-chatterbox` |
| `CHATTERBOX_TORCH_INDEX_URL` | cu124 wheel index | Torch index used for Chatterbox |
| `INSTALL_FISH` | `1` | Create `/app/.venv-fish` and clone fish-speech |
| `FISH_TORCH_INDEX_URL` | empty (PyPI, CUDA wheel) | Pre-install Fish's `torch==2.8.0`/`torchaudio` from another index |
| `APP_UID` / `APP_GID` | `1000` | UID/GID of the runtime user |

The default PyPI torch wheels are CUDA builds, so a GPU host needs a driver
compatible with the bundled CUDA version. The `*_INDEX_URL` args accept any
PyTorch wheel index (`cpu`, `cu124`, `cu128`, `cu130`, …).

**Download the models once**, into directories you will mount:

```bash
mkdir -p models voices input output data

docker run --rm -v "$PWD/models:/app/models" dubsmith models --download all
```

**Dub on a GPU host:**

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/voices:/app/voices" \
  -v "$PWD/input:/app/input" \
  -v "$PWD/output:/app/output" \
  --env-file .env \
  dubsmith --video input/talk.mp4 --target fr
```

On a CPU host, drop `--gpus all`; the pipeline picks CPU automatically. Check
that the GPU is visible before a long run with
`docker run --rm --gpus all dubsmith devices` — `none - everything will run on
the CPU` means the container cannot see the card (check `--gpus`, the NVIDIA
Container Toolkit and the host driver).

**The app in Docker** needs the port published and the data directory mounted:

```bash
docker run --rm --gpus all -p 127.0.0.1:8000:8000 \
  -v "$PWD/models:/app/models" -v "$PWD/voices:/app/voices" \
  -v "$PWD/input:/app/input" -v "$PWD/output:/app/output" \
  -v "$PWD/data:/app/data" --env-file .env \
  dubsmith dub-server
```

The image builds the frontend, so `http://127.0.0.1:8000` serves the app. Set
`DUB_HOST=0.0.0.0` inside the container (the default binds loopback and is then
unreachable from the host).

Other entry points: replace the command with `say --voice …`,
`info`, `models --list`, or use `--entrypoint bash` for a shell.

**The LLM endpoint.** `--env-file .env` passes `LLM_BASE_URL`, `LLM_API_KEY` and
`LLM_MODEL` in. A `LLM_BASE_URL` of `http://127.0.0.1:8080/v1` points at the
**container**, not the host; when the model runs on the host:

```bash
docker run --rm --gpus all --add-host=host.docker.internal:host-gateway \
  -e LLM_BASE_URL=http://host.docker.internal:8080/v1 \
  ... dubsmith --video input/talk.mp4 --target fr
```

Docker Desktop provides `host.docker.internal` already; on Linux, `--network
host` is another option.

**Caches and permissions.** The image runs as the unprivileged `dubsmith` user
with UID/GID 1000. If your host UID differs, build with
`--build-arg APP_UID=$(id -u) --build-arg APP_GID=$(id -g)`, or pass
`--user "$(id -u):$(id -g)"` at run time — and with `--user`, also pass
`-e HOME=/tmp` so Demucs and Hugging Face can write their caches. Mounted
directories must be writable for outputs, models and anchors; mount
`-v "$PWD/.cache/docker:/home/dubsmith/.cache"` to keep Demucs/Hugging Face
downloads.

**Limits.** The image is built for `linux/amd64` — `onnxruntime-gpu` and some
audio extensions have no ARM wheels; on an ARM host use
`--platform linux/amd64` and expect emulation to be slow. Models are never
included, so a first run downloads them (or expects them in the mounted
`models/`). Docker is not required for normal use.

---

Next: [Usage](usage.md) for the app tour, the command chooser and worked
examples.
