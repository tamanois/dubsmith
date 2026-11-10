# Docker

The project runs natively and Docker is **optional**; this image is for
portable, reproducible machines and for hosts where installing system packages
is inconvenient. It contains every runtime dependency, including the optional
backends, but **no model weights** — those are downloaded or mounted at runtime.

## What is in the image

- Debian bookworm with Python 3.10
- `ffmpeg`/`ffprobe`, `rubberband` and `sox`
- The main environment with `.[full,gpu-asr]`: Qwen TTS, faster-whisper,
  Demucs, funasr and SpeechBrain, plus the CUDA 12 libraries CTranslate2 needs
  and `torchcodec` for recent torchaudio
- CosyVoice 3 in the main environment (checkout under `third_party/CosyVoice`
  and the transformers 4.51 overlay)
- Chatterbox in `/app/.venv-chatterbox`
- Fish Speech in `/app/.venv-fish` (checkout under `third_party/fish-speech`)

Nothing from `models/`, `voices/`, `input/` or `output/` is baked in.

## Build

```bash
docker build -t dubsmith .
```

The build needs network access (apt, PyPI, GitHub and the PyTorch wheel
indexes) and takes a while on the first run. Later builds reuse the dependency
layers.

The default image includes all three optional backends. It is large (roughly
35–40 GB on disk in this configuration, because Chatterbox and Fish each need
their own torch installation) and takes a while to build. Slim it with the build
args below if you do not need them.

### CPU-only image

```bash
docker build -t dubsmith:cpu \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  --build-arg CHATTERBOX_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  --build-arg FISH_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  --build-arg INSTALL_GPU_ASR=0 .
```

This avoids the CUDA torch wheels and is much smaller. It runs every path, just
slower.

### Without the optional speech backends

```bash
docker build -t dubsmith:lean \
  --build-arg INSTALL_COSYVOICE=0 \
  --build-arg INSTALL_CHATTERBOX=0 \
  --build-arg INSTALL_FISH=0 .
```

The main environment still includes the `full` extras (Demucs, funasr,
SpeechBrain); this only removes the three isolated expressive backends.

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

The default PyPI torch wheels are CUDA builds. A GPU host therefore needs a
driver compatible with the bundled CUDA version. On a CPU-only host the same
image still runs, using the CPU paths. The `*_INDEX_URL` args accept any
PyTorch wheel index (`cpu`, `cu124`, `cu128`, `cu130`, …), so you can match the
host driver or trade size for CUDA support.

## Download the models

Create the directories you will mount, then run the normal download command:

```bash
mkdir -p models voices input output

docker run --rm \
  -v "$PWD/models:/app/models" \
  dubsmith models --download all
```

Optional checkpoints work the same way:

```bash
docker run --rm -v "$PWD/models:/app/models" \
  dubsmith models --download tts_cosyvoice3
docker run --rm -v "$PWD/models:/app/models" \
  dubsmith models --download tts_chatterbox
docker run --rm -v "$PWD/models:/app/models" \
  dubsmith models --download tts_fish
```

## Run

The entry point is `dub`, so the first argument is the subcommand or a dubbing
flag.

GPU host:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/app/models" \
  -v "$PWD/voices:/app/voices" \
  -v "$PWD/input:/app/input" \
  -v "$PWD/output:/app/output" \
  --env-file .env \
  dubsmith --video input/talk.mp4 --target fr
```

CPU host: drop `--gpus all` (the pipeline picks CPU automatically).

Verify the GPU is visible inside the container before a long run:

```bash
docker run --rm --gpus all dubsmith devices
```

If it prints `none - everything will run on the CPU`, the container cannot see
the GPU: check `--gpus`, the NVIDIA Container Toolkit, and the host driver.

Speech generation and inspection:

```bash
docker run --rm -v "$PWD/models:/app/models" -v "$PWD/voices:/app/voices" -v "$PWD/output:/app/output" \
  dubsmith say --voice warm-fr --text "Bonjour, ceci est un test."

docker run --rm -v "$PWD/models:/app/models" dubsmith info
docker run --rm -v "$PWD/models:/app/models" dubsmith models --list
```

A shell in the image (the entry point is replaced):

```bash
docker run --rm -it --entrypoint bash \
  -v "$PWD/models:/app/models" -v "$PWD/voices:/app/voices" \
  -v "$PWD/input:/app/input" -v "$PWD/output:/app/output" \
  dubsmith
```

## The LLM endpoint

`--env-file .env` passes `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` into the
container. A `LLM_BASE_URL` of `http://127.0.0.1:8080/v1` points at the
**container**, not the host. If the LLM runs on the host:

```bash
docker run --rm --gpus all --add-host=host.docker.internal:host-gateway \
  -e LLM_BASE_URL=http://host.docker.internal:8080/v1 \
  ... dubsmith --video input/talk.mp4 --target fr
```

Docker Desktop provides `host.docker.internal` already. `--network host` is
another option on Linux.

## Caches, permissions and models

- The image runs as the unprivileged `dubsmith` user with UID/GID 1000. If your
  host UID differs, either build with
  `--build-arg APP_UID=$(id -u) --build-arg APP_GID=$(id -g)`, or pass
  `--user "$(id -u):$(id -g)"` at run time. When overriding `--user`, also pass
  `-e HOME=/tmp` so Demucs/Hugging Face can write their caches.
- Mounted directories must be writable by that user for outputs, downloaded
  models and generated anchors.
- Demucs and any Hugging Face downloads use
  `/home/dubsmith/.cache`. Mount it to persist them:
  `-v "$PWD/.cache/docker:/home/dubsmith/.cache"`.
- Model weights are local directories under `MODELS_DIR` (default
  `/app/models`), not the Hugging Face cache, when you use
  `dub models --download`.

## Limits

- The image is built for `linux/amd64`; `onnxruntime-gpu` and some audio
  extensions do not publish ARM wheels. On an ARM host, build with
  `--platform linux/amd64` (and expect emulation to be slow).
- The full image includes three separate torch installations, so it is large.
  Use the CPU or lean build for a smaller deployment.
- Models are never included. A first run still downloads the checkpoints (or
  expects them in the mounted `models/` directory).
- Docker is not required for normal use; see [Setup](setup.md) for the native
  install.
