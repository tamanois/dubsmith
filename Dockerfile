# syntax=docker/dockerfile:1.7
#
# Dubsmith in a container. Model weights are deliberately NOT baked in: mount a
# models/ directory (and voices/, input/, output/) at runtime. See
# docs/docker.md for build and run examples.
#
# The default build includes every optional backend, because they pin
# conflicting torch/transformers versions and are kept in isolated environments
# here. Use the INSTALL_* build args to slim the image.
#
#   docker build -t dubsmith .
#   docker build -t dubsmith:cpu \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
#     --build-arg CHATTERBOX_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
#     --build-arg FISH_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
#     --build-arg INSTALL_GPU_ASR=0
#
# Runtime (GPU):
#   docker run --rm --gpus all \
#     -v "$PWD/models:/app/models" -v "$PWD/voices:/app/voices" \
#     -v "$PWD/input:/app/input" -v "$PWD/output:/app/output" \
#     --env-file .env dubsmith --video input/talk.mp4 --target fr

FROM python:3.10-slim-bookworm

LABEL org.opencontainers.image.title="Dubsmith" \
      org.opencontainers.image.description="AI dubbing and expressive voice generation with local voice cloning" \
      org.opencontainers.image.source="https://github.com/tamanois/dubsmith" \
      org.opencontainers.image.licenses="Apache-2.0"

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODELS_DIR=/app/models \
    VOICES_DIR=/app/voices

# ffmpeg/ffprobe are required for video; rubberband and sox are the optional
# stretchers. build-essential covers source builds of small audio extensions
# (pyworld) on architectures without a wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        rubberband-cli \
        sox \
        libsox-fmt-all \
        libsndfile1 \
        libgomp1 \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Optional torch pre-install. Leave empty to let the project's dependency
# resolution pull the default (CUDA) wheel; set it to the CPU index for a much
# smaller image that still runs every non-GPU path.
ARG TORCH_INDEX_URL=""
RUN set -eu; \
    if [ -n "$TORCH_INDEX_URL" ]; then \
        pip install --index-url "$TORCH_INDEX_URL" \
            --extra-index-url https://pypi.org/simple \
            torch torchaudio; \
    fi

# Copy only what the editable install needs, so ordinary source edits do not
# invalidate the (large) dependency layer.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY pipeline ./pipeline
COPY dub.py audit_qc.py audit_timbre.py ./

# `full` adds Demucs, funasr and SpeechBrain; `gpu-asr` adds the CUDA 12
# cublas/cudnn wheels that CTranslate2 needs on modern torch installations.
ARG INSTALL_GPU_ASR=1
RUN set -eu; \
    extras="full"; \
    if [ "$INSTALL_GPU_ASR" = "1" ]; then extras="full,gpu-asr"; fi; \
    pip install -e ".[$extras]"

# torchaudio 2.9+ routes load() through torchcodec. CosyVoice needs it for its
# reference audio, and Demucs/SpeechBrain use the same loader even when
# CosyVoice is not installed.
RUN pip install "torchcodec"

# ---------------------------------------------------------------------------
# Optional backends. Each installs into the location the engine already probes
# relative to the project root: the main environment for CosyVoice3, and the
# isolated .venv-chatterbox / .venv-fish environments for the other two.
# ---------------------------------------------------------------------------

ARG INSTALL_COSYVOICE=1
RUN set -eu; \
    if [ "$INSTALL_COSYVOICE" = "1" ]; then \
        mkdir -p third_party; \
        git clone --depth 1 --recursive \
            https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice; \
        pip install --target third_party/cosyvoice-overlay --no-deps \
            "transformers==4.51.3" "tokenizers==0.21.4"; \
        pip install \
            "diffusers==0.29.0" "conformer==0.3.2" "x-transformers==2.11.24" \
            "wetext==0.0.4" "inflect==7.3.1" "openai-whisper==20250625" \
            "lightning==2.2.4" "gdown==5.1.0" "matplotlib==3.7.5" \
            "pyarrow==18.1.0" "pyworld==0.3.4" "onnxruntime-gpu==1.18.0" \
            "hyperpyyaml==1.2.3" "hydra-core==1.3.2" "omegaconf==2.3.0" \
            "modelscope==1.20.0" "wget==3.2" "torchcodec" "setuptools<81"; \
    fi

ARG INSTALL_CHATTERBOX=1
ARG CHATTERBOX_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN set -eu; \
    if [ "$INSTALL_CHATTERBOX" = "1" ]; then \
        python3 -m venv .venv-chatterbox; \
        .venv-chatterbox/bin/pip install \
            --index-url "$CHATTERBOX_TORCH_INDEX_URL" \
            --extra-index-url https://pypi.org/simple \
            "torch==2.6.0" "torchaudio==2.6.0"; \
        .venv-chatterbox/bin/pip install "chatterbox-tts==0.1.7"; \
    fi

ARG INSTALL_FISH=1
# Leave empty for Fish's default (CUDA) wheel; set the CPU index for a CPU-only
# image. The rest of Fish's dependencies always come from PyPI.
ARG FISH_TORCH_INDEX_URL=""
RUN set -eu; \
    if [ "$INSTALL_FISH" = "1" ]; then \
        git clone --depth 1 \
            https://github.com/fishaudio/fish-speech.git third_party/fish-speech; \
        python3 -m venv .venv-fish; \
        if [ -n "$FISH_TORCH_INDEX_URL" ]; then \
            .venv-fish/bin/pip install \
                --index-url "$FISH_TORCH_INDEX_URL" \
                --extra-index-url https://pypi.org/simple \
                "torch==2.8.0" "torchaudio==2.8.0"; \
        fi; \
        .venv-fish/bin/pip install \
            "numpy" "torch==2.8.0" "torchaudio==2.8.0" \
            "transformers==5.19.0" "hydra-core>=1.3.2" "natsort>=8.4.0" \
            "einops>=0.7.0" "librosa>=0.10.1" "kui>=1.6.0" \
            "uvicorn>=0.30.0" "loguru>=0.6.0" "pyrootutils>=1.0.4" \
            "resampy>=0.4.3" "einx[torch]==0.2.2" "pydub" \
            "modelscope==1.17.1" "opencc-python-reimplemented==0.1.7" \
            "silero-vad" "ormsgpack" "tiktoken>=0.8.0" \
            "pydantic==2.9.2" "cachetools" "descript-audio-codec" \
            "safetensors" "tokenizers==0.23.2" "soundfile>=0.12.1"; \
        .venv-fish/bin/pip install --no-deps -e third_party/fish-speech; \
        .venv-fish/bin/pip install "torchao==0.13.0"; \
    fi

# ---------------------------------------------------------------------------
# Runtime user and entry point
# ---------------------------------------------------------------------------

ARG APP_UID=1000
ARG APP_GID=1000
# Only the runtime data directories need to be writable. Do not chown -R /app:
# that copies every installed package and venv file into a new image layer for
# no benefit.
RUN set -eu; \
    groupadd --gid "$APP_GID" dubsmith; \
    useradd --uid "$APP_UID" --gid "$APP_GID" --create-home \
        --shell /bin/bash dubsmith; \
    mkdir -p /app/models /app/voices /app/input /app/output; \
    chown dubsmith:dubsmith /app /app/models /app/voices /app/input /app/output

ENV HOME=/home/dubsmith \
    XDG_CACHE_HOME=/home/dubsmith/.cache \
    HF_HOME=/home/dubsmith/.cache/huggingface \
    TORCH_HOME=/home/dubsmith/.cache/torch \
    COSYVOICE_PYTHONPATH=/app/third_party/cosyvoice-overlay:/app/third_party/CosyVoice:/app/third_party/CosyVoice/third_party/Matcha-TTS:/app

USER dubsmith

# `dub` is the console script installed above. Pass any subcommand:
#   docker run ... dubsmith --video input/talk.mp4 --target fr
#   docker run ... dubsmith say --voice warm-fr --text "Bonjour"
#   docker run ... dubsmith models --download all
ENTRYPOINT ["dub"]
CMD ["--help"]
