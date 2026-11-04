# Third-party licenses

This project is Apache-2.0. It depends on Python packages, optional packages and
model weights that are licensed separately. The notes below are a practical
summary, not legal advice; verify the upstream license before commercial use.

## Runtime Python packages

| Package | Role | License (upstream) |
|---|---|---|
| `numpy` | numerical arrays | BSD-3-Clause |
| `torch` | model runtime | BSD-3-Clause |
| `qwen-tts` | Qwen3-TTS runtime | Apache-2.0 |
| `faster-whisper` | ASR runtime | MIT |
| `ctranslate2` | ASR backend | MIT |
| `pydantic` | configuration validation | MIT |
| `pyloudnorm` | EBU R128 loudness | MIT |
| `openai` | OpenAI-compatible client | Apache-2.0 |
| `soundfile` | audio I/O | BSD-3-Clause |
| `librosa` | audio analysis | ISC |

## Optional Python packages

| Package | Feature | License (upstream) |
|---|---|---|
| `demucs` | background separation | MIT |
| `funasr` / `modelscope` | speaker diarization | MIT / Apache-2.0 |
| `speechbrain` | overlapping-speech separation | Apache-2.0 |
| `chatterbox-tts` | expressive TTS backend | MIT |
| `torchao` | Fish int8 loading | BSD-3-Clause |

## Model weights

Model weights are **not** included in this repository. They are downloaded by
`dub.py models --download` or by the backend itself.

| Model | Used for | License (upstream) |
|---|---|---|
| `Systran/faster-whisper-large-v3` | speech recognition | MIT |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | voice cloning | Apache-2.0 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | voice design / accent anchors | Apache-2.0 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | preset speakers / instructions | Apache-2.0 |
| `Qwen/Qwen3-TTS-Tokenizer-12Hz` | standalone tokenizer | Apache-2.0 |
| `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | expressive cloning | Apache-2.0 |
| `ResembleAI/chatterbox` | expressive cloning | MIT |
| `fishaudio/s2-pro` | expressive cloning / inline tags | **Fish Audio Research License (non-commercial)** |

The Fish backend is optional and is not required by the default dubbing path.
If you need commercial rights, do not enable the Fish backend unless you have a
separate license from Fish Audio.

## System binaries

These are not installed by `pip` and are not distributed in this repository:

| Binary | Used for | Notes |
|---|---|---|
| `ffmpeg` / `ffprobe` | video/audio extraction, muxing | Build-dependent: LGPL or GPL. Check your build. |
| `rubberband` | time stretching | GPL-2.0-or-later or commercial license from the vendor |
| `sox` | time stretching fallback | GPL-2.0 / LGPL components |

If you distribute a container or bundle that includes these binaries, review
their licenses and include the required notices and source offers.

## Documentation and media

Documentation is licensed CC BY 4.0 unless a file says otherwise. Demo media
must be public-domain or used with explicit consent; include attribution and a
license next to the asset.
