# Dubsmith documentation

Start with the [README](../README.md) for the short version, then pick a guide:
See the [before/after demo](media/README.md) for an 8-second English → French dub.
| Guide | Read it when you want to… |
|---|---|
| [Setup](setup.md) | install from scratch, configure the LLM endpoint, download models, install optional backends |
| [Docker](docker.md) | build a portable image with all dependencies (model weights excluded) |
| [Usage](usage.md) | choose a command, see worked examples, iterate quickly |
| [Web UI and API](web-ui.md) | run the browser interface, use the local HTTP API, understand jobs and SSE |
| [Options](options.md) | look up a flag, default, or environment variable |
| [Voices](voices.md) | design, clone, or import a saved voice; understand accent modes and cards |
| [Voice generation](voice-generation.md) | use `dub say`, expressive backends, and per-line emotion |
| [Model licenses](model-licenses.md) | check code, weights, and commercial-use boundaries |
| [Multi-speaker](multi-speaker.md) | dub interviews and debates, handle overlapping speech |
| [Audio and timing](audio-and-timing.md) | keep music, match the room, fit long translations |
| [Performance](performance.md) | pick a device/precision, use several GPUs, survive long videos |
| [Outputs and caching](outputs-and-caching.md) | find an artifact, read `report.json`/`prosody.json`, understand the cache |
| [Testing and quality](testing-and-quality.md) | run tests, audit timbre, gate a run automatically |
| [Architecture](architecture.md) | see the pipeline, module map and design decisions |
| [Troubleshooting](troubleshooting.md) | diagnose an error, read the logs, know the limits |
| [Engineering guidelines](development/guidelines.md) | code layout, tests, logging, subprocess and backend rules |
| [Contributing](../CONTRIBUTING.md) | set up a development environment, run tests, sign off with DCO |
| [Security](../SECURITY.md) | report a vulnerability privately |

Emotion for dubbed lines — per-line delivery planned by the LLM and rendered by
CosyVoice3, Chatterbox or Fish — is covered in
[Voice generation](voice-generation.md), under
[Options → Emotion](options.md#emotion-dubbing-backend-and-prosody-plan), with a
hands-on recipe in [Testing → Real dubs](testing-and-quality.md#real-dubs-against-the-local-checkpoints).

Return to the [project README](../README.md), the [roadmap](../PLAN.md), the
[engineering guidelines](development/guidelines.md), or the
[contribution guide](../CONTRIBUTING.md).
