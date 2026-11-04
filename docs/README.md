# qwen-tts documentation

Start with the [README](../README.md) for the short version, then pick a guide:

| Guide | Read it when you want to… |
|---|---|
| [Setup](setup.md) | install from scratch, configure the LLM endpoint, download models |
| [Usage](usage.md) | choose a command, see worked examples, iterate quickly |
| [Options](options.md) | look up a flag and its default |
| [Voices](voices.md) | design or clone a saved voice, understand accent modes |
| [Multi-speaker](multi-speaker.md) | dub interviews and debates, handle overlapping speech |
| [Audio and timing](audio-and-timing.md) | keep music, match the room, fit long translations |
| [Performance](performance.md) | pick a device/precision, use several GPUs, survive long videos |
| [Outputs and caching](outputs-and-caching.md) | find an artifact, read `report.json`, understand the cache |
| [Testing and quality](testing-and-quality.md) | run tests, audit timbre, gate a run automatically |
| [Architecture](architecture.md) | see the pipeline, module map and design decisions |
| [Troubleshooting](troubleshooting.md) | diagnose an error, read the logs, know the limits |

Emotion for dubbed lines — per-line delivery planned by the LLM and rendered by
CosyVoice3, Chatterbox or Fish — is covered under
[Options → Emotion](options.md#emotion-dubbing-backend-and-prosody-plan), with a
hands-on recipe in [Testing → Real dubs](testing-and-quality.md#real-dubs-against-the-local-checkpoints).

Return to the [project README](../README.md), the [roadmap](../PLAN.md), or the
[engineering guidelines](../.agents/technical-guidelines.md).
