# Dubsmith documentation

Pick the path that matches what you want to do. Every path ends in the same
reference material, so nothing is a dead end.

## I want to dub a video

For the browser app — no flags to remember.

1. [Getting started](getting-started.md) — install, download the models, start the app
2. [Using the app](usage.md#the-app) — the pages, the options, presets, the activity log
3. [Troubleshooting → the browser app](troubleshooting.md#the-browser-app) — if the page or a job misbehaves

## I want to script or integrate it

1. [The command line](usage.md#the-command-line) — which command to run, and worked examples
2. [Reference](reference.md) — every flag, default, environment variable and HTTP route
3. [Outputs and caching](outputs-and-caching.md) — artifacts, `report.json`, cache rules

## I want better results

- [Voices](voices.md) — the voice library, cloning, accent modes, emotion, `dub say`
- [Dubbing quality](dub-quality.md) — several speakers, overlapping speech, timing, room, loudness
- [Performance](performance.md) — device and precision, several GPUs, long videos

## I want to change the code

1. [Engineering guidelines](development/guidelines.md) — layout, pipeline, tests, logging, backend rules
2. [Contributing](../CONTRIBUTING.md) — development setup, tests, DCO sign-off
3. [Security](../SECURITY.md) — private vulnerability reporting

## All pages

| Page | Read it when you want to… |
|---|---|
| [Getting started](getting-started.md) | install natively or with Docker, configure the LLM, download models, add optional backends |
| [Usage](usage.md) | learn the app page by page, or choose a command and copy an example |
| [Reference](reference.md) | look up a flag, default, environment variable or HTTP endpoint |
| [Voices](voices.md) | design, clone or import a voice; understand cards, accent modes and emotion |
| [Dubbing quality](dub-quality.md) | keep music, match the room, fit long translations, dub several speakers |
| [Performance](performance.md) | pick a device and precision, fan out over GPUs, survive a long video |
| [Outputs and caching](outputs-and-caching.md) | find an artifact, read `report.json`, gate a run with `audit_qc.py` |
| [Model licenses](model-licenses.md) | check code, weights and commercial-use boundaries |
| [Troubleshooting](troubleshooting.md) | diagnose an error, read the logs, know the limits |
| [Engineering guidelines](development/guidelines.md) | write code that fits the pipeline, tests and logging rules |

The before/after demo is in [media](media/README.md). Documentation is licensed
CC BY 4.0 unless a file says otherwise; see [the docs license](LICENSE.md).

Back to the [project README](../README.md), the [roadmap](../PLAN.md), or the
[contribution guide](../CONTRIBUTING.md).
