# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-04

### Added

- Expressive dubbing backends: `cosyvoice3`, `chatterbox`, and `fish`, selected
  with `--backend`; per-line delivery is planned from the translated text with
  `--prosody-plan auto|off|llm`.
- Optional `Dockerfile` and `docs/docker.md` for a portable image containing all
  dependencies (including the optional backends) but no model weights.
- Standalone speech generation with `dub say`, including saved voices, inline
  voice design and cloning, and the `customvoice` preset-speaker backend.
- Voice library import with `dub voices --add`, optional transcription, voice
  cards under `prompts/`, and per-voice provenance.
- Emotion/prosody provenance for dubbing and `say` (`prosody.json` and
  `provenance/`), with `--no-provenance` as an escape hatch.
- Public-release documentation, Apache-2.0 NOTICE, contribution and security
  policies, model-license table, and CI checks.

### Changed

- Public identity renamed from the temporary `qwen-tts-dub` to **Dubsmith**;
  the `dub` CLI and the `pipeline` import path are unchanged.
- README now documents the real product scope: dubbing, subtitles, voice
  generation, expressive backends, emotions, voice cards, and provenance.
- Python support is explicitly 3.10–3.12; `numpy==1.26.4` has no 3.13 wheels.
- `huggingface-hub` is now a direct dependency.
- Root-level scratch scripts and the `packages.md` stub were removed.

### Fixed

- Dubbing no longer truncates the muxed video and audio to the start of the
  last subtitle cue. `mux_audio` caps the output at the video's own duration
  instead of relying on `-shortest`, which misread the SRT stream length.
- Passing a video container to `--audio` now raises an actionable `MediaError`
  instead of leaking a raw `LibsndfileError` traceback.
- `--prosody-plan auto` now records `plan_source: "llm"` in `prosody.json` when
  the plan succeeded; previously only `llm` and failures were recorded, so a
  successful automatic plan looked like planning was off.
- CosyVoice setup instructions now install every runtime import
  (`hyperpyyaml`, `hydra-core`, `omegaconf`, `modelscope`, `wget`), `torchcodec`
  for recent torchaudio, and `setuptools<81` for lightning 2.2.4. The whisper
  pin moves to 20250625 because older releases need `pkg_resources` to build.
- A missing CosyVoice dependency now names the module and points at the setup
  guide instead of printing a traceback.
- The Fish setup guide now provides the full PortAudio-free dependency list, so
  installing on a stock machine does not stop at the `pyaudio` build, and it
  explains that the resulting `pip check` warning is harmless.
- Documentation links, test counts, and setup instructions no longer point at
  removed anchors or private sample paths.
- Fish S2-Pro is clearly marked as non-commercial and remains optional.

## [0.1.0] - 2026-10-30

- Initial implementation of video dubbing, subtitles, multi-speaker support,
  accent modes, timing, mixing, quality gates, and the local model registry.
