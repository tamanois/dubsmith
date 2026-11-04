# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-04

### Added

- Expressive dubbing backends: `cosyvoice3`, `chatterbox`, and `fish`, selected
  with `--backend`; per-line delivery is planned from the translated text with
  `--prosody-plan auto|off|llm`.
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

- Documentation links, test counts, and setup instructions no longer point at
  removed anchors or private sample paths.
- Fish S2-Pro is clearly marked as non-commercial and remains optional.

## [0.1.0] - 2026-10-30

- Initial implementation of video dubbing, subtitles, multi-speaker support,
  accent modes, timing, mixing, quality gates, and the local model registry.
