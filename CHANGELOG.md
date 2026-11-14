# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Local browser UI and HTTP API (`server/`, `web/`): FastAPI + SQLite jobs,
  one heavy job at a time in a spawned process, SSE stage/log streaming, and a
  React + Vite + Tailwind interface covering dubbing, subtitles, speech
  generation, voices, models, devices, files and settings. Every job writes to a
  unique `<stem>__<operation>__<language>__<run-id>` folder, so outputs never
  overwrite. See `docs/usage.md` (the app) and `docs/reference.md` (HTTP API).
- Shared pipeline operations for the CLI and API: `pipeline/say.py`,
  `pipeline/voice_ops.py`, `pipeline/job_ops.py`, `pipeline/info.py`, plus
  structured device/environment reports and an optional stage-event hook on the
  dubbing and subtitles pipelines.
- Frontend regression tests (Vitest + Testing Library, 248 tests) cover the API
  wrapper and SSE reconnect, the UI primitives, uploads, live job logs,
  cancellation, retry/edit/compare and filtering, pagination and bulk actions on
  every list, model deletion, playback, presets, recent/favorite inputs, settings
  export/import, theme and defaults, hash routing, the upload picker and app
  navigation. They run without a GPU, network, or built assets.
- Actionable activity: finished jobs can be retried (a fresh run folder, the
  original untouched) or opened for editing with every stored option prefilled.
  The panel expands, and a completed dub can be compared against its source.
- Activity filtering and search, server-side pagination (20 per page by default)
  and bulk clearing of finished job records.
- A header defaults bar for the default device and precision, persisted with the
  other settings, plus a System / Light / Dark theme switch.
- Inline audio and video playback for uploads, outputs and saved voices, with
  path copy buttons; bulk deletion on Files.
- Model deletion from the Models page behind a danger confirmation, reporting the
  freed space; the endpoint refuses package-backed models and paths outside the
  models directory.
- Saved presets for the dub and speech forms, and recent/favorite input chips so a
  previously used clip is one click away.
- Disk usage in the UI, JSON settings export/import, completion toasts for
  background jobs, and a dismissible first-run getting-started card.
- `python dub.py devices` lists every GPU torch can see, its VRAM, the precision
  it would run, whether ASR can use it, and the exact flag to select it. It is
  read-only and loads no model, the same contract as `dub.py info`.
- `--accent-voice LABEL` takes the reference accent's clip and its exact
  transcript from a saved voice, so `--accent-mode reference-accent` no longer
  needs `--accent-audio` and `--accent-text` repeated by hand. It implies the
  mode and is rejected alongside either of those flags.

### Changed

- Option values now have a single declaration. `pipeline/options.py` owns every
  accepted value, its aliases and its default; the CLI's `argparse` choices, the
  config validators, the API request schemas and the web forms all read from it,
  and backend ids and labels come from `GET /api/backends`. The browser mirrors
  the vocabulary in `web/src/options.ts`, which derives every dropdown from the
  declaration. Renaming or adding a value is a two-file change instead of a
  search across every form, flag and validator; `tests/test_options.py` and
  `web/test/options.test.ts` fail when the copies disagree. No user-visible value
  changed.
- Main outputs are now named after the input stem (`talk_dubbed.mp4`,
  `talk_dubbed.wav`, `talk_subtitled.mp4`) instead of the generic
  `video_dubbed.mp4` / `dub.wav`, so a run folder is self-describing. The app's
  internal `<uuid>__` upload prefix is stripped from the delivered name, and the
  Files/Activity pages still resolve the old generic names for existing folders.
- The Dub and Say forms keep the common choices always visible and move the
  technical knobs under a collapsed **Advanced options** section; Presets moved
  to the bottom of each form, and fields gained explanatory tooltips.
- Activity and Files lead with the output file name and show the operation type
  as the subtitle. Every page has a URL hash (`#/dub`, `#/files`, …) for
  bookmarking and browser back/forward.

- Device names are validated when the command line is parsed: an unknown value
  such as `vulkan0` fails immediately with an explanation instead of crashing
  inside torch after the first stage. `rocm`/`rocm:N` is accepted as an alias for
  `cuda`/`cuda:N`, because that is the only spelling torch uses for an AMD card.
- `torch` and `torchaudio` are pinned to `2.5.1`, and the `torch` floor in the
  package metadata is raised to the same release. Below 2.5, transformers 4.57's
  sliding-window attention-mask builder writes in place onto an expanded tensor,
  which aborts Qwen3-TTS synthesis with a `RuntimeError` about a written-to
  tensor sharing a memory location.
- `GET /api/outputs` now returns one entry per run folder with that run's main
  artifact and a file count, instead of every intermediate artifact. The full
  list for one run is available from `GET /api/outputs/files?job_dir=…`, which
  the Files page exposes behind a Details toggle. A run that produced no
  deliverable (only stage files) is not listed.

### Fixed

- `POST /api/jobs/dub` accepts `accent_voice` on its own for
  `reference-accent`, as `--accent-voice` already did on the command line: the
  saved voice supplies the clip and its transcript, so the request is no longer
  rejected before the voice is resolved. Naming any other accent mode alongside
  it is an error rather than a silent override, and the Dub form only offers the
  control in that mode.
- The speech form refuses to submit when the chosen voice source is empty, and
  CustomVoice requires a preset speaker and an explicit language, instead of
  starting a job that fails validation.
- The web forms no longer offer a backend and option pair the pipeline cannot
  honour. `GET /api/backends` publishes the per-backend capability table (from
  `pipeline/capabilities.py`, the same source the Say validator uses), and the Dub
  and Say forms disable any control the selected backend cannot honour, print the
  reason, and drop the value from the payload — so Qwen Base no longer shows
  prosody fields it ignores. Combinations that span several fields (source equal
  to target, a reference clip without its transcript, a borrowed accent with no
  donor, accent matching outside native mode) show the reason above Submit and
  keep it disabled. `POST /api/jobs/say` answers `422` with the same message
  instead of enqueuing a job that fails in the worker.
- The job panel can no longer grow past the viewport and hide its own header and
  close button: the card is height-capped and its body scrolls, with the header
  pinned, and Escape closes it.
- Editing a dub job no longer loses its saved voice and accent voice, and the
  restored input is shown on the upload field instead of looking empty.
- Uploaded files are selectable for a job through a browse-all picker, not only
  the handful of recents; the chip band is capped with a "more" expander.
- The light-theme loader is visible again: the spinner now follows the
  surrounding text colour instead of painting white on white.
- Selecting Play shows the player inside its row and starts playback
  immediately (Files and Voices), rather than at the top of the page needing a
  second click.
- Job logs open expanded by default while the job is queued or running, and start
  collapsed for a job that had already finished (completed, failed or cancelled);
  a log opened by the user stays open when the job ends.
- The Files page lists generated outputs before uploaded inputs.
- Activity rows for finished jobs offer a direct Download and a Files link; the
  job panel adds "Download output" and "View in Files". Edit is hidden for voice
  jobs that have no editable form.

- The job panel's Cancel button now keeps working after the live SSE connection
  drops. Cancelling reflects the row the HTTP endpoint returns, and the stream
  reconnects with backoff, resuming from the last event id so no log line or
  stage change is lost. A regression test covers the dropped-stream path.
- Pagination no longer renders `NaN`: the API always returns a numeric `total`
  and the component refuses a non-finite one, so a stale server or partial
  response cannot break the pager.
- The Devices page loads devices, packages and storage independently; one failing
  optional endpoint no longer blanks the whole page.
- Playback controls only appear for files a browser can actually play. JSON,
  SRT and other artifacts list Download and Delete only.
- The header defaults are labelled "Device" and "Precision" instead of a single
  ambiguous "Default".
- The job panel now keeps exactly the latest 300 log lines instead of 301; the
  cap test caught the off-by-one in the live log buffer.
- The upload field's file input, disclosure buttons and toggles expose accessible
  names and expanded state, so keyboard and screen-reader users can identify
  them and the UI tests can target them reliably.
- The activity list and the completion watcher only refresh quickly while a job
  is running, drop to one request every 15 s when everything is finished, and
  stop entirely while the tab is hidden, so an idle browser no longer polls the
  API every 2.5 s. Returning to the tab refreshes immediately.
- `pip install .` (non-editable) made `dub-server` import a frozen snapshot from
  `site-packages`, so source changes never reached the running server and the UI
  hit 404s on `/api/usage` and `/api/presets` while showing `NaN` pagination.
  The install is now editable and `docs/getting-started.md` tells you to rebuild
  and restart after a backend change.

## [0.2.0] - 2026-11-09

### Added

- Before/after demo media in `docs/media/` (GIF, poster, source and dubbed
  excerpts) plus `scripts/make_demo.sh` to regenerate it.
- Expressive dubbing backends: `cosyvoice3`, `chatterbox`, and `fish`, selected
  with `--backend`; per-line delivery is planned from the translated text with
  `--prosody-plan auto|off|llm`.
- Optional `Dockerfile`, documented in `docs/getting-started.md`, for a portable
  image containing all dependencies (including the optional backends) but no
  model weights.
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
