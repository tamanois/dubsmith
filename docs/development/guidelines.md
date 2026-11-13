# Engineering guidelines

These are the conventions this codebase actually follows. They replace the
earlier generic "production service" guidance, which described an API/queue
architecture that this project does not use.

## Architecture

- The importable core is the flat `pipeline/` package. `dub.py` is the CLI and
  maps flags to `JobConfig`/`Settings`; `server/` is the local HTTP API and
  `web/` is the React single-page app. The CLI and the API both call `pipeline/`
  and must never duplicate media, TTS or translation logic.
- The API is single-user and local by design: SQLite under `data/` (gitignored)
  stores jobs, settings and file metadata, one heavy job runs at a time in a
  spawned process, and stage events plus log lines stream to the browser over
  SSE. Long work never blocks request handling.
- Multi-user, remote workers and URL inputs are future additions behind three
  seams that already exist: a `user_id` column on owned rows, the
  `server/executor.py::JobExecutor` protocol, and the single input-path
  validator in `server/app.py`. Do not close those doors.
- Heavy stages live behind small engine/protocol boundaries:
  - `pipeline/tts.py` for Qwen Base cloning and CustomVoice
  - `pipeline/backends.py` for CosyVoice3, Chatterbox and Fish
  - `pipeline/asr.py`, `pipeline/diarize.py`, `pipeline/separation.py`,
    `pipeline/overlap_separation.py`, `pipeline/translate.py`
  - `pipeline/orchestrate.py` owns stage order, caching, artifacts and reporting
- Optional dependencies must degrade gracefully. A missing backend should produce
  an actionable `DependencyError` or a documented fallback, never a raw traceback.
- Backends that pin conflicting `torch`/`transformers` versions run in a child
  process (`*_worker.py`); do not import them into the main process.
- What a backend can honour is declared once, in `pipeline/capabilities.py`.
  Validators and `GET /api/backends` both read it, so never re-derive
  backend/option compatibility from string comparisons elsewhere; a new backend
  or rule belongs in that table.

## Configuration and data

- `pipeline/config.py` uses pydantic models. `Settings` comes from environment
  variables and `.env`; `JobConfig` describes one run and feeds the cache
  fingerprint.
- Every accepted option value is declared once, in `pipeline/options.py`. The
  CLI hands those tuples to `argparse`, the models validate against the alias
  tables built from them, and `server/schemas.py` exposes the enums. Never
  retype a value in another validator, CLI flag or test.
- The browser mirrors the vocabulary in `web/src/options.ts`, which derives each
  `{ value, label }` list from the declaration. Backend ids and labels come from
  `GET /api/backends`, not from a second frontend list.
- To rename or add a value: change `pipeline/options.py`, mirror it in
  `web/src/options.ts`, then update `docs/options.md`. `tests/test_options.py`
  and `web/test/options.test.ts` fail if the copies disagree.
- Real environment variables override `.env`.
- Never commit `models/`, `voices/`, `input/`, `output/`, `third_party/`, or
  `.env`. They are local data, often private.
- Voice prints are biometric data. Do not log or publish reference audio,
  transcripts, or full LLM requests without consent.

## Error handling and logging

- Raise domain errors from `pipeline/errors.py`; the CLI turns them into a clear
  message and exit code.
- No bare `except:`. Catch specific exceptions; use a `# noqa: BLE001` only when
  a best-effort path is deliberate and the comment explains why.
- Use stdlib logging with the job-scoped logger from `pipeline/logging_utils.py`.
  Every run should be separable in `run.log` and `report.json`.
- `-v` may include transcript text and full LLM requests. Treat it as private.

## Subprocess and security

- Use argument lists with `shell=False` for ffmpeg, rubberband, sox and backend
  workers. Never interpolate user text into a shell string.
- Read secrets from the environment or `.env`; never write them to artifacts.
- Validate paths that come from users; keep job outputs under the configured
  output root.

## Tests

- Pytest is the only test runner. The default suite must not need a GPU,
  downloaded models, or ffmpeg; fake the heavy pieces.
- Server tests use `create_app(data_dir=tmp_path, executor=FakeExecutor)` so the
  state machine, uploads, settings and SSE plumbing run without spawning models.
- Optional backends are exercised through fakes and `pytest.importorskip`.
- Add tests for failure paths and edge cases, not only the happy path.
- Keep tests deterministic; no network calls.
- `python -m pytest` is the release gate.

## Documentation

- Every user-visible flag belongs in `docs/options.md`.
- The README must only claim capabilities implemented by the CLI, a test, or a
  documented module. Update it in the same change as the behavior.
- Update `docs/voices.md` / `docs/voice-generation.md` for backend or emotion
  changes; `docs/outputs-and-caching.md` for new artifacts or report fields;
  `docs/troubleshooting.md` for new error messages; `docs/web-ui.md` for the
  browser UI or API surface.
- Keep `PLAN.md` status and test counts current.

## Style

- Target Python 3.10; CI covers 3.10–3.12.
- Keep lines readable; the codebase generally stays at or below 88 columns.
- Prefer early returns and small pure functions for data transforms.
- Use type hints on public functions and dataclasses.
- Run `ruff check` before opening a PR. Repo-wide formatting and mypy are
  separate follow-ups, not hidden requirements.

## Backends and models

- A new backend needs: a license review, a VRAM/disk estimate, a graceful
  missing-dependency path, tests with fakes, CLI help, and docs.
- Do not reintroduce removed backends (IndexTTS2, CosyVoice2) without a decision.
- Fish S2-Pro weights are non-commercial. Keep the backend optional and label it
  clearly; the default path must not require it.
