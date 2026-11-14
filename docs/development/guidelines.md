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

## Pipeline and module map

```text
video/audio ─► 1. audio extract ─► 2. separate ─► 3. ASR + VAD ─► 4. diarize ─┐
                               (speech | bed)                               │
  8. mux + SRT ◄─ 7. mix + duck ◄─ 6. clone + synthesise ◄─ 5. translate ◄──┘
```

| # | Stage | Module | What it does |
|---|---|---|---|
| 1 | Audio | `pipeline/media.py` | ffmpeg extract (video) or direct read (audio), mono 48 kHz |
| 2 | Separate | `pipeline/separation.py` | Demucs splits speech from music/ambience; ASR reads the clean speech stem |
| 3 | ASR | `pipeline/asr.py` | faster-whisper + Silero VAD, word timestamps, drops non-speech |
| 4 | Diarize | `pipeline/diarize.py` | embeds each segment and clusters the embeddings into speakers |
| 4b | Overlap | `pipeline/overlap_separation.py` | with `--overlap separate`, splits a flagged segment into per-speaker streams |
| 5 | Translate | `pipeline/translate.py` | batched, context-aware, length-budgeted translation |
| 6 | Synthesise | `pipeline/tts.py`, `pipeline/backends.py` | donor selection + cloning, grouped by speaker, spread over `--devices` |
| 7 | Mix | `pipeline/timing.py`, `pipeline/mixing.py` | fit each line to its slot, lay it over the ducked bed, optionally match the room |
| 8 | Deliver | `pipeline/orchestrate.py` | loudness match, subtitles, mux, report |

Design points that are easy to break by accident:

- **Accent comes from an anchor, timbre from cloning.** `native` (default) borrows
  pronunciation from a generated per-language anchor; `preserve` keeps the source
  accent, `speaker-only` drops the borrowed phonetics, `reference-accent` uses a
  clip you supply.
- **One anchor per language, generated once.** The VoiceDesign checkpoint turns a
  short passage into native speech, cached under `models/anchors/`.
- **One prompt per voice, not per line.** Synthesis is grouped by speaker, so the
  clone prompt is rebuilt only when the voice changes.
- **Timing errors never accumulate.** Each line is anchored to its own start time;
  a line that does not fit spills into the pause after it and any residue is
  trimmed. Lines are never pushed late as a series.
- **The room can be borrowed without touching the voice.** `--room match`
  measures decay on the separated speech stem and convolves the finished dub, so
  it runs after synthesis and cannot disturb the clone.

Where things live: `dub.py` (CLI), `audit_qc.py` / `audit_timbre.py` (quality
scripts), `pipeline/` (flat, importable core), `server/` (FastAPI, SQLite store,
job manager, worker), `web/` (React single-page app), `tests/`. `models/`,
`voices/`, `output/` and `data/` are local, gitignored state.

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
  `web/src/options.ts`, then update `docs/reference.md`. `tests/test_options.py`
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

- Pytest is the only backend test runner. The default suite must not need a GPU,
  downloaded models, or ffmpeg; fake the heavy pieces. It runs in about a minute.
- `python -m pytest` is the release gate. `ruff check .` must pass too.
- Server tests use `create_app(data_dir=tmp_path, executor=FakeExecutor)` so the
  state machine, uploads, settings, SSE plumbing and presets run without
  spawning models.
- Optional backends are exercised through fakes and `pytest.importorskip`.
- Add tests for failure paths and edge cases, not only the happy path.
- Keep tests deterministic; no network calls. `tests/test_docs.py` checks that
  every flag and HTTP route is documented and every doc is linked.

Run one area while iterating:

```bash
python -m pytest tests/test_tts.py -q            # voice cloning, accent modes
python -m pytest tests/test_pipeline.py -q       # end-to-end orchestration
python -m pytest tests/test_multi_speaker.py -q  # per-speaker dubbing
python -m pytest tests/test_audit.py -q          # timbre audit + quality gates
python -m pytest tests/test_voices.py -q         # saved voices and cards
python -m pytest tests/test_server.py -q         # API, job state machine, uploads
python -m pytest -k accent -q                    # everything accent-related

cd web && npm test        # Vitest + Testing Library: API, primitives, every page
cd web && npm run build   # tsc --noEmit plus the Vite production build
```

### Real dubs against the local checkpoints

The suite proves the wiring with fakes; to hear a feature, dub a short excerpt
with each backend. Use 15–25 s so the loop stays quick — ASR, diarization and
translation are paid once per run regardless:

```bash
ffmpeg -y -ss 00:01:00 -t 20 -i source.mp4 -vn -ac 1 -ar 24000 /tmp/clip.wav

python dub.py --audio /tmp/clip.wav --target fr --backend qwen       --output-dir /tmp/dub-qwen
python dub.py --audio /tmp/clip.wav --target fr --backend chatterbox --prosody-plan auto --output-dir /tmp/dub-chatterbox
python dub.py --audio /tmp/clip.wav --target fr --backend cosyvoice3 --prosody-plan auto --output-dir /tmp/dub-cosyvoice
python dub.py --audio /tmp/clip.wav --target fr --backend fish       --prosody-plan auto --output-dir /tmp/dub-fish
```

Separate `--output-dir` values stop the runs sharing a cache. Then check, in
order: the plan was produced (`-v` shows `Planned prosody for N line(s)`),
`04_dubbed/donor.json` names the backend, the expressive run sounds acted rather
than read against the Qwen baseline, `--prosody-plan off` still sounds flat, and
switching `--backend` re-synthesises instead of serving a cached dub.

## Documentation

The docs are organised by reader intent, one page each:

| File | Owns |
|---|---|
| `README.md` | the pitch, the two surfaces, the five-minute install |
| `docs/getting-started.md` | install (native and Docker), models, LLM endpoint, optional backends |
| `docs/usage.md` | the app page by page, then the CLI chooser and worked examples |
| `docs/reference.md` | every flag, default, environment variable and HTTP route |
| `docs/voices.md` | voice library, cards, `dub say`, expression, accent modes |
| `docs/dub-quality.md` | speakers, overlap, bed, room, timing, length repair |
| `docs/performance.md` | device, precision, fan-out, long-video memory |
| `docs/outputs-and-caching.md` | artifacts, `report.json`, cache rules, audit tools |
| `docs/troubleshooting.md` | symptoms in the app and the CLI, logs, limits |
| `docs/model-licenses.md` | code vs weights licenses, commercial boundaries |
| `docs/README.md` | the index and the reading paths |

- Every user-visible flag belongs in `docs/reference.md`; `tests/test_docs.py`
  enforces it from `dub.py`'s parsers, and enforces the HTTP routes from
  `server/app.py`. Adding a flag or route without documenting it fails the suite.
- A merged or renamed page must update `docs/README.md` and every inbound link;
  the same test catches orphans and broken relative links.
- The README must only claim capabilities implemented by the CLI, the API, a test
  or a documented module. Update it in the same change as the behavior.
- Keep `PLAN.md` status current, and keep the verification log there rather than
  duplicating measurements into the guides.

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
