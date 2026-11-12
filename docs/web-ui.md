# Browser UI and local API

The web app is a local, single-user browser interface over the same `pipeline/`
code the CLI uses. It is not a hosted service and has no authentication yet.

## Run it

```bash
# Python side (FastAPI + the worker)
pip install -e ".[web]"
dub-server
# or: python -m server

# Frontend in development (separate terminal)
cd web
npm install
npm run dev        # http://localhost:5173, proxies /api to :8000
```

For a single self-contained process, build the frontend first and then start the
server; FastAPI serves `web/dist` at `/`:

```bash
cd web && npm run build && cd ..
dub-server
# http://127.0.0.1:8000
```

Configuration:

| Variable | Default | Meaning |
|---|---|---|
| `DUB_HOST` | `127.0.0.1` | Bind address. Keep it loopback unless you add auth. |
| `DUB_PORT` | `8000` | HTTP port. |
| `DUB_DATA_DIR` | `data` | SQLite database, uploads and job metadata. Gitignored. |
| `DUB_OUTPUT_DIR` | `output` | Root for unique per-run output folders. |

Secrets (`LLM_API_KEY`, `HF_TOKEN`, …) stay in the environment or `.env` and are
never written to SQLite. The UI shows whether an LLM key is configured.

## What it exposes

The UI covers every CLI capability:

- **Dub** — all `JobConfig` options, grouped as Basic, Advanced and More advanced.
  Includes voice/accent, backend and prosody, timing, loudness, room, speakers,
  overlap, length repair, subtitles and per-op device fan-out.
- **Subtitles only** — the same input/target/source flow without TTS.
- **Say** — saved/designed/cloned voices, CustomVoice preset speakers, per-line
  prosody, backend-specific knobs (Chatterbox, Fish, CustomVoice).
- **Voices** — design, clone, register, list and delete.
- **Models** — status and downloads (MVP set, one model, or everything).
- **Devices** — detected hardware, precision, optional packages and binaries.
- **Files** — upload, list, download and delete.
- **Settings** — LLM endpoint/model/timeouts, audio defaults, storage and logging.

## Jobs and feedback

Every submission returns immediately with a job id. Jobs are persisted as
`QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED` or `CANCELLED`, and the state
machine rejects invalid transitions. One heavy job runs at a time in a spawned
process (the same single-model residency the CLI uses); the queue is bounded and
rejects excess work explicitly.

`GET /api/jobs/{id}/events` is an SSE stream of stage starts/ends, log lines,
errors and the final job. Progress is real stage state only; the UI never
fabricates a percentage. The activity panel keeps the full log behind a toggle.

## Outputs

Each job gets its own folder under `DUB_OUTPUT_DIR`:

```
<stem>__<operation>__<language>__<run-id>/
```

The pipeline writes its normal artifacts inside it, so two runs never overwrite
each other. `GET /api/files` lists uploads; job outputs are visible on disk and in
the job result.

## API summary

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness |
| GET | `/api/devices` | Structured device report |
| GET | `/api/info` | Environment, models, packages |
| GET | `/api/packages` | Optional packages and system binaries |
| GET | `/api/models` | Registry status |
| POST | `/api/models/download` | Download one, `all`, or `everything` |
| GET | `/api/voices` | Saved voices |
| POST | `/api/voices/design` | Design a voice (job) |
| POST | `/api/voices/clone` | Clone a voice (job) |
| POST | `/api/voices/add` | Register a clip (job) |
| DELETE | `/api/voices/{label}` | Delete a voice |
| GET | `/api/anchors` | Cached accent anchors |
| POST | `/api/anchors/generate` | Generate anchors (job) |
| POST | `/api/jobs/dub` | Start a dub |
| POST | `/api/jobs/subtitles` | Start a subtitles-only job |
| POST | `/api/jobs/say` | Start speech generation |
| GET | `/api/jobs` | Recent jobs |
| GET | `/api/jobs/{id}` | Job detail and recent events |
| POST | `/api/jobs/{id}/cancel` | Cancel |
| GET | `/api/jobs/{id}/events` | SSE stream |
| GET | `/api/files` | Uploaded files |
| POST | `/api/files/upload` | Upload a media file |
| GET | `/api/files/{id}/download` | Download an upload |
| DELETE | `/api/files/{id}` | Delete an upload |
| GET | `/api/outputs` | List generated output files |
| GET | `/api/outputs/download?path=…` | Download an output |
| DELETE | `/api/outputs?path=…` | Delete an output |
| GET/PATCH | `/api/settings` | Non-secret settings |

## Security boundaries

- Inputs must be uploaded through the app or already live under `input/`,
  `output/` or the app data directory; arbitrary filesystem paths are rejected.
- Uploads are size/type checked and stored under server-generated names.
- Deletes are restricted to the app's own directories.
- CORS is limited to local development origins.
- No auth yet: bind to `127.0.0.1`, or add authentication before exposing it.

## Tests

`tests/test_server.py` uses a temporary SQLite database and a fake executor, so
the state machine, uploads, settings and event persistence run without a GPU,
models or ffmpeg:

```bash
python -m pytest tests/test_server.py
```

The frontend is checked with Vitest and Testing Library, then type-checked and
built:

```bash
cd web
npm test
npm run build
```

The 125 frontend tests do not use a network, GPU, or built assets:

- `test/api.test.ts` covers request errors, upload failures, URL encoding and SSE
  dispatch, including malformed frames and connection loss.
- `test/ui.test.tsx` and `test/UploadField.test.tsx` cover formatting helpers,
  disclosures, toggles, badges and upload success/failure states.
- `test/JobPanel.test.tsx` and `test/JobsPage.test.tsx` cover live log growth and
  the 300-line cap, terminal messages, cancellation, result paths, activity
  polling and cleanup.
- `test/ModelsPage.test.tsx`, `DevicesPage.test.tsx`, `FilesPage.test.tsx`,
  `SettingsPage.test.tsx` and `VoicesPage.test.tsx` cover the management pages,
  including confirms, payloads, busy states and API errors.
- `test/DubPage.test.tsx` and `SayPage.test.tsx` cover every form mode and the
  request payloads for dubbing, subtitles, saved/designed/cloned voices and
  backend-specific controls.
- `test/App.test.tsx` covers navigation, the job-start hand-off, the toast and
  the panel lifecycle.
