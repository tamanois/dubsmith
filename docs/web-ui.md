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

After changing `server/` or `pipeline/`, restart `dub-server`; after changing
`web/src`, re-run `npm run build` and hard-refresh the tab. Install the project
editable (`pip install -e`) — a plain `pip install .` freezes a copy in
`site-packages` and the running server then ignores your changes, which looks
exactly like a broken UI.

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

- **Activity** — every job with its human title, input name and output file;
  filter by state, search, paginate, and bulk-clear finished records. A finished
  job can be retried in a fresh folder, opened for editing with its stored options
  prefilled, or compared against its source. The panel expands and keeps its log.
- **Dub** — all `JobConfig` options, grouped as Basic, Advanced and More advanced.
  Includes voice/accent, backend and prosody, timing, loudness, room, speakers,
  overlap, length repair, subtitles and per-op device fan-out. Saved presets and
  recent/favorite input chips sit above the options.
- **Subtitles only** — the same input/target/source flow without TTS.
- **Say** — saved/designed/cloned voices, CustomVoice preset speakers, per-line
  prosody, backend-specific knobs (Chatterbox, Fish, CustomVoice), plus presets and
  recent/favorite clips.
- **Voices** — design, clone, register, list with pagination, audition and delete.
- **Models** — status, downloads (MVP set, one model, or everything) and deletion
  behind a danger confirmation that reports the freed space.
- **Devices** — detected hardware, precision, optional packages and binaries.
- **Files** — uploads with pagination, inline playback, path copy, download and
  bulk delete. Generated outputs are listed once per run with that run's main
  file; a Details view reveals every artifact in the run.
- **Settings** — LLM endpoint/model/timeouts, audio defaults, storage, logging,
  the default device/precision, and JSON export/import.

The header always shows the default device and precision and a System / Light /
Dark theme switch; both are persisted with the other settings.

## Jobs and feedback

Every submission returns immediately with a job id. Jobs are persisted as
`QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED` or `CANCELLED`, and the state
machine rejects invalid transitions. One heavy job runs at a time in a spawned
process (the same single-model residency the CLI uses); the queue is bounded and
rejects excess work explicitly.

`GET /api/jobs/{id}/events` is an SSE stream of stage starts/ends, log lines,
errors and the final job. Progress is real stage state only; the UI never
fabricates a percentage. If the stream drops, the browser reconnects with
backoff and resumes from the last event id, and the activity panel keeps working
from plain HTTP state meanwhile (so Cancel still succeeds). Completion toasts
announce a job finishing wherever the user is in the app.

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
| DELETE | `/api/models/{name}` | Delete a checkpoint directory (reports freed bytes) |
| GET | `/api/usage` | Disk used by models, voices, outputs and uploads |
| GET | `/api/presets?kind=…` | Saved form configurations |
| POST | `/api/presets` | Save a preset |
| DELETE | `/api/presets/{id}` | Delete a preset |
| GET | `/api/favorites` | Pinned input paths |
| POST | `/api/favorites` | Pin an input path |
| DELETE | `/api/favorites?path=…` | Unpin an input path |
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
| GET | `/api/jobs?page=&page_size=&state=&q=` | Recent jobs (paginated, filterable) |
| GET | `/api/jobs/{id}` | Job detail and recent events |
| POST | `/api/jobs/{id}/retry` | Re-run the stored options as a new job |
| GET | `/api/jobs/{id}/input` | Stream the job's source file (compare view) |
| POST | `/api/jobs/{id}/cancel` | Cancel |
| POST | `/api/jobs/clear` | Delete finished job records in bulk |
| GET | `/api/jobs/{id}/events` | SSE stream |
| GET | `/api/files?page=&page_size=` | Uploaded files (paginated) |
| POST | `/api/files/upload` | Upload a media file |
| GET | `/api/files/{id}/download` | Download an upload (`?inline=1` to play it) |
| DELETE | `/api/files/{id}` | Delete an upload |
| POST | `/api/files/delete` | Delete uploads in bulk |
| GET | `/api/outputs?page=&page_size=` | One row per run, with its main artifact |
| GET | `/api/outputs/files?job_dir=…` | Every artifact in one run (details view) |
| GET | `/api/outputs/download?path=…` | Download an output (`?inline=1` to play it) |
| DELETE | `/api/outputs?path=…` | Delete an output |
| POST | `/api/outputs/delete` | Delete outputs in bulk |
| GET | `/api/voices/{label}/audio` | Stream a saved voice's clip |
| GET/PATCH | `/api/settings` | Non-secret settings |
| GET | `/api/settings/export` | Download a JSON settings document |
| POST | `/api/settings/import` | Apply a JSON settings document |

## Security boundaries

- Inputs must be uploaded through the app or already live under `input/`,
  `output/` or the app data directory; arbitrary filesystem paths are rejected.
- Uploads are size/type checked and stored under server-generated names.
- Deletes are restricted to the app's own directories; model deletion is
  restricted to the models directory and refuses package-backed models.
- Only files a job already used (under `input/`, `output/` or the data directory)
  can be streamed back as its source.
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

The 163 frontend tests do not use a network, GPU, or built assets:

- `test/api.test.ts` covers request errors, upload failures, URL encoding and SSE
  dispatch, including malformed frames and reconnect-with-resume.
- `test/ui.test.tsx`, `test/UploadField.test.tsx` and `test/PresetPanel.test.tsx`
  cover formatting helpers, disclosures, toggles, badges, upload success/failure
  states, and preset save/apply/delete.
- `test/JobPanel.test.tsx` and `test/JobsPage.test.tsx` cover live log growth and
  the 300-line cap, terminal messages, cancellation after a dropped stream,
  retry/edit/compare, result paths, filtering, pagination, bulk clearing, the
  adaptive polling cadence and cleanup.
- `test/ModelsPage.test.tsx`, `DevicesPage.test.tsx`, `FilesPage.test.tsx`,
  `SettingsPage.test.tsx` and `VoicesPage.test.tsx` cover the management pages,
  including confirms, bulk deletes, playback, export/import, payloads, busy
  states and API errors.
- `test/DubPage.test.tsx` and `SayPage.test.tsx` cover every form mode and the
  request payloads for dubbing, subtitles, saved/designed/cloned voices and
  backend-specific controls.
- `test/RecentInputs.test.tsx` and `test/Onboarding.test.tsx` cover the input
  chips, favoriting and the dismissible first-run card.
- `test/App.test.tsx` covers navigation, the global defaults and theme, the
  job-start hand-off, completion toasts (including the idle and hidden-tab
  cadences) and the panel lifecycle.
