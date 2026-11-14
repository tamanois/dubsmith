# Contributing

Thanks for helping improve Dubsmith. The project is Apache-2.0 and accepts
contributions under the Developer Certificate of Origin (DCO), not a CLA.

## Before you start

- Read `AGENTS.md` and the project guidelines in `docs/development/guidelines.md`.
- Check open issues and pull requests before starting a large change.
- For a new backend, open an issue first. The backend must have a clear license,
  a VRAM/disk estimate, a graceful-degradation path, tests with fakes, and docs
  updates. Do not reintroduce removed backends without a decision.
- Never commit voices, input media, output artifacts, model weights, `.env`, or
  third-party checkouts. They are gitignored for privacy and size reasons.

## Development setup

The tested floor is Python 3.10; CI covers 3.10–3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The test suite needs no GPU, downloaded models, or ffmpeg. Heavy components are
faked.

```bash
python -m pytest
python -m pytest tests/test_provenance.py -q
python -m pytest tests/test_cli_smoke.py -q
python -m pytest tests/test_server.py -q      # API, jobs, uploads, presets
python -m pytest tests/test_docs.py -q        # docs match flags, routes and links

cd web && npm test         # Vitest + Testing Library
cd web && npm run build    # tsc --noEmit and the production build
```

### Web UI development

Two processes: the API with auto-reload, and Vite with hot module replacement.
The API needs the `web` extra (`pip install -e ".[web]"`).

```bash
make serve        # :8000, reloads when server/ or pipeline/ changes
make web          # :5173, HMR, /api proxied to :8000
make web-build    # write web/dist for a single-process run (dub-server)
```

`pip install -e ".[web]"` matters: a plain `pip install .` freezes a copy in
`site-packages`, so the running server ignores your edits.

Optional backends live in separate environments. See
[Getting started → optional backends](docs/getting-started.md) and
[Voices](docs/voices.md). Their tests skip when the dependency is absent.

## DCO sign-off

Every commit must be signed off:

```bash
git commit -s -m "Your change"
```

The `Signed-off-by` line certifies the Developer Certificate of Origin 1.1 in
`DCO`. Contributions are licensed under Apache-2.0.

## What to update

When behavior changes, update the docs in the same change:

- `README.md` for user-visible capabilities, commands and both surfaces.
- `docs/reference.md` for flags, defaults, environment variables and HTTP routes.
- `docs/usage.md` for the app's pages and the command chooser.
- `docs/voices.md` for voice, backend, emotion and accent behavior.
- `docs/dub-quality.md` for speakers, timing, room and length repair.
- `docs/outputs-and-caching.md` for artifacts, `report.json` fields and the audit tools.
- `docs/troubleshooting.md` for new error messages, in the app as well as the CLI.
- `docs/README.md` when a page is added, merged or renamed.
- `PLAN.md` for roadmap/status changes.

`tests/test_docs.py` fails when a flag, an HTTP route or a doc link is missing or
orphaned, so run it while writing docs: `python -m pytest tests/test_docs.py -q`.

Do not claim a capability in the README unless there is a CLI flag, an API route,
a test, or a documented module that implements it.

## Code expectations

- Keep the existing flat `pipeline/` layout. `server/` and `web/` are thin layers
  over it: they call `pipeline/` and never reimplement media, TTS or translation
  logic. No new service, queue or worker model without a decision logged in
  `PLAN.md`.
- Prefer pure functions for data transforms; isolate model/ffmpeg work behind
  the existing engine/protocol boundaries.
- Use pydantic for configuration and domain errors for user-facing failures.
- Use argument lists with `subprocess`; never `shell=True`.
- Keep heavy imports lazy so `dub --help` and the test suite stay fast.
- Add tests for edge cases and failure paths, not only the happy path.
- Run the full suite before opening a PR.

## Bug reports

Include:

- OS, Python version, GPU/CPU, and `python dub.py info` output.
- The exact command, with secrets removed.
- The relevant `report.json`, `prosody.json`, or log excerpt.
- Whether optional backends (`funasr`, `demucs`, `speechbrain`, Chatterbox,
  CosyVoice3, Fish) are installed.

Do not upload real voices or private media to a public issue. Use synthetic or
public-domain samples.

## Security

Do not open a public issue for a vulnerability. See `SECURITY.md`.
