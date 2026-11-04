# Security policy

## Supported versions

Security fixes are applied to the latest release on the default branch. The
project is pre-1.0, so older releases may not receive patches.

## Reporting a vulnerability

Please use GitHub Security Advisories ("Report a vulnerability" on the
repository's Security tab). Do not open a public issue for a vulnerability.

Include:

- A description of the issue and its impact.
- Steps to reproduce, with secrets and personal data removed.
- The affected version, OS, Python version, and whether optional backends are
  installed.

## Scope

The project is a local CLI. The main security boundaries are:

- `.env` and environment variables, which may contain API keys.
- Model downloads from Hugging Face and third-party checkouts.
- `subprocess` calls to ffmpeg, rubberband, sox, and backend workers.
- Job artifacts (`output/`) and saved voices (`voices/`), which may contain
  private speech and transcripts.

## Practices

- `subprocess` calls use argument lists with `shell=False`.
- API keys are read from the environment or `.env`; they are not written to job
  artifacts.
- `-v` logging can include transcript text and full LLM requests; treat logs as
  private.
- Do not commit `voices/`, `input/`, `output/`, `models/`, `third_party/`, or
  `.env`.
- Pin or review third-party model revisions if you need reproducible supply-chain
  behavior.

## Model weights

Model weights are downloaded separately and are not covered by this project's
Apache-2.0 license. Review their licenses, especially Fish S2-Pro, which is
non-commercial. See `docs/model-licenses.md`.
