---
name: "Engineering Rules"
description: "Use when writing, changing, reviewing, refactoring, testing or debugging any code in this repository. Cross-cutting engineering standard: priority order, error handling, security, logging, performance, concurrency, data, tests, docs, dependencies and definition of done."
applyTo: "**"
---
# Engineering Rules

Project-specific conventions (architecture, Python 3.10, pytest gate, line width,
docs to update, backend and release rules) live in `docs/development/guidelines.md`.
That file wins on conflicts and is authoritative for this repo. This file is the
cross-cutting standard.

## Priority order

1. Security and data integrity
2. Correctness and reliability
3. Simplicity and maintainability
4. Observability and testability
5. Performance and scalability
6. Flexibility and future extensibility

Never trade correctness or security for a minor performance win. Never trade
simplicity for speculative scalability.

## Before writing code

- State the requested behavior and acceptance criteria; read the existing code,
  conventions and tests first.
- Reuse existing functions and modules before creating new ones. Do not open with
  new abstractions or restructuring.
- Identify affected interfaces, dependencies, tests, docs, and security or
  compatibility risks.
- For nontrivial work, name the components, approach, key decisions, risks and
  test plan, and prefer small independently verifiable steps.

## While writing code

- Change only what is necessary. No drive-by refactors, no rewriting working code
  for style, no removing unrelated user changes.
- Apply KISS first; DRY only for meaningful business logic; YAGNI for options,
  layers and infrastructure; SOLID only where it improves modularity or tests.
- One cohesive responsibility per function, descriptive names, explicit side
  effects, shallow nesting, no hidden mutation.
- No duplicate business logic between CLI paths, backends and stages.
- No dumping-ground `utils` module; shared code goes where its ownership is clear.
- Explicit types and contracts; validate at trust boundaries; never use unsafe
  casts or `assert`s to silence a type checker.
- Comments explain why, constraints and assumptions, not what the code does.
  Delete comments that no longer match behavior.
- Fail safely: on failure leave no corrupted artifacts, stale caches or partial
  outputs reported as success.

## Errors

- Handle every meaningful failure. Never swallow exceptions, never return success
  for a failed operation, never retry indefinitely.
- Catch only where you can recover, add context, clean up, or translate to a
  stable interface. Otherwise let it reach the domain error type.
- Raise from `pipeline/errors.py`. No bare `except:`; `# noqa: BLE001` only for a
  deliberate best-effort path that the comment explains.
- Set timeouts on subprocess, network and model calls. Bound retries to transient
  failures with backoff; never retry validation or authorization failures.
- Release temp files, handles, CUDA memory and child processes on failure.
- User-facing errors are actionable, with no stack traces, filesystem paths,
  tokens or infrastructure details.

## Security

- Treat all input and media as untrusted: validate type, size, format, range and
  path at the boundary.
- Never hardcode or commit secrets; read them from the environment or `.env` and
  keep them out of artifacts.
- Never log tokens, reference audio, transcripts or full prompts; voice prints are
  biometric data.
- Subprocesses: argument lists with `shell=False`, no string interpolation,
  timeouts, captured exit status.
- Keep job outputs under the configured output root and reject path traversal.
  Never write outside server-controlled directories.
- Do not add a dependency for something that existing tools do simply and safely;
  check maintenance, license, size and known vulnerabilities.
- No debug hooks, permissive CORS or verbose logging enabled by default.

## Logging and observability

- Use the job-scoped logger from `pipeline/logging_utils.py`; every run stays
  separable in `run.log` and `report.json`.
- DEBUG for diagnostics, INFO for milestones, WARNING for recoverable surprises,
  ERROR for failures needing investigation, CRITICAL for integrity or availability.
- Log an error once, at the layer that handles it, with job id, stage and duration.
- Logging must never include secrets or verbose user content unless it is the
  explicitly private `-v` path.

## Performance and concurrency

- No speculative optimization; measure a representative workload first (see
  `docs/performance.md`) and document the assumption.
- Avoid repeated model loads, whole-file reads where streaming works, and repeated
  ASR, TTS or translation passes. Reuse cached artifacts when the fingerprint
  matches.
- Bound concurrency by real limits (VRAM, CPU, RAM, disk, provider rate limits),
  not by "more parallel is faster".
- Every cache needs a purpose, an invalidation rule, and must not mask a slow
  design. Do not cache sensitive data unnecessarily.

## Data and configuration

- Configuration is not code: no hardcoded paths, endpoints, model names or limits
  that belong in `Settings` or the registry.
- Validate configuration at startup and fail with a clear, actionable message.
- Never commit `models/`, `voices/`, `input/`, `output/`, `third_party/` or `.env`.
- Define cleanup and retention for temporary and intermediate artifacts; do not
  leave junk behind indefinitely.
- Avoid destructive changes to local data without a recovery path.

## Tests

- Test behavior, not implementation details. Tests are deterministic, isolated,
  independent of order, and free of network, GPU, downloaded models and ffmpeg.
- Cover failure paths and edge cases (empty, invalid, missing, oversized,
  duplicate, cancelled, partially failed), not just the happy path.
- For a reproducible bug, add a failing test first, then fix it, then run the
  relevant suite.
- Do not mock the core logic under test; fake only expensive boundaries.
- No flaky test is acceptable as a permanent condition.
- Report exactly which commands ran and their outcome. Never claim a test passed
  unless it actually ran.
- For each feature always add tests for every uses cases and edge case, including failure paths. For each bug, add a regression test.

## Documentation

- Update documentation in the same change as the behavior. Every user-visible flag
  goes in `docs/options.md`; README claims must be backed by code or a test.
- Do not document hypothetical features as implemented.
- Explain significant design decisions where the code cannot.

## Dependencies and integrations

- Prefer an existing dependency over a new one; justify each addition, pin where
  reproducibility matters, and remove unused dependencies.
- External calls get timeouts, rate-limit and failure handling, validated
  responses, and safe logging. Protect credentials in transit and at rest.

## Definition of done

Report what changed, why that approach, which checks ran and whether they passed,
and any remaining risk, limitation or follow-up. Compiling and passing a small
test set is not proof of production readiness; disclose anything unverified.

## Future scope: frontend, API, worker and queue

Not built yet. This project is currently a local CLI (see
`docs/development/guidelines.md`). Apply these rules only when such a layer exists.

- Keep processing in `pipeline/`. The API and worker call it; they never
  reimplement media, TTS or translation logic. The CLI stays a first-class
  interface.
- Long AI or media work never blocks request handling: accept, return a job id,
  and track durable state (`QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`,
  `CANCELLED`) with valid transitions, attempt counts and error details.
- Accepted jobs are never silently lost; retries must not duplicate or corrupt
  outputs.
- Bound concurrency per resource, reject excess work explicitly, and never let a
  queue grow unbounded.
- Enforce authorization on every user-owned resource; never trust client-provided
  ownership ids; apply upload size and type limits with server-controlled names.
- If remote URLs are ever supported, restrict destinations (SSRF), size and
  timeouts; otherwise do not add the capability.
- Progress reflects real measurable state; never fabricate percentages.

## Prohibited

Complexity for its own sake; rewriting working functionality; duplicated business
logic across interfaces; hardcoded secrets; trusting unvalidated input; silently
ignored exceptions; success responses for failed operations; unbounded retries or
resource use; unsafe shell execution; leaked internals; orphaned temp files;
bypassed auth or authorization; unjustified breaking interface changes;
configuration added without a concrete need; claimed-but-unrun verification.
