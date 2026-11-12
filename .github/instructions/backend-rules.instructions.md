---
name: "Backend and Model Rules"
description: "Use when touching TTS backends, model registry, device/dtype handling, voice cards or backend worker processes: pipeline/tts.py, pipeline/backends.py, pipeline/models.py, pipeline/voices.py, pipeline/{chatterbox,cosyvoice,fish}.py and pipeline/*_worker.py."
applyTo: ["pipeline/tts.py", "pipeline/backends.py", "pipeline/*_worker.py", "pipeline/models.py", "pipeline/voices.py", "pipeline/chatterbox.py", "pipeline/cosyvoice.py", "pipeline/fish.py"]
---
# Backend and Model Rules

Concrete rules for the code that loads checkpoints, moves tensors and talks to
backend child processes. `docs/development/guidelines.md` remains authoritative.

## Model loading

- The registry in `pipeline/models.py` is the only place model names, repo ids and
  directories are defined. No `models/<something>` path literals elsewhere.
- Missing weights raise `ModelNotFoundError` carrying the exact
  `python dub.py models --download <name>` remedy. Never surface a raw traceback.
- Never import a backend whose `torch`/`transformers` pins conflict with the main
  environment. Check availability without importing (file on disk,
  `_module_available`) and run it through its `*_worker.py` child process.
- Load a checkpoint once per run and reuse it; never per line or per segment.
- Do not silently substitute a different model, precision or quantisation than the
  one requested. If a flag changes them, it is documented and logged.

## Device and memory

- Device comes from `resolve_device()`/`Settings`; never hardcode `"cuda"`. An
  explicit `--device` value passes through unchanged (no silent narrowing).
- Persisted artifacts stay device- and dtype-portable: floats as float32 numpy,
  moved with `.to(device)`/`.to(dtype)` at use time.
- Checkpoint tensors may be bfloat16, which `.numpy()` rejects. Widen through a
  `_as_numpy`-style helper instead of calling `.numpy()` on them directly.
- Call `release_gpu_memory()` between stages that each load a multi-GB model, and
  keep a model's codec on the same device as its model.
- Size batches against measured VRAM, not against what happened to fit on one
  development card.

## Derived artifacts and caches

- Any artifact derived from a checkpoint records `model_fingerprint(name)`; a
  mismatch re-encodes instead of reusing. `find_prompt` matches exactly — it never
  falls back to another backend, checkpoint or transcript.
- Cache keys cover backend, fingerprint and content (`voice_content` = clip bytes +
  whitespace-folded transcript). Adding a key dimension means updating the readers.
- A miss never deletes another checkpoint's card; cards for different fingerprints
  coexist so switching models stays instant in both directions.
- A stale or mismatched entry is treated as absent. Caches must never mask a wrong
  result, only a slow one.

## Voice data

- Voice prints are biometric: never log reference audio, speech codes, embeddings,
  transcripts or full prompts outside the explicitly private `-v` path.
- Write voice data only inside the voice directory or the configured output root.

## Backend worker protocol

- Parent and child exchange JSON on disk (`request.json` → `result.json`), invoked
  with an argument list, `shell=False`, project root as `cwd` and explicit
  `PYTHONPATH`.
- A nonzero exit raises a domain error with a bounded stderr tail; a missing or
  non-dict `result.json` is an error, never an empty result.
- Worker diagnostics belong on stderr and are logged at DEBUG; stdout is not a
  data channel.
- Every new backend ships with a fake so the default suite runs without weights,
  a GPU or network access.

## Adding a backend

Cover the checklist in `docs/development/guidelines.md` (license review, VRAM/disk
estimate, graceful missing-dependency path, help text, fakes, docs, `PLAN.md`).
Do not reintroduce removed backends (IndexTTS2, CosyVoice2) without a decision.
