# Architecture

The stage pipeline, the module map, the design points worth knowing, and where
everything lives in the repository.

## How it works

```
video/audio ─► 1. audio extract ─► 2. separate ─► 3. ASR + VAD ─► 4. diarize ─┐
                               (speech | bed)                               │
  8. mux + SRT ◄─ 7. mix + duck ◄─ 6. clone + synthesise ◄─ 5. translate (LLM) ◄┘
```

| # | Stage | Module | What it does |
|---|---|---|---|
| 1 | Audio | `pipeline/media.py` | ffmpeg extract (video) or direct read (audio), mono 48 kHz |
| 2 | Separate | `pipeline/separation.py` | Demucs splits speech from music/ambience; ASR then reads the clean speech stem |
| 3 | ASR | `pipeline/asr.py` | faster-whisper + Silero VAD, word timestamps, drops non-speech |
| 4 | Diarize | `pipeline/diarize.py` | embeds each segment and clusters the embeddings into speakers |
| 4b | Overlap | `pipeline/overlap_separation.py` | with `--overlap separate`, splits a flagged segment into per-speaker streams for re-transcription |
| 5 | Translate | `pipeline/translate.py` | batched, context-aware, length-budgeted translation |
| 6 | Synthesise | `pipeline/tts.py` | per-speaker donor selection + voice cloning, grouped by speaker — spread over several cards with `--devices` |
| 7 | Mix | `pipeline/timing.py`, `pipeline/mixing.py` | fit each line to its slot, then lay it over the ducked bed — optionally matched to the source's room |
| 8 | Deliver | `pipeline/orchestrate.py` | loudness match, subtitles, mux, report |

Three design points worth knowing:

- **Accent comes from an anchor, timbre from cloning.** By default (`native`) the
  accent is a generated per-language anchor and the timbre is the source speaker's.
  `--accent-mode preserve` keeps the source accent, `speaker-only` drops the
  borrowed phonetics, and `reference-accent` uses a native clip you supply.
- **One anchor per language, generated once.** The VoiceDesign checkpoint turns a
  short passage into native speech, so ten languages share one model instead of
  shipping ten voices; the result is cached under `models/anchors/`.
- **One prompt per voice, not per line.** Diarization gives each speaker their own
  donor and, in `reference-accent` mode, their own blended reference; synthesis is
  then grouped by speaker so the prompt is rebuilt only when the voice changes.
- **Timing errors never accumulate.** Each line is anchored to its own original
  start time; a line that does not fit spills into the pause after it and any
  residue beyond that is trimmed. Lines are never pushed late, so a long
  translation cannot make the rest of the video drift out of sync.
- **The room can be borrowed without touching the voice.** Synthetic speech is
  recorded close and dry, which reads as a voice laid over the picture. With
  `--room match`, the decay (RT60) is measured from the separated speech stem —
  the only material that carries the room — and applied as a convolution over the
  finished dub, while the stem's own pauses become a continuous room-tone floor.
  It runs after synthesis, so it cannot disturb the voice cloning.

## Project layout

```
dub.py                     CLI entry point (`dub`, `dub info`, `dub models`, `dub voices`, `dub say`)
audit_timbre.py            per-line speaker-timbre audit (--device cpu works)
audit_qc.py                automated quality gate for a finished job
LICENSE                    Apache-2.0
pipeline/
├── anchors.py             generated accent anchors (analysis + cache)
├── config.py              job + settings models, device capability, languages
├── errors.py              domain exception hierarchy
├── logging_utils.py       job-scoped structured logging
├── models.py              model registry, download, path resolution
├── media.py               ffmpeg/ffprobe wrappers, loudness, audio I/O
├── subtitles.py           SRT rendering
├── asr.py                 transcription + segmentation
├── diarize.py             speaker embeddings + clustering (multi-speaker)
├── overlap_separation.py  SepFormer backend for splitting overlapping voices
├── separation.py          speech/background split (Demucs)
├── mixing.py              ducking, bed mix and room matching (pure numpy)
├── translate.py           OpenAI-compatible translation
├── tts.py                 per-speaker donor selection + voice cloning
├── voice.py               one voice: the facts that build its clone prompt
├── voices.py              saved voices: clip + manifest + stored encoding
├── provenance.py          emotion/prosody provenance artifact (schema + recorder)
├── parallel.py            fan-out: tasks, the worker pool, per-device counters
├── timing.py              stretching backends + timeline assembly
├── audit.py               timbre measurement + quality gates
└── orchestrate.py         stages, caching, reporting
tests/                     pytest suite (1105 tests, no GPU or models needed)
models/                    downloaded weights (gitignored)
voices/                    saved voices: clip, manifest and encoding (gitignored)
output/                    job artifacts (gitignored)
docs/                      user documentation (this folder)
clone.py, main.py          standalone TTS demos kept for reference
pyproject.toml             packaging, entry points, optional extras, pytest config
.github/workflows/ci.yml   test suite on a CPU-only runner
PLAN.md                    roadmap: MVP + P1–P10
.agents/technical-guidelines.md   engineering standards
```

---

Back to the [docs index](README.md), or [Troubleshooting](troubleshooting.md).
