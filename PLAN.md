# AI Video Translation — Plan & Roadmap

> Companion to `AGENTS.md` (project vision) and `.agents/technical-guidelines.md`
> (production engineering rules). This file records the MVP scope and the phased
> plan to reach the full vision. Keep it updated as decisions change.

---

## 0. Implementation status

**MVP (P0) — implemented.** See [`README.md`](README.md) for setup and usage.

| Area | Status | Where |
|---|---|---|
| CLI + model download | ✅ | `dub.py` (`models --list/--download`, `--video/--audio`) |
| Config, env, languages | ✅ | `pipeline/config.py` |
| Model registry / resolution | ✅ | `pipeline/models.py` |
| Media (ffmpeg, loudness, SRT) | ✅ | `pipeline/media.py`, `pipeline/subtitles.py` |
| ASR + VAD gating | ✅ | `pipeline/asr.py` |
| Translation (OpenAI-compatible) | ✅ | `pipeline/translate.py` |
| Voice cloning + donor selection | ✅ | `pipeline/tts.py` |
| Timing / isochrony | ✅ | `pipeline/timing.py` |
| Orchestration, caching, report | ✅ | `pipeline/orchestrate.py` |
| Tests | ✅ 330 passing | `tests/` |

Verification performed:

- Full pytest suite green (no GPU, models or ffmpeg needed — heavy parts are faked).
- `ffmpeg` integration exercised against real `ffmpeg` 4.4.2: probe, audio extract
  (48 kHz mono), and mux with `-c:v copy`.
- `rubberband` 2.0.0 verified: `--time 1.10` on a 2.000 s clip yields 2.200 s, so
  the `ratio = target/source` convention in `pipeline/timing.py` is correct.
- Missing-model path produces an actionable error naming the exact download command.
- **Real end-to-end runs** (real Whisper large-v3 + real LLM + real Qwen3-TTS,
  only the HTTP transport swapped for a local model via the injected translator):
  - Audio mode on a 23 s French voice clip → English dub in ~15 s (faster than
    realtime). Round-trip ASR of the output returns intelligible English matching
    the translation, confirming the voice clone is correct.
  - Video mode on a video with an audio track → `video_dubbed.mp4` with the video
    stream copied and the dub muxed in; round-trip ASR confirms the new audio.
- Whisper download verified end to end: `whisper_large_v3` transcribes French and
  Chinese correctly (language detection at p=0.99).

Issues found and fixed during e2e:

- **`report.json` reported the wrong loudness.** It measured the pre-normalisation
  mix, so it disagreed with the delivered `dub.wav`. Now reports the delivered
  value plus target, applied gain, source loudness, and a `loudness_peak_limited`
  flag explaining when −16 LUFS was unreachable without clipping.
- **`--device auto` crashed on GPU ASR** when CTranslate2's CUDA major (12) differs
  from torch's (13). Now retries after preloading vendored CUDA libraries, then
  falls back to CPU with a warning instead of aborting.
- Video-only inputs now fail with an actionable message (many "video" files,
  including the TED sample used for testing, carry no audio stream at all).

Still to do before real acceptance: point `LLM_BASE_URL` at a running
OpenAI-compatible server (no `llama-server` or GGUF model was available on this
machine, so translation was validated with a local transformers model behind the
real translator code path). A real multi-sentence single-speaker video with audio
is still needed for a full-quality sign-off.

### First real video: 4:47 TED talk, English → French

Run against the user's own `llama-server` (Gemma-4 E4B): 63 segments, ~140 s
total (**2× faster than realtime**) — ASR 23 s, translate 42 s, synthesise 63 s,
mix 7 s.

This run exposed the biggest quality issue so far, and the fix changed the timing
design:

- **Symptom.** 33 of 63 lines were being cut mid-word, deleting **21.1 s** of
  speech. Cause: EN→FR translation runs ~15–40% longer, the stretch clamp caps
  compression at 0.85, and `_fit_length` then hard-trimmed to the exact slot.
- **First fix (wrong).** Allowing unlimited overrun removed all truncation but
  let lines push each other later: **max drift 5.31 s, 36/63 lines >1 s late.**
  A dub 5 s behind the video is worse than a missing syllable.
- **Final design.** `fit_line(max_overrun=…)` receives the *gap* before the next
  utterance. A line may spill into that silence for free; only the residue is
  trimmed, because overrunning it would desync the video. Lines are never shifted
  late (`shifted_segments: 0`), so sync is exact.

| Metric | Before | After |
|---|---|---|
| Lines truncated mid-speech | 33/63 | 27/63 |
| Speech cut | 21.1 s | **11.4 s** |
| Speech reclaimed into pauses | 0 s | **7.7 s** |
| Lines >1 s out of sync | 36/63 | **0/63** |
| Max drift | 5.31 s | **0.000 s** |

Remaining trimming (11.4 s) is inherent to translating verbose English into
French at a natural speech rate. The principled fix is roadmap **P3**: feed the
measured duration back to the translator and ask for a shorter line, rather than
compressing or cutting it. Two new knobs (`--min-stretch`, `--max-stretch`) and
report fields (`overrun_*`, `trimmed_*`) make the trade-off visible meanwhile.

Also fixed from this run: a GPU OOM warning during synthesis (the ASR model's
~3 GB was still in PyTorch's cache when TTS loaded) — stages now call
`release_gpu_memory()`, and a TTS batch that OOMs is halved rather than failing
the job.

### Review fixes: loudness, background, accent

The user reviewed the dubbed video. Three findings, all addressed:

**1. Too quiet (fixed).** The dub sat 3.25 dB below the source. The mix peaked at
−0.06 dBFS from isolated TTS transients, so the peak guard allowed ~0 dB of the
+6.74 dB needed, and the fixed −16 LUFS target ignored the source's own level.
Fixes: `media.limit_peaks()` (envelope limiter), per-line limiting at −3 dBFS, and
a `match-source` loudness policy that defaults to the original track's measured
level. Result: `loudness_delta_vs_source` −0.0, crest factor 22.7 → 19.4 dB (the
original is 18.6), zero clipped samples.

**2. Music and ambience vanished (fixed — was roadmap P1).** `build_timeline`
assembled a speech-only track and `mux_audio` replaced the whole audio stream, so
the soundtrack was discarded by design. New `pipeline/separation.py` (Demucs
`htdemucs`, stereo-native at 44.1 kHz, resampled around the model) splits speech
from background; `pipeline/mixing.py` lays the dub over the bed, ducked 6 dB under
dialogue with 150 ms ramps. ASR and donor selection now read the clean speech
stem, which also reduced segment count 63 → 58. Separation costs 11.8 s for a
287 s video (~24× realtime).

**3. Source accent carried into the target language (improved for free).** Accent
lives in the reference audio: with no `--ref-audio` the donor *is* the source
speaker, and ICL cloning conditions on their speech codes, so their English
phonetics came along. Added `--accent-mode speaker-only`, which keeps only the
speaker embedding and lets the model use its own target-language phonetics.

Measured on the same French lines with the same donor:

| Mode | Timbre similarity to donor | French acoustic fit (avg logprob) |
|---|---|---|
| `preserve` (default) | 0.9911 | −0.139 |
| `speaker-only` | 0.9873 | **−0.061** |
| *(same speaker, different utterance)* | 0.9869 | — |

Speaker-only retains timbre within natural same-speaker variation while fitting
French phonetics noticeably better, so the accent problem may need no
voice-conversion stage at all. Listening is still the ground truth; the full
two-stage path (VoiceDesign accent reference + voice conversion) remains the
fallback if it proves insufficient.

Deviations from the original plan (deliberate, documented):

- **`silero-vad` dependency dropped.** faster-whisper bundles an ONNX Silero VAD
  and enables it via `vad_filter=True`, so a separate package was redundant.
- **Stretching has three backends** (`rubberband` → `sox` → `librosa`) instead of
  a hard `rubberband` requirement, so audio-only jobs run with no system binaries.
- **`pyloudnorm` + numpy for loudness** instead of an ffmpeg `loudnorm` pass: it
  works identically for audio-only input and avoids a second ffmpeg invocation.
- **Cache fingerprint is per-job, not per-stage.** Changing the target language
  re-runs ASR too. Acceptable for a CLI; refine before P6.

---

## 1. Vision (from AGENTS.md)

Take a video, detect the language of the speakers, identify each voice, and
produce a new video where speakers speak a **target language** in their **own
voice timbre**, ideally with the **target language's accent**.

- Prefer **open models** that run on **consumer hardware** (current box: 2× RTX 3060, 12 GB each).
- The LLM must be behind an **OpenAI-compatible** interface, so we can switch
  between a local server and a cloud model.

---

## 2. Decisions (decision log)

| # | Topic | Decision | Rationale |
|---|-------|----------|-----------|
| D1 | Native target accent | **Nice-to-have**, not MVP | Doubles voice-pipeline complexity (needs a voice-conversion stage) |
| D2 | Source language(s) | **Single language** for MVP; per-utterance LID later | Simpler; leave hooks for multi-source-lang videos |
| D3 | Product shape | **A tool** (local CLI), not a service | No FastAPI/Celery/billing in MVP; design module boundaries so service-ification is a wrapping exercise later |
| D4 | First target language | **French** | User's native voice asset + test corpus |
| D5 | Speakers | **Single speaker** in MVP | Diarization is P1 |
| D6 | Lip sync | **Out of MVP** (dub audio only) | Most expensive stage (~10–60× real-time) |
| D7 | Translation engine | OpenAI-compatible client → local `llama-server` or cloud | Satisfies D-AGENTS requirement, swappable by config |
| D8 | TTS | Qwen3-TTS Base (`generate_voice_clone`) | Already proven working in this repo |
| D9 | Accent default | `preserve` (clone source speaker directly) | Matches D1; `native` mode comes in P2 |

---

## 3. MVP

### 3.1 Scope

**In**
- One video in → one dubbed video out, via a single CLI command.
- Single source language (auto-detected at video level, overridable by flag).
- Single speaker; target language **French**.
- Voice cloning with the speaker's own timbre (accent preserved — D9).
- Audio re-timed to the source timeline (no lip sync).
- Subtitles: original (SRT) + translated (SRT) as a bonus deliverable.

**Out (deferred to later phases)**
- Multi-speaker diarization (P1), background music/SFX preservation (P1).
- Native target accent via voice conversion (P2).
- Iterative length-control loop, reverb/loudness matching, QC metrics (P3).
- Lip sync (P4), multi-source-language videos (P5), service-ification (P6).

### 3.2 Pipeline

```
video.mp4
   │  ffmpeg: extract audio, resample 48 kHz mono, normalize loudness
   ▼
audio_48k.wav
   │  Silero VAD → cut utterances, drop silence/music
   ▼
segments.json                # [{id, start, end}]
   │  faster-whisper (word_timestamps=True) → transcript + timings
   │  + language detected (video-level, single lang)
   ▼
transcript.json              # [{id, start, end, text, words[]}]
   │  LLM translation (OpenAI-compatible), one call per scene/batch
   │  prompt: role, register, target language, per-segment length budget
   ▼
script.json                  # [{id, start, end, source, target}]
   │  Qwen3-TTS Base: build clone prompt once, synthesize each line
   ▼
dubbed_segments/*.wav        # one file per segment
   │  rubberband: fit each segment's duration to its source slot (anchored)
   ▼
dubbed_segments_sync/*.wav
   │  ffmpeg: place on timeline, mix, mux back onto video; write SRTs
   ▼
output/
  ├─ video_dubbed.mp4
  ├─ subs_source.srt
  └─ subs_target.srt
```

The donor voice is either provided (`--ref-audio` + `--ref-text`) or, by default,
auto-extracted from the video: the longest clean segment's audio + its Whisper
transcript. `ref-text` must match the donor clip exactly (ICL mode).

### 3.3 Tech stack

| Stage | Tool | Notes |
|-------|------|-------|
| Media | `ffmpeg` + `soundfile` + `pyloudnorm` | ffmpeg is NOT yet on PATH — install first |
| VAD | `silero-vad` | tiny, robust |
| ASR | `faster-whisper` (`large-v3`) | word timestamps; avoids the pyannote/HF-token dependency for now |
| Language ID | Whisper's detected language (+ optional `fasttext` `lid.176` cross-check) | single, video-level for MVP |
| Translation | `openai` client → any OpenAI-compatible base URL | local `llama-server` or cloud |
| TTS | `qwen-tts` (Qwen3TTSModel, Base) | proven in `clone.py` |
| Timing | `rubberband-cli` (fallback `sox tempo`) | keep stretches within ~±15–20% |
| Mux | `ffmpeg` | restore video track, burn/embed subtitles |

### 3.4 Job config

A small JSON/YAML manifest per job (also the future queue-message shape):

```yaml
video: input.mp4
target_language: fr
source_language: auto            # or explicit, e.g. "en"
ref_audio: null                  # auto-extract from video if null
ref_text: null                   # required when ref_audio is given (ICL mode)
keep_subtitles: true
output_dir: output/
```

### 3.5 Repository layout (tool-sized, service-ready)

```text
qwen-tts/
├── dub.py                    # CLI entrypoint (argparse)
├── pipeline/
│   ├── config.py             # Pydantic job schema
│   ├── media.py              # ffmpeg wrappers (extract, mux, loudness)
│   ├── asr.py                # VAD + faster-whisper
│   ├── translate.py          # OpenAI-compatible client + prompt
│   ├── tts.py                # Qwen3TTSModel wrapper
│   ├── timing.py             # rubberband fitting
│   └── orchestrate.py        # runs stages, writes artifacts, caches
├── models/                   # local checkpoints (existing)
└── PLAN.md
```

Module boundaries mirror the future `app/{api,services,ai_workers,media}`
layout, so P6 is mostly "add a queue + API in front of these functions".

### 3.6 Definition of Done

- One command runs the whole thing: `python dub.py --video in.mp4 --target fr`.
- Produces `video_dubbed.mp4` + two SRTs.
- Same voice timbre as the source speaker (subjective listen + SECS check).
- Audio synchronized to the source timeline within ~100 ms at segment anchors.
- Handles a ≤ 10-minute video on one RTX 3060 without OOM.

### 3.7 Prerequisites to install

```
ffmpeg  rubberband-cli  faster-whisper  silero-vad  pyloudnorm  openai
```

---

## 4. Improvement roadmap

### Overview

| Phase | Focus | Key additions | Complexity |
|-------|-------|---------------|------------|
| **P0** | MVP (this doc §3) | Single-speaker dub, French, no lip sync | — |
| **P1** | Multi-speaker + background | Diarization, voice bible, Demucs bg preservation | M |
| **P2** | Native accent | VoiceDesign accent anchor + voice conversion; per-speaker accent knob | M |
| **P3** | Quality & timing | Isochrony loop, reverb/loudness matching, QC + eval harness | M |
| **P4** | Lip sync | LatentSync/MuseTalk, gated on face visibility | L |
| **P5** | Multi-source-language | Per-utterance LID, per-segment source language | S–M |
| **P6** | Service-ification | FastAPI + Celery/Redis, model residency, consent gate | L |

### P1 — Multi-speaker & background preservation

- **Diarization**: pyannote `speaker-diarization-3.x` + 3D-Speaker embeddings;
  build a **voice bible** `{speaker_id → {gender, timbre, accent, register}}`.
- **Per-speaker cloning**: one clone prompt per speaker, reusing the same
  reference-across-lines pattern from `clone.py`.
- **Overlap**: flag overlapping speech; pick dominant speaker in v1, separate
  later (MossFormer2/SepFormer).
- **Background bed**: Demucs split dialogue vs. music/SFX; re-mix only dialogue.
- **Output**: multi-voice dub with preserved background.

### P2 — Native target accent (nice-to-have per D1)

- **Accent anchor**: download `Qwen3-TTS-12Hz-1.7B-VoiceDesign`; generate a
  native-target reference with an accent description ("voix masculine, français
  natif, accent parisien").
- **Voice conversion**: Seed-VC / kNN-VC to swap source timbre onto the
  native-accent synthesis.
- **Product knob**: per-video and per-speaker `accent: preserve | native`.
- **Output**: their voice, native target accent.

### P3 — Quality & timing

- **Isochrony loop**: translate → synthesize → measure → re-prompt if too long
  (max N iterations), then rubberband only for the remainder.
- **Acoustics**: RT60 estimate + convolutive reverb matching; EBU R128 loudness.
- **QC stage + eval harness**: SECS (speaker similarity), round-trip ASR WER,
  timing-drift histogram, loudness compliance; threshold → review queue.
- **Output**: measurable quality gates and a model bake-off harness.

### P4 — Lip sync

- Detect face size/visibility per shot; run lip sync **only** on close-ups.
- **LatentSync** (quality) or **MuseTalk** (speed); Wav2Lip as fallback.
- **Output**: dubbed video with regenerated lip motion; batch/overnight job.

### P5 — Multi-source-language videos

- Per-utterance **LID** (VoxLingua107 / FastText `lid.176`); translation node
  accepts per-segment source language.
- **Output**: mixed-language source videos translate correctly.

### P6 — Service-ification

- Wrap the tool behind **FastAPI** (async) + **Celery/Redis** per
  `technical-guidelines.md`.
- **Model residency**: LLM in an OpenAI-compatible model server; audio models
  pinned per worker type; lip-sync in its own pool (see "model residency"
  decision in the appendix).
- **Job state machine**, content-addressed caching, progress reporting.
- **Consent gate** + synthetic-media disclosure; billing/accounts if needed.

---

## 5. Risks

1. **Isochrony** is the top quality risk — de-risk in P0/P3, not later.
2. **Accent fidelity** is only indirectly controllable; validate the P2 path on
   real clips before committing architecture to it.
3. **Speaker consistency** over long videos may drift; per-speaker LoRA
   fine-tuning on minutes of their audio is the fallback.
4. **Lip sync** dominates cost; keep it gated and last.
5. **Whisper hallucination** on silence/music — gate transcription with VAD +
   no-speech probability from day one.
6. **Licensing**: keep to permissive weights (Qwen3-TTS Apache-2.0, CosyVoice,
   Whisper); avoid CPML (XTTS) for anything commercial.
7. **Consent/legal**: voice is biometric data (GDPR) — design the consent flag
   early even if the UI lands in P6.

---

## 6. Appendix — full target architecture (reference)

### 6.1 Stage pipeline

```
[VIDEO IN] → 1 demux/normalize → 2 source separation (Demucs)
           → 3 VAD/segmentation → 4 ASR + word timestamps (WhisperX)
           → 5a LID per utterance  +  5b diarization + speaker registry
           → 6 dubbing-aware translation (LLM)
           → 7 TTS with native-accent reference → 8 voice conversion (source timbre)
           → 9 timing/isochrony → 10 mix (bg bed + R128 + reverb)
           → 11 lip sync (close-ups only) → 12 mux + subtitles + QC → [VIDEO OUT]
```

### 6.2 Layered mapping (for P6)

```
client → app/api (FastAPI, async) → app/services (job state machine, DAG, voice bible)
                                   → app/ai_workers (one engine per stage, pinned to GPU)
                                   → app/media (ffmpeg wrappers)
```

### 6.3 Model residency decision

- "Stateless" means **stateless about jobs**, not about models — a worker may
  hold a model in VRAM and still be replaceable if job state lives in Redis.
- LLM: OpenAI-compatible model server (local `llama-server` or cloud).
- Audio models: pinned per worker type (GPU0: ASR/diarization; GPU1: TTS/VC).
- Lip-sync diffusion: separate pool (memory-heavy, batch-friendly).
