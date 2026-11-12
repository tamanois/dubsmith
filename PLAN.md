# AI Video Translation — Plan & Roadmap

> Companion to `AGENTS.md` (project vision) and
> [`docs/development/guidelines.md`](docs/development/guidelines.md)
> (engineering rules). This file records the MVP scope and the phased
> plan to reach the full vision. Keep it updated as decisions change.
>
> Measurement notes refer to local clips under `input/`; those files are not
> distributed with the repository.

---

## 0. Implementation status

**MVP (P0) — implemented.** See [`README.md`](README.md) for the quick start and
[`docs/`](docs/README.md) for the full documentation.

| Area | Status | Where |
|---|---|---|
| CLI + model download | ✅ | `dub.py` (`models --list/--download`, `--video/--audio`) |
| Local browser UI + HTTP API | ✅ | `server/` (FastAPI, SQLite jobs, SSE), `web/` (React), `docs/web-ui.md` |
| Config, env, languages | ✅ | `pipeline/config.py` |
| Model registry / resolution | ✅ | `pipeline/models.py` |
| Media (ffmpeg, loudness, SRT) | ✅ | `pipeline/media.py`, `pipeline/subtitles.py` |
| Subtitle embedding (soft default, burn opt-in) | ✅ | `pipeline/media.py`, `pipeline/subtitle_pipeline.py`, `--subtitles-mode` |
| ASR + VAD gating | ✅ | `pipeline/asr.py` |
| Translation (OpenAI-compatible) | ✅ | `pipeline/translate.py` |
| Voice cloning + donor selection | ✅ | `pipeline/tts.py` |
| Voice library: design / clone / import, cards | ✅ | `pipeline/voices.py`, `dub.py voices` |
| Standalone speech generation + emotion | ✅ | `dub.py say`, `pipeline/prosody.py` |
| Expressive dubbing backends | ✅ | `pipeline/backends.py`, `--backend`, `--prosody-plan` |
| Emotion/prosody provenance | ✅ | `pipeline/provenance.py`, `prosody.json`, `provenance/` |
| Speaker diarization (P1) | ✅ | `pipeline/diarize.py` |
| Overlap detection + separation (P1) | ✅ | `pipeline/diarize.py`, `pipeline/overlap_separation.py`, `pipeline/orchestrate.py` |
| Timing / isochrony | ✅ | `pipeline/timing.py` |
| Room matching (P3d) | ✅ | `pipeline/mixing.py`, `--room match` |
| Quality gate (P3e) | ✅ | `audit_qc.py`, `pipeline/audit.py` |
| Native accent by default (P2) | ✅ | `pipeline/anchors.py`, `Pipeline._stage_accent_anchor`, `dub.py anchors`, `--accent-match` |
| Expressive native accent (`--expressive-accent blend`) | ✅ | `pipeline/orchestrate.py`, `pipeline/cosyvoice.py`, `--expressive-accent` |
| Orchestration, caching, report | ✅ | `pipeline/orchestrate.py` |
| Portability & distribution (P7) | ✅ | `pipeline/config.py`, `dub.py info`, `pyproject.toml`, `LICENSE`, `.github/workflows/ci.yml` |
| Tests | ✅ 1485 tests (all passed) | `tests/` |

Verification performed:

- **Never lose a word** (2026-09-29, `input/louisavy-11m.mp4`, 11 min, 2 voices,
  English → French, `native`): speech deleted **40.63 s → 4.69 s**, lines that lost
  speech 179 → **31**, lines losing >10% 135 → **23**, worst single line 97.5% →
  51.4%, synthesis 699 s → **461 s**, and the sync gate still passes with lateness
  bounded at `--max-overrun` (0.25 s; `shifted_seconds_max` 0.25). Four changes did
  it: the delivery-rate probe (cloned voices measured at 3.57 / 4.00 words/s against
  the accent clip's 2.97, so translations landed on the ceiling — words ÷ ceiling
  median 1.00, mean 0.95 — instead of 1.5–1.8× over it); the pre-synthesis word
  check (39 of 144 offenders reworded, 945 of 1047 words kept); dead-air trimming
  (74 s removed); and a bounded overrun instead of a cut. ASR re-segmented between
  runs (489 / 471 / 478 lines), so the deltas are indicative, not controlled.
- **A cascade found on the way**: with the allowance spent per *step*, a run of
  tight lines pushed the stream steadily later (27 lines in a row, worst 1.89 s).
  The allowance is now bounded per *line* — a line that is already late loses it and
  is cut instead — and the mix stage tracks the placement cursor to know. Priced on
  the job's own durations: allowance 0 → 66 lines / 8.90 s cut; 0.25 → 31 / 4.69 s;
  0.5 → 15 / 2.63 s; 0.75 → 4 / 0.40 s.
- Full pytest suite green (no GPU, models or ffmpeg needed — heavy parts are faked).
- **Blend strength A/B** (2026-09-29, `input/veo-studio-creation.mp4`, 8 s, 2 lines,
  same 4.7 s donor, `native` unless stated): a human accent clip clears a
  `donor_ratio` of 0.8 in one pass (0.8992), the generated anchor reached only 0.861
  after eight passes at a 0.9 target, and under the old 0.7 default it stopped at
  0.79 — the setting whose voice a listener called lost. `--accent-match gender`
  measured *worst* (0.7571, margin −0.001) and returned a slow anchor; so it is an
  accent lever, not a timbre one. Defaults are now `--blend-ratio 0.85`,
  `--blend-iters 5`; a plain `native` run with them landed at 0.8989 on pass one.
- `ffmpeg` integration exercised against real `ffmpeg` 4.4.2: probe, audio extract
  (48 kHz mono), and mux with `-c:v copy`.
- `rubberband` 2.0.0 verified: `--time 1.10` on a 2.000 s clip yields 2.200 s, so
  the `ratio = target/source` convention in `pipeline/timing.py` is correct.
- Missing-model path produces an actionable error naming the exact download command.
- **Multi-speaker run** on the repo's real two-person clip (`input/louisavy.mp4`,
  60 s, English → French, `reference-accent`): 2 speakers found over 15 segments,
  a distinct donor per voice (segments 11 and 4, each one that speaker delivered),
  and one 11.9 s / 13.3 s blend clip each at donor ratios 0.85 / 0.75. Per-speaker
  audit (`audit_timbre.py`) reports 0 of 8 and 0 of 7 lines closer to the accent
  voice, each measured against its own voice's scale.
- **Real end-to-end runs** (real Whisper large-v3 + real LLM + real Qwen3-TTS,
  only the HTTP transport swapped for a local model via the injected translator):
  - Audio mode on a 23 s French voice clip → English dub in ~15 s (faster than
    realtime). Round-trip ASR of the output returns intelligible English matching
    the translation, confirming the voice clone is correct.
  - Video mode on a video with an audio track → `video_dubbed.mp4` with the video
    stream copied and the dub muxed in; round-trip ASR confirms the new audio.
- Whisper download verified end to end: `whisper_large_v3` transcribes French and
  Chinese correctly (language detection at p=0.99).
- Voice identity is measured, not only listened to: `audit_timbre.py` scores every
  synthesised line against the blend clip and the accent clip on a language-matched
  scale. Numbers in P3b and P3c.

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
- **The `flash-attn` banner is import-time noise, not a missing feature.**
  `qwen_tts` eagerly imports both tokenizer generations, and the 25 Hz one prints
  "flash-attn is not installed" from its Whisper encoder while being imported. The
  12 Hz checkpoint this pipeline requires never runs that code — `tokenizer_12hz/`
  has no such module — and the transformer tower already uses PyTorch SDPA
  (`attn_implementation="sdpa"`), so **installing flash-attn would not make this
  pipeline faster**. The print is a bare `print`, not a log record, so silencing
  it needs a one-line stdout guard around the import in `pipeline/tts.py`.
  **Not yet applied.**

**Acceptance for single-speaker videos is met.** `LLM_BASE_URL` points at a running
OpenAI-compatible server, and the user has signed off on real end-to-end runs (a
4:47 TED talk plus two short clips) in `reference-accent` mode. What remains
unverified is the *long-video* half of the DoD (§3.6): nothing longer than 4:47 has
been run, and that run still relied on the synthesis OOM-retry path.

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
  trimmed, because overrunning it would desync the video. A line may now borrow up
  to `--max-overrun` seconds of the next line's time before anything is deleted, so
  `shifted_segments` is non-zero by design and the gate judges `shifted_seconds_max`
  rather than the count (see the P3 follow-up).

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

Measured on the same French lines with the same donor (one controlled run, so the
three rows are directly comparable):

| Mode | Timbre similarity to donor | French acoustic fit (avg logprob) | English-forced logprob (higher = more English-like) |
|---|---|---|---|
| `preserve` (default) | 0.9841 | −0.138 | −0.351 |
| `speaker-only` | 0.9876 | −0.081 | −0.262 |
| **`reference-accent`** | 0.9854 | **−0.057** | −0.215 |
| *(same speaker, different utterance)* | 0.9808 | — | — |

All three modes retain timbre above the natural same-speaker baseline (0.9808).
Forced to French, `reference-accent` gives the best French acoustic fit; forced to
English, it is the *least* English-like of the three. Both signals point the same
way: the accent moves furthest toward native French, while the voice stays the
speaker's. Listening is still the ground truth, but the direction is unambiguous.

Speaker-only retains timbre within natural same-speaker variation while fitting
French phonetics noticeably better, so the accent problem may need no
voice-conversion stage at all. Listening is still the ground truth; the full
two-stage path (VoiceDesign accent reference + voice conversion) remains the
fallback if it proves insufficient.

**4. Still audibly accented (addressed by `reference-accent`).** The user
listened again and `speaker-only` was not enough. Asking the model to pick its
own phonetics gives up control over delivery; the fix is to *supply* the
pronunciation instead of hoping for it.

The checkpoint splits what we need: `ref_code` carries pronunciation and prosody,
while `ref_spk_embedding` carries vocal identity. `x_vector_only_mode`/`icl_mode`
only decide which of the two the model *uses*, so they can be sourced from
different clips. New `TTSEngine.prepare_hybrid_prompt()` reads the accent clip in
ICL mode and the timbre clip in x-vector mode, then rebuilds one prompt item via
`dataclasses.replace(accent_item, ref_spk_embedding=timbre_item.ref_spk_embedding)`.
Because the timbre side is only ever an embedding, it needs no transcript.

| `--accent-mode` | Speech codes (accent, pacing) | Speaker embedding (timbre) |
|---|---|---|
| `preserve` | the reference voice / auto donor | same clip |
| `speaker-only` | none — the model decides | the reference voice / auto donor |
| `reference-accent` | `--accent-audio` | the reference voice / auto donor |

Additive by design: no mode was renamed, removed, or re-defaulted, and
`--accent-audio` without the matching mode is a hard error rather than a silent
no-op. The accent clip is recorded in `report.json` under `donor.accent`, and
`_donor_info()` re-derives it on cache hits so a second run still reports it.

Full-pipeline runs on the 4:47 TED talk (English → French), both end to end with
real Whisper, real LLM, real Demucs and real Qwen3-TTS:

| | `preserve` + slow French donor | `reference-accent` (hybrid) |
|---|---|---|
| Segments | 59 | 60 |
| Speech trimmed to keep sync | 22.98 s | **13.36 s** |
| Speech reclaimed into pauses | 10.51 s | 7.32 s |
| Lines shifted late / max drift | 0 / 0.000 s | 0 / 0.000 s |
| Loudness vs source | −0.0 dB | −0.0 dB |
| Background preserved | ✅ | ✅ |
| Round-trip WER (per segment) | 1.6% | 1.2% |
| Wall clock | — | 2 m 48 s (2× realtime) |

Read that table with care: the two runs used different segmentations, so it is
*not* a controlled comparison. The controlled experiment is the A/B table above;
these rows only show that the mode works end to end without regressing sync,
loudness, background or intelligibility. Round-trip WER is measured per segment
rather than on the assembled track, where Whisper hallucinates into the silence
gaps.

Known cost: the accent clip's *pacing* rides along with its accent. A slow
reference lengthens every line, which means more compression and more trimming —
the same coupling documented for `--ref-audio`. Mitigation is a naturally-paced
accent clip; the structural fix — sizing each line's word ceiling from the voice
that will speak it rather than from the clip — is in place (see the rate probe in
the pipeline diagram and §0's "never lose a word" entry).

**Confirmed by the user on the same video, with a faster accent clip.** Swapping
the slow personal recording for a 14.8 s narrated news clip (~3.6 words/s, and an
MP3, confirming the accent path is not WAV-only) cut trimmed speech to **2.32 s**,
with 0 drift, loudness delta −0.0 and the background preserved. That is the best
configuration measured so far, and it settles the pacing question empirically:

| Reference used as | Pace | Speech trimmed |
|---|---|---|
| auto donor (the source speaker) | ~3.5 words/s | 11.4 s |
| slow personal recording, as donor | ~2.0 words/s | 23.0 s |
| slow personal recording, as accent clip | ~2.0 words/s | 13.4 s |
| fast narrated clip, as accent clip | ~3.6 words/s | **2.3 s** |

Two things follow. Borrowing only the accent is cheaper than borrowing the whole
voice (13.4 s vs 23.0 s for the same slow clip). And reference *pace* is a trade
rather than a win — the README presents it that way. The "prefer a brisk narrated
clip" advice above was later corrected: see P3 below, which shows the pace knob
trades silence against deleted words and cannot fix either.

### P3 — length feedback to the translator (IMPLEMENTED)

The problem was diagnosed with real data rather than assumed. On the TED run the
report's `clamped_segments` split into **21 lines too short** (harmless: they left
silence) and **8 too long** (harmful: they lost words). Damage was concentrated —
2 of 58 lines carried 75% of all deleted speech, one losing **59% of its own
words** — so the aggregate `trimmed_seconds_total` (0.83% of audio) looked fine
while the result was audibly broken on those lines. Conclusion: optimise
worst-line deletion, not the total.

Three findings shaped the design:

- **Pace cannot fix it.** Sweeping the accent clip's pace across five values moved
  every metric monotonically, but in opposite directions: slowing the clip cut
  silence 15.2 s → 4.3 s while deleted speech rose 0.7 s → 11.2 s. At the best
  setting only 29/55 lines fitted, so ~47% miss at *every* pace. The mismatch is
  textual, not temporal. (Re-pacing is therefore ruled out; the accent and timbre
  probes showed it is cost-free but useless.)
- **Deletion is worse than silence.** Silence is a pause; deleted words are lost
  information. That asymmetry sets the priority order.
- **Synthesis is stochastic.** Identical settings gave 0.71 s and 2.32 s of
  trimming in two runs, so single-run deltas are unreliable; only large,
  monotonic effects across many lines can be trusted.

What was built (options 1+2, additive and default-on):

1. **Instrumentation first** (`Placement.synth_seconds`/`deficit`, plus a report
   split of `clamped_long/short_segments`, `worst_line_deleted_ratio`,
   `lines_over_10pct_deleted`, `speech_rate_wps`). Without this the two failure
   directions were indistinguishable, and "not degrading" was unfalsifiable.
2. **Targeted shorten-and-retry** (`--refine`, default on). Lines that overflow
   slot + following pause are sent back to the translator together with their
   previous wording, the source line, and a word ceiling derived from the measured
   overflow; only those lines are re-synthesised. Safe by construction: lines that
   already fit are never touched, a rewrite is kept only if it measurably reduces
   the overflow, and a failure is a non-fatal no-op. Rewrites start from the
   original text so re-runs cannot double-shorten; the spoken version is stored in
   `03b_spoken.json` and used for the target subtitles.
3. **Gated word budget** (`--length-budget words`, default `off`). Converts the
   slot into a word ceiling measured from a target-language reference clip, since
   a language model can count words but cannot estimate seconds. The general form
   was deliberately *not* shipped: with the default auto-donor the only reference
   is the source speaker, whose rate describes the wrong language, so the budget
   is skipped in that case rather than guessed.

Verified end to end on the 4:47 TED talk, identical settings apart from the flag:

| Metric | `--no-refine` | with repair |
|---|---|---|
| Speech deleted | 3.76 s | **0.0 s** |
| Worst single line deleted | 59% | **0%** |
| Lines losing >10% of speech | 5 | **0** |
| Output exceeding slot + pause | 4.79 s | 0.51 s |
| Silence left in slots | 18.35 s | 17.11 s |
| Round-trip WER (per segment) | 1.5% | 1.2% |
| Drift / loudness / background | — | unchanged |

3 of 57 lines were rewritten in one extra translator call. Cost: the synthesis
stage went from 56 s to 73 s. Caveat: the two runs segmented slightly differently
(56 vs 57 lines) because ASR re-ran, so the sweep is not perfectly controlled; the
within-run repair statistics and the worst-line ratio are the clean evidence. The
rewrites were inspected by hand and preserve meaning, e.g. #39 went from 21 to 14
words with the same content.

#### P3 follow-up — fix the ceiling, then enforce it (2026-09-29)

The repair above worked on the *symptom*: it rewrote lines after synthesising them
and then re-synthesised the rewrites. On the 11-minute job it still left 40.6 s of
deleted speech, so the mechanism was measured rather than guessed:

- **The ceiling was wrong.** It was derived from the accent clip's own reading rate
  (2.97 words/s on `models/anchors/fr.wav`), but the clone speaks faster: 3.65
  words/s measured on that job. A ceiling a fifth too tight is one the model
  ignores, and it did — the translations came out 1.5–1.8× the ceiling.
- **Two length rules were being sent at once.** `SYSTEM_PROMPT` rule 3 asked for the
  *source's* length (±20%), which a fast source speaker makes unachievable for a
  slower target voice; rule 7 gave the ceiling. The model followed rule 3.
- **Then the answer arrived too late.** Overflow was only measured after synthesis,
  so 192 lines were spoken twice to recover 26.5 s.
- **Dead air was charged to the budget.** A synthesised clip often opens with 0.1–0.35 s
  of digital silence (`0.5 s` on one measured line); placed at the slot start it is
  audible dead air, and the mixer compressed real speech to pay for it (0.82 where
  the speech alone needed 0.97).
- **`min_stretch` was doing two jobs.** At 0.85 it was both "how far may a line be
  compressed" and "where does the trimming start", so the 40.6 s that a lower floor
  could absorb was simply cut. No single line needed more than 0.73 and none lost
  more than 1 s — the losses were small and numerous.

What shipped (all default-on, each measured):

1. **Rate probe** (`03a_rate.json`, stage `rate-probe`): one 40-word render per
   cloned voice, timed, *before* translating. `_stage_blend` moved ahead of
   `_stage_translate` so the probe renders through the real clone prompt; both
   stages share one model load, and `_ensure_speaker_prompt` stops the synthesis
   from rebuilding a prompt the probe just primed.
2. **One length rule**: with a budget present, rule 3 *becomes* the ceiling rule.
   Ceiling = `window * delivery_wps / budget_speedup`, where the window is the slot
   plus the pause before the next line and `budget_speedup` (0.85) is the tempo the
   budget is willing to spend on every line.
3. **Pre-synthesis check** (`03d_prefit.json`, stage `prefit`): one batched
   `shorten()` call for everything over budget, adopting a rewrite only when it is
   measurably shorter. `03_script.json` keeps the model's own wording for
   provenance; `_spoken_overrides()` merges the pre-fit and measured rewrites, the
   latter winning, and the subtitles follow.
4. **Compression floor** 0.75 (`JobConfig.min_stretch`, `DEFAULT_BUDGET_SPEEDUP` for
   the budget tempo) and `fit_ratio()`: compress toward the slot, spend the
   following pause before going below the floor, and never compress a line more
   than it needs. One rule, shared by `fit_line`, the repair's predictor and the
   report, so they cannot disagree.
5. **Dead-air trimming** (`timing.trim_silence`, applied before fitting in the mix
   and before measuring in the repair): relative to each clip's own peak, with a
   50 ms guard and a floor on how much may be removed.
6. **Reporting**: `delivery_wps`, `budget_wps`, `compress_floor`,
   `accelerated_segments`, `needed_below_floor_segments`, `needed_ratio_min`,
   `silence_trimmed_*`, `worst_trimmed_lines`, a `prefit` block, and `needed_ratio`
   per line in `timing.json`. A mix that loses speech now logs a WARNING naming the
   five worst ids instead of only reporting a total.
7. **Shift guard** (`SHIFT_FLAG_GUARD_S`): a line fitted to end exactly on the next
   one's start can land a sample past it; that is rounding, not desync, and calling
   it a shift would fail the QC gate on arithmetic.

Verified on the 11-minute job (numbers in §0). The remaining 6.74 s was confined to
lines whose source slot is impossible — `needed_ratio` down to 0.15, e.g. four
words in 0.16 s because Whisper squeezed that stretch of speech — so two further
rounds followed, both measured first:

**Lateness instead of deletion.** The 52 remaining lines each missed their window by
a *median of 0.11 s* (p90 0.22, p99 0.39), in 43 chains whose longest was four
lines, so lateness rarely compounds. Two remedies were priced against the real
fitted durations:

- *Compress deeper* cannot do it: the median stubborn line needs 0.64 (56% faster),
  and even a 0.65 floor still cuts 28 lines / 3.02 s.
- *Allow an overrun* `B` seconds per line: 0.15 → 1.82 s cut, **0.25 → 0.70 s**,
  0.35 → 0.41 s, unlimited → 0 with one line 1.15 s late.

Shipped `max_overrun` = 0.25 s (`--max-overrun`; `0` restores the old exact sync).
`_stage_mix` passes `max_overrun = gap + allowance` and
`available = window + allowance`, and `_overrun_seconds` mirrors both through
`fit_ratio` so the repair no longer fires for a line the mixer can now fit. The sync
gate had to change with it: `max_shifted_segments` (a count, limit 0) is now
`max_shift_seconds` (`audit_qc.py --max-shift`), with a 0.02 s tolerance so a run
sitting exactly on its policy does not fail on rounding.

**The first run taught the difference between per-step and per-line.** Spending the
allowance per step let a run of tight lines push the stream steadily later: 248
lines late, a chain of 27, the worst 1.89 s. The allowance is now bounded *per line*
— `_stage_mix` tracks the placement cursor, a line that is already late loses its
allowance, and it is cut instead — so no chain can exceed it. The final run: 31 lines
losing 4.69 s, lateness bounded at 0.25 s, sync gate pass. The curve, simulated on
the job's own fitted durations and confirmed against the run: allowance 0 → 66 lines
/ 8.90 s; 0.25 → 31 / 4.69 s; 0.5 → 15 / 2.63 s; 0.75 → 4 / 0.40 s.

**The timing was wrong, not the text.** Whisper squeezed consecutive lines into less
time than their words need — 9 words in 0.9 s, and on the real job **60 runs over 94
lines, worst 70.0 words/s** against a 4.7 w/s speaker. `timing.redistribute_packed`
re-times such a run before translation: each line takes a share of the run's span
plus the pause after it, proportional to its words (`distribute_seconds`, with a
0.3 s floor). The run keeps its start and never passes the next line's start, so
nothing outside it moves and **no sync is traded** — it only fixes boundaries that
were impossible, and widens the word budget for the lines that had none. Reported in
`report.json` → `packed`, absent when the boundaries were already plausible.

Deviations from the original plan (deliberate, documented):
- **`silero-vad` dependency dropped.** faster-whisper bundles an ONNX Silero VAD
  and enables it via `vad_filter=True`, so a separate package was redundant.
- **Stretching has three backends** (`rubberband` → `sox` → `librosa`) instead of
  a hard `rubberband` requirement, so audio-only jobs run with no system binaries.
- **`pyloudnorm` + numpy for loudness** instead of an ffmpeg `loudnorm` pass: it
  works identically for audio-only input and avoids a second ffmpeg invocation.
- **Cache fingerprint is per-job, not per-stage.** Changing the target language
  re-runs ASR too. Acceptable for a CLI; refine before P6.

### P3b — Two-stage blended reference for `reference-accent` (IMPLEMENTED)

**The problem, as heard.** In `reference-accent` the voice changes part-way
through a video: some lines ship the source timbre with the borrowed accent,
others come out as the accent clip's own voice. To a listener the same speaker
sounds like two different people.

**The mechanism.** A `reference-accent` prompt item carries two signals sourced
from *different clips*: the accent clip's `ref_code` + `ref_text`, which is a
complete in-context example of that speaker talking (`generate_icl_prompt`,
`modeling_qwen3_tts.py:1972`, injected at `:2188`), and the donor's
`ref_spk_embedding`, which is a single token (`:2103`). ICL conditioning is the
strong signal; the embedding is a competing hint. Which one a given line follows
is decided by sampling, and the shipped generation config samples
(`do_sample: true`, `temperature 0.9`, `top_k 50`) with no seeding anywhere in the
pipeline. Measured on the TED talk: segments 0 and 6 collapsed while 1–5 did not,
all five in the **same** `generate_voice_clone` call with an identical prompt and
a `batch_size` of 8. Batching is not the cause — items are left-padded behind a
proper `talker_attention_mask` (`:2241`), so they are independent decodes. The
aggregate timbre figure (0.9854) hid this completely: a handful of collapsed lines
is exactly what an average cannot show.

**A second, independent source of flips.** `_refine_overlong` re-synthesises every
overlong line with a fresh `engine.synthesize(...)` call — a new random draw — and
adopts the result **only if it is shorter** (`if new_overflow >= overflow - 1e-3:
continue`). Nothing checked that the re-roll kept the timbre, so a good first take
could be silently replaced by a collapsed one. In the reference job, ids 30, 36,
37 and 38 were re-rolled this way.

**The fix: reconcile the two signals once, as audio.** Stage 1 renders the accent
clip's transcript through the hybrid prompt (accent codes + donor embedding) and
writes the result to `reference_blend.wav`. Stage 2 clones every line from that
clip, so `ref_code` and `ref_spk_embedding` are both derived from it — they cannot
disagree, and there is nothing left for a line to decide. This is structurally
different from policing the output: it removes the conflict rather than
re-rolling until the conflict resolves favourably.

Because this single clip fixes the voice of the whole video, a collapse here would
be *global* rather than local, so stage 1 is sampled several times
(`--blend-takes`, default 3) and the take kept is the one with the best margin —
`cos(take, donor) − cos(take, accent)` — using the checkpoint's own speaker encoder
(`extract_speaker_embedding`, 24 kHz). A margin that is still negative after
scoring is recorded in the report. That margin is a **ranking** score only: it
orders takes against each other, and every take reads the same text, so any
language penalty cancels. Its absolute value is not comparable across clip pairs
and is deliberately not warned on — see the measurement below, where a good take
scored −0.011. `--accent-blend single-stage` reproduces the previous behaviour
exactly, which is what makes an A/B listen on one video possible without a branch
checkout.

**Costs and properties.** One extra generation (~15 s of audio, a few seconds of
wall clock against a ~68 s synthesis stage). The blend is a cached stage, so a
rerun reuses the clip instead of quietly selecting a different voice — previously
nothing pinned the voice between runs at all. Adding config fields changes the job
fingerprint, so the first run after this re-does every stage once.

**Touched:** `TTSEngine.build_blended_reference` / `speaker_embedding` /
`_cosine` / `BlendedReference` (`pipeline/tts.py`), `Pipeline._stage_blend` and the
extracted `_resolve_donor` (`pipeline/orchestrate.py`), `accent_blend` /
`blend_takes` + `--accent-blend` / `--blend-takes` (`pipeline/config.py`,
`dub.py`). Artifacts: `04_dubbed/reference_blend.wav`, `04_dubbed/blend.json`,
`report.json → donor.blend`.

**Status: implemented, unit-tested, and measured on a real run.** The suite
covers take selection (best margin, ties, degenerate embeddings, encoder
absent), prompt construction, the cached-reuse path, the single-stage escape
hatch, the config guards, the orphaned-clip cleanup, and — from P3c — the donor
ratio, the iteration loop and its early stop.

Measured on the 4:47 TED talk, English → French, `reference-accent`, using the
repo's real accent clip. Stage 1 cost 16.4 s of a 168 s run and produced a 12.7 s
clip; a rerun serves it from cache. The auto-selected donor came out as the same
segment (19, 99.47–107.57 s) as the pre-existing baseline, so both runs could be
compared against the same reference voice.

The decisive measurement is **language-matched**: each dubbed line (French) scored
against the blend clip R (French, donor timbre) and the accent clip (French,
accent speaker). Comparing a French line to the *English* donor would fold a
language penalty into a speaker measurement — see the lesson below. Two anchors
bound the scale: a pure accent-speaker utterance scores −0.0230 and R itself
+0.0228.

| Run | Lines | min | median | max | Lines closer to the accent speaker |
|---|---|---|---|---|---|
| single-stage (baseline) | 55 | −0.0049 | +0.0156 | +0.0301 | **5 (9.1%)** |
| two-stage (default) | 59 | **+0.0176** | **+0.0257** | +0.0310 | **0** |

The two-stage run has no line on the accent side of the scale, and its whole
distribution sits in a tight band far from the collapse anchor, where the baseline
had a tail crossing zero. Caveats: this is *not* a controlled A/B — ASR
re-segmented 55 → 59 lines and the baseline folder came from a separate
"transcript fix" run — and R is itself an intermediate voice (0.968 to the
speaker's English utterances, 0.974 to the accent clip), so absolute margins are
not "fraction of speaker A". The sign of the change and the absence of a tail are
the evidence, not the exact figures.

**Engineering lesson: an absolute threshold needs a language-matched reference.**
The take-selection score is `cos(take, donor) − cos(take, accent)`. That is sound
for *ranking* takes, because every take reads the same text and so carries the
same language penalty. Its absolute value is not comparable across clip pairs. The
first cut of this change warned whenever the winning margin was ≤ 0, and it duly
fired on a perfectly good run (best margin −0.011). A control run on the real
clips showed why: the encoder's same-speaker band is 0.9889 (different utterances)
against a different-speaker floor of 0.9542, a separation of only 0.035 — so a
−0.011 margin is measurement noise next to the language difference, not evidence
of a collapsed voice. The warning was removed and the field documented as a ranking
score. Any future threshold here should be self-calibrating (compare a take to the
accent clip's own internal same-speaker similarity) rather than a constant.

**Pre-existing bug found while auditing.** `04_dubbed/` was never cleaned up, so
clips orphaned by an earlier, differently-split run stayed in the folder — the
audit found 73 files where the run had produced 59. They were never mixed into the
output (the mix walks the transcript, not the directory), but they mislead anything
that inspects the folder, including a per-line audit. `_stage_synthesize` now
deletes anything it did not write.

**Shipped with this change:** `audit_timbre.py`, the language-matched per-line
audit described above (plus the encoder control it was calibrated against, as a
one-off). Nothing in the repo could previously answer "did the voice change?"
except a listener.

**Supersedes earlier work.** A best-of-K reranker over all 55 lines (engine
parameter `voice_takes`) was written and then deleted. It spent its sampling
budget re-rolling every line to find one that happened not to flip; the two-stage
path spends it on the single line that decides the voice of the whole video, which
is both cheaper and a fix rather than a filter. Its `_refine_overlong` timbre
guard is likewise unnecessary now: with a single agreeing reference there is no
timbre to lose on a re-roll.

### P3c — Blend strength: iterating Stage 1 back toward the speaker (IMPLEMENTED)

**What the two-stage fix left behind.** Removing the per-line flip exposed a
simpler and more audible imbalance: the borrowed accent dominated. Measured on
`dsuzu`, the stage-1 clip sat at 0.985 against the accent clip and 0.960 against
the donor, while the accent speaker's own same-speaker band is 0.989 — so the clip
the whole video was cloned from *was*, as far as the speaker encoder could tell,
the accent speaker. Stage 2 then reproduced that faithfully on every line, which is
the "the speaker's voice is barely there" the user reported.

**Why a single pass is biased.** The hybrid prompt is asymmetric by construction:
the accent clip contributes a full in-context example — its speech codes (~165
steps for a 14 s clip) *plus* its transcript — while the donor contributes one
embedding token. The model reproduces the voice it was shown and treats the
embedding as a nudge. `--blend-takes` cannot compensate: every take leans the same
way, because the bias is in the prompt rather than in the sampling.

**The fix: iterate, and stop on a measured target.** Each pass re-reads the
previous clip's codes and re-injects the donor's embedding, so the voice walks back
toward the speaker while the unchanged accent transcript keeps pronunciation
anchored. The stop condition needed the language-matched scale the earlier audit
had already shown to be missing — the donor clip speaks the *source* language, so
any similarity to it folds in a language penalty. The scale is therefore built from
two target-language clips: the accent clip at 0, and at 1 the donor's own voice,
rendered once in x-vector mode (no accent codes) from the accent transcript. The
blend is projected onto that line to give `donor_ratio`.

Defaults are `--blend-ratio 0.5` (at least half-way to the speaker) and
`--blend-iters 4` (cap). The *best* pass is kept rather than the last, since passes
are stochastic. `--blend-iters 1` restores single-pass behaviour, and the anchor
probe is skipped entirely when nothing would read it.

**Verified on `dsuzu`.** One extra pass — 2 of a possible 4, stopping as soon as
the target was cleared — moved the blend off the accent speaker and onto the
speaker's own voice:

| | single pass | 2 passes |
|---|---|---|
| similarity to donor | 0.960 | **0.989** |
| similarity to accent clip | 0.985 | 0.959 |
| `donor_ratio` | ~0.0 | **0.95** |

At 0.989 the French blend sits inside the donor's own same-speaker band (0.989,
English, a different utterance) and outside the accent clip's. Per line, all three
clips of that job scored 0.037–0.043 on the audit's ±0.041 scale — at the reference
end, none closer to the accent voice. Confirmed by ear.

**Cost.** On a 19 s clip the blend stage dominates (46 s against 5.5 s of
synthesis); on a multi-minute video the reverse holds, and each extra pass is one
more synthesis of the accent transcript (~5 s here).

**Knobs, plainly.** `--blend-iters` bounds the work, `--blend-ratio` decides when
to stop, and whichever binds first wins. `--blend-ratio 0` runs every pass blind
and keeps the least accent-like one — the escape hatch for tuning by hand.
`report.json` → `donor.blend` carries `iters`, `target_ratio` and `donor_ratio`,
and a run that misses the target logs which way to turn the knob.

### P1 — Multi-speaker dubbing (IMPLEMENTED)

**The gap.** `AGENTS.md` asks to "identify each voice", but until now one donor
served the whole video: speaker A's lines were cloned from speaker B whenever B
talked longest. Every design choice below exists to avoid a second voice silently
replacing the first.

**Choosing a diarization backend, by measurement.** Three candidates were tried
against the repo's real two-person clip rather than against a benchmark number:

| Candidate | Verdict |
|---|---|
| **Reuse Qwen3-TTS's own speaker encoder** | **Refuted.** All pairwise cosines fell in 0.963–0.991 and the AHC merge profile ran 0.009→0.029 with *no* gap — the two people are not separable by it. A tempting shortcut, since the model is already loaded |
| **`wespeaker`** | Rejected: not on PyPI (HTTP 404), git-install only, which would make `requirements.txt` depend on a moving commit |
| **`funasr` + ERes2NetV2** | **Chosen.** ~30 transitive deps, but every heavyweight one is behind an `extra ==` marker, so it installs without dragging in `datasets`/`onnx`/`torchvision`. Verified non-destructive to the existing environment |

With `iic/speech_eres2netv2_sv_zh-cn_16k-common` (192-dim) the same clip's pairs
span **0.341–0.874**, and the average-linkage merge profile is flat to `0.379` then
jumps to `0.468`. A threshold anywhere in that gap returns the same two speakers,
which is what makes a *constant* defensible — hence `--speaker-threshold 0.40` as
the default rather than an auto-tuned value. The dependency was measured before it
was trusted, and the environment was pinned first (`pip install -c`) so a
resolution conflict would fail loudly rather than silently upgrade numpy.

**Design: embed and cluster, don't re-segment.** The pipeline is already
per-segment, so all diarization has to produce is a *label* per segment:

- Each segment is embedded and the embeddings are clustered, so the labels align
  with the text by construction. A segment is never split — a line is assigned,
  not re-cut — which is what keeps `02_transcript.json` and the subtitles valid.
- Average-linkage AHC over cosine distance, then clusters shorter than
  `--min-speaker-seconds` are folded into their nearest neighbour: below the 3 s
  donor window there is no voice to clone, so such a cluster is noise, not a
  speaker. (The default was 3.0 when this was written and is 20.0 now — see §3.4 —
  because a donor of a second or two exists but does not describe a voice.)
- Labels are assigned by **first appearance**, so `SPK_0` is whoever talks first
  and a rerun numbers them the same way.
- A segment whose embedding fails inherits the previous segment's speaker, and a
  leading run of failures takes the first assigned one. One bad two-second window
  must not invent a third voice.
- Everything except the `funasr` call is a pure function over arrays, which is why
  the clustering is unit-tested exhaustively without a model.

The orchestrator takes the labeler by injection (`Pipeline(diarizer=…)`, mirroring
`separator`), so the suite never loads the embedding model — the same seam that
keeps Demucs out of the tests.

**Design: per-speaker voices, one prompt each.** Synthesis is grouped by speaker,
so the prompt — which *is* the voice — is rebuilt when the voice changes and not
once per line. A pinned `--ref-audio` is the same prompt for everyone, so it is
built once rather than once per group; re-encoding the reference audio per group
would be pure waste. Each speaker gets:

- their **own donor**, selected from the segments *that speaker* delivered;
- in `reference-accent` mode, their **own blend clip**
  (`reference_blend_<speaker>.wav`), built by the same measured iteration as P3c.
  A single shared blend cannot work: its codes and its embedding both describe one
  speaker, which is precisely what makes it stable — and what makes it wrong for a
  second voice. A one-voice job keeps the original `reference_blend.wav` filename,
  so `audit_timbre.py` and the docs are unaffected.

**The voice bible was folded into `donor.json`.** The plan called for a separate
`voices.json`. `donor.json` already means "which voice this job used, and why", so
it now carries `speaker_count`, `primary_speaker` and a `speakers` map (donor
segment, segments owned, speech seconds, accent and blend records per voice). The
dominant speaker's fields are *also* mirrored flat at the top level, so
`donor.source` / `donor.accent` / `donor.blend` keep working for single-speaker
jobs and for anything already reading them. `report.json` gains a `speakers`
summary that is `null` — not a one-entry map — when the video has one voice, so
previous reports are byte-for-byte unchanged.

**Cost.** One embedding pass over the speech stem (cheap next to ASR), plus one
prompt and, in `reference-accent` mode, one blend clip per voice. Batching is
preserved *within* a speaker, so throughput matches the single-speaker path.

**Backward compatibility.** `--diarizer auto` is the default but degrades to one
voice when `funasr` is absent, with a warning naming the install command — the
optional-dependency pattern already used for Demucs and `flash-attn`.
`--diarizer none` is authoritative: it overrides even an injected labeler, because
it is an explicit request to dub every line in one voice.

**Not attempted, deliberately.** Speaker attribution is per segment, so
overlapping speech goes to one speaker; the speaker *count* comes from the
clustering rather than being declared, with `--speaker-threshold` as the knob.
Both are noted in the README's limitations rather than papered over. Overlap is
now *detected* rather than assumed away — see P1b below — but it is still dubbed
in one voice.

**Supersedes an earlier hypothesis.** The first attempt was to reuse the Qwen3-TTS
cloning encoder for diarization; it was refuted by measurement (see the table
above) before any code was written on top of it. The lesson is recorded in D12.

**Two defects the real run exposed**, both fixed rather than worked around:

1. **`device="auto"` silently broke every embedding.** funasr passes the string to
   torch, which has no `"auto"` device, and it fails *after* downloading and loading
   the weights. Because the per-window handler absorbed the exception at debug
   level, the symptom was a single-voice dub and a log line blaming the audio —
   indistinguishable from a video that genuinely has one speaker. Now the device is
   resolved before funasr sees it (`resolve_device`), the model is built once
   *outside* the per-window handler so a load failure propagates, and the first
   window failure is logged at warning level.
2. **A truncated render could become the voice of the whole video.** For one
   speaker the accent transcript came back as a 1.4 s clip, and because the ratio
   search ranks on embedding similarity alone, a fragment that scores well won. A
   fragment is a *failed* render rather than a candidate voice, so renders more than
   half the length of the longest sibling shorter than it are now dropped before
   scoring. Re-running the same job gives 11.9 s and 13.3 s blends at ratios 0.85
   and 0.75.

Both were found only by running the real clip: the unit tests passed throughout,
because they inject a fake embedder and a scripted model.

### P1b — Overlap detection and reporting (IMPLEMENTED)

**The gap.** `_stage_diarize` assigns **one** speaker per ASR segment, so a segment
that contains two voices is given to whoever dominates it: the second voice is
cloned in the wrong timbre, or lost entirely. Nothing recorded that this had
happened, so the failure was invisible — an aggregate timbre score cannot show a
handful of mis-attributed lines, exactly as it could not show the per-line accent
flips that P3b fixed.

**First attempt, refuted by measurement.** The obvious design — re-embed each
segment in short windows and cluster them the way the labels are clustered —
failed on the real clip: it flagged **14 of 16** segments. The reason is worth
recording. On the two-person clip a 1 s window splits one speaker's *own* voice by
up to **1.08** cosine distance, more than three times the 0.34 that separates the
two speakers, so a short-window “split” reports embedding noise rather than a
second voice. Same-speaker window spread falls monotonically with window length
(0.88 at 1.5 s, 0.85 at 2 s, 0.63 at 2.5 s), and so does the share of clean
segments whose decision flips (6/13, 3/13, 1/13). The lesson is the same shape as
D12's: measure the embedder before building a rule on it.

**Detection: long windows compared against speaker centroids.** Two changes, both
forced by that measurement. The window is **2.5 s** with a 1.25 s hop
(`OVERLAP_WINDOW_S` / `OVERLAP_HOP_S`) — the shortest window at which noise stops
driving the answer — and the comparison is *supervised*: the labels already say
who the speakers are, so each window is compared against a **centroid** (the
average of a speaker's own windows) rather than against other windows. Averaging
is the whole point; it cancels the per-window noise that makes window-to-window
comparison meaningless. A segment is flagged when a `MIN_OVERLAP_S` (0.6 s)
**consecutive run** of its windows fits another speaker's centroid better than its
own, *and* its own speaker still owns the majority of it — a segment whose windows
all match another voice is a mislabelled segment, which is the diarization stage's
problem, not an overlap. A one-voice job is `False` by construction: with no second
centroid, no window can match one, so a noisy solo recording can never be flagged.
No new dependency and no new model: the same ERes2NetV2 embedder, called once with
every window of every segment.

**v1 behaviour: report, don't repair.** A flagged line keeps the dominant-speaker
treatment it would have had anyway, so this stage cannot make anything worse — the
only change is that the limitation is measured. Flags are persisted to
`02c_overlap.json` (its own cached stage, so the labels' cache entry is never
coupled to it), annotated on `Segment.overlapped`, logged at warning level naming
the ids, and summarised in `report.json → overlap` (`segments`, `seconds`,
`speech_ratio`, `mode: "dominant"`). The field is absent — not `null` — for a
one-voice job, and the stage is skipped entirely when there is no real labeler
(`--diarizer none`, or `funasr` missing), so single-voice reports stay exactly as
they were.

**Cost.** One extra embedding pass over the speech stem: ~3 windows per 5 s
segment, batched into a single `embed` call — 4.5 s on the real 60 s clip. Cheap
next to ASR or TTS, and cached like every other stage.

**Sensitivity is the price, and it is the honest limit of this embedder.** An
interruption shorter than about a window is missed, because a shorter window
cannot be trusted with ERes2NetV2. On the real two-person clip the detector flags
1 of 16 segments — conservative on purpose, since a false “overlap” would send a
listener hunting a defect that is not there. Frame-level accuracy would need a
dedicated overlap model (pyannote's `OverlappedSpeechDetection`), rejected here for
the same reason D12 rejected pyannote for diarization.

**Correction: this section described code that did not exist yet.** The constants
and the `overlaps` method were never committed. `_stage_overlap` called
`getattr(labeler, "overlaps", None)`, the real `EmbeddingDiarizer` had no such
method, so on real runs the stage returned `{}` and never wrote `02c_overlap.json`
at all. Only the *test fake* implemented `overlaps`, which is why the suite stayed
green while the feature was absent. It is now implemented as described above, with
the algorithm unit-tested against a scripted embedder (no model needed) and its
constants exported (`OVERLAP_WINDOW_S` 2.5, `OVERLAP_HOP_S` 1.25, `MIN_OVERLAP_S`
0.6). The 1-of-16 figure above is the design target to re-confirm on a real run, not
a shipped measurement.

**Re-confirmed on a real run (2026-09-27), and the answer is 0 of 16.** On
`input/louisavy.mp4` the detector finds 2 speakers (9 / 7 segments) and flags
**none**. Instrumenting every segment shows why: for all 16, every 2.5 s window
matched its *own* speaker's centroid (`other-span` 0.00 everywhere), so no window
preferred the other voice. The clip's interruptions are shorter than the window,
which is the documented sensitivity limit rather than a bug. Consequence: on this
clip the separation stage has nothing to act on, so `--overlap separate` is a
no-op here. The stage itself was exercised on a crafted clip with one genuine
flag: with the real SepFormer both separated streams matched the **dominant**
speaker better than the other (0.67 / 0.53 vs 0.56 / 0.49), so `assign_streams`
returned `None` and the line was left whole — the false-positive guard working as
designed, but the happy path (a real overlap split into two voices) is still
unvalidated on real audio.

### P1c — Overlapping-speech separation (IMPLEMENTED)

**The gap.** A flagged line was dubbed in one voice: the second voice was lost.

**Backend, chosen after the plan's premise failed.** The roadmap said MossFormer2
was "already reachable through the optional `funasr` dependency". It is not, in
this environment. funasr 1.4.16 registers **no** separation model, its
`funasr.bin.ss_inference_launch` is absent, and `funasr/models/mossformer/e2e_ss.py`
cannot even be imported (`funasr.models.base_model` is missing). ModelScope (already
a dev dependency) registers only **8 kHz** separation, and its `SeparationPipeline`
imports that same missing funasr launcher. The only 16 kHz, two-speaker option is
**SpeechBrain's SepFormer** (`speechbrain/sepformer-whamr16k`), so `speechbrain`
joins `funasr` and `demucs` as an *optional* dependency: missing it degrades to
`--overlap dominant` with a warning rather than failing a job.

**Design.** `pipeline/overlap_separation.py` wraps the checkpoint behind a
`SpeakerSeparator` protocol (`NullSpeakerSeparator` when absent). The stage runs
only on the flagged segments and before the ASR model is released:

1. separate the segment into two streams; embed each and match it to a speaker
   against the diarization **centroids** — the same averaging that makes detection
   stable;
2. require the streams to be *distinguishable*: the second stream must fit another
   speaker better than the dominant one, or the segment is left alone. That is what
   stops a false-positive flag from splitting a one-person line;
3. re-transcribe each stream (batched into one Whisper load) and build one line per
   voice with absolute timings.

Those lines then flow through the existing translate / synthesise / mix path. The
dominant line reuses its parent's segment id, so donor, timing and report keys stay
stable; the second voice becomes an `overlay` line with a fresh id. `build_timeline`
places overlay lines **at their true time** and sums them instead of pushing them
later — the simultaneity is the point — while their own fit gets no gap spill, so
they cannot desync the following line. Donor selection and the source subtitles keep
using the original ASR segments, so a donor is never taken from a filtered stream.

**Artifacts and reporting.** `02d_overlap_segments.json` holds the per-voice lines
(`{parent_id, speaker, start, end, text}`, plus `separated` and `mode`). The report's
`overlap` record gains `mode: "separate"`, `separated` and `lines` when the split
happened, and stays `mode: "dominant"` when it did not — including when the backend
is missing, so a report never claims a voice it did not recover.

**Limitations, stated plainly.** SepFormer is a fixed two-speaker model, so a third
voice on the same instant is not recovered. Its output is 16 kHz, so re-transcription
is narrow-band; the separated streams are never mixed into the dub (that is always
synthesised TTS). The stage is best-effort: any failure leaves the line exactly as
the dominant-only path would have produced it.

**Supersedes the roadmap's backend choice** (funasr/MossFormer2) for the reason
above; see D15.

### P3d — Room matching: the dub sounds laid over the picture (IMPLEMENTED)

**The complaint.** "The voice is just over the video and not part of it" — reported
by ear, and present since the beginning rather than introduced by the
diarization work. The speaker *identity* is right; the sense of *place* is missing.

**Why, in the code.** Three causes compound, and all three are structural rather
than bugs:

| # | Cause | Where |
|---|---|---|
| 1 | The synthetic voice is dry — nothing anywhere adds reverb | `pipeline/tts.py` output is close-mic'd and anechoic; no reverb code exists in the repo |
| 2 | The original's room was discarded together with its voice | `separation.py`: speech = `("vocals",)`, background = `drums\|bass\|other`. The `vocals` stem is the voice *plus its reverb tail and the room tone it was masking* — the only material that says "this room" — and the dub never uses it. So "background preserved" means **music** preserved |
| 3 | Ducking pulls the bed down exactly where the voice is | `mixing.py`: 6 dB with 150 ms ramps under every utterance. Right for a documentary narrator, backwards for someone standing in the room |

Secondary, same direction: no microphone-perspective matching (distance, off-axis
colouration, codec bandwidth), a line level that never varies against a speaker who
moves and emotes, and no breaths — the gaps behind the voice are digitally silent
*and* ducked.

**Confirm before building.** This cannot be settled by argument, only by ear.
Listen to the three stems: `01b_speech.wav` (voice + room, discarded),
`01c_background.wav` (music only, what actually plays under the dub) and `dub.wav`.
Then A/B the one cause that is already a flag — `--duck-db 0` keeps the bed at full
level under the voice. If that alone fixes the feel, cause 3 dominates and the fix
is a default, not a stage.

**The fix ladder, cheapest first:**

1. **Soften the duck** — lower the default depth, or use a shallower curve. One line.
2. **Room tone as a floor** — reuse the speech stem's non-speech parts as a
   continuous ambience bed under the dub. Nearly free, needs no reverb algorithm,
   and it glues a dry voice into a space surprisingly well.
3. **Convolutive reverb matching** — estimate RT60 from the source speech, build one
   impulse response, convolve each dubbed line at a fixed wet/dry. Post-processing by
   necessity: the checkpoint cannot synthesise reverberant speech.

All three belong in `pipeline/mixing.py`, as a post-synthesis stage, so they cannot
disturb the voice cloning already tuned in P3b/P3c. Steps 1–2 are worth doing
regardless of whether step 3 follows.

**Worth a knob, not a default:** per-video `--room {off, match, none}` — a dry,
close voice-over is the *correct* choice for narrated content, so it should stay
reachable rather than being silently replaced.

**Status: IMPLEMENTED.** All three ladder steps ship together behind
`--room {off, match}`, defaulting to `off`. The plan's third value (`none`) was
folded into `off`: two names for the same behaviour is a trap, not a feature.

Decisions taken during implementation, each to keep the change small and safe:

- **Graceful degradation, not a hard requirement.** `--room match` needs the
  separated speech stem; without one (usually `--no-background`) it logs a warning
  and delivers a dry dub with `report.json → room.applied: false`. That matches the
  project's established pattern for optional features (Demucs, `funasr`,
  `flash-attn`) and never fails a run that would otherwise have succeeded.
- **One convolution over the assembled track, not per line.** The reverb is an FFT
  convolution of the finished speech timeline, so the tail decays across the pauses
  between utterances exactly as it would in the room. Output length is unchanged, so
  the timeline geometry — and the dub's duration — is untouched; only the final tail
  is cut.
- **The RT60 is measured, and overridable.** The estimator was rewritten after the
  first version failed on real audio. The classic Schroeder backward integral —
  the textbook method — integrates *speech activity* rather than reverberation when
  applied to a whole speech stem, because energy keeps arriving as long as the
  speaker talks. Measured on this repo's own stems (`output/louisavy`, 60 s, and
  `output/change-the-world`, 22 s of Demucs `vocals`) it **saturated at the upper
  clamp both times**, i.e. a cathedral for every video. The shipped estimator
  instead measures the short-time energy envelope over a 0.5 s window and takes the
  **median** decay rate: the room's decay, not the fastest transient and not the
  slow drift of a pause. Recovering 0.2 / 0.5 / 0.8 / 1.2 / 2.0 / 3.0 s exactly on
  synthetic decays of known RT60, and returning 2.40 s and 1.83 s on the two real
  stems above. A decay too slow for one window to see falls back to the upper clamp
  (long), never to the lower one — reading "very reverberant" as "no reverb" would
  be a silent, wrong-direction failure. `--room-rt60` pins it.
- **The wet level is the tuning knob.** Because the deliver stage re-normalises
  loudness to the source, the reverb only has to be *present and stable*: the
  synthetic impulse response is normalised to unit energy, which makes
  `--room-wet-db` an input-independent level rather than one that depends on the
  tail length.
- **`room` is absent from the report on a dry run** — not `null` — the same
  convention as `speakers` and `overlap`, so reports written before this feature
  existed read unchanged.

**Measured on real audio**, not only in unit tests. The mix stage was run twice
over a copy of the real `change-the-world` job — same stems, same dub clips, same
rubberband stretcher, only `--room` differing:

| Region | `--room off` | `--room match` | Ratio |
|---|---|---|---|
| Dialogue (10.0–11.0 s) | 0.08115 | 0.08115 | **1.00×** — the voice is untouched |
| Pause (17.5–18.1 s) | 0.00045 | 0.00564 | **12.5×** — room tone + tail + shallower duck |
| Tail after the last line (21.95–22.0 s) | 0.00046 | 0.00126 | **2.7×** — a real reverb tail |
| Dub length | 22.01 s | 22.01 s | unchanged |

Sync is unaffected (`shifted_segments: 0`, `trimmed: 0.00 s`). The measured RT60 on
that job is 1.83 s.

**Verified end to end** on `input/louisavy.mp4` (60 s, English → French, default
`preserve` accent mode): a full `python dub.py --video input/louisavy.mp4 --target
fr --room match` run completed in ~70 s with every artifact written, 13 segments,
2 speakers, per-speaker donors, `--refine` recovering 2.02 s, and
`Room matching: RT60 2.41s` in the log. `05_room.json` and `report.json → room`
both carry the measured decay.

| Run | Segments | Dub | Speech trimmed | Worst line | Lines > 10% | Shifted | Duck |
|---|---|---|---|---|---|---|---|
| `--room off` | 14 | 60.37 s | 1.57 s | 23.3% | 1 | 0 | 6.0 dB |
| `--room match` | 13 | 60.01 s | 1.19 s | 50.0% | 2 | 0 | 3.0 dB |

Read that as *evidence of independence*, not as a controlled A/B: the two runs
re-segmented differently (14 vs 13 lines, and different translations), which is
the documented non-determinism of re-running ASR. What it establishes is that
**truncation is not introduced by room matching** — the dry baseline fails the
gate too (`audit_qc.py` exits 1 on both, on `truncation`). Room matching is
applied after `fit_line`, so it cannot influence trimming at all; the numbers
agree.

**Two pre-existing problems surfaced by running it for real**, both outside this
feature and neither caused by it:

1. **The Definition of Done's own command was broken.** `dub.py` passes every flag
explicitly, including its `None` defaults, and Pydantic counts an explicitly-passed
`None` as "set" in `model_fields_set` — so the accent-blend guard fired and
`python dub.py --video in.mp4 --target fr` raised `ConfigError` for **every accent
mode except `reference-accent`**. Recent real runs all used `reference-accent`,
which is why it went unnoticed. The guard now keys off the value, and the
flag→config mapping is extracted into `job_config_from_args()` so the CLI shape is
tested directly.
2. **Synthesis can still OOM on a single line**, and the batch-halving fallback
bottomed out twice (8 → 4 → 2 → 1) while the second GPU sat idle. Re-running the
same job with its transcript and translation cached then succeeded with **zero**
OOM retries, so this is transient allocator state rather than a text or model
limit — the same ⚠️ DoD item ("≤ 10-minute video on one RTX 3060 without OOM") that
remains unverified, now with a reproduction.

Deliberately not done: microphone-perspective matching (distance, off-axis
colouration, codec bandwidth), per-line level variation against a speaker who
emotes, and breaths. Each refines the same "sense of place" problem rather than
adding a missing capability.

### P3e — Automated quality gate (IMPLEMENTED)

**The gap.** The signals existed — SECS per line, round-trip WER, timing drift,
loudness compliance in `report.json` — but nothing thresholded them, so "the run was
fine" was a feeling rather than a checked claim, and `audit_timbre.py` was a
hand-run script.

**What was built.** The measurement core moved out of `audit_timbre.py` into
`pipeline/audit.py`, so both CLIs share one implementation of the language-matched
measurement instead of drifting apart. `audit_qc.py` was added on top: it reads a
finished job's `report.json`, applies the gates below, prints a pass/fail table, and
exits non-zero when one fails.

| Gate | Checks | Default |
|---|---|---|
| `sync` | No line more than the allowance late | `shifted_seconds_max <= 0.25` |
| `truncation` | No line loses a meaningful share of its words | worst ≤ 10%, none over 10% |
| `loudness` | The dub sits at the source's level | within 1.5 dB |
| `timbre` | No line followed the accent voice | 0% of lines (needs `--timbre`) |

Three decisions worth recording:

- **It never blocks the pipeline.** The dub is written before the gate runs, so a
  failure is a verdict on the delivery rather than a reason to discard the work —
  which is also what makes it usable in CI and by hand on any completed job.
- **Overrun is not gated, and lateness is gated by *amount*.** A line spilling into
  the pause before the next utterance costs nothing and is *how* sync is preserved.
  Since the mixer may also borrow up to `--max-overrun` seconds from the next line's
  time rather than delete a word, the sync gate compares `shifted_seconds_max`
  against that policy instead of counting moved lines — a run that borrowed 0.1 s is
  not the same defect as one that started a line a second late.
- **A gate that cannot run is `skip`, never `pass`.** Missing timbre data or an
  unmeasurable source level must not read as a clean bill of health.

**A latent bug fixed on the way.** `collapse_anchor` averaged two 7 s windows of the
accent clip, so any accent clip shorter than 14 s produced a `NaN` anchor — and the
README's own advice is to use a short, brisk clip. It now measures the clip whole
when it is shorter than one window.

**Still open in P3:** the model bake-off harness, and room matching's own listening
verification (the A/B procedure is in P3d above).

### P3f — First listen: the reverb was rejected (FIXED)

**Feedback from the user, on a real dubbed video:** "voices have echo and speech is
distorted". This is the by-ear gate P3d asked for, and it failed the first cut.

**What was wrong.** The ladder's third rung was enabled by default with settings that
had never been heard:

| Problem | Evidence | Fix |
|---|---|---|
| 2.4 s of **undamped** reverb, flat to 12 kHz | `synthetic_ir` was raw noise; a flat burst reads as a metallic hiss, not a room | Low-pass the response at 4 kHz (−8.7 dB of tilt measured between the 0.2–1 kHz and 9–11 kHz bands, against −0.3 dB before) |
| **RT60 estimated far too long** | 2.41 s for ordinary talking-head footage — the median over all decay windows folds in the shallow dips *between words*, where the voice continues, so the estimator was measuring speech envelope dynamics rather than the room | Take the 75th percentile: the fast, sustained decays that follow speech offsets. Same stem now reads 1.65 s; still a *starting point*, not a measurement to trust |
| **The room-tone floor could carry the original voice** | It was the stem's complement-of-dialogue regions, so speech the transcript never covered (breaths, dropped utterances, overlaps) and the speaker's own decay came along at −30 dB — a ghost of the English voice under the French one | Gate on **level**: only frames far below the stem's peak survive, which is what genuine room tone is. Plus renormalise to the requested level so the gate and the gain cannot compound into inaudibility, with a cap so a stem with no quiet frames yields silence instead of amplified residue |
| A reverb level nobody had calibrated | −12 dB was a guess | **Opt-in, default `0` = no reverb** |

**Measured after the fix**, on the real `louisavy` job (a controlled speech-only
render, so the same lines are compared):

| Run | Post-offset energy | Overall |
|---|---|---|
| `--room off` | −31.1 dBFS | −27.7 dBFS |
| `--room match` (default, no reverb) | −31.1 dBFS | −27.5 dBFS |
| `--room match --room-wet-db -20` | −30.7 dBFS | −27.4 dBFS |
| `--room match --room-wet-db -12` (the old default) | −30.0 dBFS | −26.6 dBFS |

**The honest reading:** the reverb's contribution to a 0.4 s post-offset window is
smaller than those RMS figures suggest (+1.1 dB even at −12 dB), so part of what was
heard was probably the *floor* carrying the original voice — the other thing this
change removes. Both mechanisms are now gone from the default path, but only an ear
can confirm it, which is why the tail is opt-in rather than re-tuned by guesswork.

**Status: fixed, 825 tests pass, re-verified end to end** (`RT60 1.65s measured,
room tone -30.0 dB, duck 3.0 dB, reverb off`). `--room match` now means the two
steps the plan always said were "worth doing regardless of whether step 3 follows".

### P7 — Portability and distribution (IMPLEMENTED)

**The gap.** The pipeline worked end to end on the author's box, and nowhere else
was proven. Three things blocked a second user, and only one of them was code: no
`LICENSE`, so the project was not legally usable at all; no packaging, so it could
not be installed or version-reported; and a device layer that asked a single
binary question (`cuda` or not) and answered it with a hardcoded `bfloat16`.

**Three real defects, each of which hits "another GPU" and not this one.**

| Defect | Effect | Fix |
|---|---|---|
| `resolve_dtype` returned bfloat16 for *any* `cuda*` device | Unsupported or emulated-slow on Pascal (6.1) and Volta/Turing (7.0–7.5) — exactly the second-hand cards a "runs on consumer hardware" project attracts | Capability-gated: bfloat16 from sm_80, float16 from sm_70, float32 below. ROCm uses CDNA (9.4) as its bfloat16 line |
| `DemucsSeparator._device` collapsed `cuda:N` to `cuda` | `--device cuda:1` silently ran Demucs on GPU 0, colliding with whatever else was resident — the second GPU was unusable | The index is preserved; `auto` resolves through the probe |
| ASR passed the raw device string to CTranslate2 | `--device mps` failed *after* the weights loaded, because CTranslate2 has no Metal backend | New `asr.asr_device` maps cuda/rocm to `cuda` and everything else to `cpu`, so a Metal user gets a working dub instead of a traceback |

**A capability layer instead of string matching.** `pipeline/config.py` now has
`DeviceInfo` (`kind`, `index`, `name`, `total_memory`, `capability`),
`probe_device`, `available_accelerators`, `supports_bf16` / `supports_fp16`,
`preferred_dtype` and `pick_dtype`. Every previous `startswith("cuda")` is now a
question asked of that probe, and `release_gpu_memory` returns the caching
allocator on Metal and XPU too, not only CUDA. `pipeline.tts.resolve_dtype`
delegates to it, so the engine and the report cannot disagree about the precision.

**`auto` is deliberately conservative, and that is a decision rather than an
omission.** It picks a device *every* stage can use — CUDA/ROCm when present, else
CPU. It does not pick Metal or XPU even when they are available, because
faster-whisper runs on CTranslate2, which offers only CUDA and CPU: an `auto` that
chose Metal would break a Mac that works today. Explicit `--device mps` is still
honoured, and `dub.py info` reports what is present so the choice is discoverable.
Per-stage device resolution is the fuller fix, and is not built.

**`dub.py info`.** One command answering the only question a user on unfamiliar
hardware can ask: Python and torch versions (with the CUDA/ROCm build tag), the
accelerators present, the resolved device with its memory and compute capability,
the bf16/fp16 flags, the dtype and ASR compute type that *would* be chosen, which
optional packages import, which system binaries are on `PATH`, and which models are
present. It loads no model, so it is safe to run first — and it is what makes a bug
report answerable.

**`--dtype`, and why it is fingerprinted.** `--dtype {auto,bf16,fp16,fp32}`
overrides the automatic choice; an unusable request degrades to the best the device
supports and says so at warning level rather than failing the job. `dtype` is a
`JobConfig` field, so it is part of the job fingerprint: a run at one precision
cannot silently reuse clips synthesised at another. Adding the field changes every
existing fingerprint, so the first run after this re-does each stage once — the same
one-off cost documented for P3b.

**Distribution.** `LICENSE` (Apache-2.0, matching the Qwen3-TTS and Whisper
weights). `pyproject.toml` gains `[project]`, a version single-sourced from
`pipeline.__version__`, `requires-python = ">=3.10"`, PEP 639 SPDX licence
metadata, console scripts (`dub`, `dub-audit-qc`, `dub-audit-timbre`) and the
optional backends as extras (`background`, `diarization`, `separation`, `gpu-asr`,
`full`, `dev`) — so `pip install .` is a real install, and the graceful degradation
the code always had is finally visible in the metadata. Dependencies are pinned to
the verified combination; `torch` stays a floor with the CPU index documented,
because freezing it would drag CUDA wheels onto a CPU-only machine. `speechbrain`
moved from a hard requirement to the `separation` extra, matching how the code
already treated it (D18). CI runs the whole suite on a CPU-only runner for Python
3.10 and 3.12.

**A deliberate omission: no lock file.** A `pip freeze` lock generated on this box
contains the CUDA wheels, which is the opposite of portable. Pinning the direct
dependencies while documenting the torch index achieves the actual goal — a
reproducible install — without freezing one machine's hardware into the
requirements.

**Cost.** One dependency-free probe per run (a `torch` import plus, on CUDA, three
cheap queries), one new `report.json` key, and a one-off full cache invalidation
from the new `dtype` field.

**Verified.** 825 tests pass (69 new). `tests/test_device.py` exercises Pascal,
Volta, Ampere, CDNA, Metal, XPU and CPU, plus a broken probe and a missing torch,
against a fake `torch` module — so the hardware rules are checked on a CPU-only
runner rather than asserted. `tests/test_cli_smoke.py` covers the command surface:
`models --list`, `--version`, `info`, `--dtype` reaching the config, and dtype
invalidating the fingerprint. The metadata and wheel were built directly and
checked (`Name: dubsmith`, `Version: 0.1.0`, three console scripts, `pipeline/`
and the three CLI modules present, `LICENSE` included) with deprecation warnings
promoted to errors.

**Deviations from the plan, both deliberate.** The distribution is named
`dubsmith`, not `qwen-tts`: that name belongs to the TTS checkpoint's own
package, which is a dependency here. And the lock file was dropped, for the reason
above.

#### Two audit defects the first CPU runs exposed

Running the gate on a real CPU job found two problems no unit test had. Both are in
the audit tools rather than the pipeline, and both only show up when the tool is
used the way a second person would use it:

1. **The audit CLIs could not be pointed at a device.** `audit_timbre.py` and
   `audit_qc.py --timbre` both hardcoded `device="auto"`, so on a machine whose
   GPUs were already occupied the audit simply failed — which was exactly the
   situation the measurement was wanted in. Both now accept `--device`.
2. **`audit_qc.py --timbre` aborted a `preserve` job instead of skipping it.** A
   `preserve` job never borrowed an accent, so there is no clip to measure the
   lines against; `audit_job` raised and the process exited 2. That contradicts the
   gate's own contract — *a gate that cannot run is `skip`, never `pass`* — and
   made a mixed batch of jobs uncheckable. `qc()` now consults the new
   `timbre_available()` first and reports `skip` **with the reason**, so the other
   three gates still report and the exit code stays 0. `audit_job` still raises,
   because a dedicated single-job tool asking an unanswerable question should say
   so; only the gate degrades.

### P8 — Long-video memory: the allocator, not the video (IMPLEMENTED)

**The gap.** The DoD asks for a ≤10-minute video on one RTX 3060 without OOM. The
longest run was 4:47, and it hit `CUDACachingAllocator` OOM retries that the
batch-halving fallback absorbed. An 11-minute job (`input/louisavy-11m.mp4`, 476
segments, English → French, ~58 min wall) turned that into a measurement.

**What the log said.** During synthesis, hundreds of lines like:

```
[W CUDACachingAllocator.cpp:3934] memory allocation failed with OOM on device 0
  while trying to allocate 1639972864 bytes (free: 1209729024, total: 12744982528)
```

Three numbers, and they are the whole diagnosis:

| | bytes | GiB |
|---|---|---|
| Card total | 12,744,982,528 | 11.87 |
| Free at the failure | 1,209,729,024 | **1.13** |
| Requested tensor | 1,639,972,864 … 1,723,858,944 | **1.53 – 1.61** |

Every failing request was ~0.4 GiB larger than the *entire* free memory on the
card, and the bursts repeated hundreds of times in ~15 ms at a constant `free:`
value. The job still finished: the allocator recovers internally, so no exception
ever reached `TTSEngine._generate` — which is why the existing halving fallback
never ran, and why `report.json` looked perfectly clean. Retrying at that edge is
one unlucky batch away from a failed run.

**Two causes, both structural.**

1. **The clone prompt's reference is decoded once per line.**
   `generate_voice_clone` concatenates the prompt's `ref_code` onto *every* line's
   codes and only trims the reference back off the waveform afterwards, so a batch
   of eight lines cloned from an 21 s clip decodes ~170 s of audio however short
   the lines are. Measured directly (real checkpoint, the job's own blend clip as
   prompt, RTX 3060 bfloat16):

   | Lines in one call | Audio produced | Peak reserved |
   |---|---|---|
   | 1 | 0.2 s | 4.67 GiB |
   | 2 | 0.5 s | 5.33 GiB |
   | 4 | 1.3 s | 6.74 GiB |
   | 8 | 3.0 s | **9.55 GiB** |

   ~0.65 GiB per line, set by the reference length, essentially independent of what
   the lines produce. The first fix attempt — capping the *lines'* predicted
   duration — changed nothing (`peak_reserved_bytes` 11.55 → 11.58 GiB, failures
   66 on the next run), because the reference term dominates.
2. **Nothing was released between batches.** `release_gpu_memory()` ran at stage
   boundaries and inside the OOM handler only, so across hundreds of calls the
   caching allocator accumulated a pool that had free memory in it but no block
   large enough for the next request. Free memory was measured as low as 54 MB.

**The fix.**

| Change | Where |
|---|---|
| A batch is budgeted in **decoded** audio: the reference is charged to every line on top of the lines' predicted duration | `TTSEngine._take_batch` |
| A measured **ceiling** (`0.7` of the card): a call whose peak exceeds it halves the size of everything after it, and never grows back | `TTSEngine._adapt_batch_size`, `_resolve_ceiling` |
| Cached blocks go back to the driver when free memory falls below 3 GB | `TTSEngine._release_when_tight` |
| `configure_allocator()` applies `PYTORCH_ALLOC_CONF=expandable_segments:True` for CUDA runs, before the first allocation; an explicit user value always wins, and ROCm/XPU/CPU are untouched | `pipeline/config.py`, called from `Pipeline.run()` and both audit CLIs |
| A batch that OOMs shrinks `batch_size` **for the rest of the run**, and batches are taken lazily so the smaller size applies to them | `TTSEngine._generate`, `synthesize` |
| `--tts-batch` (default 8) is the manual ceiling | `JobConfig.tts_batch` |
| `report.json` → `tts` records `oom_retries`, `ceiling_shrinks`, `cache_releases`, `reference_seconds`, `seconds_per_char`, `peak_batch_seconds` and `peak_reserved_bytes` | `Pipeline._tts_stats` |

**Three attempts, two of them wrong** - recorded because the wrong ones are
instructive. Budgeting the *lines'* predicted duration changed nothing: failures
went 66 -> 72 and the peak reserved went 11.55 -> 11.58 GiB, because the lines were
never the term that mattered. Only after reading `generate_voice_clone` and
measuring the prompt itself did the reference-per-line cost appear (table above).

**Verified on the 11-minute job** (`input/louisavy-11m.mp4`, 476 lines, English ->
French, RTX 3060, bfloat16, with `--min-speaker-seconds 20` so synthesis is reached
in minutes instead of an hour):

| | Before the fix | After |
|---|---|---|
| `CUDACachingAllocator` failures | thousands | **0** |
| `report.json` -> `tts` | *absent* | `batch_size 2`, `reference_seconds 17.2`, `seconds_per_char 0.0536`, `batches 267`, `peak_batch_seconds 13.04`, `oom_retries 0`, `cache_releases 1`, `ceiling_shrinks 1`, `peak_reserved_bytes 10,053,746,688` |
| Peak memory held | 11.55 GiB of 11.87 | **9.36 GiB** |
| Synthesis stage | 393 s, with 66 failures | 616 s, with none |

`ceiling_shrinks 1` is the controller visible in the log:

```
A batch of 4 line(s) filled 9.4 GiB, past the 8.3 GiB ceiling for this card; using 2 line(s) per batch from here on.
```

The halving follows the batch that **overshot**, not the configured cap: halving the
cap would have left the next batch as large as the one that just filled the card, and
the earlier version of this rule needed two shrinks where this one needs one. It
never grows back, so a run cannot oscillate around the edge.

**The cost is real and is the trade.** Synthesis went from 393 s to 616 s: 0.65 GiB
of reference decoding per line is not negotiable, so the choice is between more model
calls and a full card. On a card with room the controller never fires and nothing
changes; on this 12 GB card a long job now costs ~55% more synthesis time than the
version that failed 66 allocations.

### P9 — Multi-device synthesis (IMPLEMENTED)

**The gap.** Synthesis is the stage that decides how long a job takes - 616 s on the
11-minute run, and 217 s of the 318 s on the *short* accent run, where the blend
dominates - and it ran on one card. A second RTX 3060 sat idle while the first halved
its batches to stay inside its own memory ceiling (P8), and the one recorded
reproduction of a synthesis failure was a batch that bottomed out at a single line
while the other card did nothing.

**What ships.** `--devices cuda:0,cuda:1` (`JobConfig.devices`). Each device gets its
own worker *process* with its own copy of the checkpoint, and the stages that dominate
the clock hand work to whichever worker is free:

| Stage | Work unit |
|---|---|
| Blend | One task per voice - the voices share nothing, so nothing has to be split |
| Synthesis | One task per voice, or per *piece* of a voice's lines when there are fewer voices than devices |
| Rate probe | One task per prompt, so a voice shared by several speakers is measured once, not once per card |

The split is by **work**, not by model: nothing is sharded inside a checkpoint,
because a clone prompt belongs to the device that built it. The parent keeps ownership
of the job folder, the manifest and the stage cache and never loads an engine of its
own, so a worker that dies cannot corrupt a cache entry. Dispatch is a pull - a worker
is handed the next task the moment it reports a result - so a voice with unusually
long lines cannot leave a card idle behind it, and a task that a card refuses (a
single line past its ceiling) is retried on another card before the job fails.

**What deliberately did not move.** ASR, diarization, separation and the accent
anchors each run once, so they take the first device of the list - that is what
`primary_device` means, and it is why `--devices cuda:1,cuda:0` also decides which
card they use. Chunking ASR across cards was costed and dropped: Whisper re-segments
its own chunk boundaries, so it would need a seam merge and would change the
transcript, and the stage is a small share of the wall clock.

**Verified.**

- **The 994 existing tests pass unchanged.** A job without `--devices` resolves to a
  one-element list and takes the inline path, so the single-device behaviour is the
  *same* code, not a reimplementation beside it.
- **A real end-to-end dub ran on two RTX 3060s** (`input/change-the-world.mp4`, 22 s,
  `--accent-mode preserve`, `--devices cuda:0,cuda:1`). `report.json` → `tts.devices`
  lists both cards; the work split three lines to one card and two to the other
  (`batches` 2 and 1); the per-card peaks were 5.31 GiB and 4.93 GiB, each well inside
  its own 8.3 GiB ceiling and independent of the other; `oom_retries` and
  `ceiling_shrinks` were 0; and `audit_qc.py` exited 0 (sync, truncation and loudness
  pass, timbre skipped).
- **A controlled A/B measured the split itself**: the same five synthesis tasks, same
  texts, reference and model, changing only the number of workers - **19.4 s on one
  card, 14.9 s on two**. The remaining fixed cost is the checkpoint load (~7-8 s per
  worker, paid once per job), which is a large share of a 19 s job and a rounding error
  on the 11-minute one; on this run it landed inside the rate-probe stage, which took
  18.3 s for a single render, leaving synthesis itself at 10.0 s.
- **The pool is tested without a model**: dispatch across devices, retry on another
  card, the all-cards-failed error, a card that cannot open its engine, and
  teardown-on-close all run against stub workers in spawned processes, so CI covers
  them on one CPU.

**Still unmeasured.** The same A/B on a *long* job is where the split should approach
its ceiling: at 22 s the per-task work is smaller than the model load, so the measured
gain was 23% rather than the ~50% the design targets. A many-voice job on two cards
(the blend, one task per voice) is also unrun - the blend is the stage that costs the
most per voice, and it is the one case where no splitting is needed at all.
before the field existed, so the feature invalidated no existing cache.

### P10 — Saved voice prints: design and cloning (IMPLEMENTED)

**The gap.** Both underlying models were already wrapped — `VoiceDesignEngine`
turns a text description into a voice, and `TTSEngine` clones one from a clip and
its transcript — but every use was inline and throwaway. `clone.py` and `main.py`
were standalone demos, and the only persisted voice assets were the per-language
accent anchors. There was no way to name a voice once and reuse it.

**What ships.** A *voice print* is a short reference clip plus its exact
transcript, saved under a label in `pipeline/voices.py`:

- `voices/<label>/voice.wav` — the reference clip (the canonical
  `ANCHOR_TEXT[language]` passage, rendered in the voice; 20–30 s).
- `voices/<label>/voice.json` — label, language, source (`design` | `clone`),
  transcript, instruct, creation time, measured gender and pace band.

Two creation paths and three ways to use one:

```bash
python dub.py voices --design --prompt "a warm female French voice" --lang fr --label warm-fr
python dub.py voices --clone --audio me.wav --clone-text "…" --lang fr --label me
python dub.py voices --list
python dub.py voices --remove warm-fr

python dub.py say --voice warm-fr --text "Bonjour…"
python dub.py say --design "a calm narrator" --text "…" --lang en --save-as narrator
python dub.py --video talk.mp4 --target en --voice warm-fr
```

**Why it is low-risk.** `--voice` is resolved in `job_config_from_args` into
`ref_audio` + `ref_text`, the representation the pipeline already understands, so
no stage, cache fingerprint or fan-out worker changed. The store is pure data
handling (injectable engine, no model import), modeled on `anchors.py`, so the
whole feature is testable without a GPU.

**Accent.** A saved voice supplies the timbre, and the print's own language
decides the accent path: a print already in the target language is cloned
directly (its speech codes carry a native accent, and the costly blend is
skipped), while a print in another language uses `--accent-mode native` and
borrows pronunciation from a generated target-language anchor. An explicit
`--accent-mode` always wins. This reuses the existing clone/hybrid prompts rather
than adding a second accent rule.

**Defaults.** Labels are slugified and de-duplicated (`-2`, `-3`), so an
auto-named voice cannot silently overwrite another; an explicitly named voice is
never overwritten without `--force`. `dub.py say` writes `output/say/<name>.wav`.

**Deliberately out of scope.** The CustomVoice checkpoint's nine named speakers
(a preset-voice source, not a described one), embedding-only storage (the
`qwen_tts` API needs reference audio), and embedding-based de-duplication.

**Verified.**

- **The full suite passes** (1453 tests at the time of the public release): 36
  cover the store and 24 the new CLI surface, all without a model or a GPU.
- **Real voice design** on an RTX 3060: `voices --design` saved a 27.1 s French
  print; `say --voice` round-tripped through Whisper with the exact text.
- **Real cloning** from `input/Delphine-Charming.mp3`: a 22.1 s print; `say`
  round-tripped exactly.
- **Real dub integration** (15 s French clip, `--voice`, real ASR/TTS/timing/
  mixing; translation replaced by an echo because no local LLM endpoint was
  running): 5 segments, synthesis 16.1 s, and the speaker-encoder cosine between
  the delivered dub and the print was **0.9908** — above the 0.9808 natural
  same-speaker baseline measured earlier.
- **The language rule paid for itself immediately.** Before it, the same
  same-language job spent **122.9 s** in the blend and warned it reached only a
  0.385 donor ratio; the rule skips the blend entirely.

### Still open (verified against the code, 2026-10-09)

The capability work is implemented; this is the verified remainder, so the table
above is not read as "finished".

**Verification not yet closed**

- **The multi-device fan-out has not been measured on a long job** (P9). It is
  verified on two GPUs for a 22 s clip, where the 23% it saved is bounded by the
  checkpoint load; on a long job the split has the room to approach 50%, and a
  many-voice job (the blend, one task per voice) has not been run on two cards at all.
- **P3f has not been re-listened to.** The fix is measured, but by its own admission
  only an ear can confirm it — which is why the reverb tail is opt-in.
- **P1c's happy path is unvalidated on real audio.** Only the false-positive guard has
  run against the real SepFormer; a genuine two-voice split is covered by fakes, and
  the detector flags 0/16 on `input/louisavy.mp4` (interruptions shorter than the
  2.5 s window).
- **The §3.6 hardware matrix has one real gap: a second physical machine.** The CPU
  rows and both GPU precisions are measured and all four runs pass `audit_qc.py`,
  but every one of them ran on this box, so a driver- or vendor-level difference is
  still unexercised.

**Code (small and self-contained)**

- The `flash-attn` banner guard is described above but **not applied**, and the same
  missing stdout wrapper lets funasr's `load_data` dump and tqdm bars through in
  `pipeline/diarize.py` (`model.generate(...)`, no `disable_pbar`).
- **Cache fingerprint is per-job, not per-stage** (deliberate; refine before P6).

**Phases not started**

- **P3** model bake-off harness — the only remaining P3 item.
- **P4** lip sync, **P5** multi-source-language (per-utterance LID), **P6**
  service-ification (FastAPI + Celery/Redis).
- **P2's remaining nicety**: the optional voice-conversion path is unnecessary —
  the generated anchor supplies native prosody directly — so P2 is closed.

**Engineering standards** ([`docs/development/guidelines.md`](docs/development/guidelines.md))

- No `mypy --strict`, Black or Ruff: `pyproject.toml` carries packaging metadata and
  pytest settings, and `requirements-dev.txt` only `pytest`, `funasr`, `modelscope`.
- No coverage measurement, so the >85% target is unverified. CI now exists
  (`.github/workflows/ci.yml`: Python 3.10 and 3.12 on a CPU-only runner) and runs
  the whole suite, but it does not measure coverage.
- No `scripts/`; a `Dockerfile` provides an optional portable image (models
  excluded) and is not used by CI.
- No real tiny-fixture integration test; every heavy part is faked.
- `structlog`/`Loguru` are not used (stdlib logging), but every record already carries
  a `job_id` via `pipeline/logging_utils.py`.

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
| D9 | Accent default | `native` — a generated per-language accent anchor supplies pronunciation, the source speaker supplies timbre; `--accent-match rate|gender` fits the anchor to the source voice | Supersedes the earlier `preserve` default. VoiceDesign makes D1's native accent the zero-input default without shipping a voice per language: one checkpoint plus a small text table yields all ten anchors, cached under `models/anchors/`. `preserve` / `speaker-only` / `reference-accent` remain as opt-ins, and the blend knobs now cover `native` too |
| D10 | `reference-accent` prompt | **Two-stage** by default: reconcile accent + timbre into one clip, then clone from it | Conditioning each line on two disagreeing sources lets sampling pick the winner per line, which is heard as the speaker changing mid-video. Kept `--accent-blend single-stage` as a documented escape hatch for A/B |
| D11 | Blend strength | **Iterate Stage 1 and stop on a measured target**: `--blend-ratio 0.5` (position between the accent voice at 0 and the speaker's own voice at 1), capped by `--blend-iters 4` | One pass is biased toward the accent clip by construction (a ~165-token in-context example versus a single embedding token), which left the speaker's voice barely audible. The target is measured on a language-matched scale — the donor anchor is the donor's voice speaking the target language in x-vector mode — because a similarity against the source-language donor measures the language gap too. Defaults change behaviour and cost: every `reference-accent` run now iterates |
| D12 | Diarization | **Embed and cluster ASR segments** with `funasr` + ERes2NetV2 (`iic/speech_eres2netv2_sv_zh-cn_16k-common`), `--speaker-threshold 0.40` | Measured against the repo's two-person clip rather than assumed: reusing the Qwen3-TTS cloning encoder was **refuted** (all pairs 0.963–0.991, no merge gap) and `wespeaker` is not on PyPI. ERes2NetV2 separates the same clip at 0.341–0.874 with a merge plateau (flat to 0.379, then 0.468), so a constant threshold is defensible. Embedding the existing segments avoids a second segmentation pass, which is what keeps the labels aligned with the text instead of re-cutting lines |
| D13 | Multi-speaker voices | **Per-speaker donor, and one blend clip per voice** in `reference-accent` mode; `donor.json` is the voice bible, with the dominant speaker also mirrored flat | A shared blend clip cannot serve two voices — its codes and embedding describe one speaker, which is exactly what makes it stable, and exactly what makes it wrong for the second voice. Keeping the dominant speaker's fields at the top level of `donor` means single-speaker jobs, `audit_timbre.py` and the existing docs are unchanged; `report.json → speakers` stays `null` for one voice rather than becoming a one-entry map |
| D14 | Diarization dependency | **Optional**, degrading to one voice with a warning | `funasr` pulls ~30 transitive packages; a user who wants a single-voice dub should not need them. Same pattern as Demucs and `flash-attn`: the feature is available, its absence is survivable and loud. `--diarizer none` is authoritative so an explicit request to keep one voice is never second-guessed |
| D15 | Overlap detection | **Compare long (2.5 s) windows against speaker centroids, supervised by the labels**; report and keep the dominant speaker (v1) | The first design — re-cluster short windows — was **refuted by measurement**: 1 s windows split one speaker's own voice by up to 1.08 cosine distance versus 0.34 between speakers, and it flagged 14 of 16 real segments. Same-speaker spread falls with window length (0.88 at 1.5 s → 0.63 at 2.5 s) and so does the flip rate, so 2.5 s is the shortest trustworthy window. Comparing against a centroid rather than another window is what cancels the remaining noise, and being supervised by the labels makes a one-voice job structurally unflaggable. The cost is sensitivity — sub-window interruptions are missed — which is documented rather than hidden; pyannote's overlap model is the upgrade path if frame-level accuracy is ever needed. **Extended by P1c/D18**: `--overlap separate` splits a flagged segment with SepFormer, re-transcribes each stream and dubs both voices in parallel |
| D16 | Room matching | **Post-synthesis, derived from the speech stem, off by default** — `--room match` applies a measured RT60 convolution plus a room-tone floor and a shallower duck; without a stem it degrades to dry with `room.applied: false` | The room lives in the *speech* stem (voice + reverb tail + the room tone it masked), which the pipeline previously discarded, so recovering it needs no extra model and no new dependency. Adding it after synthesis keeps it clear of the voice-cloning tuning in P3b/P3c. A dry voice-over is correct for narrated content, so it stays the default rather than being silently replaced |
| D17 | Quality gate | **A separate, non-blocking executable** (`audit_qc.py`) reading `report.json`, exiting non-zero on failure, and reporting `skip` (never `pass`) when a gate cannot run; the measurement core is shared with `audit_timbre.py` in `pipeline/audit.py` | A gate is only useful if it can fail without destroying work, and if "not measured" cannot be mistaken for "healthy". Sharing one implementation of the language-matched measurement is what stops the two CLIs from disagreeing about the same job |
| D18 | Overlap separation | **SpeechBrain SepFormer (16 kHz) behind an optional `--overlap separate`**, placed simultaneously with the dominant line; the default stays `dominant` | The plan's "no new dependency via funasr" premise was **refuted by inspection**: funasr 1.4.16 registers no separation model and lacks `ss_inference_launch`, and ModelScope ships 8 kHz checkpoints only. SepFormer 16 kHz is the one self-contained two-speaker model at that rate, so `speechbrain` becomes an optional dependency like `funasr`/`demucs`. Separation runs only on flagged segments, and a segment whose streams cannot be matched to distinct speakers keeps the dominant-only treatment, so the stage cannot make a job worse. The default stays `dominant` until a bake-off shows separation is reliably better |
| D19 | Device portability | **Capability-probed, with a conservative `auto`**: bfloat16 only at CUDA sm_80+ (ROCm 9.4+), float16 at sm_70+, float32 otherwise; `cuda:N` honoured exactly; `auto` picks only what *every* stage supports (CUDA/ROCm, else CPU); Metal and XPU reachable via an explicit `--device` and reported by `dub.py info` | The old rule asked one binary question and answered it with bfloat16, which is unsupported on the Pascal and Volta cards a "consumer hardware" project attracts most. Metal is excluded from `auto` because faster-whisper runs on CTranslate2, which has no Metal backend: choosing it automatically would break a Mac that works today. The rules are unit-tested against fake hardware profiles, so a CPU-only CI runner still proves them |
| D20 | Distribution | **Apache-2.0, pip-installable, optional backends as extras, CI on a CPU-only runner**; direct dependencies pinned but **no lock file** | An unlicensed repository is not usable by anyone, whatever the code does. Extras make the degradation the pipeline already had (`demucs`, `funasr`, `speechbrain` are all optional by design) visible in the metadata. A `pip freeze` lock built on this box would freeze CUDA wheels into the requirements, which is the opposite of portable — pinning the direct dependencies and documenting the torch index gets the reproducibility without the trap |
| D21 | Subtitle embedding | **Subtitles are independent of dubbing, and embedded softly by default**; `--subtitles-mode burn` renders them into the frames. A new `dub.py subtitles` command runs ASR (+ translation) only. Soft mux copies every stream and replaces an existing track in the target language; burn-in opts into a video re-encode | Subtitles are useful without a dub, so coupling them to the TTS pipeline would waste the most expensive stages. Soft mux is lossless and toggleable, and preserving other-language tracks keeps information the viewer may still want; replacing the target-language track avoids duplicate streams across runs. Burn-in exists for players that ignore soft tracks, and its re-encode is why it stays opt-in |

---

## 3. MVP

### 3.1 Scope

**In**
- One video in → one dubbed video out, via a single CLI command.
- Single source language (auto-detected at video level, overridable by flag).
- One speaker, or several via diarization; target language **French**.
- Voice cloning with the speaker's own timbre.
- Accent handling: `native` (default — a generated accent anchor for pronunciation,
  the speaker's own timbre), `preserve`, `speaker-only`, or `reference-accent`
  (borrow a supplied native clip's pronunciation while keeping the voice's timbre).
- Overlong lines repaired by re-wording rather than by cutting speech (`--refine`).
- Overlapping-speech separation: a segment carrying two voices is split with
  SepFormer and both voices are dubbed (`--overlap separate`).
- Audio re-timed to the source timeline (no li, embedded in the dubbed video by
  default as a soft (toggleable) track, or burned into the frames via
  `--subtitles-mode burn`
- Subtitles: original (SRT) + translated (SRT) as a bonus deliverable.

**Out (deferred to later phases)**
- Iterative length control, reverb/loudness matching, QC metrics (P3).
  The length-repair half is now implemented (see the P3 section above), and both
  loudness matching and room matching ship (`--room match`, P3d). The automated QC
  gate is implemented too (P3e); only the model bake-off harness remains.
- Lip sync (P4), multi-source-language videos (P5), service-ification (P6).
  (Fully automatic native accent is no longer deferred: it is the default.)

### 3.2 Pipeline

```
video.mp4
   │  ① ffmpeg: probe + decode → 48 kHz mono WAV
   ▼
01_audio.wav
   │  ② Demucs htdemucs: split speech from music/ambience (44.1 kHz stereo-native)
   ├─► 01b_speech.wav    — feeds the stages below, and donor selection
   ├─► 01c_background.wav — held back for the mix at ⑦
   │
   │  ③ faster-whisper large-v3 (+ bundled Silero VAD) → text + language
   ▼
02_transcript.json           # [{id, start, end, text, no_speech_prob}]
   │  ④ diarize — embed each segment (funasr ERes2NetV2) and cluster the
   │     embeddings into speakers; folds away clusters too short to clone
   ▼
02b_speakers.json            # {segment_id: SPK_n}; segments carry their speaker
   │  ④b flag overlap — re-embed each segment in 2.5 s windows and compare them
   │     against the speaker centroids; a long run fitting another voice is marked
   ▼
02c_overlap.json             # {overlapped_ids, segments, seconds, speech_ratio}
   │  ④c separate overlap — --overlap separate only: split each flagged segment
   │     with SepFormer, match each stream to a speaker by its centroid, and
   │     re-transcribe it (the second voice is not in Whisper's single output)
   ▼
02d_overlap_segments.json    # [{parent_id, speaker, start, end, text}]
   │  ⑤ re-time impossible stretches: a run of consecutive lines whose words could
   │     not have been spoken at 6.5 w/s or more is redistributed over its own span
   │     plus the pause after it, so the word budget and the slots agree with reality.
   │     Nothing outside the run moves, so no sync is traded (see P3 follow-up)
   ▼
lines (in memory; 02_transcript.json keeps what ASR said)
   │  ⑤b blend — reference-accent only: accent codes + each speaker's embedding
   │     rendered once per voice, then re-iterated toward that speaker until
   │     `donor_ratio` reaches --blend-ratio (see P3c)
   ▼
04_dubbed/reference_blend[_<speaker>].wav
   │  ⑤b rate probe — one render per cloned voice of the accent text, timed to
   │     get the words per second that voice actually speaks. Every translation
   │     is sized by it, so it has to run before translating
   ▼
03a_rate.json                # {speaker: words per second}
   │  ⑥ LLM translation (OpenAI-compatible), chunked, one word ceiling per line
   ▼
03_script.json               # [{id, start, end, source, target}]
   │  ⑥b pre-fit check — lines whose translation cannot fit their window are
   │     reworded *before* synthesis, so nothing is ever spoken twice
   ▼
03d_prefit.json              # offenders, improved, words kept, the rewrites
   │  ⑦ Qwen3-TTS Base: one clone prompt per speaker, that speaker's lines
   │     synthesised as a batch; overlong lines that survived the pre-fit check
   │     are re-worded and re-synthesised (--refine), each in its own voice
   ▼
04_dubbed/seg_*.wav
   │  ⑧ drop each line's dead air, fit it to its window (rubberband → sox →
   │     librosa) compressing to the floor, spending the following pause before
   │     more tempo, and borrowing up to --max-overrun from the next line's time
   │     before anything is cut; place on the timeline, duck the bed under dialogue,
   │     match EBU R128 loudness (`--room match` also convolves the source's
   │     measured decay and adds a room-tone floor, before the bed is ducked
   │     more gently)
   ▼
tim│     and embed them — soft (default) or burned in (--subtitles-mode burn)
   ▼
output/
  ├─ video_dubbed.mp4        # dub audio + soft subtitle track (default)
  ├─ video_burned.mp4        # with --subtitles-mode burn (re-encoded)
  ├─ subs_source.srt         # still written as sidecars
  ├─ video_dubbed.mp4
  ├─ subs_source.srt
  ├─ subs_target.srt
  ├─ report.json             # quality signals (see README)
  └─ manifest.json           # stage cache, keyed by a job fingerprint
```

The donor voice is either provided (`--ref-audio`) or, by default, auto-selected
**per speaker** from that speaker's own segments: the best clean speech segment by
length, loudness and no-speech probability. `--ref-text` must match the donor clip
exactly in ICL mode — except in `speaker-only` and `reference-accent` modes, where
the donor contributes only a speaker embedding (`x_vector_only_mode=True`) and no
transcript is involved.

In `reference-accent` mode the donor no longer conditions the lines directly. The
accent clip's codes and the donor's embedding are reconciled once into
`reference_blend.wav` (stage ⑥), and every line is cloned from that clip, so its
codes and its embedding cannot disagree. See P3b for why, and P3c for the
blend-strength iteration that keeps the speaker's own voice dominant. With several
speakers there is one such clip per voice, because one clip cannot describe two
people (D13).

### 3.3 Tech stack

| Stage | Tool | Notes |
|-------|------|-------|
| Media | `ffmpeg` + `soundfile` + `pyloudnorm` | ffmpeg 4.4.2 verified (probe, extract, mux with `-c:v copy`). Loudness is numpy + `pyloudnorm`, so audio-only jobs need no ffmpeg |
| Separation | Demucs `htdemucs` | 44.1 kHz stereo-native, resampled around the model, ~24× realtime. Optional: without it the background is discarded |
| VAD | bundled ONNX Silero via `faster-whisper`'s `vad_filter` | a separate `silero-vad` package was dropped as redundant |
| ASR | `faster-whisper` (`large-v3`, CTranslate2) | word timestamps; preloads vendored CUDA libs, then falls back to CPU |
| Language ID | Whisper's detected language | single, video-level for MVP |
| Diarization | `funasr` + ERes2NetV2 (`iic/speech_eres2netv2_sv_zh-cn_16k-common`) | 192-dim embeddings at 16 kHz, ~30 MB cached under `~/.cache/modelscope`. Optional: without it every line is dubbed in one voice, with a warning |
| Translation | `openai` client → any OpenAI-compatible base URL | local server or cloud; chunked, with a JSON-repair pass |
| TTS | `qwen-tts` (Qwen3TTSModel, **Base** required) | cloning plus the two-stage blend. `flash-attn` is irrelevant here — see §0 |
| Timing | `rubberband` → `sox` → `librosa` | three interchangeable backends, auto-detected; stretches clamped to ±15–18% |
| Loudness | `pyloudnorm` (EBU R128) + a numpy limiter | `match-source; soft subtitle track muxed by default (`mov_text` in MP4), or burned in with `--subtitles-mode burn` (re-encodes the video). Other-language subtitle tracks are preserv
| Mix | numpy ducking | 6 dB under dialogue, 150 ms ramps |
| Room matching | numpy FFT convolution + a synthetic impulse response | `--room match`: room-tone floor and a 3 dB duck, plus an RT60 measured from the speech stem (clamped to 0.2–3.0 s). The reverb tail is opt-in (`--room-wet-db`, default `0` = none). No new dependency |
| Quality gate | `audit_qc.py` over `pipeline/audit.py` | sync / truncation / loudness / timbre thresholds; exit 0/1, never blocks the dub |
| Mux / subs | `ffmpeg`, `pipeline/subtitles.py` | video stream copied, not re-encoded |

### 3.4 Job config

This is `pipeline/config.py`'s `JobConfig` (Pydantic), populated from CLI flags,
and its fingerprint keys the stage cache — so changing any field below re-runs the
affected stages. Also the future queue-message shape.

```yaml
# Input — exactly one of video / audio
video: input.mp4
audio: null

# Languages
target_language: fr              # required; code or full name
source_language: null            # auto-detect when null

# Voice reference
ref_audio: null                  # auto-select a donor from the video when null
ref_text: null                   # required with ref_audio in `preserve` mode
accent_mode: native              # native | preserve | speaker-only | reference-accent
accent_match: off                # off | rate | gender (native only)
accent_audio: null               # required for reference-accent
accent_text: null                # required with accent_audio
accent_blend: two-stage          # two-stage | single-stage
blend_takes: 3                   # first-pass candidates
blend_iters: 5                   # max passes pulling toward the speaker
blend_ratio: 0.85                # stop target; 0 disables the search

# Speakers
diarizer: auto                   # auto | none (none = every line in one voice)
speaker_threshold: 0.70          # cosine distance above which voices differ
min_speaker_seconds: 20.0        # shorter clusters join their nearest neighbour

# Overlapping speech
overlap: dominant                # dominant | separate (--overlap)
overlap_separator: auto          # auto | sepformer | none

# Timing
min_stretch: 0.75                # compression floor: past it the wording gives, not the audio
max_stretch: 1.18
max_overrun: 0.25                # lateness a line may borrow from the next line's time
length_budget: words             # off | words
budget_speedup: 0.85             # tempo the per-line word ceiling is willing to spend

# Length repair
refine: true
refine_max_passes: 1
refine_threshold: 0.05

# ASR gating
no_speech_threshold: 0.6

# Audio bed
keep_background: true
separator: auto                  # auto | demucs | none
duck_db: 6.0
background_gain_db: 0.0

# Room matching (off = dry voice-over; match = the source's acoustic space)
room: off                        # off | match
room_rt60: null                  # null = measure it from the speech stem
room_wet_db: 0.0                 # 0 = no reverb (opt-in); -20 for a subtle tail
room_duck_db: 3.0                # shallower than duck_db, by design
room_tone_gain_db: -30.0         # level of the room-tone floor

# Loudness
loudness_target: null            # match-source when null | lufs
target_lufs: null

# Models, device and output
device: auto                     # auto | cpu | cuda | cuda:N | mps | xpu
devices: null                    # list to spread blend+synthesis over (--devices)
dtype: auto                      # auto | bf16 | fp16 | fp32
whisper_model: large-v3
llm_model: null                  # overrides LLM_MODEL for this run
subtitles_mode: soft             # soft (default) | burn | off
tts_batch: 8                     # line-count ceiling on a synthesis batch
keep_subtitles: true
output_dir: output/
force: false                     # ignore the stage cache
```

### 3.5 Repository layout (tool-sized, service-ready)

```text
dubsmith/
├── dub.py                    # CLI entrypoint (argparse) + model downloads
├── audit_timbre.py           # per-line speaker-timbre audit (--device cpu works)
├── audit_qc.py               # automated quality gate for a finished job
├── pipeline/
│   ├── config.py             # Pydantic job schema, env settings, languages
│   ├── models.py             # model registry, download, path resolution
│   ├── errors.py             # domain exception hierarchy
│   ├── logging_utils.py      # per-job logging (text or JSON)
│   ├── media.py              # ffmpeg wrappers, loudness, resampling
│   ├── asr.py                # faster-whisper + VAD gating
│   ├── diarize.py            # speaker embeddings + clustering (multi-speaker)
│   ├── separation.py         # Demucs speech/background split
│   ├── mixing.py             # ducking, bed mix and room matching (pure numpy)
│   ├── translate.py          # OpenAI-compatible client, chunking, shorten()
│   ├── tts.py                # Qwen3-TTS wrapper, donor selection, blend
│   ├── backends.py           # CosyVoice3 / Chatterbox / Fish engine construction
│   ├── voice.py              # one voice: the facts that build its clone prompt
│   ├── voices.py             # voice library, cards, provenance
│   ├── provenance.py         # per-line prosody and transformation records
│   ├── parallel.py           # fan-out: tasks, worker pool, per-device counters
│   ├── timing.py             # stretching backends, timeline assembly
│   ├── subtitles.py          # SRT writer
│   ├── audit.py              # timbre measurement + quality gates
│   └── orchestrate.py        # stages, artifacts, caching, report
├── tests/                    # pytest suite (no GPU, models or ffmpeg)
├── models/                   # local checkpoints (gitignored)
└── PLAN.md
```

This is intentionally a flat `pipeline/` package for a local CLI, not a
service-oriented `app/` tree. The engineering conventions this project actually
follows are in [`docs/development/guidelines.md`](docs/development/guidelines.md).
Optional backends run in child processes; the default Qwen path stays in-process.

### 3.6 Definition of Done

- One command runs the whole thing: `python dub.py --video in.mp4 --target fr`.
- Produces `video_dubbed.mp4` + two SRTs.
- Same voice timbre as the source speaker — subjective listen ✅ **plus** SECS ✅
  (`audit_timbre.py`, run by hand).
- Audio synchronized to the source timeline within ~100 ms at segment anchors ✅
  (measured drift 0.000 s at the time of this check, i.e. before the bounded
  overrun existed; a line may now start up to `--max-overrun` late, and
  `report.json` → `shifted_seconds_max` says what a run actually used).
- Quality signals thresholded into a gate rather than inspected by hand ✅
  (`audit_qc.py`, P3e). The gate reports; it does not block delivery, so a
  failure is a verdict on the finished job rather than a lost run.
- Handles a ≤ 10-minute video on one RTX 3060 without OOM — ✅ **verified at 11
  minutes** (P8). The batch ceiling is measured rather than fixed, cached blocks
  are returned to the driver under pressure, and the allocator runs with
  expandable segments. `report.json` → `tts.oom_retries` is the number to check;
  the 11-minute run reports `0` where it previously flooded the log with failed
  1.6 GiB allocations.
- Runs on hardware other than the author's — **verified on CPU and on the
  float16 path**. The device *rules* are proven for every profile
  (`tests/test_device.py`, fake hardware), and there are now four real end-to-end
  runs through `audit_qc.py`, all passing: two on CPU (float32) and two on the GPU
  (float16 and bfloat16, same clip). No second *physical* machine has been used, so
  a driver- or vendor-level difference is still unexercised.

#### Hardware matrix

Each row is one real end-to-end run whose output was put through `audit_qc.py`. A
cell that has not been run is `unverified`, never assumed.

| Hardware | Runs end to end | Sync (`shifted_segments`) | Truncation (`worst_line_deleted_ratio`) | Loudness (dB vs source) | Timbre | Wall clock | Peak memory |
|---|---|---|---|---|---|---|---|
| RTX 3060, bfloat16, CUDA | ✅ 11:00 video, 476 lines, `min-speaker-seconds 20` | 0.000 s | 64.7% worst line, 21.6 s cut - the *translation* is too long for its slots (unrelated to memory; text identical to the pre-fix run) | −0.0 | not audited | **~21 min** (1258 s, incl. 241 s of blend for the parked 4-voice issue) | **9.36 GiB reserved, 0 OOM failures** |
| RTX 3060, bfloat16, CUDA | ✅ 4:47 TED + 60 s clip | 0.000 s | 0% with `--refine` | −0.0 | 0 lines closer to accent | 2× realtime | ⚠️ OOM retries |
| 12-core CPU, float32, `preserve` | ✅ 22.1 s clip (`change-the-world.mp4`) | 0; 1.84 s spilled into pauses, 0.00 s trimmed | 0.0%; 0 lines over 10% | −0.0 | n/a — needs a `reference-accent` job | **2:10** (130 s), 0.17× realtime | **12.7 GB** RSS |
| 12-core CPU, float32, `reference-accent` | ✅ same clip, accent clip + transcript | 0; 0.00 s overrun, 0.00 s trimmed | 0.0%; 0 lines over 10% | +0.08 | **0 of 5** lines closer to the accent voice; margin median +0.0513 on a ±0.056 scale | **5:18** (318 s), 0.07× realtime | **17.2 GB** RSS |
| Pre-Ampere GPU, **emulated**: 3060 pinned to `--dtype fp16 --device cuda:0`, `reference-accent` | ✅ same 22.1 s clip, accent clip + transcript | 0; 0.00 s overrun, 0.00 s trimmed | 0.0%; 0 lines over 10% | +0.05 | **0 of 5** lines closer to the accent voice; margin median +0.0444 on a ±0.044 scale | **1:03** (63 s) | 8.7 GB |
| RTX 3060, bfloat16, `reference-accent` (control for the row above) | ✅ same clip, same accent | 0; 0.00 s | 0.0%; 0 lines over 10% | −0.0 | **0 of 5** closer to the accent; margin median +0.0330 on a ±0.030 scale | **0:54** (54 s) | 8.7 GB |

**The float16 row is an emulation, not a pre-Ampere card.** `--dtype fp16` on the
3060 exercises the same code path a Turing or Volta card would take — the dtype is
accepted, recorded and used — but it cannot reproduce a driver difference, a
missing kernel, or an older CUDA runtime. That residue is what "no second physical
machine" still means.

**Emulating it surfaced a zero-cost bonus check: `--device cuda:0` works.** Both
GPU runs pinned GPU 0 explicitly and `report.json` → `device.resolved` reads
`cuda:0`, so the index survives all the way through — including into Demucs, which
is where it used to be silently dropped (P7). Verified on real hardware rather than
by unit test.

#### Does the precision move the voice? Measured, not assumed

This was flagged in P7 as needing an A/B before it could be trusted. Two full runs
are **not** that A/B: they differ in translation too (`speech_rate_wps` 4.17 vs
4.57), so the margin difference between them is not attributable to the dtype. The
controlled test holds the text, the reference clip and the draw count fixed and
changes only the precision:

| Comparison | min | median | max |
|---|---|---|---|
| fp16 vs fp16 — two draws (noise floor) | 0.9823 | 0.9898 | 0.9928 |
| bf16 vs bf16 — two draws (noise floor) | 0.9893 | 0.9935 | 0.9948 |
| **fp16 vs bf16 — the question** | 0.9819 | **0.9902** | 0.9942 |
| fp16 vs bf16, second draws | 0.9876 | 0.9928 | 0.9944 |

Three French sentences, one reference clip, two draws per precision, scored with the
checkpoint's own speaker encoder. **Cross-dtype similarity sits inside the
same-dtype run-to-run spread**, so the precision does not measurably move the voice:
the sampling variance that already exists between two draws at *one* precision is as
large as the difference between precisions. This is the same shape of evidence as
the P3b lesson about the encoder's noise band — a difference is only a difference if
it exceeds the floor.

Small print: three sentences, one speaker, one reference. Enough to justify not
treating fp16 as a quality risk; not enough to claim a general law.

**What the CPU runs establish.** Both went through `audit_qc.py` at exit 0. The
`preserve` run passed sync, truncation and loudness with timbre skipped (there is
no accent clip to measure against); the `reference-accent` run passed all four
gates including timbre, with `donor_ratio` 0.9682 reached in the first blend pass.
`--refine` fired in the `preserve` run (2 offenders, 2 improved, 1.70 s recovered)
and was not needed in the accent run. `report.json` → `device` correctly recorded
`cpu` / `float32` / `int8` in both, which is the P7 report block working on a real
run rather than in a unit test.

**The cost is where you would not expect it.** CPU TTS is ~4.3× realtime measured
on a single line (3.92 s of audio in 16.8 s, float32, after a 9.9 s model load) —
not the two orders of magnitude the prerequisites first assumed, and that number
has been corrected in the README. The binding constraint is **memory**, not speed:
17.2 GB peak for the accent run, because the blend stage holds the checkpoint plus
its own renders. The blend is also 68% of the accent run (217.0 s of 318 s,
against 60.9 s of per-line synthesis), which matches the earlier note that the
blend dominates on short clips.

Stage split, `preserve` / `reference-accent`, in seconds: separate 9.9 / 9.5,
transcribe 14.3 / 13.4, diarize 7.8 / 6.6, translate 7.4 / 7.3, blend — / 217.0,
synthesize 88.0 / 60.9, mix 0.7 / 0.4.

### 3.7 Prerequisites to install

```
ffmpeg                      # system binary, required for video in/out
rubberband                  # optional system binary; sox or librosa otherwise
python -m pip install .     # numpy, torch, soundfile, librosa, pydantic,
                            # qwen-tts, faster-whisper, pyloudnorm, openai
python dub.py info          # what will this machine actually use?
```

CPU-only machine: install the CPU torch wheel first, or the default wheel pulls
several gigabytes of CUDA that cannot be used.

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install .
```

Optional backends are extras: `pip install ".[background]"` (Demucs, keeps music
and ambience), `".[diarization]"` (funasr, per-speaker voices), `".[separation]"`
(SpeechBrain, splits overlapping speakers), `".[gpu-asr]"` (CUDA 12 runtime for
faster-whisper), or `".[full]"` for all of them.

---

## 4. Improvement roadmap

### Overview

| Phase | Focus | Key additions | Complexity |
|-------|-------|---------------|------------|
| **P0** | MVP (this doc §3) | Single-speaker dub, French, no lip sync | — |
| **P1** | Multi-speaker + background | Diarization **done**, voice bible **done**, per-speaker cloning **done**, overlap detection **done**, overlapping-speech separation **done** | M |
| **P2** | Native accent | **done** — generated anchor by default, per-speaker `--accent-match` | M |
| **P3** | Quality & timing | Isochrony loop, room matching (P3d) and the automated QC gate (P3e) **done**; remaining: the model bake-off harness | M |
| **P4** | Lip sync | LatentSync/MuseTalk, gated on face visibility | L |
| **P5** | Multi-source-language | Per-utterance LID, per-segment source language | S–M |
| **P6** | Service-ification | FastAPI + Celery/Redis, model residency, consent gate | L |
| **P7** | Portability & distribution | Device capability probing, `dub.py info`, `--dtype`, packaging + console scripts, CI, Apache-2.0 licence | S–M |
| **P9** | Multi-device synthesis | **done** — one worker process per card for the blend and the synthesis; the stages that run once stay on the first device | S–M |

### P1 — Multi-speaker & background preservation

- **Diarization**: ✅ **done** — `pipeline/diarize.py` embeds each ASR segment with
  ERes2NetV2 and clusters the embeddings (D12). The planned pyannote +
  3D-Speaker route was dropped: pyannote needs a Hugging Face token and a heavier
  stack for the same per-segment answer, and `wespeaker` is not on PyPI. The
  backend is pluggable, so a better embedder is a one-class change.
- **Voice bible**: ✅ **done** — `04_dubbed/donor.json` → `speakers`, plus the
  `speakers` summary in `report.json` (D13). The planned separate `voices.json`
  was unnecessary: `donor.json` already answers "which voice, and why".
- **Per-speaker cloning**: ✅ **done** — one prompt per speaker, lines batched
  within a speaker, and one blend clip per voice in `reference-accent` mode.
- **Overlap**: ✅ **done** — each segment is re-embedded in 2.5 s windows and
  compared against the speakers the labels establish, so a segment carrying two
  voices is recorded in `02c_overlap.json` and `report.json → overlap` instead of
  being silently attributed to one person (P1b). With `--overlap separate` the
  flagged segment is split into one stream per voice with SpeechBrain's 16 kHz
  SepFormer, each stream is re-transcribed and matched to a speaker, and both
  voices are translated, synthesised and placed simultaneously (P1c). Without the
  optional backend, `--overlap separate` warns and behaves as `--overlap
  dominant`.
- **Background bed**: ✅ **done** — `pipeline/separation.py` splits dialogue from
  music/SFX with Demucs, and `pipeline/mixing.py` mixes the dub back over the bed,
  ducked 6 dB under dialogue with 150 ms ramps (`--keep-background`, default on).
- **Output**: multi-voice dub with preserved background.

**P1 is complete**: overlapping speech is detected (P1b) *and*, with `--overlap
separate`, split so both voices are dubbed (P1c).

### P2 — Native target accent (IMPLEMENTED)

**The gap.** Native accent was a nice-to-have (D1) because it seemed to need a
native recording per language, or a voice-conversion stage. The VoiceDesign
checkpoint removes both: it synthesises speech from a text *description*, so one
model yields an accent anchor for every supported language.

**What ships.**

- **Accent anchor** (`pipeline/anchors.py`): a short, phonetically varied passage
  per language (`ANCHOR_TEXT`) is rendered once by
  `Qwen3-TTS-12Hz-1.7B-VoiceDesign` with a style description (`instruct_for`) and
  cached under `models/anchors/`. The anchor is used exactly like a supplied
  `--accent-audio` clip: it feeds the existing two-stage blend, so no new synthesis
  path was needed.
- **Lazy generation**: `ensure_anchor` builds the engine only on a cache miss, so a
  warm cache never loads the checkpoint, and the two TTS models are never resident
  together (`_stage_accent_anchor` releases the GPU before the blend loads base).
- **Setup path**: `dub.py anchors --generate [--lang …]` / `--list` pre-warms
  anchors explicitly; a run generates a missing one lazily instead of failing the
  setup step.
- **Product knob**: `--accent-mode native` is the default; `--accent-match
  off|rate|gender` generates a **per-speaker** anchor whose delivery fits the
  source: `rate` from the speaker's measured words-per-second, `gender` from a
  median-F0 estimate. Anchors are keyed by `(language, rate_band, gender)`.
- **Word budget**: because a target-language anchor always exists, `--length-budget
  words` is now the default, so the translator gets an actionable ceiling.

**Output**: their voice, native target accent, with no user-supplied clip.

The exported ``--accent-mode reference-accent`` with a supplied clip remains for
users who want to choose the accent themselves.

### P3 — Quality & timing

- **Isochrony loop**: ✅ **done** — `--refine` re-words overlong lines instead of
  cutting them, and `--length-budget words` gives the translator a ceiling it can
  act on. See the P3 section in §0 for the measurements.
- **Acoustics**: EBU R128 loudness ✅ (`match-source` by default, with a peak
  limiter) and **room matching ✅ done** — `--room match` measures the decay from
  the speech stem and applies it as a convolution, adds a room-tone floor and
  ducks more gently, all post-synthesis. Off by default, because a dry voice-over
  is correct for narrated content. See P3d in §0.
- **QC stage + eval harness**: **the gate is ✅ done** — `audit_qc.py` thresholds
  SECS per line, timing drift, deleted-speech share and loudness compliance into
  pass/fail, sharing its measurement core with `audit_timbre.py` via
  `pipeline/audit.py`. **Remaining: the model bake-off harness**, which the gate's
  `--json` output and exit code are designed to be called from.
- **Output**: a measurable quality gate ✅; the bake-off harness remains.

### P4 — Lip sync

- Detect face size/visibility per shot; run lip sync **only** on close-ups.
- **LatentSync** (quality) or **MuseTalk** (speed); Wav2Lip as fallback.
- **Output**: dubbed video with regenerated lip motion; batch/overnight job.

### P5 — Multi-source-language videos

- Per-utterance **LID** (VoxLingua107 / FastText `lid.176`); translation node
  accepts per-segment source language.
- **Output**: mixed-language source videos translate correctly.

### P6 — Service-ification

- If the project ever adds a service, use **FastAPI** (async) + Celery/Redis or
  equivalent, and keep the pipeline functions independent so they can be wrapped
  without moving model code into the API layer.
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
3. **Speaker consistency** over long videos may drift. Largely addressed: the
   `reference-accent` blend pins one clip as the reference for every line, so the
   voice no longer re-decides per line or per run. Per-speaker LoRA fine-tuning on
   minutes of their audio remains the fallback.
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
