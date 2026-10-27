# Troubleshooting

Symptoms, causes and fixes, plus how the logs work and what the project does not
attempt yet.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Missing model: 'whisper_large_v3' …` | Run `python dub.py models --download all` |
| `Required binary 'ffmpeg' was not found on PATH` | `sudo apt-get install -y ffmpeg` (audio-only jobs still work) |
| `… contains no audio stream, so there is nothing to transcribe` | The file is video-only. Verify with `ffprobe -v error -show_entries stream=codec_type file.mp4`; mux an audio track in first |
| `CUDA is unusable for speech recognition … libcublas.so.12` | Informational — ASR fell back to CPU. Install `nvidia-cublas-cu12 nvidia-cudnn-cu12` for GPU ASR (Setup §3) |
| `No speech detected in the input` | No audible speech, or the wrong audio track. Check with `ffprobe` |
| `Translation response was not valid JSON` | Lower `LLM_CHUNK_SIZE` in `.env`, or use a model better at JSON |
| `Translation request to … failed` | The LLM server isn't reachable — check `LLM_BASE_URL` |
| `--ref-audio requires --ref-text` | Add the exact transcript, or drop both flags to auto-pick a donor |
| `--accent-audio only applies to --accent-mode reference-accent` | Add that flag, or remove `--accent-audio` |
| `--accent-mode reference-accent requires --accent-audio` | Add the native-target clip, or use another mode |
| `--accent-audio requires --accent-text` | Transcribe the accent clip; the codes cannot be aligned without it |
| `does not support voice cloning` | Pointed at a CustomVoice/VoiceDesign checkpoint; use `tts_base` |
| `Unsupported language 'xx'` | Use one of the ten supported languages |
| `No saved voice named …` | List the library with `python dub.py voices --list`, or create the voice first |
| CUDA out of memory | `--device cpu`, or free the GPU (a large local LLM often occupies it). On a multi-GPU box, `--device cuda:1` pins the run to the other card |
| On a long video, repeatedly | See [Memory on a long video](performance.md#memory-on-a-long-video): check `report.json` → `tts.oom_retries` and `tts.ceiling_shrinks`. The pipeline budgets each batch by decoded audio and releases cached blocks when free memory falls below 3 GB; if the counters are still non-zero, lower `--tts-batch`, or use a shorter reference clip — it is charged once per line |
| An unexpected amount of VRAM is in use | `--device cuda:1` now genuinely uses the second GPU; the log line and `report.json` → `device` both report the card that was used |
| `dtype fp16 is not usable on cpu (cpu); using fp32 instead` | Informational, not an error: half-precision on a CPU is slow and patchy, so float32 is used. Drop `--dtype` to let the capability decide |
| The GPU is not being used at all | Run `python dub.py info` — it reports the accelerator, the device it would pick, and the precision. If it says `cpu`, torch either sees no GPU or the wrong wheel is installed (see Setup §2 and §3) |
| Speech recognition is on the CPU when a GPU exists | CTranslate2 links against a specific CUDA major. Run `python dub.py info` and check the `asr device` line; Setup §3 has the fix |
| Dub sounds rushed or slow | Check `report.json` → `clamped_segments`; tighten the translation |
| Some words are cut off | `report.json` → `worst_trimmed_lines` names them (`id`, `trimmed`, `needed_ratio`); `needed_below_floor_segments` counts them. A `needed_ratio` below `compress_floor` means the translation was longer than even a 0.75 compression can hold, so the wording, not the tempo, is what to change: check `prefit` (did the pre-synthesis check run, and how much wording did it keep?) and `--length-budget`. Lowering `--min-stretch` further buys room at the cost of sounding hurried |
| A line lost a lot of words despite a low `trimmed_seconds_total` | Look at `worst_line_deleted_ratio` and `lines_over_10pct_deleted`: damage is usually concentrated in one or two lines, not spread evenly |
| Lines sound slow or have gaps | `report.json` → `deficit_seconds_total`: the translation is shorter than its slot, i.e. the word ceiling was conservative. Check `delivery_wps` (the probe's measurement) against `speech_rate_wps` (what the voice actually delivered): if the probe reads well under the delivery, raise `--budget-speedup`. See also [Reference pacing](voices.md#reference-pacing) |
| A rewrite did not help | Check `report.json` → `refine` (the measured pass) and → `prefit` (the pre-synthesis one). `attempted: false` means no line overflowed, or the translator cannot rewrite; `improved: 0` means the rewrites did not fit either, and the pre-fit check was skipped or its candidates were not shorter |
| `delivery_wps` looks wrong for my video | It is measured from one 40-word probe render per voice, so a very expressive or very short-spoken video can differ from `speech_rate_wps` (what the lines actually delivered). Delete `<job>/03a_rate.json` to re-probe, or set `--budget-speedup`/`--min-stretch` yourself and leave the budget out of it with `--length-budget off` |
| Dub drifts out of sync | Up to `--max-overrun` seconds per line is deliberate: it keeps the last word of a line that would otherwise lose it. Check `shifted_seconds_max` — if it is over the limit, the source speech genuinely overlaps itself, and `--max-overrun 0` restores exact sync at the cost of the line tails |
| Dub sounds quiet | Check `loudness_delta_vs_source`; `0` means it matches the original. Force a level with `--loudness lufs --lufs -16` |
| Music/ambience is missing | Install `demucs` (`pip install demucs`); without it the background cannot be separated and is discarded |
| Music fights the voice | Raise `--duck-db` (default 6 dB) |
| `Missing model: 'tts_voice_design' …` | The default `native` mode needs it to generate the accent anchor. Run `python dub.py models --download tts_voice_design` (or `--download all`) |
| `--accent-match only applies to --accent-mode native` | Use `--accent-mode native` (the default), or drop `--accent-match` |
| The dub still has the speaker's accent | The default `native` mode already borrows a generated anchor. If it is not enough, try `--accent-match gender`, then `--accent-mode speaker-only`, or supply your own clip with `--accent-mode reference-accent` |
| Still sounds foreign in `reference-accent` mode | Check `--accent-text` matches the clip word for word — mismatched text misaligns the borrowed codes. A longer, more natural accent clip also helps |
| The voice changes part-way through the video | Only possible in `reference-accent` with `--accent-blend single-stage`, where every line is free to follow the accent codes or the speaker embedding. The default `two-stage` removes that choice — each voice's lines are cloned from that voice's own `reference_blend[_<speaker>].wav`, so the voice cannot change between lines. Listen to that clip on its own to confirm stage 1 picked up the speaker's timbre; if it did not, raise `--blend-takes` or try a different `--accent-audio`. To measure it rather than judge it, see [Auditing a dub's voice](testing-and-quality.md#auditing-a-dubs-voice-line-by-line) |
| A voice changes *between speakers* rather than mid-line | Expected, unless the two voices are the same person: each speaker is cloned from their own donor. If the count is wrong, see the speaker-threshold rows above |
| The borrowed accent dominates — the speaker's own voice is barely there | The blend was built too close to the accent voice. Check `report.json` → `donor.blend.donor_ratio`: a low value means it never got far enough to the speaker. Raise `--blend-iters`, or lower `--blend-ratio` if you want more of the accent |
| The blend reached a donor ratio of only … | Informational: the target was not met within `--blend-iters` passes, and the best pass was used anyway. Raise `--blend-iters` or lower `--blend-ratio` |
| `--accent-blend only applies to --accent-mode reference-accent` | Add that flag, or drop `--accent-blend` |
| `--blend-takes only applies to --accent-mode reference-accent` | Add that flag, or drop `--blend-takes` |
| `--blend-iters only applies to --accent-mode reference-accent` | Add that flag, or drop `--blend-iters` |
| `--blend-ratio only applies to --accent-mode reference-accent` | Add that flag, or drop `--blend-ratio` |
| Every line is dubbed in one voice, but the video has several speakers | Diarization is off or unavailable. Check the log for `Diarization (none)` — it means `--diarizer none` was set, or `funasr` is not installed. Install it (`pip install -r requirements-dev.txt`), then re-run. The detection happens before synthesis, so a cached job needs `--force` to pick up a newly installed backend |
| One person is being split into two voices | Raise `--speaker-threshold` (e.g. `0.6`); a distance-based threshold is genuinely material-dependent. `02b_speakers.json` shows how the segments were labelled. (Overlap detection is supervised by those labels but does not use this threshold.) |
| A line sounds like the wrong person, or you can hear two people on it | `report.json` → `overlap` counts segments that hold a second voice; `02c_overlap.json` lists their ids. By default such a line is dubbed in its dominant speaker's voice — run with `--overlap separate` (needs the optional `speechbrain`) to split it and dub both voices. Detection is deliberately conservative (a 2.5 s run that fits another speaker better), so short interruptions are not reported |
| Two people are merged into one voice | Lower `--speaker-threshold` (e.g. `0.3`), and check `--min-speaker-seconds` is not folding the quieter speaker away. `--diarizer none` confirms the rest of the pipeline is unaffected |
| A quiet speaker's lines are in the wrong voice | Their cluster was probably shorter than `--min-speaker-seconds` and got folded into the nearest voice. Lower it, at the cost of cloning from a shorter window |
| `funasr is not installed, so every line will be dubbed in one voice` | Informational fallback, not an error: the job still runs. Install `funasr` for multi-speaker dubbing, or pass `--diarizer none` to make the intent explicit |
| The dub is rushed, or words are missing | Check `accelerated_segments` and `needed_ratio_min`: the voice is being sped up because the written line is longer than the window holds. Raise `--budget-speedup` only if `deficit_seconds_total` is also high; otherwise the lines are simply long for their slots. See [When a translation is too long for its slot](audio-and-timing.md#when-a-translation-is-too-long-for-its-slot) |
| `audit_qc.py` reports `SKIP timbre — no accent clip to measure against` | Expected for a `preserve` job: nothing was borrowed, so "did the line follow the accent voice?" has no answer. Pass `--accent <clip>` to force one. The other three gates still report, and the exit code stays 0 |
| The timbre audit fails while the GPUs are busy | Pass `--device cpu` to `audit_qc.py` or `audit_timbre.py`. The measurement is slower there but identical |

Logs are written to stderr and include a job id, and are mirrored to
`<job folder>/run.log`. That file is **appended** to, never replaced: one input
is normally dubbed many times while its options are tuned, so every run keeps its
lines and opens with a `Run started` line to mark the boundary between them. Once
it passes `--log-max-bytes` (2 MiB by default) it is renamed to
`run-<YYYYmmdd-HHMMSS>.log` and a fresh `run.log` begins — at a run boundary, so a
single run is never split. `--log-file PATH` puts the copy somewhere else,
`--no-log-file` writes none, and `LOG_FILE` sets the default path. Set
`LOG_JSON=1` for one JSON object per line, on the terminal and in the file alike.

---

## Notes and limitations

- **Voice cloning requires consent.** Clone only voices you have permission to
  use. Voice is biometric data under GDPR, and synthetic-dubbing disclosure is
  increasingly required by platforms. A consent gate is planned for P6.
- **Licence**: this project is Apache-2.0 (see `LICENSE`).
- **Hardware portability is deliberate, and tested without the hardware.** The
  dtype follows the GPU's compute capability rather than its name (bfloat16 is
  unsupported on Pascal and Volta, and choosing it unconditionally was the most
  likely "works for me, not for you" failure). `--device cuda:N` is honoured
  exactly, so a second GPU can be pinned. Metal and XPU are usable but never
  chosen automatically, because speech recognition runs on CTranslate2. The rules
  are covered by `tests/test_device.py` against fake hardware profiles, so a
  regression is caught on a CPU-only CI runner.
- **Model licences**: Qwen3-TTS and Whisper weights are Apache-2.0 / MIT.
  Avoid CPML-licensed models (e.g. XTTS-v2) if you need commercial rights.
- **Multi-speaker dubbing attributes speakers per segment**, not per word, so
  overlapping speech is shown as a flag on the segment. By default such a line is
  dubbed in the voice of its dominant speaker and reported (`report.json` →
  `overlap`, `02c_overlap.json`); `--overlap separate` splits the two voices apart
  with SepFormer, re-transcribes each and dubs both. The speaker count comes from
  clustering rather than being declared. Both are tracked in `PLAN.md`.
- **Known gaps**, all tracked in `PLAN.md`: lip sync, multi-language source
  videos, and iterative length control in the translation loop. Background
  preservation, native-accent borrowing and multi-speaker dubbing are
  implemented (see [Accent modes](voices.md#accent-modes) and
  [Dubbing a video with several speakers](multi-speaker.md#dubbing-a-video-with-several-speakers)).
- **Native accent is generated, not downloaded.** The default `native` mode
  synthesises a short anchor per language with the VoiceDesign checkpoint, cached
  under `models/anchors/`, so no voice is shipped per language and
  `reference-accent` with `--accent-audio` remains for choosing one yourself.
  `--accent-match` fits the anchor's rate (and optionally gender) to the source
  voice. There is no native-accent-free default path: to keep the source accent,
  opt in with `--accent-mode preserve`.
- **Room matching is off by default and adds no reverb unless asked.**
  `--room match` adds a room-tone floor and a gentler duck, both derived from the
  separated speech stem. A dry, close voice-over is the correct choice for narrated
  content, which is why it stays the default. The reverb tail is a separate, opt-in
  step (`--room-wet-db -20`): an earlier default of −12 dB was rejected on a real
  listen as an audible echo. `--no-background` removes the stem, so room matching
  then degrades to a dry dub and says so (`report.json` → `room.applied: false`).
- **Changing any setting re-runs the whole job** — the cache fingerprint is
  per-job, not per-stage. Fine for a CLI; worth refining before P6.

---

Back to the [docs index](README.md), or the [project README](../README.md).
