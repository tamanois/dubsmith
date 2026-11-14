# Outputs and caching

Where a job's files land, how to read `report.json` without listening, and how the
stage cache decides what to re-run.

## Outputs

A job's files all land in one folder, whose name depends on which surface started
it — the CLI keys the folder on the input stem, while the API adds the operation,
the language and a run id so two runs never collide:

| Started by | Folder |
|---|---|
| CLI (`dub.py --video talk.mp4`) | `output/talk/` |
| App / API (`POST /api/jobs/dub`) | `output/talk__dub__French__<run-id>/` |

Everything else on this page applies to both. A CLI job folder looks like this:

```
output/talk/
├── 01_audio.wav         # normalised 48 kHz mono source audio
├── 01b_speech.wav       # separated speech stem (when background is preserved)
├── 01c_background.wav   # separated music/ambience stem
├── 02_transcript.json   # ASR result: segments, words, timings, language
├── 02b_speakers.json    # segment id -> speaker, from diarization
├── 02c_overlap.json     # which segments carry a second voice (multi-speaker runs)
├── 02d_overlap_segments.json   # only with --overlap separate: the per-voice lines
├── 03_script.json       # source -> target text per segment
├── 03a_rate.json        # words/s measured for each cloned voice (what budgets come from)
├── 03b_spoken.json      # lines the length repair re-wrote (only those)
├── 03c_refine.json      # what the length repair achieved
├── 03d_prefit.json      # lines re-worded before synthesis, and how much of the wording survived
├── 04_dubbed/           # one synthesised clip per segment (+ .done marker)
│   ├── donor.json       # which voice was cloned for each speaker, and from what
│   ├── prosody.json     # latest per-line prosody: settings, LLM reply, text transforms
│   ├── provenance/      # timestamped prosody.json snapshots, one per run
│   ├── reference_blend.wav   # only in reference-accent: the clip every line is cloned from
│   ├── reference_blend_SPK_1.wav   # ...one per speaker once there are several
│   └── blend.json       # only in reference-accent: how each clip was chosen
├── 05_mix.wav           # assembled dub track, before normalisation
├── 05_room.json         # only with --room match: the decay measured and levels used
├── timing.json          # per-segment stretch ratio, placement and deficit
├── <input>_dubbed.wav   # final loudness-normalised dub audio
├── <input>_dubbed.mp4   # dubbed video (video input only)
├── subs_source.srt
├── subs_target.srt
├── report.json          # counts, loudness, stretch stats, stage timings
├── manifest.json        # stage cache
├── run.log              # the log from stderr, appended across runs
└── run-20260929-070308.log    # only once it passes --log-max-bytes
```

The deliverables are named after the input stem, so a run folder is
self-describing: `talk.mp4` yields `talk_dubbed.mp4` (or `talk_dubbed.wav` for an
audio input), and the subtitles-only mode writes `talk_subtitled.mp4` (or
`talk_burned.mp4` with `--subtitles-mode burn`).

`prosody.json` is the emotion record: for every segment it states the planned text,
the text actually spoken, the effective `ProsodySpec`, and the exact tokens/tags
the renderer added. `provenance/` keeps one timestamped copy per synthesis, so an
earlier run's plan is never overwritten. `--no-provenance` disables both.

`report.json` is the quickest way to judge a run without listening to it:

| Field | Meaning |
|---|---|
| `segment_count` | Utterances found after VAD gating |
| `source_language` | Language Whisper detected |
| `dub_seconds` | Length of the dub track (should match the source) |
| `loudness_source_lufs` | Loudness of the original audio |
| `loudness_dub_lufs` | Loudness actually delivered |
| `loudness_target_lufs` | What the dub aimed for |
| `loudness_policy` | `match-source` or `lufs` |
| `loudness_delta_vs_source` | Positive when the dub is louder than the original |
| `loudness_gain_db` | Total gain applied to reach the target |
| `loudness_peak_limited` | `true` when the limiter engaged to avoid clipping |
| `background_preserved` | `true` when the original music/ambience was kept |
| `duck_db` | Attenuation applied under dialogue (`null` when no background). With `--room match` this is the shallower `--room-duck-db`, i.e. the depth actually used |
| `room` | Present only with `--room match`: the mode, whether it was applied, the measured `rt60_s`, whether the reverb tail is on (`reverb`), and the `wet_db` / `tone_gain_db` / `duck_db` used. Absent — not `null` — on a dry run, so previous reports read unchanged. `applied: false` means the speech stem was missing (usually `--no-background`) and the dub was delivered dry |
| `accent_mode` | `native`, `preserve`, `speaker-only`, or `reference-accent` |
| `speakers` | One entry per voice with its segment count and speech time, or `null` for a one-voice job — so single-voice reports read exactly as they did before diarization existed. The names (`SPK_0`, …) are the ones in `02b_speakers.json` |
| `overlap` | Present only on multi-speaker runs: how many segments were found to carry a second voice (`segments`, `seconds`, `speech_ratio`) and how they were handled. `mode: "dominant"` means each flagged line was dubbed in its dominant speaker's voice; `mode: "separate"` (with `--overlap separate`) means both voices were recovered and dubbed, adding `separated` and `lines` counts. Absent — not `null` — on a one-voice job |
| `donor` | Which reference voice was used: an explicit file, or the auto-selected segment (with its timestamps and transcript) |
| `donor.speaker_count` | How many voices the run found |
| `donor.primary_speaker` | The speaker with the most speech. Its own fields are also copied to the top level of `donor`, so `donor.source` / `donor.accent` / `donor.blend` keep naming the voice a single-speaker job would have used |
| `donor.speakers` | Per-voice detail: the donor segment chosen for each, how many segments they own, how long they speak, and the accent and blend records for that voice. This is the job's voice bible |
| `donor.accent` | Present in `native` and `reference-accent` modes: for a supplied clip, its path and whether a transcript was provided; for a generated anchor, `source` (`anchor` or `anchor-matched`), the `language`, and the matched `rate_band` / `gender` |
| `donor.blend` | Present in every blending mode (`native` and `reference-accent`): the strategy used, the passes and takes considered, and how the kept clip scored. `donor_ratio` places it between the accent voice (`0`) and the speaker's own voice (`1`) — the field to watch, and the one to raise `--blend-iters` for. `margin` only ranks takes against each other; a negative value is not by itself evidence of a problem |
| `tts` | How synthesis batched and what memory it reached: `batch_size`, `max_decode_seconds`, `reference_seconds` (the clone clip, charged per line), `seconds_per_char` (measured, not assumed), `batches`, `peak_batch_seconds`, `oom_retries`, `cache_releases`, `ceiling_shrinks`, and `peak_reserved_bytes`. `oom_retries` or `ceiling_shrinks` above zero means the run touched the memory ceiling — lower `--tts-batch`. See [Memory on a long video](performance.md#memory-on-a-long-video). With `--devices` the counters of every worker are summed, `peak_reserved_bytes` is the worst card rather than the total, and `devices` plus `per_device` name the cards that did the work |
| `clamped_segments` | Lines that hit a stretch limit (either direction) |
| `clamped_long_segments` | …of which were too long to compress into their slot |
| `clamped_short_segments` | …of which were too short to slow down enough to fill it |
| `compress_floor` | The compression floor this run used (`--min-stretch`), which is what `needed_ratio` should be read against |
| `delivery_wps` | Words per second measured for each cloned voice by the probe — the number every word budget is derived from |
| `budget_wps` | …divided by `budget_speedup`: the ceiling the translator was actually given, per voice |
| `budget_speedup` | The tempo the budget was allowed to spend on every line |
| `accelerated_segments` | Lines the mixer sped up at all |
| `needed_below_floor_segments` | Lines whose *translation* was longer than the floor allows — the only ones that can lose words, and the number to drive to zero |
| `needed_ratio_min` | The hardest compression any line needed, before the floor was applied |
| `prefit` | What the check *before* synthesis did: `attempted`, `offenders`, `improved`, `llm_calls`, and `words_kept` (the share of the wording that survived) |
| `silence_trimmed_segments` | Lines that had dead air trimmed from their ends before fitting |
| `silence_trimmed_seconds_total` | Seconds of dead air removed. Silent, so nothing is lost — but leaving it in would spend the compression budget (and, at the floor, the wording) on a pause |
| `worst_trimmed_lines` | The worst offenders by name: `id`, `trimmed` seconds and the `needed_ratio` that caused it |
| `deficit_segments` | Lines that left silence in their slot |
| `deficit_seconds_total` | Total silence left inside slots |
| `overrun_segments` | Lines that spilled into the pause after them (free, no speech lost) |
| `overrun_seconds_total` | Total speech fitted into those pauses |
| `surplus_seconds_total` | Total speech exceeding slot + pause, i.e. at risk of being cut |
| `trimmed_segments` | Lines where the residue had to be cut to keep the video in sync |
| `trimmed_seconds_total` | Seconds of speech lost to those cuts |
| `worst_line_deleted_ratio` | The largest share of a single line that was deleted |
| `lines_over_10pct_deleted` | Lines that lost more than a tenth of their speech |
| `speech_rate_wps` | Words per second the cloned voice actually delivered, measured on the placed audio (dead air excluded, stretch included) — compare it with `delivery_wps` to see whether the probe was right |
| `refine` | What the length repair did: offenders, improved, seconds reclaimed |
| `shifted_segments` | Lines that start later than their source, because they used the overrun allowance |
| `shifted_seconds_total` / `shifted_seconds_max` | How late they run, in total and at worst. `max` is the number the sync gate judges |
| `max_overrun` | The lateness policy this run was given (`--max-overrun`) |
| `packed` | Present only when ASR timing had to be repaired: how many runs of lines Whisper squeezed into less time than their words need, how many lines that was, the worst words/second and the seconds added back. Absent — not `null` — when the boundaries were already plausible |

Two of these deserve attention:

- **`clamped_segments` above zero** means the translation was too long for its
  slot, so timing hit the stretch limit. Tighten the translation (or shorten the
  lines) rather than raising `--max-stretch`, which would sound unnatural.
- **`loudness_peak_limited: true`** means the limiter reduced a few peaks so the
target could be reached without clipping. Speech is peakier than mastered audio,
and without limiting a handful of transients would hold the whole program several
dB too quiet. The delivered loudness should still match `loudness_target_lufs`;
if it does not, the material was too dynamic to reach the target cleanly.
- **`trimmed_seconds_total`** is the one to watch. Translated speech is often
  longer than the source (French runs roughly 15–40% longer than English), and no
  amount of tempo can absorb an arbitrarily long line. A line is first compressed
  to the floor, then allowed to spill into the pause that follows it — free,
  since nobody is talking — and only the remainder is cut. Three things keep that
  remainder at zero, in order: the word ceiling the translator is given, the
  pre-synthesis check that rewords whatever still misses it, and the floor itself.
  `needed_below_floor_segments` counts the lines that got past all three, and
  `worst_trimmed_lines` names them. What is left is what the overrun allowance could
  not cover: a line may run up to `--max-overrun` seconds past its window instead of
  losing its tail, and the lines that do are counted in `shifted_segments` with their
  lateness in `shifted_seconds_max`.

## Caching

Stages are cached in `<job folder>/manifest.json`, keyed by a fingerprint of
every job setting. Re-running an identical command does nothing at all — that is
what makes iterating on a long video cheap.

The fingerprint is a hash of the whole configuration, so **changing any option
re-runs the whole job**: the target language, the accent mode, the reference or
accent clip, the stretch bounds, the device. There is no partial invalidation, so
a run that only changes `--accent-mode` re-does the ASR and translation as well.
That is deliberate — it keeps a result reproducible from one command — but it
means experimenting costs a full run each time. Slice the input short
(see [Fast iteration](usage.md#fast-iteration)) when you plan to try several settings.

The one exception is `--devices`: a job that never passes it hashes exactly what
it hashed before the option existed, so adding the feature invalidated no cache.
Asking for a fan-out *does* change the hash, because per-worker batching differs
from a single worker's — the same reason `--tts-batch` is part of it.

Because the fingerprint distinguishes settings, you can safely point several
different runs at the same `--output-dir`: each input gets its own `<stem>`
folder, and a run whose settings differ from the last one starts fresh rather
than reusing stale audio.

```bash
python dub.py --video talk.mp4 --target fr --force        # ignore the cache entirely
rm -rf output/talk                                        # or just start clean
```

---

## Judging a run

`report.json` makes a run readable without listening to it. Two scripts turn it —
and, optionally, a timbre measurement — into a verdict.

### Auditing the voice, line by line

`audit_timbre.py` measures whether each synthesised line kept the speaker's
timbre, using the checkpoint's own speaker encoder. It needs a GPU and the base
checkpoint, because it loads the same model the pipeline does:

```bash
python audit_timbre.py "output/<job-folder>"
python audit_timbre.py "output/<job-folder>" --device cpu   # when the GPUs are busy
```

It reports, per line, `cos(line, reference) − cos(line, accent)` where both
references are in the **target language** (the blend clip and the accent clip).
Matching the language is the point: comparing a French line against an English
donor mixes a language difference into a speaker measurement and makes every line
look like a collapse. Two anchors give the number a scale — a pure accent-voice
utterance scores about −0.023, and the reference against itself about +0.023 — so
"closer to the accent voice" is a measured position, not a threshold.

Measured on this repo's TED talk, with the same donor and accent clip:

| Run | Lines | min margin | median | Closer to the accent voice |
|---|---|---|---|---|
| `--accent-blend single-stage` | 55 | −0.0049 | +0.0156 | 5 (9.1%) |
| `two-stage` (default) | 59 | **+0.0176** | **+0.0257** | **0** |

Read it as a smoke test rather than a controlled experiment: ASR re-segmented the
two runs differently, so the text differs. Pass `--reference` and `--accent`
explicitly to measure a single-stage job against another job's blend clip.

A job with several speakers is audited **one speaker at a time**, each against
their own `reference_blend_<speaker>.wav`, because a line cloned from one voice
must not be judged against another's. Each group is its own scale — two voices
have two different "pure accent" floors — so the anchors are printed per group.
`--reference` overrides that and measures everyone against one clip, which is only
useful for a deliberate A/B.

### Checking a run automatically

`audit_qc.py` turns the signals in `report.json` — and, optionally, the timbre
audit above — into pass/fail gates. It never blocks the pipeline: the dub is
already written, so a failure is a verdict on the delivery rather than a reason to
lose the work. Exit status is `0` when every gate passes, `1` when one fails, and
`2` when the job could not be checked at all, which makes it usable in CI.

```bash
python audit_qc.py "output/<job-folder>"              # cheap gates, no model
python audit_qc.py "output/<job-folder>" --timbre      # adds the per-line timbre gate
python audit_qc.py "output/<job-folder>" --timbre --device cpu   # measure on the CPU
python audit_qc.py "output/<job-folder>" --json        # machine-readable
```

| Gate | Checks | Default limit |
|---|---|---|
| `sync` | No line starts more than the allowance late | `shifted_seconds_max <= 0.25` (override with `--max-shift`) |
| `truncation` | No line loses a meaningful share of its words | worst line ≤ 10%, no line over 10% |
| `loudness` | The dub sits at the source's level | within 1.5 dB (skipped if unmeasurable) |
| `timbre` | No line followed the accent voice instead of the speaker's | 0% (needs `--timbre`) |

Filling the pause after a line is *not* a failure — overrunning into silence costs
nothing and is how sync is preserved — so `overrun` is reported as context rather
than gated. Every limit can be overridden (`--max-shift`, `--max-deleted-ratio`,
`--max-deleted-lines`, `--max-loudness-delta`, `--max-accent-share`). A gate that
could not run (no timbre measurement, unmeasurable source loudness) is reported as
**skipped**, never as a pass.

In the app, the same `report.json` is behind a finished job on **Activity**; the
scripts take its folder path, which the job detail shows.

---

Next: [Usage](usage.md) for the option-to-command map, or
[Performance](performance.md) for device and memory choices.
