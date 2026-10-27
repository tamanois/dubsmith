# Audio and timing

Keeping the original bed, matching the room, and fitting a longer translation
into the slots the source leaves behind.

## Placing the dub in the room

Synthesised speech is close-mic'd and dry. Laid over a soundtrack it can read as a
voice *on top of* the picture rather than someone *in* it — the speaker's identity
is right, the sense of place is missing. `--room match` fixes that as a
post-synthesis step, so it cannot disturb the voice cloning. It does two things:

- the **room-tone floor**: the separated speech stem's quiet passages (genuine
  ambience, not the speaker) become a continuous bed under the dub;
- a **gentler duck** (`--room-duck-db`, 3 dB instead of 6), because the room is
  part of the voice and a deep duck pulls it away again.

```bash
python dub.py --video talk.mp4 --target fr --room match
```

**The reverb tail is opt-in and off by default.** An earlier version enabled it at
−12 dB, and a real listen rejected it: 2.4 s of undamped, full-bandwidth reverb
rings under every line and reads as an echo rather than a room. `--room-wet-db -20`
adds a tail if you want one — the impulse response is now low-passed at 4 kHz,
because a real room absorbs treble and flat noise does not.

It is **off by default** for the same reason the dry voice-over is the default: a
close voice-over is correct for narrated content, so it should stay reachable
rather than being silently replaced. Two things to know before enabling it. The
RT60 is estimated from the source's own decay, so it is a starting point — trust
your ears and tune with `--room-rt60` and `--room-wet-db`. And it needs the speech
stem, which only exists when the background is kept: with `--no-background` the dub
is delivered dry and `report.json` says `room.applied: false` rather than inventing
a space.

## When a translation is too long for its slot

Translated speech is routinely longer than the source: French needs roughly 15–40%
more time than English for the same content. Four things stand between that extra
length and a cut, in the order they apply.

1. **The word ceiling.** Each line's *window* — its slot plus the pause before the
   next line — is converted into a word count using the rate the **cloned voice**
   was measured at: one probe render per speaker, before translating
   (`03a_rate.json`). The translator is told that number, and the ceiling
   *replaces* the "stay within 20% of the original" hint rather than joining it.
   Sending both was the original bug: the model followed the source-length hint and
   overran the ceiling by 1.5–1.8×, which the mixer could only answer by deleting
   words.
2. **The pre-synthesis check** (`03d_prefit.json`). Whatever still exceeds its
   ceiling is reworded before it is ever spoken, in one batched translator call. A
   rewrite is adopted only if it is measurably shorter, the original translation
   stays in `03_script.json`, and the reworded version is what the subtitles show.
   Checking here rather than afterwards is the difference between one synthesis
   pass and two.
3. **The compression floor** (`--min-stretch`, default `0.75`). The mixer speeds a
   line up, never past the floor, and spends the pause before the next line before
   spending more tempo — a pause nobody talks over is free, while tempo always
   costs clarity.
4. **The measured repair** (`--refine`, default on). Lines that still overrun are
   reworded against *real* audio: their spoken length, the time actually available,
   and a ceiling derived from that measurement. Only those lines are re-synthesised.
5. **A bounded overrun** (`--max-overrun`, default `0.25 s`). A line that still does
   not fit may run up to a quarter of a second past its window, which the next line
   absorbs by starting that much later. This is the last guard before the cut: on
   the 11-minute job it is what stands between 52 damaged lines and four. It is
   lateness, not clarity — the alternatives were a 0.64 tempo on half those lines
   (audible) or deleting their last word.

Dead air goes first, for every line: a synthesised clip often opens with a tenth to
a third of a second of digital silence (`silence_trimmed_seconds_total` reports
what was removed). Left in place it is audible dead air *and* it is charged to the
compression budget — measured on one line, 0.5 s of silence forced a 0.82
compression where the speech alone needed 0.97.

The writes are all kept separate so a run explains itself: `03_script.json` is what
the model said, `03d_prefit.json` what the pre-synthesis check changed, and
`03b_spoken.json` what the measured repair changed. The subtitles follow the last
of those, so they never display words the viewer cannot hear.

### When the *source* timing is the problem

Whisper's boundaries occasionally squeeze consecutive lines into less time than
their words need — measured on the 11-minute job, nine words in 0.9 s, four of them
in 0.16 s with the next line starting 0.02 s later. No tempo, wording or allowance
fits that, because the text is right and the timing is not.

Those stretches are repaired **before** anything is translated: a run of
consecutive lines whose words could not have been spoken at 6.5 words/s or more
(the speaker on that job averages 4.7) is re-timed, each line taking a share of the
run's span and of the pause that follows it, in proportion to its own word count.
The run keeps its start, never passes the next line's start, and every line outside
it keeps its own time — so the repair **trades no sync at all**, it only fixes
boundaries that were impossible. `report.json` → `packed` records what it did, and
the block is absent when the boundaries were already plausible.

The repair also widens the word budget for those lines: a slot that held four words
in 0.16 s now holds a sentence, which is what stops the translator being asked for
the impossible.

Three properties make it safe to leave on:

- **Lines that already fit are never touched** — their audio is bit-for-bit
  identical to a run with `--no-refine`, which is asserted in the test suite.
- **A failed repair is a no-op.** If the translator is unreachable, or a custom
  translator cannot rewrite, the run logs a warning and keeps the audio it has.
- **Nothing is re-shortened on a re-run.** Both rewriting passes always start from
  the original translation, so repeated runs cannot shorten an already-short line.

Measured on the 4:47 TED talk, English → French, identical settings apart from
the flag:

| Metric | `--no-refine` | with repair |
|---|---|---|
| Speech deleted | 3.76 s | **0.0 s** |
| Worst single line deleted | **59%** | **0%** |
| Lines losing >10% of their speech | 5 | **0** |
| Lines that had to be cut | 8 | 0 |
| Output exceeding slot + pause | 4.79 s | 0.51 s |
| Silence left in slots | 18.35 s | 17.11 s |
| Round-trip intelligibility (WER) | 1.5% | 1.2% |
| Drift / loudness / background | — | unchanged |

The repair rewrote 3 of 57 lines in one extra translator call. Its cost is a
second synthesis pass over just those lines, which took the stage from 56 s to
73 s.

**The same problem on an 11-minute two-speaker video**, which is where the word
ceiling mattered. Both runs are the same input and settings, English → French,
`native` accent; ASR re-ran, so the two segmented slightly differently (489 vs 471
lines) and the deltas are indicative rather than perfectly controlled:

| Metric | Before | After |
|---|---|---|
| Speech deleted | 40.63 s | **4.69 s** |
| Lines that lost speech | 179 | **31** |
| Lines losing >10% of their speech | 135 | **23** |
| Worst single line deleted | 97.5% | 51.4% |
| Speech fitted into the following pauses | 43.97 s | 26.82 s |
| Dead air removed from clip ends | — | 74.12 s (277 lines) |
| Lines starting within `--max-overrun` of their source | 0 | 228 (worst 0.25 s) |
| Synthesis stage | 699 s | **461 s** |
| Sync gate | pass | **pass** |

Four things did most of the work: the probe measured the cloned voices at 3.57 and
4.00 words/s where the accent clip reads 2.97, so the ceilings sent to the
translator rose and the translations landed **on** them (words ÷ ceiling: median
1.00, mean 0.95, against 1.5–1.8× before); the pre-synthesis check reworded 39 of
the 144 lines that still missed, keeping 945 of their 1047 words; dropping dead air
from clip ends removed 74 s that had been charged to the compression budget; and
`--max-overrun` turned the lines that could still not fit into a quarter-second of
lateness instead of a deleted tail. Synthesis got *faster* despite the extra
translator call, because 192 lines used to be spoken twice.

### Trading deletions for lateness

The remaining lines are ones whose **source** region is genuinely over-packed: the
dub needs about 1.5× the time the original had, so either its tail is cut or it
starts late. `--max-overrun` moves along that curve, and the lateness is bounded
*per line* — a line that is already late loses its allowance and is cut instead, so
a long stretch of tight lines cannot walk the dub out of sync. Simulated over the
job's own fitted durations (and confirmed by the run at 0.25 s):

| `--max-overrun` | Lines that lose speech | Speech lost | Worst lateness |
|---|---|---|---|
| `0` (exact sync) | 66 | 8.90 s | 0 |
| `0.25` (default) | 31 | 4.69 s | 0.25 s |
| `0.5` | 15 | 2.63 s | 0.50 s |
| `0.75` | 4 | 0.40 s | 0.75 s |

Raising it is a one-flag decision per job, and the sync gate judges the result:
`audit_qc.py --max-shift 0.75` accepts 0.75 s of lateness, and `shifted_seconds_max`
says what a run actually used.

---

Next: [Outputs and caching](outputs-and-caching.md), or
[Testing and quality](testing-and-quality.md).
