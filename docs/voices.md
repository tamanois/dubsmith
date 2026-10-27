# Voices

The saved voice library, the four accent modes, and how to borrow a native accent
without losing the speaker's timbre.

## Voice library (saved voices)

A **voice print** is a short reference clip plus its exact transcript, saved
under a label. Create one once, then reuse it by name — in `say`, or in a
dubbing run with `--voice`. Two ways to make one:

```bash
# 1. design a voice from a description (VoiceDesign checkpoint)
python dub.py voices --design \
  --prompt "a warm female French voice, slow and clear" \
  --lang fr --label warm-fr

# 2. clone a voice from a clip and its exact transcript (Base checkpoint)
python dub.py voices --clone --audio input/Delphine-Charming.mp3 \
  --clone-text "Bonjour, je suis Delphine. Comment tu vas ?" \
  --lang fr --label delphine

python dub.py voices --list                 # the library
python dub.py voices --remove warm-fr       # delete one
```

Prints live in `voices/` by default (override with `VOICES_DIR` or
`--voices-dir`), one folder per voice:

```
voices/warm-fr/voice.wav    # the reference clip (canonical passage, 20–30 s)
voices/warm-fr/voice.json   # label, language, source, transcript, instruct, gender, pace
```

Reference them:

```bash
# standalone speech — no video, no translation
python dub.py say --voice warm-fr --text "Bonjour, ceci est un test."

# or describe/clone inline, and save it at the same time
python dub.py say --design "a calm male narrator" --lang en \
  --text "…" --save-as narrator
python dub.py say --clone me.wav --clone-text "…" --lang en \
  --text "…" --save-as me

# use a saved voice in a dub
python dub.py --video talk.mp4 --target en --voice warm-fr
```

The saved voice supplies the **timbre**. If the print was recorded in the target
language, it is cloned directly: its own speech codes already carry the native
accent, so there is no blend to build. If it was recorded in another language,
pronunciation comes from a generated native anchor for the target language
(`--accent-mode native`) instead. Pass `--accent-mode` explicitly to override
either choice — for example `preserve` to always keep the print's own accent.

Defaults: `dub.py say` writes `output/say/<name>.wav`; a label is derived from
the description or audio filename unless `--label`/`--save-as` names it; an
existing label is never overwritten without `--force`. Overwriting a voice keeps
its path, so re-run a dub with `--force` to rebuild stages that used the old
clip.

## Accent modes

Accent and timbre are stored separately by the checkpoint: pronunciation and
prosody ride in the reference *speech codes*, while vocal identity rides in the
*speaker embedding*. That split is what makes these modes possible.

| `--accent-mode` | Speech codes (accent, pacing) | Speaker embedding (timbre) | Use when |
|---|---|---|---|
| `native` *(default)* | a **generated** native accent anchor | the reference voice, or the auto donor | The default: the speaker's timbre with a native target accent, no clip supplied |
| `preserve` | the reference voice, or the auto-selected donor | same clip | You want the original speaker, including their accent |
| `speaker-only` | none — the model chooses | the reference voice | The source accent is too strong and you want the model's own phonetics |
| `reference-accent` | **`--accent-audio`** | the reference voice, or the auto donor | You want to choose the borrowed accent yourself |

`native` asks for nothing: the anchor is synthesised by the VoiceDesign checkpoint
from a short native-language passage (see
[Automatic native accent](#automatic-native-accent)). The other three modes remain
available and unchanged. `speaker-only` is the cheap option — no extra clip, but
the model chooses the delivery — and `reference-accent` costs you one clip and its
transcript, buying an accent you pick by hand.

### Automatic native accent

The default mode generates its own accent reference, so the speaker's timbre comes
from the video and the pronunciation comes from a native narration the model
writes for the target language:

```bash
python dub.py --video talk.mp4 --target fr          # native accent, out of the box
python dub.py --video talk.mp4 --target fr --accent-match rate    # fit its pace
python dub.py --video talk.mp4 --target fr --accent-match gender  # pace and gender
```

The anchor is a short passage synthesised once per language by
`Qwen3-TTS-12Hz-1.7B-VoiceDesign` and cached under `models/anchors/`, so it costs
one generation per language and nothing afterwards. Pre-generate the languages you
care about during setup:

```bash
python dub.py anchors --generate                 # every supported language
python dub.py anchors --generate --lang fr --lang de
python dub.py anchors --list                     # what is cached
```

With `--accent-match`, the anchor is instead generated per speaker to fit the
source voice: `rate` matches its measured speaking rate, and `gender` matches its
(coarse) gender as well. This costs one anchor per distinct (language, rate band,
gender) combination, cached like the plain ones.


`reference-accent` asks the model to satisfy two signals at once: the accent
clip's speech codes and the speaker's own embedding. Those describe different
people, and any single line is free to follow either one — heard as the speaker
changing mid-video. So by default the mode **builds the blend once**
(`--accent-blend two-stage`): the accent transcript is rendered in the speaker's
voice, the take that kept their timbre is kept, and every line is then cloned from
that one clip. Its codes and its embedding come from the same audio, so there is
nothing left to disagree. `--accent-blend single-stage` restores conditioning each
line on both sources directly, which is useful for an A/B listen.

## Borrow a native accent without losing the voice

This is the manual alternative to the default: use it when you already have a
native-target recording whose pronunciation you want, instead of the generated
anchor. Hand over the clip; the voice stays the speaker's, the pronunciation
becomes the clip's. A news or podcast segment works better than a slow, careful
personal reading — see [Reference pacing](#reference-pacing):

```bash
python dub.py --video talk.mp4 --target fr \
  --accent-mode reference-accent \
  --accent-audio "native-narration.mp3" \
  --accent-text "the exact words spoken in native-narration.mp3"
```

Both flags are required together, and `--accent-text` must be the exact
transcript of `--accent-audio` — it is what aligns the codes being borrowed.
`--ref-audio` is optional: leave it out and the timbre is auto-selected from the
video, exactly as in the other modes. Keep the accent clip short; 10–20 seconds
is plenty, and a longer clip mostly costs synthesis time.

Before dubbing anything, the mode renders your accent transcript in the speaker's
own voice and writes the result to `reference_blend.wav` inside the job folder.
That clip is what every line is then cloned from.

Building that clip is not neutral between the two voices. The accent clip arrives
as a complete example — its speech codes *and* its transcript — while the speaker
contributes a single embedding, so a single pass tends to come out closer to the
accent clip than to the speaker. The mode therefore repeats it: each pass re-reads
the previous clip and re-injects the speaker's embedding, walking the voice back
toward the speaker while the accent transcript keeps pronunciation anchored. It
stops as soon as `--blend-ratio` is reached (default `0.85`, meaning well past
half-way from the accent voice to the speaker's own) or after `--blend-iters`
passes (default 5), whichever comes first, and keeps the best pass rather than the
last one. `--blend-takes` (default 3) additionally renders several candidates in
the first pass and keeps the one closest to the speaker. The clip is cached with
the rest of the job, so a rerun reuses it instead of quietly reselecting a
different voice.

How far the blend got is measured rather than guessed. `report.json` →
`donor.blend` reports `donor_ratio`: `0` is the accent clip's voice, `1` is a
French utterance in the speaker's own voice. Both ends of that scale are in the
target language, which is what makes it meaningful — a raw similarity against the
*source-language* donor would measure the language gap as much as the voice. If a
run stops short of the target, the log says so: raise `--blend-iters`, or lower
`--blend-ratio` if you would rather keep more of the accent. The `margin` field is
a different thing — a ranking score for choosing between takes, not a verdict,
since its absolute value is not comparable across clip pairs.

The quickest check on a run is still to listen to `reference_blend.wav` on its own:
twelve seconds of the speaker explaining something in the target language answers
"whose voice is this?" directly.

### Choosing the blend strength

`--blend-ratio` is the knob you will actually turn; `--blend-iters` is the safety
limit. Start from the defaults (`0.85`, `5`) and move in one direction at a time,
judging by the log's `donor ratio` line and the `reference_blend.wav` clip:

| You want | Do |
|---|---|
| The default: a balance that keeps the speaker clearly | leave both as-is (`--blend-ratio 0.85 --blend-iters 5`) |
| The speaker's voice stronger — the accent clip is leaking through | raise `--blend-ratio` (e.g. `0.92`), and `--blend-iters` with it |
| The speaker to dominate, accent as a hint only | `--blend-ratio 0.95 --blend-iters 8` |
| More of the borrowed accent, a bit of the speaker | lower `--blend-ratio` (e.g. `0.3`) |
| The exact previous single-pass behaviour | `--blend-iters 1` |
| Tune by hand rather than by the measured ratio | `--blend-ratio 0 --blend-iters N` — every pass runs and the least accent-like one is kept |

**Why the defaults moved.** They were `0.7` / `4` while the documented accent
source was a human clip, which clears a 0.8 target in a single pass. A *generated*
anchor (`native`, the default mode) is a complete model performance and drags much
harder: measured on one 8 s clip, 2 lines, same donor, the blend reached
`donor_ratio` 0.899 in one pass with a human accent clip, but only 0.861 after five
passes against the anchor — and at the old 0.7 target it stopped at 0.79, which is
the setting the speaker's voice sounded faintest under. Five passes is where that
curve flattened: passes six to eight improved neither the ratio nor the margin.
The cost is per *voice*, not per line: `takes + (iters - 1)` extractions of the
accent transcript per speaker.

If the log says the run *stopped short of the target* (the ratio line came in
under `--blend-ratio`), raise `--blend-iters` or lower `--blend-ratio`. A ratio
above the target means the search succeeded and stopped early; raising
`--blend-ratio` would push further toward the speaker, at the cost of one extra
synthesis of the accent transcript per pass.

## Reference pacing

The model imitates the reference clip's speaking rate, so a slow, deliberate
recording lengthens every dubbed line. Longer lines need more compression, and
once the stretch limit is reached the surplus has to be trimmed. This applies to
the donor (`--ref-audio`) and the accent clip (`--accent-audio`) alike.

Measured on this repo's TED talk — same source, same voice, only the reference
pacing differs:

| Reference used as | Pace | Speech trimmed |
|---|---|---|
| auto donor (the source speaker) | ~3.5 words/s | 11.4 s |
| slow personal recording, as donor | ~2.0 words/s | **23.0 s** |
| slow personal recording, as accent clip | ~2.0 words/s | 13.4 s |
| fast narrated clip, as accent clip | ~3.6 words/s | **2.3 s** |

Two conclusions worth carrying into your own reference choices. Borrowing only an
accent is cheaper than borrowing the whole voice: the same slow clip cost 13.4 s
as an accent source but 23.0 s as the donor. And reference pace is a **trade, not
a win**: a faster reference shortens the translation so less speech needs cutting,
but it also leaves the voice finishing early — on this video the fast clip left
15.2 s of silence spread over 23 lines, in slots where the original speaker is
still talking on screen.

Those two failures are not equally bad. Silence is a pause; deleted words are
information loss. Measured across five clip speeds, roughly half the lines missed
their slot at *every* pace, so pacing alone can never fix this — which is why the
pipeline no longer sizes its translations from the reference clip at all. It
measures the **cloned voice** with a probe render per speaker (`03a_rate.json`),
builds each line's word ceiling from that measurement, and checks the translation
against the ceiling before synthesising it. Reference pace is now a pronunciation
and delivery choice rather than a length lever; see
[When a translation is too long for its slot](audio-and-timing.md#when-a-translation-is-too-long-for-its-slot).

Need a transcript and never wrote one down? Transcribing the clip with the same
model the pipeline uses is accurate enough in practice — the run behind the table
above used a machine transcript of the accent clip. Just avoid paraphrasing: the
text has to match the audio it describes, so correct any obviously normalised
numbers or names by hand.

---

Next: [Multi-speaker](multi-speaker.md), or [Audio and timing](audio-and-timing.md).
