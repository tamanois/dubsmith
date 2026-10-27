# Multi-speaker dubbing

## Dubbing a video with several speakers

An interview, a debate, a two-hander: with more than one voice in the source, a
single donor cannot represent them. The pipeline finds the speakers and gives
each one their own voice. This is on by default, so a one-voice video behaves
exactly as it did before.

```bash
python dub.py --video debate.mp4 --target fr              # one voice per speaker
python dub.py --video debate.mp4 --target fr --diarizer none   # force one voice
```

The log names what it found, and `report.json` → `speakers` breaks it down:

```
Diarization (embedding): 2 speaker(s) over 12 segment(s) - SPK_0 7, SPK_1 5
Auto-selected donor for SPK_0: segment 3 (7.3s)
Auto-selected donor for SPK_1: segment 8 (5.0s)
```

**How it decides.** Each ASR segment is embedded with a speaker-verification
model and the embeddings are clustered, so every segment is assigned rather than
re-segmented — a line is never cut in half, and the labels stay aligned with the
text by construction. A cluster shorter than `--min-speaker-seconds` is folded
into its nearest neighbour rather than becoming a speaker with no usable voice to
clone from. That floor is generous by default (20 s, against the 3 s a donor needs
at all) because a one- or two-second donor does not describe a voice — the blend
built from it stalls near the accent, which is what "the voice is lost" looks like
in `report.json`. Segments whose embedding fails inherit the previous speaker's, so
one bad two-second window cannot invent a third voice.

The default backend is ERes2NetV2, which ships inside `funasr`. It is an optional
dependency: without it every line is dubbed in one voice and the log says so
rather than failing the job. Install it with `pip install -r requirements-dev.txt`
or uncomment it in `requirements.txt`.

The threshold is the one number worth knowing about. It is a cosine *distance* on
embeddings, and it is not portable across material — on the two-person clip in
this repo the merge profile is flat to `0.379` and then jumps to `0.468`, so
everything in that gap returns the same two speakers and a constant anywhere inside
it is defensible. The default is `0.7`, chosen on the 11-minute job rather than on
that clip: `0.40` split the video into **42** voices where `0.70` returned the **5**
people actually in it, and a listener confirmed that as the point where the extra
voices stopped appearing. If a video comes out over-split (one person becoming two)
raise `--speaker-threshold`; if two people are merged into one, lower it.
`02b_speakers.json` in the job folder records the labels the run used, so you can
see the damage without re-running.

**What each speaker costs.** Diarization adds one embedding pass over the speech
stem, which is cheap next to ASR. Synthesis is then grouped by speaker — one
prompt per voice rather than one per line — so a two-person video costs two
prompts and two batches instead of one of each. In
[`reference-accent`](voices.md#accent-modes) mode each voice needs its own blended
reference, because a clip whose codes and embedding both describe one speaker
cannot serve another:

```
04_dubbed/reference_blend_SPK_0.wav
04_dubbed/reference_blend_SPK_1.wav
```

Each is built by the same measured search as the single-voice case, with its own
`donor_ratio`, and `--blend-*` applies to all of them. A one-voice job keeps the
original `reference_blend.wav` name, so existing scripts and
`audit_timbre.py` are unaffected.

**Overlapping speech is detected, not guessed away.** Speakers are attributed per
segment, not per word, so a segment where two people talk over each other cannot
be given to both of them. Each segment is therefore re-embedded in 2.5 s windows
and every window is checked against the speakers the labels already establish: a
line is flagged when a *run* of its windows fits another of those speakers better
than its own, while its own speaker still owns the majority of it. Flagged lines
are listed in `02c_overlap.json`, counted in `report.json` → `overlap`, and named
in the log:

```
Overlapping speech detected in 1 of 16 segment(s): #1. Each is dubbed in the
voice of its dominant speaker, so the second voice is not separately dubbed.
```

The 2.5 s window is measured, not chosen by taste. On the two-person clip in this
repo a 1 s window splits one speaker's *own* voice by up to 1.08 cosine distance —
more than three times the 0.34 that separates the two speakers — so short windows
report noise rather than a second voice (re-clustering them flagged 14 of 16
segments). Comparing against a speaker *centroid* rather than against other
windows is the other half of the fix, because an average over many windows is
what cancels that noise; and a one-voice job is never flagged at all, since with
no second centroid there is nothing for a window to match.

The trade-off is sensitivity, and it is deliberate: an interruption shorter than
about a window is not caught, and a segment whose windows *all* match another voice
is treated as a labelling problem rather than an overlap. On this repo's clip the
detector flags 1 of 16 segments. By default a flagged line is still dubbed in one
voice, but the limitation is measured rather than silent; `--overlap separate`
splits it instead. A one-voice job skips the check entirely, so its report is
unchanged.

**Both voices are dubbed with `--overlap separate`.** A flagged segment is split
into one stream per voice by SpeechBrain's 16 kHz SepFormer, each stream is
re-transcribed (the second voice's words are not in Whisper's one-utterance
output), and each stream is matched to a speaker by comparing its embedding to the
same centroids the labelling used. Both lines are then translated and synthesised
in their own voice, and placed *simultaneously* on the timeline rather than one
being pushed behind the other — the overlap is genuine, so the dub keeps it. The
two voices are recorded in `02d_overlap_segments.json`, and the run's `report.json`
→ `overlap` gains `mode: "separate"`, `separated` and `lines`.

Separation is best-effort by construction: a segment whose streams cannot be told
apart (a likely false-positive flag), or whose second stream recovers no speech, is
left exactly as it was. `speechbrain` is an optional dependency — without it
`--overlap separate` warns and behaves as `--overlap dominant`.

**What is *not* attempted.** The number of speakers comes out of the clustering
rather than being declared, so `--speaker-threshold` is the knob to reach for when
the count looks wrong. Separation handles two overlapping voices per segment
(SepFormer is a fixed two-speaker model); a third voice on the same instant is not
recovered.

Measured on this repo's TED talk, same French lines and same donor, so the three
rows are directly comparable:

| Mode | Timbre similarity to donor | French acoustic fit (logprob, higher is better) |
|---|---|---|
| `preserve` | 0.9841 | −0.138 |
| `speaker-only` | 0.9876 | −0.081 |
| `reference-accent` | 0.9854 | **−0.057** |
| *(same speaker, different utterance)* | 0.9808 | — |

All three keep the timbre above the natural same-speaker baseline; the hybrid mode
moves the pronunciation furthest toward native French.

---

Next: [Audio and timing](audio-and-timing.md), or [Performance](performance.md).
