# AI Video Dubbing — MVP

Dub a **single-speaker** video into another language while keeping the speaker's
**own voice timbre**. Runs entirely on local, open models; no cloud required.

See [`PLAN.md`](PLAN.md) for the full vision, roadmap (P1–P6) and design
decisions, and [`AGENTS.md`](AGENTS.md) for the project context.

---

## MVP scope

| | |
|---|---|
| **Input** | One video, or one audio file |
| **Speakers** | Single speaker (multi-speaker is roadmap P1) |
| **Source language** | One language, auto-detected (multi-language is P5) |
| **Target language** | English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian |
| **Voice** | Cloned from the source speaker, with three accent modes (see [Accent modes](#accent-modes)) |
| **Output** | Dubbed video/audio + source & translated SRT subtitles, original music/ambience preserved |
| **Not included** | Lip sync (P4), multi-speaker diarization (P1) |

---

## Prerequisites

### 1. System binaries

These are **not** installable from `pip` and need root. On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg rubberband-cli sox
```

| Binary | Required? | Used for | If missing |
|---|---|---|---|
| `ffmpeg` / `ffprobe` | **Required for video** | audio extraction, muxing, probing | video input fails with a clear error; **audio-only input still works** |
| `rubberband` | Recommended | best-quality time stretching | falls back to `sox`, then to a pure-Python stretcher |
| `sox` | Recommended | alternative stretcher | falls back to the Python stretcher |

The pipeline **degrades gracefully**: it never hard-fails because a *stretcher* is
missing, and audio-only jobs work with no system binaries at all. Only video
input genuinely requires `ffmpeg`.

> Check what you have:
> ```bash
> for b in ffmpeg ffprobe rubberband sox; do printf '%-11s %s\n' "$b" "$(command -v $b || echo MISSING)"; done
> ```

### 2. Python

Python **3.10+** is required (developed and tested on 3.10.12).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> If dependency resolution is awkward, install the two pins first — this is what
> `packages.md` refers to:
> ```bash
> pip install numpy==1.26.4 typing_extensions
> pip install -r requirements.txt
> ```

**GPU is optional.** With CUDA the pipeline uses `cuda` + `float16`; without it,
everything still runs on CPU (`int8` ASR, `float32` TTS) — just much slower.

Python packages installed by `requirements.txt`:

| Package | Role |
|---|---|
| `torch` | model runtime |
| `qwen_tts` | voice-cloning TTS |
| `faster-whisper` | ASR (bundles its own Silero VAD) |
| `pyloudnorm` | EBU R128 loudness normalisation |
| `openai` | client for the OpenAI-compatible translation endpoint |
| `librosa`, `soundfile` | audio I/O, resampling, fallback stretching |
| `pydantic` | config validation |

For development and tests:

```bash
pip install -r requirements-dev.txt
```

### 3. Optional: CUDA 12 libraries for GPU speech recognition

CTranslate2 (which faster-whisper uses) links against **CUDA 12** specifically,
while recent `torch` wheels ship **CUDA 13**. On such a machine ASR logs:

```
Library libcublas.so.12 is not found or cannot be loaded
```

The pipeline handles this by itself: it preloads vendored CUDA libraries, and if
that is not possible it **falls back to CPU** with a clear warning (identical
output, just slower). To get GPU ASR instead, install the matching runtime:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

No `LD_LIBRARY_PATH` export is required — the pipeline loads these libraries into
the process itself (see `pipeline/asr.py`). Check the log lines to see which
device was actually used.

### 4. A translation endpoint

The translation stage calls any **OpenAI-compatible** `/v1/chat/completions`
server. Point it at a local server or a cloud provider via `.env`:

```bash
cp .env.example .env
```

```dotenv
LLM_BASE_URL=http://127.0.0.1:8080/v1
LLM_API_KEY=not-needed
LLM_MODEL=local-model
```

A local `llama.cpp` server works well:

```bash
llama-server -m /path/to/model.gguf --port 8080
```

Real environment variables always take precedence over `.env`.

---

## Download the models

Model weights are **not** stored in this repository (`models/` is gitignored).
One command fetches everything the MVP needs:

```bash
python dub.py models --download all
```

This downloads:

| Name | Hugging Face repo | Approx. size |
|---|---|---|
| `whisper_large_v3` | `Systran/faster-whisper-large-v3` | ~3 GB |
| `tts_base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | ~3.5 GB |

Other commands:

```bash
python dub.py models --list                # show every model and whether it's present
python dub.py models --download tts_base   # one model only
python dub.py models --download everything # incl. optional models
python dub.py models --download --force all
```

Optional models (registered but not needed for the MVP):

| Name | Repo | Why you might want it |
|---|---|---|
| `tts_custom_voice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 9 premium named voices + emotion instructions |
| `tts_voice_design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | described voices; accent anchor for P2 |
| `tts_tokenizer` | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | standalone tokenizer (already bundled in `tts_base`) |

You can also copy a model directory in by hand — anything present under
`MODELS_DIR` is detected automatically. Override the location with `MODELS_DIR`
in `.env`, which every command then honours. The download command also takes
`--models-dir` directly:

```bash
python dub.py models --download all --models-dir /data/models
```

Check which models are visible (and what to do about the ones that aren't):

```bash
python dub.py models --list
```

A dubbing run fails with an actionable message naming the exact download command
if a required model is missing, so you can also just try a dub and see.

---

## Usage

### Which command do I want?

| If you want… | Use | What it changes |
|---|---|---|
| a first working dub, zero setup | `--video talk.mp4 --target fr` | detects the source language, picks the donor voice from the video, preserves the original music/ambience |
| a voice that isn't the video's speaker | `--ref-audio ref.wav --ref-text "…"` | clones `ref.wav` for both timbre and accent |
| to drop the source accent | `--accent-mode speaker-only` | keeps the timbre, lets the model choose target-language phonetics |
| a native accent in the original voice | `--accent-mode reference-accent --accent-audio nat.wav --accent-text "…"` | timbre from the video's speaker, pronunciation from `nat.wav` |
| no music behind the voice | `--no-background` | discards the separated bed instead of ducking it |
| an absolute loudness target | `--loudness lufs --lufs -16` | forces a level instead of matching the source |
| to redo a run from scratch | `--force` | ignores the stage cache |
| to judge a run without listening | *nothing* | read `report.json` in the job folder |

Every option is described in [Options in full](#options-in-full);
`python dub.py --help` prints the same list.

### Commands, grouped by purpose

**Set up**

```bash
python3 -m venv .venv && source .venv/bin/activate   # one-time
pip install -r requirements.txt                      # runtime dependencies
pip install -r requirements-dev.txt                  # + pytest

python dub.py models --download all                  # ~6.5 GB of weights
python dub.py models --list                          # what is present already?
```

**Dub something**

```bash
# video -> dubbed video, source language auto-detected
python dub.py --video talk.mp4 --target fr

# audio only: no ffmpeg needed, writes dub.wav + subtitles
python dub.py --audio interview.wav --target es

# trust your own language detection instead of Whisper's
python dub.py --video talk.mp4 --source en --target de

# skip the subtitle files
python dub.py --video talk.mp4 --target ja --no-subtitles
```

With no `--ref-audio`, the reference voice is taken automatically from the
longest clean speech segment and its transcript comes from the ASR pass, so
there is nothing to prepare by hand.

**Choose the voice**

```bash
# clone a specific recording instead of the speaker in the video
python dub.py --video talk.mp4 --target fr \
  --ref-audio "voice.wav" \
  --ref-text "the exact words spoken in voice.wav"

# keep the speaker's timbre, drop their accent
python dub.py --video talk.mp4 --target fr --accent-mode speaker-only

# keep the speaker's timbre, borrow a native accent
python dub.py --video talk.mp4 --target fr \
  --accent-mode reference-accent \
  --accent-audio "native.wav" \
  --accent-text "the exact words spoken in native.wav"
```

Any format `libsndfile` reads works for reference and accent clips:
`.wav` `.flac` `.mp3` `.ogg` `.aiff`, a URL, or a `(samples, rate)` tuple.
`--ref-text` must be the **exact transcript** of `--ref-audio` because the model
conditions on both (ICL mode); the same is true of `--accent-text` for
`--accent-audio`. See [Accent modes](#accent-modes) for which mode needs which.

**Control the audio bed and level**

```bash
python dub.py --video talk.mp4 --target fr --no-background    # voice only
python dub.py --video talk.mp4 --target fr --duck-db 10       # bury the music further
python dub.py --video talk.mp4 --target fr --separator none   # treat the source as one track
python dub.py --video talk.mp4 --target fr --loudness lufs --lufs -16
```

**Control timing**

```bash
python dub.py --video talk.mp4 --target fr --min-stretch 0.75  # allow faster speech, trim less
python dub.py --video talk.mp4 --target fr --max-stretch 1.10  # cap how far lines are slowed
python dub.py --video talk.mp4 --target fr --stretcher sox     # pick a specific backend
```

**Repeatable and diagnostic runs**

```bash
python dub.py --video talk.mp4 --target fr --force     # ignore the cache
python dub.py --video talk.mp4 --target fr -v          # debug logging
python dub.py --video talk.mp4 --target fr --device cpu # rule the GPU out
```

### Fast iteration

A 5-minute video takes a few minutes to dub, which is slow to iterate on. Slice
out a short section and dub that instead — the pipeline behaves identically,
just faster:

```bash
ffmpeg -ss 30 -t 60 -i talk.mp4 -c copy clip60.mp4
python dub.py --video clip60.mp4 --target fr
```

Note that `-c copy` cuts on the nearest keyframe, so `clip60.mp4` may start a
second or two earlier than requested. That is fine for trying out options.

### Accent modes

Accent and timbre are stored separately by the checkpoint: pronunciation and
prosody ride in the reference *speech codes*, while vocal identity rides in the
*speaker embedding*. That split is what makes these three modes possible.

| `--accent-mode` | Speech codes (accent, pacing) | Speaker embedding (timbre) | Use when |
|---|---|---|---|
| `preserve` *(default)* | the reference voice, or the auto-selected donor | same clip | You want the original speaker, including their accent |
| `speaker-only` | none — the model chooses | the reference voice | The source accent is too strong and you want the model's own phonetics |
| `reference-accent` | **`--accent-audio`** | the reference voice, or the auto donor | The dub still sounds foreign and you have a native-target clip to borrow pronunciation from |

`speaker-only` is the cheap option: it needs no extra clip, but it also gives up
control over delivery, so pacing drifts with the model. `reference-accent` costs
you one clip and its transcript, and buys a delivery you can actually choose.

### Borrow a native accent without losing the voice

Hand over a recording of a native speaker. A news or podcast segment works
better than a slow, careful personal reading — see
[Reference pacing](#reference-pacing). The voice stays the speaker's; the
pronunciation becomes the clip's:

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

### Reference pacing

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
information loss. So prefer a reference that leans *slightly* fast, and let
[the length repair](#when-a-translation-is-too-long-for-its-slot) handle the lines
that still come out long — it removes the deletion without adding silence, which
no choice of reference pace can do. Measured across five clip speeds, roughly half
the lines missed their slot at *every* pace, so pacing alone cannot fix this.

Failing all that, lower `--min-stretch` to allow faster speech.

Need a transcript and never wrote one down? Transcribing the clip with the same
model the pipeline uses is accurate enough in practice — the run behind the table
above used a machine transcript of the accent clip. Just avoid paraphrasing: the
text has to match the audio it describes, so correct any obviously normalised
numbers or names by hand.

### Options in full
**Input and language** — exactly one of `--video` / `--audio` is required.

| Option | Default | What it achieves |
|---|---|---|
| `--video PATH` | – | Video input. Enables muxing, so you get `video_dubbed.mp4`. Needs `ffmpeg`. |
| `--audio PATH` | – | Audio-only input. Produces `dub.wav` + subtitles, needs no `ffmpeg`. |
| `--target LANG` | `fr` | Language to dub into. Accepts a code (`fr`) or a name (`French`). |
| `--source LANG` | auto | Overrides the detected source language. Useful when a clip is too short or too noisy for reliable detection. |

**Voice** — see [Accent modes](#accent-modes) for how these interact.

| Option | Default | What it achieves |
|---|---|---|
| `--ref-audio PATH` | auto | Clones this recording instead of the speaker found in the video. |
| `--ref-text TEXT` | – | Exact transcript of `--ref-audio`. Required with it in `preserve` mode; ignored in the other two modes. |
| `--accent-mode MODE` | `preserve` | Where pronunciation comes from: `preserve` (the reference voice), `speaker-only` (the model's own phonetics), `reference-accent` (the `--accent-audio` clip). |
| `--accent-audio PATH` | – | Native target-language clip to borrow pronunciation from. Requires `--accent-mode reference-accent`. |
| `--accent-text TEXT` | – | Exact transcript of `--accent-audio`. Required with it. |

**Audio bed and loudness**

| Option | Default | What it achieves |
|---|---|---|
| `--no-background` | off | Discards the original music/ambience. By default it is separated out and ducked under the dub instead. |
| `--separator BACKEND` | `auto` | `auto`, `demucs` (split speech from the bed), or `none` (treat the source as a single track). |
| `--duck-db DB` | `6.0` | How far the background drops while dialogue plays. Higher = the voice sits more forward. |
| `--loudness POLICY` | `match-source` | `match-source` puts the dub at the original track's measured level; `lufs` forces `--lufs` instead. |
| `--lufs VALUE` | `-16` | Absolute loudness target in LUFS. Only used with `--loudness lufs`. |

**Timing** — these trade speech clarity against staying in sync.

| Option | Default | What it achieves |
|---|---|---|
| `--min-stretch RATIO` | `0.85` | Most a line may be compressed to fit its slot. Lower = faster speech, so fewer words get trimmed, but it can start to sound hurried. |
| `--max-stretch RATIO` | `1.18` | Most a line may be slowed to fill its slot. Raising this makes speech sound unnatural, so prefer fixing the translation. |
| `--stretcher BACKEND` | `auto` | `auto`, `rubberband`, `sox`, `librosa`, `none`. `none` disables stretching (lines then only overrun or trim). Falls back automatically if a binary is missing. |

**Length repair** — fixes a long translation by wording rather than by cutting audio.

| Option | Default | What it achieves |
|---|---|---|
| `--refine` / `--no-refine` | on | Rewrites the lines that would otherwise have words cut, then re-synthesises just those. Lines that already fit are untouched; a rewrite is kept only if it measurably reduces the overflow. `--no-refine` restores the previous trim-only behaviour. |
| `--refine-max-passes N` | `1` | How many times a line may be rewritten before falling back to trimming. `0` disables the repair. |
| `--refine-threshold SECONDS` | `0.05` | How far a line must overrun its slot *and* the pause after it before it is worth rewriting. Raise it to repair fewer, worse lines. |
| `--length-budget MODE` | `off` | `words` also tells the translator how many words fit each slot, measured from a target-language reference clip. `off` sends only the duration, as before. Needs `--accent-audio` or a target-language `--ref-audio`; without one it falls back to `off` automatically. |

**Models and hardware**

| Option | Default | What it achieves ||---|---|---|
| `--device DEVICE` | `auto` | `auto`, `cuda:0`, or `cpu`. `cpu` is much slower but proves the GPU is not the problem. |
| `--whisper-model NAME` | `large-v3` | ASR model id. A smaller one is faster and less accurate. |
| `--llm-model NAME` | from `.env` | Overrides `LLM_MODEL` for this run. |
| `MODELS_DIR` (env) | `models` | Where weights live. Set in `.env`; the download command also takes `--models-dir`. |

**Output and caching**

| Option | Default | What it achieves |
|---|---|---|
| `--output-dir DIR` | `output` | Where job folders are written. Each input gets its own `<stem>` folder. |
| `--no-subtitles` | off | Skips both SRT files. |
| `--force` | off | Ignores the stage cache and rebuilds everything. |
| `-v` / `--verbose` | off | Debug logging, including which device each stage actually used. |

### When a translation is too long for its slot

Translated speech is routinely longer than the source: French needs roughly 15–40%
more time than English for the same content. The stretch clamp absorbs part of
that, and a line is allowed to spill into the pause that follows it, but anything
beyond that used to be **cut** — words deleted to keep the video in sync.

The length repair avoids the cut by fixing the *text* instead:

1. After the first synthesis pass, each line's spoken length is measured against
   its slot plus the following pause.
2. Lines that would overflow are sent back to the translator together, with their
   previous wording, the source line, and a word ceiling derived from how much
   time is actually available.
3. Only those lines are re-synthesised. A rewrite is adopted **only** if it
   measurably reduces the overflow, so a line can never come out worse.

Three properties make it safe to leave on:

- **Lines that already fit are never touched** — their audio is bit-for-bit
  identical to a run with `--no-refine`, which is asserted in the test suite.
- **A failed repair is a no-op.** If the translator is unreachable, or a custom
  translator cannot rewrite, the run logs a warning and keeps the audio it has.
- **Nothing is re-shortened on a re-run.** Rewrites always start from the original
  translation, recorded in `03_script.json`, and the spoken version is kept
  separately in `03b_spoken.json` — which is also what the target subtitles use,
  so they never display words the viewer cannot hear.

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

### Verified commands
These were all run on a 4:47 English TED talk → French, on 2× RTX 3060. The
figures come from each run's `report.json`.

| Command | Result |
|---|---|
| `--video talk.mp4 --target fr` | 11.4 s of speech trimmed; sync exact; background preserved |
| `--video talk.mp4 --target fr --ref-audio slow.wav --ref-text "…"` | 23.0 s trimmed — the slow reference lengthened every line |
| `--video talk.mp4 --target fr --accent-mode speaker-only` | timbre 0.9876, French fit −0.081 |
| `--video talk.mp4 --target fr --accent-mode reference-accent --accent-audio slow.wav --accent-text "…"` | timbre 0.9854, French fit −0.057, 13.4 s trimmed |
| `--video talk.mp4 --target fr --accent-mode reference-accent --accent-audio fast.mp3 --accent-text "…"` | **best overall**: 2.3 s trimmed, 0 drift, loudness matched, background preserved |
| `--video talk.mp4 --target fr --accent-mode reference-accent --accent-audio fast.mp3 --accent-text "…" --refine` (default) | **0.0 s trimmed**: 3 lines rewritten, worst-line deletion 59% → 0%, WER 1.5% → 1.2% |

The last row is the configuration to aim for: a fast-paced native clip as the
accent source gives near-native pronunciation with almost no speech lost. Full
run takes about 2 minutes 45 seconds.

---

## Outputs

Everything for one job lands in `output/<input-stem>/`:

```
output/talk/
├── 01_audio.wav         # normalised 48 kHz mono source audio
├── 01b_speech.wav       # separated speech stem (when background is preserved)
├── 01c_background.wav   # separated music/ambience stem
├── 02_transcript.json   # ASR result: segments, words, timings, language
├── 03_script.json       # source -> target text per segment
├── 03b_spoken.json      # lines the length repair re-wrote (only those)
├── 03c_refine.json      # what the length repair achieved
├── 04_dubbed/           # one synthesised clip per segment (+ .done marker)
├── 05_mix.wav           # assembled dub track, before normalisation
├── timing.json          # per-segment stretch ratio, placement and deficit
├── dub.wav              # final loudness-normalised dub audio
├── video_dubbed.mp4     # dubbed video (video input only)
├── subs_source.srt
├── subs_target.srt
├── report.json          # counts, loudness, stretch stats, stage timings
└── manifest.json        # stage cache
```

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
| `duck_db` | Attenuation applied under dialogue (`null` when no background) |
| `accent_mode` | `preserve`, `speaker-only`, or `reference-accent` |
| `donor` | Which reference voice was used: an explicit file, or the auto-selected segment (with its timestamps and transcript) |
| `donor.accent` | Present only in `reference-accent` mode: the accent clip's path and whether a transcript was supplied |
| `clamped_segments` | Lines that hit a stretch limit (either direction) |
| `clamped_long_segments` | …of which were too long to compress into their slot |
| `clamped_short_segments` | …of which were too short to slow down enough to fill it |
| `deficit_segments` | Lines that left silence in their slot |
| `deficit_seconds_total` | Total silence left inside slots |
| `overrun_segments` | Lines that spilled into the pause after them (free, no speech lost) |
| `overrun_seconds_total` | Total speech fitted into those pauses |
| `surplus_seconds_total` | Total speech exceeding slot + pause, i.e. at risk of being cut |
| `trimmed_segments` | Lines where the residue had to be cut to keep the video in sync |
| `trimmed_seconds_total` | Seconds of speech lost to those cuts |
| `worst_line_deleted_ratio` | The largest share of a single line that was deleted |
| `lines_over_10pct_deleted` | Lines that lost more than a tenth of their speech |
| `speech_rate_wps` | Words per second the cloned voice actually delivered |
| `refine` | What the length repair did: offenders, improved, seconds reclaimed |
| `shifted_segments` | Lines pushed later to avoid overlapping another |

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
  longer than the source (French runs roughly 15–40% longer than English), and
  the stretch limit caps how far a line can be compressed. A line is first
  allowed to spill into the pause that follows it — free, since nobody is talking
  — and only the remainder is cut, because letting it run on would push every
  later line out of sync with the video. Nothing is ever shifted late: sync is
  exact. To reduce trimming, ask the translator for terser output (roadmap P3
  adds an automatic length-feedback loop), or lower `--min-stretch` to allow
  faster speech.

### Caching

Stages are cached in `<job folder>/manifest.json`, keyed by a fingerprint of
every job setting. Re-running an identical command does nothing at all — that is
what makes iterating on a long video cheap.

The fingerprint is a hash of the whole configuration, so **changing any option
re-runs the whole job**: the target language, the accent mode, the reference or
accent clip, the stretch bounds, the device. There is no partial invalidation, so
a run that only changes `--accent-mode` re-does the ASR and translation as well.
That is deliberate — it keeps a result reproducible from one command — but it
means experimenting costs a full run each time. Slice the input short
(see [Fast iteration](#fast-iteration)) when you plan to try several settings.

Because the fingerprint distinguishes settings, you can safely point several
different runs at the same `--output-dir`: each input gets its own `<stem>`
folder, and a run whose settings differ from the last one starts fresh rather
than reusing stale audio.

```bash
python dub.py --video talk.mp4 --target fr --force        # ignore the cache entirely
rm -rf output/talk                                        # or just start clean
```

---

## How it works

```
video/audio ──► 1. audio extract ──► 2. separate ──► 3. ASR + VAD ──► 4. translate (LLM)
                                  (speech | bed)                        │
            7. mux + SRT ◄── 6. mix + duck ◄── 5. clone + synthesise ◄───┘
```

| # | Stage | Module | What it does |
|---|---|---|---|
| 1 | Audio | `pipeline/media.py` | ffmpeg extract (video) or direct read (audio), mono 48 kHz |
| 2 | Separate | `pipeline/separation.py` | Demucs splits speech from music/ambience; ASR then reads the clean speech stem |
| 3 | ASR | `pipeline/asr.py` | faster-whisper + Silero VAD, word timestamps, drops non-speech |
| 4 | Translate | `pipeline/translate.py` | batched, context-aware, length-budgeted translation |
| 5 | Synthesise | `pipeline/tts.py` | donor selection + Qwen3-TTS voice cloning, per-line limiting |
| 6 | Mix | `pipeline/timing.py`, `pipeline/mixing.py` | fit each line to its slot, then lay it over the ducked background |
| 7 | Deliver | `pipeline/orchestrate.py` | loudness match, subtitles, mux, report |

Two design points worth knowing:

- **Accent comes from the donor clip, timbre from cloning.** With no
  `--ref-audio`, the source speaker is the donor, so the accent is *preserved*.
  `--accent-mode speaker-only` drops the donor's phonetics altogether;
  `--accent-mode reference-accent` replaces them with a native clip's.
- **Timing errors never accumulate.** Each line is anchored to its own original
  start time; a line that does not fit spills into the pause after it and any
  residue beyond that is trimmed. Lines are never pushed late, so a long
  translation cannot make the rest of the video drift out of sync.

---

## Tests

All tests run **without** a GPU, downloaded models, or ffmpeg — heavy components
are injected as fakes and binaries are mocked. The whole suite runs in about
15 seconds.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

```
400 passed
```

Run a single area while iterating:

```bash
python -m pytest tests/test_tts.py -q          # voice cloning, accent modes
python -m pytest tests/test_pipeline.py -q     # end-to-end orchestration
python -m pytest -k accent -q                  # everything accent-related
```

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Missing model: 'whisper_large_v3' …` | Run `python dub.py models --download all` |
| `Required binary 'ffmpeg' was not found on PATH` | `sudo apt-get install -y ffmpeg` (audio-only jobs still work) |
| `… contains no audio stream, so there is nothing to transcribe` | The file is video-only. Verify with `ffprobe -v error -show_entries stream=codec_type file.mp4`; mux an audio track in first |
| `CUDA is unusable for speech recognition … libcublas.so.12` | Informational — ASR fell back to CPU. Install `nvidia-cublas-cu12 nvidia-cudnn-cu12` for GPU ASR (Prerequisites §3) |
| `No speech detected in the input` | No audible speech, or the wrong audio track. Check with `ffprobe` |
| `Translation response was not valid JSON` | Lower `LLM_CHUNK_SIZE` in `.env`, or use a model better at JSON |
| `Translation request to … failed` | The LLM server isn't reachable — check `LLM_BASE_URL` |
| `--ref-audio requires --ref-text` | Add the exact transcript, or drop both flags to auto-pick a donor |
| `--accent-audio only applies to --accent-mode reference-accent` | Add that flag, or remove `--accent-audio` |
| `--accent-mode reference-accent requires --accent-audio` | Add the native-target clip, or use another mode |
| `--accent-audio requires --accent-text` | Transcribe the accent clip; the codes cannot be aligned without it |
| `does not support voice cloning` | Pointed at a CustomVoice/VoiceDesign checkpoint; use `tts_base` |
| `Unsupported language 'xx'` | Use one of the ten supported languages |
| CUDA out of memory | `--device cpu`, or free the GPU (a large local LLM often occupies it) |
| Dub sounds rushed or slow | Check `report.json` → `clamped_segments`; tighten the translation |
| Some words are cut off | `report.json` → `trimmed_segments`: the translation is too long for its slot. Check `worst_line_deleted_ratio` too — a small total can hide one badly damaged line. Leave the length repair on (it is on by default), shorten the translation, or lower `--min-stretch` |
| A line lost a lot of words despite a low `trimmed_seconds_total` | Look at `worst_line_deleted_ratio` and `lines_over_10pct_deleted`: damage is usually concentrated in one or two lines, not spread evenly |
| Lines sound slow or have gaps | `report.json` → `deficit_seconds_total`: the translation is shorter than its slot. See [Reference pacing](#reference-pacing), or use `--length-budget words` with a target-language reference |
| The repair did not help | Check `report.json` → `refine`. `attempted: false` means no line overflowed (nothing to fix) or the translator cannot rewrite; `improved: 0` means the rewrites did not fit either, so the old behaviour applied |
| Dub drifts out of sync | Should not happen — lines are never shifted late. If `shifted_segments > 0`, check `timing.json` for genuinely overlapping source speech |
| Dub sounds quiet | Check `loudness_delta_vs_source`; `0` means it matches the original. Force a level with `--loudness lufs --lufs -16` |
| Music/ambience is missing | Install `demucs` (`pip install demucs`); without it the background cannot be separated and is discarded |
| Music fights the voice | Raise `--duck-db` (default 6 dB) |
| Dub still has the speaker's accent | Try `--accent-mode speaker-only`, which keeps the timbre but drops the donor's phonetics. If that is not enough, use `--accent-mode reference-accent` with a native clip |
| Still sounds foreign in `reference-accent` mode | Check `--accent-text` matches the clip word for word — mismatched text misaligns the borrowed codes. A longer, more natural accent clip also helps |
| The dub is rushed, or words are missing | The reference clip's pacing is being imitated, and a slow reference lengthens every line. Use a faster-paced reference — see [Reference pacing](#reference-pacing) |

Logs are written to stderr and include a job id. Set `LOG_JSON=1` for
machine-readable logs.

---

## Project layout

```
dub.py                     CLI entry point
pipeline/
├── config.py              job + settings models, language normalisation
├── errors.py              domain exception hierarchy
├── logging_utils.py       job-scoped structured logging
├── models.py              model registry, download, path resolution
├── media.py               ffmpeg/ffprobe wrappers, loudness, audio I/O
├── subtitles.py           SRT rendering
├── asr.py                 transcription + segmentation
├── translate.py           OpenAI-compatible translation
├── tts.py                 donor selection + voice cloning
├── timing.py              stretching backends + timeline assembly
└── orchestrate.py         stages, caching, reporting
tests/                     pytest suite (400 tests)
models/                    downloaded weights (gitignored)
output/                    job artifacts (gitignored)
clone.py, main.py          standalone TTS demos kept for reference
PLAN.md                    roadmap: MVP + P1–P6
.agents/technical-guidelines.md   engineering standards
```

---

## Notes and limitations

- **Voice cloning requires consent.** Clone only voices you have permission to
  use. Voice is biometric data under GDPR, and synthetic-dubbing disclosure is
  increasingly required by platforms. A consent gate is planned for P6.
- **Model licences**: Qwen3-TTS and Whisper weights are Apache-2.0 / MIT.
  Avoid CPML-licensed models (e.g. XTTS-v2) if you need commercial rights.
- **Known MVP gaps**, all tracked in `PLAN.md`: multi-speaker diarization, lip
  sync, multi-language source videos, and iterative length control in the
  translation loop. Background preservation and native-accent borrowing are
  implemented (see [Accent modes](#accent-modes)).
- **Accent borrowing needs a supplied clip.** `reference-accent` takes
  pronunciation from `--accent-audio`; it does not yet choose or generate a
  native voice by itself. Fully unattended native accent is roadmap P2.
- **Changing any setting re-runs the whole job** — the cache fingerprint is
  per-job, not per-stage. Fine for a CLI; worth refining before P6.
