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

The design prompt is a natural-language instruction, so it can also describe
tone and delivery — "an angry male French voice, speaking fast" works, because
Qwen VoiceDesign is conditioned on timbre, emotion and prosody together. This is
the only backend where a voice is created from words; CosyVoice and Chatterbox
clone from audio and cannot design.

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

## Emotion with CosyVoice2

The Qwen Base checkpoint copies its prosody from the reference clip, so a flat
reference reads as a flat voice. `dub say --backend cosyvoice` swaps in
CosyVoice2, which accepts a **per-line emotion/style instruction** on top of the
cloned voice:

```bash
# one line, one emotion: --prosody maps the word onto a trained instruction
python dub.py say --voice warm-fr --backend cosyvoice \
  --text "Nous avons gagné !" --prosody "emotion=happy"

# several lines, an emotion each (a single value broadcasts to every line)
python dub.py say --voice warm-fr --backend cosyvoice \
  --text $'Bonjour.\nAttends, quoi ?' --prosody "emotion=calm" --prosody "emotion=happy"
```

Prefer `--prosody` on this backend. `--prosody emotion=<word>` is translated into
the checkpoint's own trained instruction (for example `请非常开心地说一句话。`),
with aliases such as `excited` mapped onto `happy`. A bare `--emotion` is instead
passed through **verbatim** as the instruction text, which is only useful if you
write the instruction yourself — a single English adjective is not something this
checkpoint was trained on.

`--text` is split on newlines; each line is spoken as its own utterance with its
own emotion (one value broadcasts to every line, and a neutral or `calm` line
clones plainly). `--clone` works the same, and `--clone-text` is optional with
CosyVoice. Output: per-line WAVs plus a combined `speech.wav` under
`output/say/<name>/`.

The same `voices` prints serve both backends — a print is just a reference clip.
Requires the CosyVoice2 package, its transformers overlay and the checkpoint (see
[Setup](setup.md#cosyvoice2-optional-expressive-speech)); the engine runs it in a
child process because its `transformers` pin differs from Qwen's.

**Languages.** CosyVoice2 covers Chinese, English, Japanese, Korean and Cantonese
well. French, German, Spanish, Italian, Portuguese and Russian are not officially
supported and sound poor; use the default Qwen backend for those.

## Expressive backends: CosyVoice3, Chatterbox, Fish and IndexTTS2

These backends target the same problem — speech that sounds *alive* rather than
read — and all clone the voice from a reference clip.

| Backend | Languages | Controls | Setup |
|---|---|---|---|
| `--backend cosyvoice3` | 9, including **French** | emotion, pace, volume, `[breath]`, `[laughter]` | `python dub.py models --download tts_cosyvoice3` |
| `--backend chatterbox` | 23, including **French** | emotion exaggeration (one intensity) | `python dub.py models --download tts_chatterbox` + the isolated env below |
| `--backend fish` | ~100 | emotion + paralinguistics as **inline tags** (`[excited]`, `[whisper]`, `[sad]`, `[angry]`, `[laughing]`, `[pause]`) | `python dub.py models --download tts_fish` + the isolated env below |
| `--backend indextts2` | zh, en, ja (2.0); + es, ar in 2.5 | **separate emotion reference** (`--emotion-audio`, `--emotion-alpha`) + `--duration` seconds | `python dub.py models --download tts_index_tts2` + the isolated env below |

```bash
# CosyVoice3, with real per-line direction
python dub.py say --voice warm-fr --backend cosyvoice3 \
  --text $'Bonjour.\nAttends, quoi ?' \
  --prosody "emotion=calm, volume=soft" \
  --prosody "emotion=excited, pace=fast, breath"

# Chatterbox: emotion maps onto its exaggeration knob
python dub.py say --voice warm-fr --backend chatterbox \
  --text "Nous avons gagné !" --emotion happy

# Fish Speech: the emotion becomes an inline tag
python dub.py say --voice warm-fr --backend fish \
  --text "Je n'arrive pas à y croire !" --emotion excited

# IndexTTS2: the voice comes from --voice, the emotion from a second clip
python dub.py say --voice warm-fr --backend indextts2 \
  --text "Pourquoi tu ne me l'as pas dit ?" \
  --emotion-audio colere.wav --emotion-alpha 0.8 --duration 3.2
```

### `--prosody`

`--prosody` is the structured form of `--emotion`. It accepts comma-separated
`key=value` pairs or bare flags, and may be repeated once per line (a single
value broadcasts to every line):

```
emotion=<word>   pace=normal|slow|fast   volume=normal|soft|loud
breath   pause   laughter
```

### Values each backend accepts

`--prosody` takes one spec per line (a single value broadcasts):

| Field | Accepted values | CosyVoice 2/3 | Chatterbox | CustomVoice |
|---|---|---|---|---|
| `emotion` | free-form | trained set only: `happy`, `sad`, `angry`; aliases such as `excited`→happy, `calm`→none | intensity only: neutral 0.5, calm 0.4, sad 0.3, happy 0.8, excited 0.9, angry 0.85 | free-form tone, e.g. "Speak in an angry tone." |
| `pace` | `normal`, `slow`, `fast` | trained speed instructs | no pace control; ±0.05 intensity | phrased as "Speak slowly/quickly." |
| `volume` | `normal`, `soft`, `loud` | trained volume instructs | no volume control; ±0.10 intensity | phrased as "Speak softly/loudly." |
| `exaggeration` | `0.0`–`1.0` | ignored | exact value, overrides the emotion mapping | ignored |
| `breath` | flag | `[breath]` at sentence boundaries | ignored | ignored |
| `pause` | flag | a comma inserted in a long, unpunctuated line | ignored | ignored |
| `laughter` | flag | `[laughter]` appended to the line | ignored | ignored |

**Fish Speech** reads `emotion` as a prefix tag (`happy`/`excited` → `[excited]`,
`angry` → `[angry]`, `sad` → `[sad]`, `whisper` → `[whisper]`), `pause` as
`[pause]` at the first clause boundary, and `laughter` as a trailing
`[laughing]`. `pace`, `volume` and `breath` have no documented tag and are
dropped; an unknown emotion is dropped with a warning.

**IndexTTS2** ignores `--prosody`/`--emotion`/`--prosody-plan` entirely: its
emotion comes from `--emotion-audio` (see below). A warning is logged when both
are given.

Aliases: `rate`/`speed` for `pace`, `loudness` for `volume`, `mood`/`tone` for
`emotion`, `expression`/`expressiveness` for `exaggeration`. A bare word is taken
as the emotion (`--prosody excited`).

> **CosyVoice emotions are limited on purpose.** The checkpoint only has trained
> instruct sentences for happy, sad and angry; `excited` maps onto happy and
> `calm` carries no emotion clause (so "calm" means "no emotion instruction").
> An arbitrary word is dropped with a warning rather than sent as an instruction
> the model would ignore.

**How each line is rendered.** A neutral line clones with `inference_zero_shot`
(the reference transcript keeps its accent); a line with only
`[breath]`/`[laughter]` tokens uses `inference_cross_lingual`; a line with
emotion/volume uses the instruct. CosyVoice 3 runs with its text frontend off,
because the only non-Chinese frontend path is the English normaliser, which
mangles French. The reference transcript comes from the saved voice's manifest
(or `--clone-text`).

For a non-English line the instruct is **not** sent through
`inference_instruct2`, which drops the reference speech tokens and is unstable on
French (dropped or garbled words). Instead the instruction is paired with the
reference transcript in a zero-shot call, so the speech tokens still ground the
voice and accent. The line also gets a closing language clause (e.g.
`请用法语说这句话。`), and the English volume wording is replaced by Chinese
wording — the English clause pulls the output toward English and is sometimes
read aloud. **Pace is applied by time-stretching the audio** rather than as an
instruct clause, which likewise destabilises non-English speech. Give the voice a
transcript (a saved voice has one; `--clone-text` sets one) and pass `--lang` so
the target language is known; without a transcript it falls back to
`inference_instruct2`.

**CosyVoice2** — languages: Chinese, English, Japanese, Korean, Cantonese.
Accepts the instruct sentence plus `[breath]` / `[laughter]`.

**CosyVoice3** — languages: Chinese, English, Japanese, Korean, German, Spanish,
**French**, Italian, Russian. Accepts the instruct sentence plus these inline
tokens: `[breath]`, `[quick_breath]`, `[laughter]`, `[cough]`, `[sigh]`,
`[noise]`, `[lipsmack]`, `[mn]`, `[hissing]`, `[vocalized-noise]`, `[accent]`,
`<strong>`, `</strong>`, and phoneme tokens. The CLI exposes `breath`, `pause`
and `laughter`; the others can be written directly into `--text`.

**Chatterbox** — languages (23): `ar`, `da`, `de`, `el`, `en`, `es`, `fi`, `fr`,
`he`, `hi`, `it`, `ja`, `ko`, `ms`, `nl`, `no`, `pl`, `pt`, `ru`, `sv`, `sw`,
`tr`, `zh`. Its only style control is `exaggeration` (0.0–1.0, default 0.5);
`breath`/`pause`/`laughter` have no equivalent and are dropped. The value is
configurable: `--prosody exaggeration=0.8` sets it per line, and
`--chatterbox-exaggeration 0.8` fixes it for the whole run. `--chatterbox-cfg-weight`
and `--chatterbox-temperature` expose the other generation knobs Chatterbox
accepts (defaults 0.5 and 0.8).

**CustomVoice** (`--backend customvoice`) — a preset Qwen voice plus a free-form
delivery instruction. Pick one of the nine speakers below with `--speaker`, then
describe the delivery in one of three ways. Structured: `--prosody`/`--emotion`
values are rendered into a short sentence for you ("Speak in an excited tone.
Speak quickly."). Verbatim: `--style-instruct` is sent to the checkpoint exactly
as typed, which is the natural-language form it was trained on and the more
expressive option. Generated: `--style-plan llm` asks the configured model to
write those verbatim instructions from the text itself. It does not clone or
design a voice.

```bash
# structured: the spec is turned into an instruction sentence
python dub.py say --backend customvoice --speaker Vivian --lang en \
  --text "Congratulations, you won!" --prosody "emotion=excited, pace=fast"

# verbatim: a full, multi-sentence instruction, passed through unchanged
python dub.py say --backend customvoice --speaker Ryan --lang en \
  --text "I told you not to open that door." \
  --style-instruct 'Speak with restrained anger. Begin quietly, with controlled frustration. Gradually increase the intensity, emphasizing "not", then finish with a disappointed, exhausted tone.'

# generated: the LLM writes the verbatim instruction, one per line
python dub.py say --backend customvoice --speaker Ryan --lang en \
  --text $'I told you not to open that door.\nNow look what happened.' \
  --style-plan llm
```

`--style-instruct` is repeatable — one value per line, or a single value
broadcast to every line. It is valid only with `--backend customvoice`, and it
wins: `--style-plan`, `--prosody`, `--emotion` and `--prosody-plan` are ignored
for that run (the verbatim instruction is already the richer form, so there is
nothing to merge). It is also *not* what `--instruct` means: `--instruct` is an
alias of `--emotion`, and `--style-instruct` has the alias `--raw-instruct`.

> CustomVoice has preset voices only — it cannot sound like *your* speaker, and
> an instruction cannot be combined with a clone. To get a rich instruction in
> your own timbre, create an emotional reference print with the VoiceDesign
> checkpoint and clone from it.

| Speaker | Voice | Native language |
|---|---|---|
| `Vivian` | Bright, slightly edgy young female | Chinese |
| `Serena` | Warm, gentle young female | Chinese |
| `Uncle_Fu` | Seasoned male, low and mellow | Chinese |
| `Dylan` | Youthful Beijing male, clear and natural | Chinese (Beijing dialect) |
| `Eric` | Lively Chengdu male, slightly husky brightness | Chinese (Sichuan dialect) |
| `Ryan` | Dynamic male, strong rhythmic drive | English |
| `Aiden` | Sunny American male, clear midrange | English |
| `Ono_Anna` | Playful Japanese female, light and nimble | Japanese |
| `Sohee` | Warm Korean female, rich emotion | Korean |

Any speaker can speak any of the model's 10 languages, but each is best in its
native language. `--speaker` matching is case-insensitive, and an unknown name is
rejected with the available list. The runtime list comes from the checkpoint
(`get_supported_speakers()`), so it stays correct if the checkpoint changes.

`--emotion`/`--prosody` are accepted on the `customvoice`, `cosyvoice`,
`cosyvoice3` and `chatterbox` backends; `qwen` (the Base clone backend) has no
prosody control and rejects them. `--style-instruct` is `customvoice`-only.

The LLM planner is told the active backend's capabilities, so it only proposes
controls that backend can honour.

### `--prosody-plan llm`

Pass `--prosody-plan llm` to let the configured OpenAI-compatible model decide
per-line prosody from the text itself. The plan is written first; any explicit
`--prosody` or `--emotion` value then overrides it, so a hand-written direction
always wins over the model's guess. This is what produces variation *across*
sentences rather than one delivery for the whole clip.

```bash
python dub.py say --voice warm-fr --backend cosyvoice3 \
  --text $'First line.\nSecond line.\nThird line.' --prosody-plan llm
```

Requires `LLM_BASE_URL` / `LLM_MODEL` (or `.env`), exactly like translation. Pass
`-v` to log the exact request sent to the LLM (system + user messages) and its
raw reply. CosyVoice drops an unknown emotion word with a warning; the planner is
told the backend's real values so it normally does not propose one.

With `--backend customvoice`, `--style-instruct` skips planning altogether: the
verbatim instruction is used as typed, and nothing is sent to the model. To have
the model write that instruction instead, use `--style-plan llm` below; it is the
verbatim counterpart of this flag and does not go through the structured plan.

### `--style-plan llm`

`--style-plan llm` has the configured model write the verbatim delivery
instruction that `--style-instruct` takes by hand. It is the richest option: the
checkpoint reads a natural-language instruction directly, so the model can
describe emotion, energy, pacing, pauses and emphasis in one call.

```bash
# one instruction per line (default)
python dub.py say --backend customvoice --speaker Ryan --lang en \
  --text $'First line.\nSecond line.\nThird line.' --style-plan llm

# one instruction for the whole text
python dub.py say --backend customvoice --speaker Ryan --lang en \
  --text $'First line.\nSecond line.\nThird line.' \
  --style-plan llm --style-plan-scope whole
```

An explicit `--style-instruct` still wins over `--style-plan llm`, and a style
plan wins over `--prosody`/`--emotion`/`--prosody-plan` (a verbatim instruction
cannot be merged with a structured spec).

Above `LLM_PLAN_CHUNK_SIZE` lines (default 10, or `--plan-chunk N`) the planner
does two passes: it first reads the whole text and writes a short overview of the
overall mood and arc, then writes instructions per chunk with that overview and
the chunk's position in hand. That keeps a long passage on one emotional arc
instead of drifting chunk by chunk. `--prosody-plan llm` uses the same strategy.

Requires `LLM_BASE_URL` / `LLM_MODEL` (or `.env`), exactly like translation. Pass
`-v` to log the overview and every chunk request/reply. A line the model leaves
out gets an empty instruction, which the checkpoint renders as its default
delivery.

### Fish Speech inline tags

Fish Speech has no side-channel instruction: emotion and paralinguistics are
**tags in the text**, and Fish documents them as controlling "prosody and
emotion anywhere in the text". One line can therefore turn from sad to angry
mid-sentence, which a single per-clip style cannot express.

`--fish-tag-mode` chooses how those tags reach the model:

| Mode | What it does |
|---|---|
| `render` (default) | Renders the neutral spec as a leading tag, so `--prosody`, `--emotion` and `--prosody-plan llm` work unchanged. |
| `keep` | Sends the tags you typed in `--text` as written. `--emotion`/`--prosody`/`--prosody-plan` are rejected: the text already carries its own direction. |
| `llm` | The configured model rewrites each line with tags, keeping every word. A guard rejects any line whose words changed and falls back to the original. |

```bash
# render: per-line tags from the spec
python dub.py say --voice warm-fr --backend fish \
  --text $'Je n\'arrive pas à y croire !\nAttends… tu as entendu ?' \
  --prosody "emotion=excited" \
  --prosody "emotion=whisper, pause"

# render: the LLM writes the per-line emotion, which becomes the tags
python dub.py say --voice warm-fr --backend fish \
  --text $'Première ligne.\nDeuxième ligne.' --prosody-plan llm

# keep: the tags are already in the text, exactly as Fish documents them
python dub.py say --voice warm-fr --backend fish --fish-tag-mode keep \
  --text $'[sad] Je pensais que ça serait différent… [angry] mais on est là.'

# llm: the model places the tags, including a mid-line turn
python dub.py say --voice warm-fr --backend fish --fish-tag-mode llm \
  --text $'Je pensais que ça serait différent… mais on est là.'
```

Documented tags: `[excited]`, `[angry]`, `[sad]`, `[whisper]`, `[laughing]`,
`[pause]`. In `keep` mode an unknown tag such as `[sarcastic]` is reported in the
log because Fish may speak it aloud, and a `<|...|>` token is rejected outright
since Fish reserves those for its own prompt.

Always quote a `--text` that contains a tag: `[excited]` is a shell glob.
A paragraph pasted without line breaks is split into sentences for planning and
still produces one clip. `--fish-temperature` and `--fish-top-p` expose the
sampler (model defaults 0.7 / 0.7).

### IndexTTS2: emotion reference and duration

IndexTTS2 separates **who** speaks (the `--voice`/`--clone` reference) from
**how** (`--emotion-audio`, a second clip whose delivery is blended in).
`--emotion-alpha` (0–1, default 1.0) sets how strongly it applies.

```bash
python dub.py say --voice warm-fr --backend indextts2 \
  --text "Pourquoi tu ne me l'as pas dit ?" \
  --emotion-audio colere.wav --emotion-alpha 0.8
```

`--duration SECONDS` fits the line to a slot. The released IndexTTS2.0 does
**not** implement native duration control (the project marks it "not yet enabled
in this release"), so the backend time-stretches the generated line with the
pipeline's own stretcher — the same mechanism the dubbing pipeline already uses.
Repeat `--duration` once per line, or pass one value to apply everywhere.

### Isolated Chatterbox environment

Chatterbox pins `torch==2.6` and `transformers==5.2`, which cannot coexist with
the Qwen checkpoint (`transformers==4.57`) or the CosyVoice overlay (4.51.3).
Keep it in its own environment; the backend finds it automatically at
`.venv-chatterbox`, or wherever `CHATTERBOX_PYTHON` points:

```bash
python3 -m venv .venv-chatterbox
.venv-chatterbox/bin/pip install torch==2.6.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
.venv-chatterbox/bin/pip install chatterbox-tts==0.1.7
python dub.py models --download tts_chatterbox   # weights, ~3 GB
```

### Fish Speech environment

Fish pins `torch==2.8` and `transformers==5.19`, so it also lives in its own
environment at `.venv-fish` (or wherever `FISH_PYTHON` points):

```bash
python3 -m venv .venv-fish
git clone --depth 1 https://github.com/fishaudio/fish-speech.git third_party/fish-speech
.venv-fish/bin/pip install -e third_party/fish-speech
python dub.py models --download tts_fish   # fishaudio/s2-pro, ~11 GB
```

Fish's `pyaudio` dependency needs PortAudio headers (`apt install
portaudio19-dev`). Without them, install the runtime dependencies from its
`pyproject.toml` and then `.venv-fish/bin/pip install --no-deps -e
third_party/fish-speech`; the worker never uses audio input.

S2-Pro needs ~24 GB of VRAM. On a 12 GB card it fails with a clear
out-of-memory error; pass `--device cpu` instead (confirmed to work, though a
2 s line takes about 90 s on a 12-core CPU). The worker caps the checkpoint's
32k context at 4096 tokens for line-level work (override with
`FISH_MAX_SEQ_LEN`). The model runs in `pipeline.fish_worker`, so the Qwen stack
is untouched.

### IndexTTS2 environment

IndexTTS pins `torch==2.8` and `transformers==4.52`, so it also lives in its own
environment at `.venv-indextts` (or `INDEXTTS_PYTHON`):

```bash
python3 -m venv .venv-indextts
git clone --depth 1 https://github.com/index-tts/index-tts.git third_party/index-tts
.venv-indextts/bin/pip install -e third_party/index-tts
python dub.py models --download tts_index_tts2   # IndexTeam/IndexTTS-2
```

The download lands the `checkpoints_2` layout, so `config.yaml` is at the root of
`models/IndexTTS2/`.

CosyVoice, Chatterbox, Fish and IndexTTS2 all run in child processes for the same
version-conflict reason. `--save-as` and `--design` are not supported on the
expressive backends or on `customvoice`; pass `--voice`/`--clone`, or `--speaker`
for `customvoice`.

Both `say` paths log to `output/say/<name>/say.log` (or
`output/say/<name>.log` for Qwen). Pass `-v` to include the parsed prosody spec,
the rendered instruct sentence and the worker's per-line timings; the worker
reports checkpoint load separately from generation.

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
