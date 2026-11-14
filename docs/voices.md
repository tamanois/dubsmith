# Voices

The saved voice library, voice cards, standalone speech with `dub say`, the
expressive backends, per-line emotion, and the four accent modes with the native
blend. A voice made here is reusable: `say` speaks it, and `--voice` dubs with
it.

`dub say` needs no video and no translation, and no LLM endpoint unless you use
`--prosody-plan llm` or `--style-plan llm`.

## Voice library (saved voices)

A **voice** is a reference clip plus its transcript, saved under a label. A cloned
voice also keeps the checkpoint's *encoding* of that clip — the speech codes and
the speaker embedding — so using it later encodes nothing at all. Create one
once, then reuse it by name in `say`, or in a dubbing run with `--voice`. Three
ways to make one:

```bash
# 1. design a voice from a description (VoiceDesign checkpoint)
python dub.py voices --design \
  --prompt "a warm female French voice, slow and clear" \
  --lang fr --label warm-fr

# 2. encode a donated clip once (Base checkpoint) — nothing is re-synthesised
python dub.py voices --clone --audio me.wav \
  --clone-text "Bonjour, je m'appelle Camille." \
  --lang fr --label me

# 3. register a clip verbatim: no model, no GPU, no processing
python dub.py voices --add me.wav \
  --clone-text "Bonjour, je m'appelle Camille." --lang fr --label me
python dub.py voices --add me.wav --transcribe --label me  # text from the audio

python dub.py voices --list                 # the library
python dub.py voices --remove warm-fr       # delete one
```

| Mode | Stores | Reusing it costs |
|---|---|---|
| `--design` | a rendered passage in the described voice | one encode of that passage |
| `--clone` | the donor clip, its transcript, **and the encoding** | nothing |
| `--add` | the donor clip and an optional transcript | one encode, then cached |

`--add` is the fast path: it decodes the file, downmixes it to mono and writes it
into the library — no checkpoint is loaded. `--transcribe` derives the transcript
with the local Whisper model and detects the language, so `--lang` becomes
optional; without a transcript the voice clones from the speaker embedding alone
(x-vector mode), which is less faithful but needs no text.

`--verify` transcribes the clip even when `--clone-text` is given, and warns when
the two disagree. That check exists because ICL cloning aligns the speech codes
*against the transcript*: a mismatched transcript produces the wrong voice rather
than an error.

The design prompt is a natural-language instruction, so it can also describe
tone and delivery — "an angry male French voice, speaking fast" works, because
Qwen VoiceDesign is conditioned on timbre, emotion and prosody together. This is
the only backend where a voice is created from words; CosyVoice and Chatterbox
clone from audio and cannot design.

Voices live in `voices/` by default (override with `VOICES_DIR` or
`--voices-dir`), one folder per voice:

```
voices/delphine/voice.wav                   # the reference clip (the donor, as given)
voices/delphine/voice.json                  # label, language, source, transcript, gender, pace
voices/delphine/prompts/qwen-<fingerprint>.npz   # Qwen's encoding: codes + embedding
voices/delphine/provenance/provenance-<timestamp>.json   # how the voice was made
```

`provenance/` records each save: the mode, the design prompt (or the reference
clip and where its transcript came from), the model and fingerprint, and the
measured gender/pace. It is timestamped per run, so `--force` keeps the history.

`voice.wav` is the voice, and it works with every backend. A **card** under
`prompts/` is derived data: one checkpoint's encoding of that clip, which lets it
skip the encode pass. Cards are named after the backend and the model
configuration that wrote them, because they are not portable — Qwen's speech
codes mean nothing to Fish, and a card from one checkpoint is not a card for
another.

That naming is also why a card is never thrown away. Upgrading or re-downloading
the weights writes a new card beside the old one, and switching back reuses the
first one without re-encoding anything. Everything you could want from a card is
inside it: the backend, the model, a hash of the model's configuration, and a
hash of the clip + transcript it was derived from. A card whose clip or text no
longer matches is ignored (it would be a different voice), and an unreadable or
deleted card costs one encode, not the voice.

The first Qwen `say` on a voice without a card writes one, so a voice encodes at
most once per Qwen model (`VOICES_PROMPT_CACHE=0` turns that off). The
expressive backends currently read `voice.wav` directly. `voices --list` shows
the card state: `clone + qwen card`, or `×2` when the voice has cards for two
models.

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

With `--save-as`, a cloned voice is stored as the donor plus its encoding, so the
next use of that label starts from the encoding rather than re-encoding the clip.

`--clone-text` is optional everywhere. Without it the voice is cloned from the
speaker embedding alone, which keeps the timbre but loses the closer ICL match;
`say` logs a warning when it takes that path. Saved voices follow the same rule:
a voice with a transcript clones in ICL mode, one without falls back to the
embedding.

The saved voice supplies the **timbre**. If it was recorded in the target
language, it is cloned directly: its own speech codes already carry the native
accent, so there is no blend to build. If it was recorded in another language,
pronunciation comes from a generated native anchor for the target language
(`--accent-mode native`) instead; there the stored embedding saves the timbre
half of the hybrid prompt. A voice with no transcript cannot do ICL at all, so a
dub uses `--accent-mode speaker-only` for it unless you ask for something else.
Pass `--accent-mode` explicitly to override any of these choices — for example
`preserve` to always keep the voice's own accent.

Defaults: `dub.py say` writes `output/say/<name>.wav`; a label is derived from
the description or audio filename unless `--label`/`--save-as` names it; an
existing label is never overwritten without `--force`. Overwriting a voice keeps
its path, so re-run a dub with `--force` to rebuild stages that used the old
clip.

A donor that carries music, noise or a second speaker keeps all of it: the
encoding is faithfully the donor's. Trim or separate the clip before registering
it if you want a cleaner voice — `--clone` no longer renders a canonical passage
first, which is what used to hide that.

Measured on one line from the same donor, with the checkpoint's own speaker
encoder: 0.977 similarity for a voice cloned from a stored encoding, against
0.975 for the older approach that re-rendered the voice into a passage and then
encoded that. A single line is not a benchmark; the structural difference is that
the stored encoding *is* the donor's embedding, so no generation sits between the
donor and the prompt.

## Expressive backends: CosyVoice3, Chatterbox and Fish

These backends target the same problem — speech that sounds *alive* rather than
read — and all clone the voice from a reference clip.

| Backend | Languages | Controls | Setup |
|---|---|---|---|
| `--backend cosyvoice3` | 9, including **French** | emotion, pace, volume, `[breath]`, `[laughter]` | `python dub.py models --download tts_cosyvoice3`; [Getting started](getting-started.md#cosyvoice-3-optional-expressive-speech) |
| `--backend chatterbox` | 23, including **French** | emotion exaggeration (one intensity) | `python dub.py models --download tts_chatterbox`; [Getting started](getting-started.md#chatterbox-optional-expressive-speech) |
| `--backend fish` | ~100 | delivery as **inline tags** in `--text` (34 documented tags + free-form) | `python dub.py models --download tts_fish`; [Getting started](getting-started.md#fish-speech-s2-pro-optional-expressive-speech) |

```bash
# CosyVoice3, with real per-line direction
python dub.py say --voice warm-fr --backend cosyvoice3 \
  --text $'Bonjour.\nAttends, quoi ?' \
  --prosody "emotion=calm, volume=soft" \
  --prosody "emotion=excited, pace=fast, breath"

# Chatterbox: emotion maps onto its exaggeration knob
python dub.py say --voice warm-fr --backend chatterbox \
  --text "Nous avons gagné !" --emotion happy

# Fish Speech: the direction is a tag in the text
python dub.py say --voice warm-fr --backend fish --fish-tag-mode keep \
  --text "[excited] Je n'arrive pas à y croire !"
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

| Field | Accepted values | CosyVoice 3 | Chatterbox | CustomVoice |
|---|---|---|---|---|
| `emotion` | free-form | trained set only: `happy`, `sad`, `angry`; aliases such as `excited`→happy, `calm`→none | intensity only: neutral 0.5, calm 0.4, sad 0.3, happy 0.8, excited 0.9, angry 0.85 | free-form tone, e.g. "Speak in an angry tone." |
| `pace` | `normal`, `slow`, `fast` | trained speed instructs | no pace control; ±0.05 intensity | phrased as "Speak slowly/quickly." |
| `volume` | `normal`, `soft`, `loud` | trained volume instructs | no volume control; ±0.10 intensity | phrased as "Speak softly/loudly." |
| `exaggeration` | `0.0`–`1.0` | ignored | exact value, overrides the emotion mapping | ignored |
| `breath` | flag | `[breath]` at sentence boundaries | ignored | ignored |
| `pause` | flag | a comma inserted in a long, unpunctuated line | ignored | ignored |
| `laughter` | flag | `[laughter]` appended to the line | ignored | ignored |

**Fish Speech** has no prosody channel at all: its delivery lives in the text as
`[tag]` cues, so `--prosody`, `--emotion` and `--prosody-plan` are rejected for
this backend. Write the tags yourself (`--fish-tag-mode keep`, the default) or
let the model write them (`--fish-tag-mode llm`). See
[Fish Speech inline tags](#fish-speech-inline-tags) for the vocabulary.

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
per-line prosody from the text itself. For `say` the default is `off`; for
dubbing it is `auto`, which plans when the endpoint is reachable and falls back
to neutral otherwise (`llm` fails instead). The plan is written first; any
explicit `--prosody` or `--emotion` value then overrides it, so a hand-written
direction always wins over the model's guess. This is what produces variation
*across* sentences rather than one delivery for the whole clip.

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

Fish Speech has no side-channel instruction: delivery is written as **tags inside
the text**. That is the checkpoint's native interface, so this backend exposes
only tags - `--prosody`, `--emotion` and `--prosody-plan` are rejected.

`--fish-tag-mode` chooses where the tags come from:

| Mode | What it does |
|---|---|
| `keep` (default) | Sends `--text` exactly as written, tags included. |
| `llm` | The configured model inserts the tags, keeping every word; a line whose words changed falls back to the original. |

```bash
# keep: you write the tags
python dub.py say --voice warm-fr --backend fish \
  --text $'[sad] Je pensais que ça serait différent… [angry] mais on est là.'

# keep with a free-form description
python dub.py say --voice warm-fr --backend fish \
  --text $'[whisper in small voice] Tu n\'es pas censé savoir ça.'

# llm: the model places the tags
python dub.py say --voice warm-fr --backend fish --fish-tag-mode llm \
  --text $'Je pensais que ça serait différent… mais on est là.'
```

The model card documents **free-form descriptions** ("15,000+ unique tags") plus
this common set: `[pause]` `[emphasis]` `[laughing]` `[inhale]` `[chuckle]`
`[tsk]` `[singing]` `[excited]` `[laughing tone]` `[interrupting]` `[chuckling]`
`[excited tone]` `[volume up]` `[echo]` `[angry]` `[low volume]` `[sigh]`
`[low voice]` `[whisper]` `[screaming]` `[shouting]` `[loud]` `[surprised]`
`[short pause]` `[exhale]` `[delight]` `[panting]` `[audience laughter]`
`[with strong accent]` `[volume down]` `[clearing throat]` `[sad]` `[moaning]`
`[shocked]`. Anything else short is passed through as its own tag (`[tender]`,
`[nervous]`), which is why an emotion without a documented tag still changes the
delivery.

Always quote a `--text` that contains a tag: `[excited]` is a shell glob. A
`<|...|>` token is rejected outright, since Fish reserves those for its own
prompt. A paragraph pasted without line breaks is split into sentences for
planning and still produces one clip. `--fish-temperature` and `--fish-top-p`
expose the sampler (model defaults 0.7 / 0.7).

### Optional backend environments

Chatterbox and Fish pin conflicting `torch`/`transformers` versions, so each runs
in its own child environment. Installation steps are in
[Getting started → Chatterbox](getting-started.md#chatterbox-optional-expressive-speech) and
[Getting started → Fish](getting-started.md#fish-speech-s2-pro-optional-expressive-speech).

CosyVoice, Chatterbox and Fish all run in child processes for the same
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
| `reference-accent` | **`--accent-audio`** or **`--accent-voice`** | the reference voice, or the auto donor | You want to choose the borrowed accent yourself |

`native` asks for nothing: the anchor is synthesised by the VoiceDesign checkpoint
from a short native-language passage (see
[Automatic native accent](#automatic-native-accent)). The other three modes remain
available and unchanged. `speaker-only` is the cheap option — no extra clip, but
the model chooses the delivery — and `reference-accent` costs you one clip and its
transcript, buying an accent you pick by hand.

### Expressive backends and accent modes

The modes above are implemented by the Qwen prompt builder. CosyVoice3,
Chatterbox and Fish clone from a single reference clip, so `native`,
`reference-accent` and `speaker-only` cannot be applied by the backend itself. By
default (`--expressive-accent donor`) they clone the source-language donor and
keep its accent; a run logs a warning when the requested mode will not reach the
backend. `--accent-mode preserve` is the explicit way to ask for that.

To keep the speaker's timbre and get a native target accent, use
`--expressive-accent blend` with `--accent-mode native` or `reference-accent`. The
pipeline builds the same two-stage blend the Qwen path uses (Qwen Base plus the
generated or supplied accent clip) and hands that target-language WAV to the
expressive backend as its reference. This loads Qwen Base for the one-off blend,
then frees it before the expressive checkpoint loads. It requires
`--accent-blend two-stage` (the default); `single-stage` leaves no clip to clone.

CosyVoice3 has a cheaper default for neutral lines: when the donor speaks another
language and the mode is not `preserve`, it uses `inference_cross_lingual`, which
drops the reference speech codes so the model uses its own target-language
phonetics. Lines with an emotion/pace/volume instruction keep the zero-shot path
so their prosody stays grounded, and may keep some source accent. Fish has no
language parameter, so `--expressive-accent blend` is the only way to give it a
native target accent while preserving the speaker's timbre.

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

If the accent clip is already a saved voice, one flag replaces both: the voice's
manifest stores its transcript, so `--accent-voice` supplies the clip and the text
together and implies `--accent-mode reference-accent`:

```bash
python dub.py --video talk.mp4 --target fr \
  --accent-mode reference-accent \
  --accent-voice deep-fr
```

The voice must speak the target language, because the anchor is what supplies the
pronunciation; a voice in another language is rejected rather than silently
borrowing the wrong accent. `--accent-voice` is independent of `--voice`, so you
can take the timbre from one saved voice and the accent from another.
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
[When a translation is too long for its slot](dub-quality.md#when-a-translation-is-too-long-for-its-slot).

Need a transcript and never wrote one down? Transcribing the clip with the same
model the pipeline uses is accurate enough in practice — the run behind the table
above used a machine transcript of the accent clip. Just avoid paraphrasing: the
text has to match the audio it describes, so correct any obviously normalised
numbers or names by hand.

---

Next: [Dubbing quality](dub-quality.md), or [Performance](performance.md).
