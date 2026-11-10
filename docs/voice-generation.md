# Voice generation and emotion

`dub say` generates speech without translating a video. It uses the same voice
library, expressive backends, and provenance records as a dub, so a voice you
create here can be reused with `--voice` in a dubbing run.

See the [before/after demo](media/README.md) for a dubbing example, then continue
with the deep reference in [Voices](voices.md), [Options → `say`](options.md), and
[Setup → optional backends](setup.md).

## Quick start

```bash
# Create a voice from a prompt (VoiceDesign)
python dub.py voices --design \
  --prompt "a warm female French voice, slow and clear" \
  --lang fr --label warm-fr

# Clone from a clip and its exact transcript (Base)
python dub.py voices --clone --audio me.wav --clone-text "…" --lang fr --label me

# Register a clip without loading a model or using a GPU
python dub.py voices --add me.wav --transcribe --label me

# Reuse it
python dub.py say --voice warm-fr --text "Bonjour, ceci est un test."
```

`say` needs no LLM endpoint unless you use `--prosody-plan llm` or
`--style-plan llm`.

## Backends

| Backend | Where | Languages | Controls | Weights license |
|---|---|---|---|---|
| `qwen` (default) | dub + say | 10 | voice clone, accent modes | Apache-2.0 |
| `customvoice` | say only | 10 | preset speaker + instruction | Apache-2.0 |
| `cosyvoice3` | dub + say | 9, incl. French | emotion, pace, volume, `[breath]`, `[laughter]` | Apache-2.0 |
| `chatterbox` | dub + say | 23, incl. French | expression intensity | MIT |
| `fish` (S2-Pro) | dub + say | ~100 | inline `[tags]` in the text | **non-commercial** |

The default path does not require Fish. See
[Model licenses](model-licenses.md) before commercial use.

## Manual emotion

`--prosody` is repeatable: one value per line, or one value broadcast to every
line. It accepts `emotion=`, `pace=`, `volume=`, `exaggeration=`, and the bare
`breath`, `pause`, and `laughter` flags.

```bash
python dub.py say --voice warm-fr --backend cosyvoice3 \
  --text "Je suis tellement contente !" \
  --prosody "emotion=happy, pace=fast, breath"

python dub.py say --voice warm-fr --backend chatterbox \
  --text "Nous avons gagné !" \
  --emotion happy --prosody exaggeration=0.8
```

`--emotion` is shorthand for `emotion=` (alias `--instruct`). It is rejected by
`qwen` and `fish`; Fish takes direction from tags in the text.

## Planned emotion

```bash
python dub.py say --voice warm-fr --backend cosyvoice3 \
  --text $'Première ligne.\nDeuxième ligne, plus tendue.' \
  --prosody-plan llm
```

For `say`, `--prosody-plan` defaults to `off`; `llm` asks the configured
OpenAI-compatible model for one spec per line. Explicit values still win. Long
passages get a whole-passage overview before chunk planning.

For **dubbing**, `--prosody-plan auto` is the default: the planner runs when the
LLM is reachable and falls back to neutral otherwise. `off` speaks neutral;
`llm` fails if the endpoint is unreachable.

## CustomVoice instructions

`customvoice` pairs a preset speaker with a free-form delivery instruction and
does not clone a voice. It is `say` only.

```bash
python dub.py say --backend customvoice --speaker Ryan --lang en \
  --text "I told you not to open that door." \
  --style-instruct "Speak with restrained anger, then fade to exhaustion."

python dub.py say --backend customvoice --speaker Ryan --lang en \
  --text $'First line.\nSecond line.' \
  --style-plan llm
```

`--style-instruct` is sent verbatim and wins over `--prosody`. `--style-plan llm`
writes the instruction for you; `--style-plan-scope whole` writes one instruction
for the entire text.

## Fish inline tags

Fish has no side channel. Put tags in the text (`--fish-tag-mode keep`, the
default), or let the model insert them (`--fish-tag-mode llm`). The planner is
guarded so words cannot change.

```bash
python dub.py say --voice warm-fr --backend fish \
  --text "[sad] Je pensais que ça serait différent… [angry] mais on est là."
```

## Voice cards and provenance

A voice directory contains:

```text
voices/<label>/voice.wav                         # the reference clip
voices/<label>/voice.json                        # label, language, transcript, source
voices/<label>/prompts/<backend>-<hash>.npz      # cached encode for one backend/model
voices/<label>/provenance/provenance-<time>.json # how the voice was made
```

A card is only used by the backend and model configuration that wrote it, so
switching checkpoints keeps both cards. Qwen writes and consumes cards; the
expressive backends currently read `voice.wav` directly. Deleting a card costs
one encode, not the voice.

For a `say` run, the log is `output/say/<name>/say.log` (or
`output/say/<name>.log` for Qwen). `-v` includes the planned spec and the
worker's per-line timings, and may include the text itself; treat it as private.
