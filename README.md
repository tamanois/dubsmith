# AI Video Dubbing

Dub a video into another language while keeping each speaker's **own voice timbre**.
Runs entirely on local, open models; no cloud required.

Take a video, detect the language of the speakers, identify each voice, and
produce a new video where the speakers speak a target language in their own
voice — with a native accent when possible.

- [Vision and roadmap](PLAN.md) · [Documentation](docs/README.md) · [Engineering guidelines](.agents/technical-guidelines.md)

---

## What it does

| Capability | Detail |
|---|---|
| **Input** | One video, or one audio file |
| **Speakers** | One or several: detected automatically, each cloned from their own voice (`--diarizer none` forces one) |
| **Languages** | Source auto-detected; target: English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian |
| **Voice** | Cloned from the source speaker, with a generated native accent by default. Saved voice prints can be designed or cloned once and reused by name |
| **Output** | Dubbed video/audio + source and translated SRT subtitles. The target subtitles are embedded in the video as a switchable track by default (other languages kept), or burned in with `--subtitles-mode burn`. Original music/ambience preserved |
| **Not included** | Lip sync (P4), multi-language source videos (P5) |

---

## Quick start

From a fresh clone to a first dub.

**Requirements**

- Python **3.10+**
- `ffmpeg` and `ffprobe` for video input (audio-only needs neither)
- ~10 GB of disk for the model weights
- A GPU is optional but strongly recommended (~12 GB VRAM for the default path)
- An **OpenAI-compatible** LLM endpoint for translation

**1. Clone and install**

```bash
git clone <repository-url> qwen-tts
cd qwen-tts
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

Optional extras (each degrades gracefully when absent):

```bash
pip install ".[full]"     # Demucs background separation, funasr diarization, SpeechBrain overlap
```

**2. Point the translator at a model**

```bash
cp .env.example .env
```

```dotenv
LLM_BASE_URL=http://127.0.0.1:8080/v1
LLM_API_KEY=not-needed
LLM_MODEL=local-model
```

Any OpenAI-compatible server works, including a local `llama.cpp`:
`llama-server -m /path/to/model.gguf --port 8080`.

**3. Download the models (~10 GB, once)**

```bash
python dub.py models --download all
```

**4. Check the machine**

```bash
python dub.py info          # device, precision, optional packages, models
```

**5. Dub**

```bash
python dub.py --video talk.mp4 --target fr
```

The result is `output/talk/video_dubbed.mp4`, with `dub.wav`, subtitles and
`report.json` beside it. A first run also generates the French accent anchor,
cached under `models/anchors/`.

For the full setup matrix — CPU-only installs, CUDA 12 libraries for GPU ASR,
optional models, `MODELS_DIR` — see [docs/setup.md](docs/setup.md).

---

## Main commands

**Dub a video or audio file**

```bash
python dub.py --video talk.mp4 --target fr          # dubbed video + embedded soft subtitles
python dub.py --audio interview.wav --target es     # audio only, no ffmpeg
python dub.py --video talk.mp4 --source en --target de
python dub.py --video talk.mp4 --target fr --subtitles-mode burn   # burn text into frames
```

**Add subtitles without dubbing**

```bash
python dub.py subtitles --video talk.mp4 --target fr
python dub.py subtitles --video talk.mp4 --target fr --subtitles-mode burn
python dub.py subtitles --audio interview.wav --target es            # SRT only
```

A subtitles-only run recognises speech, translates it, writes `subs_target.srt`,
and embeds it in the video. Existing subtitle tracks in other languages are kept;
a track already in the target language is replaced. No TTS model is loaded.

**Choose the voice**

```bash
# clone a specific recording (the transcript must match the audio exactly)
python dub.py --video talk.mp4 --target fr --ref-audio me.wav --ref-text "…"

# keep the speaker's timbre, borrow a native accent
python dub.py --video talk.mp4 --target fr \
  --accent-mode reference-accent --accent-audio native.mp3 --accent-text "…"

# keep the source accent
python dub.py --video talk.mp4 --target fr --accent-mode preserve
```

**Create and reuse a saved voice**

```bash
python dub.py voices --design --prompt "a warm female French voice" --lang fr --label warm-fr
python dub.py voices --clone --audio me.wav --clone-text "…" --lang fr --label me
python dub.py voices --list

python dub.py say --voice warm-fr --text "Bonjour, ceci est un test."
python dub.py say --voice warm-fr --backend cosyvoice --text "…" --prosody "emotion=excited"
python dub.py say --voice warm-fr --backend cosyvoice3 --text "…" --prosody "emotion=happy, pace=fast, breath"
python dub.py say --voice warm-fr --backend chatterbox --text "…" --prosody "emotion=happy, exaggeration=0.8" --lang fr
python dub.py say --backend customvoice --speaker Vivian --lang en --text "…" --prosody "emotion=excited, pace=fast"
python dub.py say --backend customvoice --speaker Ryan --lang en --text "…" \
  --style-instruct "Speak with restrained anger, then fade to exhaustion."
python dub.py say --backend customvoice --speaker Ryan --lang en --text "…" \
  --style-plan llm
python dub.py say --voice warm-fr --backend fish --text "…" --emotion excited
python dub.py say --voice warm-fr --backend indextts2 --text "…" \
  --emotion-audio colere.wav --emotion-alpha 0.8 --duration 3.2
python dub.py --video talk.mp4 --target en --voice warm-fr
```

Expressive backends (`cosyvoice`, `cosyvoice3`, `chatterbox`, `fish` and
`indextts2` clone the voice from the same reference clip; `customvoice` pairs a
preset speaker with an instruction):

| Backend | Languages | What it adds |
|---|---|---|
| `customvoice` | 10 | preset Qwen speaker + tone/emotion instruction (no clone) |
| `cosyvoice` (CosyVoice2) | zh, en, ja, ko, yue | per-line emotion/style instruct |
| `cosyvoice3` | 9, **incl. French** | emotion, pace, volume, `[breath]`, `[laughter]` |
| `chatterbox` | 23, **incl. French** | configurable expression intensity (+ natural prosody) |
| `fish` (Fish Speech S2-Pro) | ~100 | cloned voice + inline `[excited]`/`[whisper]`/`[sad]` tags |
| `indextts2` | zh, en, ja | cloned voice + a separate emotion reference + `--duration` fit |

CustomVoice speakers: `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric` (Chinese),
`Ryan`, `Aiden` (English), `Ono_Anna` (Japanese), `Sohee` (Korean) — descriptions
and native languages in
[docs/voices.md](docs/voices.md#expressive-backends-cosyvoice3-and-chatterbox).

`--prosody` is the structured per-line control (repeat once per line, or pass one
value to apply everywhere):

| Field | Accepted values |
|---|---|
| `emotion` | free-form; CosyVoice uses the trained `happy`/`sad`/`angry` (aliases map) |
| `pace` | `normal`, `slow`, `fast` |
| `volume` | `normal`, `soft`, `loud` |
| `exaggeration` | `0.0`–`1.0` (Chatterbox's own intensity knob) |
| `breath`, `pause`, `laughter` | bare flags |

`--prosody-plan llm` lets the configured OpenAI-compatible model choose those
values from the text; an explicit `--prosody`/`--emotion` then overrides it.
`--style-instruct TEXT` (`customvoice` only) sends a full delivery instruction
verbatim instead, which is the natural-language form that checkpoint expects and
the most expressive option; it wins over `--prosody`. `--style-plan llm`
(`customvoice` only) has the model write that instruction for you, one per line
(`--style-plan-scope whole` writes a single one for the whole text); above
`LLM_PLAN_CHUNK_SIZE` lines (default 10, or `--plan-chunk`) it first reads the
whole text for an overview, then plans each chunk against it. Note that
`--instruct` is an alias of `--emotion`, not of `--style-instruct`.
Per-backend limitations (Chatterbox has no breath/laughter tokens; CosyVoice2
languages) and the full value lists are in [docs/voices.md](docs/voices.md).

**Multi-speaker videos**

```bash
python dub.py --video debate.mp4 --target fr            # one voice per speaker
python dub.py --video debate.mp4 --target fr --overlap separate
```

**Inspect and audit a run**

```bash
cat "output/talk/report.json"                  # the quickest way to judge a run
python audit_qc.py "output/talk"               # pass/fail gates, no model
python audit_qc.py "output/talk" --timbre      # adds the per-line timbre gate
```

**Models and the environment**

```bash
python dub.py models --list
python dub.py info
```

Every flag and default is in [docs/options.md](docs/options.md);
`python dub.py --help` prints the same list.

---

## Tests

The suite needs no GPU, downloaded models or ffmpeg — heavy components are faked.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

```text
1270 passed
```

Run one area while iterating:

```bash
python -m pytest tests/test_tts.py -q
python -m pytest tests/test_pipeline.py -q
python -m pytest tests/test_voices.py -q
python -m pytest -k accent -q
```

The quality tools and their gates are covered in
[docs/testing-and-quality.md](docs/testing-and-quality.md).

---

## Documentation

| Guide | What it covers |
|---|---|
| [Setup](docs/setup.md) | prerequisites, install, LLM endpoint, models |
| [Usage](docs/usage.md) | command chooser, worked examples, fast iteration |
| [Options](docs/options.md) | every flag and its default |
| [Voices](docs/voices.md) | saved voice prints, cloning, accent modes |
| [Multi-speaker](docs/multi-speaker.md) | diarization, speaker thresholds, overlap |
| [Audio and timing](docs/audio-and-timing.md) | music/ambience, room matching, long translations |
| [Performance](docs/performance.md) | devices, precision, several GPUs, long-video memory |
| [Outputs and caching](docs/outputs-and-caching.md) | job artifacts, `report.json`, the stage cache |
| [Testing and quality](docs/testing-and-quality.md) | tests, timbre audit, `audit_qc.py` |
| [Architecture](docs/architecture.md) | pipeline, module map, project layout |
| [Troubleshooting](docs/troubleshooting.md) | errors, logs, notes and limitations |

The [docs index](docs/README.md) has the same list with reading paths.

---

## Project layout

```text
dub.py                  CLI entry point (dub, models, voices, say, info)
audit_qc.py             automated quality gate for a finished job
audit_timbre.py         per-line speaker-timbre audit
pipeline/               ASR, translation, TTS, timing, mixing, orchestration
tests/                  pytest suite (no GPU or models needed)
models/                 downloaded weights (gitignored)
voices/                 saved voice prints (gitignored)
output/                 job artifacts (gitignored)
docs/                   user documentation
PLAN.md                 vision, roadmap and design decisions
```

Full module map: [docs/architecture.md](docs/architecture.md).

---

## License and responsible use

- **License**: Apache-2.0; see [LICENSE](LICENSE).
- **Consent**: clone only voices you have permission to use. Voice is biometric
  data under GDPR, and synthetic-dubbing disclosure is increasingly required.
- **Model licenses**: Qwen3-TTS and Whisper weights are Apache-2.0 / MIT. Avoid
  CPML-licensed models (e.g. XTTS-v2) if you need commercial rights.
- **Known gaps** are tracked in [PLAN.md](PLAN.md): lip sync, multi-language
  source videos, and iterative length control in the translation loop.
