# Options in full

Every flag, its default, and what it achieves. `python dub.py --help` prints the
same list at the terminal.

## Input and language

Exactly one of `--video` / `--audio` is required.

| Option | Default | What it achieves |
|---|---|---|
| `--video PATH` | – | Video input. Enables muxing, so you get `video_dubbed.mp4`. Needs `ffmpeg`. |
| `--audio PATH` | – | Audio-only input. Produces `dub.wav` + subtitles, needs no `ffmpeg`. |
| `--target LANG` | `fr` | Language to dub into. Accepts a code (`fr`) or a name (`French`). |
| `--source LANG` | auto | Overrides the detected source language. Useful when a clip is too short or too noisy for reliable detection. |

## Voice

See [Accent modes](voices.md#accent-modes) for how these interact.

| Option | Default | What it achieves |
|---|---|---|
| `--ref-audio PATH` | auto | Clones this recording instead of the speaker found in the video. |
| `--ref-text TEXT` | – | Exact transcript of `--ref-audio`. Required with it in `preserve` mode; ignored by the other modes. |
| `--voice LABEL` | – | Reuse a saved voice as the timbre source, including its stored encoding when it has one. Cannot be combined with `--ref-audio`; see [Voice library](voices.md#voice-library-saved-voices). |
| `--accent-mode MODE` | `native` | Where pronunciation comes from: `native` (a generated accent anchor — the default), `preserve` (the reference voice), `speaker-only` (the model's own phonetics), `reference-accent` (the `--accent-audio` clip, or `--accent-voice`). |
| `--accent-match MODE` | `off` | With `--accent-mode native`, shape the generated anchor to the source voice: `rate` matches its speaking rate, `gender` also matches its gender. |
| `--accent-audio PATH` | – | Native target-language clip to borrow pronunciation from. Requires `--accent-mode reference-accent`. |
| `--accent-text TEXT` | – | Exact transcript of `--accent-audio`. Required with it. |
| `--accent-voice LABEL` | – | Take both the accent clip and its exact transcript from a saved voice, so neither `--accent-audio` nor `--accent-text` is needed. Implies `--accent-mode reference-accent`, and cannot be combined with either of them. |
| `--accent-blend MODE` | `two-stage` | `two-stage` renders the accent transcript once in the speaker's voice and clones every line from that clip, so its codes and its embedding cannot disagree. `single-stage` conditions each line on both sources directly. |
| `--blend-takes N` | `3` | How many times the one-off blend is rendered before keeping the take closest to the speaker. Only that clip is sampled this way, not the lines. |
| `--blend-iters N` | `5` | Maximum passes used to build the blend. Each pass re-reads the previous clip and re-injects the speaker's embedding, pulling the voice back toward the speaker. `1` disables the iteration. See [Choosing the blend strength](voices.md#choosing-the-blend-strength). |
| `--blend-ratio T` | `0.85` | How far toward the speaker's own voice the blend should reach, from `0` (the accent voice) to `1` (a target-language utterance in the speaker's voice). Iteration stops there. `0` disables the search. See [Choosing the blend strength](voices.md#choosing-the-blend-strength). |

With `--voice`, a saved voice already in the target language clones directly
(`preserve`), while one in another language uses the `native` anchor. A voice
with no transcript cannot do ICL at all, so it uses `speaker-only`. An explicit
`--accent-mode` always wins.

## Emotion (dubbing backend and prosody plan)

By default the dub clones each speaker with the Qwen Base checkpoint: the voice
is kept, but the model has no emotion control. Pick an expressive `--backend`
and `--prosody-plan` lets the configured model decide how each translated line
should be delivered.

| Option | Default | What it achieves |
|---|---|---|
| `--backend NAME` | `qwen` | `qwen` clones with the Base checkpoint (no prosody). `cosyvoice3` adds instruct emotion plus `[breath]`/`[laughter]` tokens. `chatterbox` adds a single expression intensity. `fish` reads inline `[tag]` cues. All four clone each speaker from a clip of their own voice. |
| `--prosody-plan {off,llm,auto}` | `auto` | `auto` asks the LLM for per-line delivery when it is reachable and falls back to neutral otherwise; `llm` fails if it is unreachable; `off` speaks neutral. Ignored by `qwen`. |
| `--chatterbox-exaggeration 0..1` | derived | `chatterbox` only: fixed expression intensity for every line, overriding the planned emotion. |
| `--chatterbox-cfg-weight FLOAT` | `0.5` | `chatterbox` guidance weight. |
| `--chatterbox-temperature FLOAT` | `0.8` | `chatterbox` sampling temperature. |
| `--fish-temperature FLOAT` | `0.7` | `fish` sampling temperature. |
| `--fish-top-p FLOAT` | `0.7` | `fish` nucleus sampling probability. |
| `--fish-quantize {int8,off}` | `int8` | `fish` weight format: `int8` fits a 12 GB card, `off` loads the bf16 weights. |

Fish S2-Pro weights are non-commercial; see [Model licenses](model-licenses.md).
The default `qwen` path does not use Fish.

The plan is written from the **translated** lines; a long video is chunked with
a whole-passage overview so the delivery keeps one arc. Fish has no side
channel, so its plan is the text itself: tags are written into each line, guarded
so the words cannot change.

Accent: the expressive backends clone from the speaker's donor clip, so the Qwen
accent machinery (generated anchors, two-stage blend, rate probe) does not run.
`cosyvoice3` keeps the reference accent on its zero-shot path; instruct-based
emotion can pull the delivery toward the target language. `chatterbox` and
`fish` keep the reference's timbre and accent.

`qwen` has no prosody channel: with the default `auto` the plan is skipped
silently, while an explicit `--prosody-plan llm` logs a warning and is ignored.

## Saved voices (`dub.py voices`)

| Option | Default | What it achieves |
|---|---|---|
| `--design` | mode | Create a voice from `--prompt` (VoiceDesign checkpoint). |
| `--clone` | mode | Encode `--audio` once and store the encoding (Base checkpoint). |
| `--add AUDIO` | mode | Register an existing clip verbatim: no model, no GPU, no processing. |
| `--list` | mode | List the library (the default). |
| `--remove LABEL` | mode | Delete a saved voice. |
| `--clone-text TEXT` | – | Exact transcript of the reference clip. Required with `--clone` unless `--transcribe` is given; optional with `--add`. |
| `--transcribe` | off | Derive the transcript with the local ASR, and use the language it detects when `--lang` is omitted. |
| `--verify` | off | Transcribe anyway and warn when `--clone-text` does not match what was heard. |
| `--lang LANG` | see note | Language the voice is recorded in. `--design`/`--clone` default to `fr`; `--add` needs it, or `--transcribe`. |
| `--label NAME` | derived | Name to save under; a slug of the description, or the audio's filename. |
| `--force` | off | Overwrite an existing voice with the same label. Cards already written are kept, and one that no longer matches the new clip is ignored rather than used. |

Encodings are stored per backend and per model, as
`voices/<label>/prompts/<backend>-<fingerprint>.npz`. A card is only used by the
backend and checkpoint that wrote it, and switching models keeps both cards.

## Speakers

See [Dubbing a video with several speakers](multi-speaker.md#dubbing-a-video-with-several-speakers).

| Option | Default | What it achieves |
|---|---|---|
| `--diarizer BACKEND` | `auto` | `auto` finds the speakers with the embedding model; `none` dubs every line in one voice, which is the pre-P1 behaviour. Without `funasr` installed, `auto` warns and behaves as `none`. |
| `--speaker-threshold D` | `0.7` | Cosine distance above which two embeddings count as different people. Raise it if one speaker is being split in two; lower it if two speakers are being merged. |
| `--min-speaker-seconds S` | `20.0` | A voice must own at least this much speech to be dubbed separately; a shorter cluster joins its nearest neighbour. The floor is well above the 3 s needed to clone at all, because a donor of a second or two does not describe a voice — its blend stalls near the accent (measured at `donor_ratio` 0.23 on the 11-minute job, where the dub audibly kept the anchor's voice). `0` keeps every cluster. |
| `--overlap MODE` | `dominant` | What to do with a segment that carries two voices at once. `dominant` dubs the whole line in its dominant speaker's voice; `separate` splits it and dubs both voices (needs the optional `speechbrain`). See [Dubbing a video with several speakers](multi-speaker.md#dubbing-a-video-with-several-speakers). |
| `--overlap-separator BACKEND` | `auto` | `auto` uses SpeechBrain's SepFormer when installed, `sepformer` requires it, `none` disables it. Only used with `--overlap separate`. |

## Audio bed and loudness

| Option | Default | What it achieves |
|---|---|---|
| `--no-background` | off | Discards the original music/ambience. By default it is separated out and ducked under the dub instead. |
| `--separator BACKEND` | `auto` | `auto`, `demucs` (split speech from the bed), or `none` (treat the source as a single track). |
| `--duck-db DB` | `6.0` | How far the background drops while dialogue plays. Higher = the voice sits more forward. |
| `--room MODE` | `off` | `match` places the dub in the source's own acoustic space — a reverb tail, a continuous room-tone floor, and a shallower duck. `off` keeps the dry voice-over, which is the right choice for narrated content. Needs the background to be kept (the default). |
| `--room-rt60 SECONDS` | measured | Reverberation time to use instead of measuring one from the speech stem. An escape hatch for a source whose decay cannot be estimated. |
| `--room-wet-db DB` | `0` | Reverb level relative to the dry voice. **`0` means no reverb** — the tail is opt-in, because it can only be judged by ear and the first default was rejected on a real listen as an audible echo. Try `-20` to enable a subtle tail. |
| `--room-duck-db DB` | `3.0` | How far the bed drops under dialogue when `--room match`. Shallower than `--duck-db`, because the room is part of the voice. |
| `--room-tone-db DB` | `-30.0` | Level of the room-tone floor lifted from the speech stem's pauses. |
| `--loudness POLICY` | `match-source` | `match-source` puts the dub at the original track's measured level; `lufs` forces `--lufs` instead. |
| `--lufs VALUE` | `-16` | Absolute loudness target in LUFS. Only used with `--loudness lufs`. |

## Timing

These trade speech clarity against staying in sync.

| Option | Default | What it achieves |
|---|---|---|
| `--min-stretch RATIO` | `0.75` | Compression floor: how much faster than natural a line may be played to fit its slot. It is a clarity limit rather than the point where words are cut — past it the wording has to give instead, because the alternative is deleting speech. Lower = faster speech, and it can start to sound hurried. |
| `--max-stretch RATIO` | `1.18` | Most a line may be slowed to fill its slot. Raising this makes speech sound unnatural, so prefer fixing the translation. |
| `--max-overrun SECONDS` | `0.25` | How far a line may run past its window — into the next line's time — before the excess is cut. The pause after the line is spent first (free); this covers the rest, because a fraction of a second of lateness is a smaller defect than a deleted word. `0` restores exact sync and cuts the tails instead. |
| `--stretcher BACKEND` | `auto` | `auto`, `rubberband`, `sox`, `librosa`, `none`. `none` disables stretching (lines then only overrun or trim). Falls back automatically if a binary is missing. |

## Length repair

Fixes a long translation by wording rather than by cutting audio.

| Option | Default | What it achieves |
|---|---|---|
| `--refine` / `--no-refine` | on | Rewrites the lines that would otherwise have words cut, then re-synthesises just those. Lines that already fit are untouched; a rewrite is kept only if it measurably reduces the overflow. `--no-refine` restores the previous trim-only behaviour. |
| `--refine-max-passes N` | `1` | How many times a line may be rewritten before falling back to trimming. `0` disables the repair. |
| `--refine-threshold SECONDS` | `0.05` | How far a line must overrun its slot *and* the pause after it before it is worth rewriting. Raise it to repair fewer, worse lines. |
| `--length-budget MODE` | `words` | `words` (the default) also tells the translator how many words fit each line, measured from the voice that will speak it — one probe render per speaker, before translating. `off` sends only the duration, which measurably loses the ends of sentences. Falls back to the accent clip's own rate when there is no target-language text to probe with. |
| `--budget-speedup RATIO` | `0.85` | Tempo the word budget is willing to spend on *every* line: the ceiling is how many words fit at that speed-up. Keep it above `--min-stretch` so most lines stay at or above 0.85 and only the stubborn ones reach the floor. |

## Models and hardware

| Option | Default | What it achieves |
|---|---|---|
| `--device DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N`, `mps`, or `xpu`. `cuda:1` pins a specific GPU. `cpu` is much slower but proves the GPU is not the problem. See [Devices and precision](performance.md#devices-and-precision) |
| `--devices DEV[,DEV...]` | unset | Spreads the blend and the synthesis over several devices, one worker process each: `--devices cuda:0,cuda:1`. A worker takes the next piece of work as it frees up, so the split is by work rather than by model. The first device carries the stages that cannot be split (ASR, diarization, separation, anchors). Repeating a name is allowed and is how the fan-out is tested on one machine. Unset leaves `--device` as the whole answer. See [Using more than one card](performance.md#using-more-than-one-card) |
| `--dtype DTYPE` | `auto` | Precision for the voice-cloning checkpoint: `auto`, `bf16`, `fp16`, or `fp32`. `auto` reads the device's capability. An explicit value is honoured unless the device cannot run it |
| `--whisper-model NAME` | `large-v3` | ASR model id. A smaller one is faster and less accurate — the main lever on a CPU-only machine. |
| `--llm-model NAME` | from `.env` | Overrides `LLM_MODEL` for this run. |
| `--tts-batch N` | `8` | Most lines synthesised in one model call. The engine additionally budgets a batch by the audio that will be **decoded**, which includes the clone reference once per line — that term dominates, so this is the ceiling and the escape hatch. Lower it if a long video still runs out of GPU memory; see [Memory on a long video](performance.md#memory-on-a-long-video). |
| `MODELS_DIR` (env) | `models` | Where weights live. Set in `.env`; the download command also takes `--models-dir`. |

### Environment variables

Real environment variables win over `.env`. `dub info` shows the resolved paths
and optional packages.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible endpoint used for translation and optional planning |
| `LLM_API_KEY` | `not-needed` | API key for that endpoint |
| `LLM_MODEL` | `local-model` | Model name sent to the endpoint |
| `LLM_TIMEOUT_S` | `180` | Per-request timeout |
| `LLM_CHUNK_SIZE` | `40` | Lines per translation request |
| `LLM_PLAN_CHUNK_SIZE` | `10` | Lines per prosody/style planning request |
| `LOUDNESS_TARGET` | `match-source` | `match-source` or `lufs` |
| `LOUDNESS_LUFS` | `-16` | Absolute target when `LOUDNESS_TARGET=lufs` |
| `KEEP_BACKGROUND` | `true` | Keep and duck the original music/ambience |
| `BACKGROUND_SEPARATOR` | `auto` | `auto`, `demucs`, or `none` |
| `DUCK_DB` | `6.0` | Background attenuation under dialogue |
| `MODELS_DIR` | `models` | Weight directory |
| `VOICES_DIR` | `voices` | Saved voice library |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_JSON` | `0` | `1` emits one JSON object per line |
| `LOG_FILE` | unset | Also write the log here; CLI default is the job folder |
| `CHATTERBOX_PYTHON` | `.venv-chatterbox/bin/python` | Interpreter for the Chatterbox worker |
| `FISH_PYTHON` | `.venv-fish/bin/python` | Interpreter for the Fish worker |
| `FISH_MAX_SEQ_LEN` | `4096` | Context cap for Fish line-level work |
| `COSYVOICE_PYTHONPATH` | unset | Extra paths prepended to the CosyVoice child process |
| `HF_TOKEN` | unset | Hugging Face token for gated model downloads (read by `huggingface_hub`) |
| `HF_HUB_OFFLINE` | unset | Set to `1` to prevent network access to the Hub |

## Output and caching

| Option | Default | What it achieves |
|---|---|---|
| `--output-dir DIR` | `output` | Where job folders are written. Each input gets its own `<stem>` folder. |
| `--subtitles-mode MODE` | `soft` | How the target subtitles reach the video: `soft` muxes a toggleable track and copies every other stream (other subtitle languages are kept), `burn` draws them into the frames and re-encodes the video, `off` writes the SRT files only. |
| `--no-subtitles` | off | Skips both SRT files and implies `--subtitles-mode off`. |
| `--force` | off | Ignores the stage cache and rebuilds everything. |
| `--no-provenance` | off | Skips the timestamped provenance artifact (`prosody.json` + `provenance/`) that records the per-line prosody, the LLM reply and the text transformations. |
| `--log-file PATH` | `<job folder>/run.log` | Where the run log is also written, alongside stderr. Created with its parent directory, appended to across runs (each one opens with a `Run started` line, so they stay separable), and stamped with the time so a line can be matched against `report.json` → `stage_seconds`. `LOG_FILE` sets the same default, which is why the log never invalidates a cached stage. |
| `--log-max-bytes N` | `2097152` | Size at which the log is rotated. Once the file reaches it, it is renamed to `run-<YYYYmmdd-HHMMSS>.log` and a fresh one begins — at a run boundary, so a run is never split across two files. `0` keeps a single file forever. |
| `--no-log-file` | off | Writes no log file at all — the escape hatch when a caller wants stderr alone. |
| `-v` / `--verbose` | off | Debug logging, including the device each stage used, per-batch/per-task timings and worker detail. Third-party HTTP/model loggers stay at WARNING so the output is about this pipeline. |

`dub.py say` accepts the same `--log-file`, `--no-log-file`, `--log-max-bytes` and
`-v` flags. Its default log is `output/say/<name>/say.log` for the expressive
backends and `output/say/<name>.log` for Qwen; each run records the backend, the
model, the prosody plan, per-line render times and a total.

### `dub.py say` options

`say` takes `--text` and exactly one source: a saved `--voice`, an inline
`--design`, an inline `--clone`, or — with `--backend customvoice` — a preset
`--speaker`.

| Option | Default | What it achieves |
|---|---|---|
| `--backend NAME` | `qwen` | `qwen` (clone), `customvoice` (preset speaker + instruction), `cosyvoice3` (clone + per-line prosody), `chatterbox` (clone + expression intensity), `fish` (clone + inline tags). |
| `--prosody SPEC` | — | Per-line `emotion`/`pace`/`volume`/`exaggeration` and bare `breath`/`pause`/`laughter` flags; repeat per line or broadcast one. |
| `--emotion TEXT` | — | Shorthand for `emotion=` (alias `--instruct`); repeatable/comma-separated. Rejected by `qwen` and `fish` (Fish takes its direction from tags in `--text`). |
| `--style-instruct TEXT` | — | `customvoice` only: the delivery instruction, sent **verbatim** to the checkpoint. Repeatable (one per line, or one value broadcast). Wins over `--style-plan`/`--prosody`/`--emotion`/`--prosody-plan`. |
| `--style-plan {off,llm}` | `off` | `customvoice` only: `llm` writes those verbatim instructions for you, which is the form the checkpoint expects. An explicit `--style-instruct` still wins. Requires the LLM settings. `-v` logs every request and reply. |
| `--style-plan-scope {per-line,whole}` | `per-line` | With `--style-plan llm`: one instruction per line (chunked, with a whole-passage overview when long), or a single instruction for the entire text. |
| `--plan-chunk N` | `LLM_PLAN_CHUNK_SIZE` (10) | Lines per planning request before `--style-plan llm`, `--prosody-plan llm` and `--fish-tag-mode llm` switch to the overview + chunked strategy. |
| `--prosody-plan {off,llm}` | `off` | `llm` asks the configured OpenAI-compatible model to plan per-line prosody; explicit values still win. Skipped when `--style-instruct` or `--style-plan llm` is given. Long passages get a whole-passage overview first. `-v` logs the raw request and reply. |
| `--speaker NAME` | — | Preset voice for `--backend customvoice` (e.g. `Vivian`, `Ryan`). |
| `--chatterbox-exaggeration 0..1` | derive | Fixes Chatterbox's expression intensity for every line. |
| `--chatterbox-cfg-weight FLOAT` | `0.5` | Chatterbox guidance weight. |
| `--chatterbox-temperature FLOAT` | `0.8` | Chatterbox sampling temperature. |
| `--fish-temperature FLOAT` | `0.7` | Fish Speech sampling temperature. |
| `--fish-top-p FLOAT` | `0.7` | Fish Speech nucleus sampling probability. |
| `--fish-tag-mode {keep,llm}` | `keep` | `fish` only. `keep` speaks `--text` exactly as written, tags included; `llm` has the model insert the tags, guarded so the words cannot change. Fish has no `--prosody`/`--emotion`/`--prosody-plan` channel. |
| `--fish-quantize {int8,off}` | `int8` | `fish` only. `int8` loads 8-bit weight-only (~6 GB resident, ~11 GB peak) so the model fits a 12 GB card; `off` loads the bf16 weights Fish documents as needing ~24 GB. CPU runs always use bf16. |
| `--clone-text TEXT` | — | Transcript of `--clone`. Optional: without it `qwen` clones from the speaker embedding alone (lower similarity), and CosyVoice loses the reference's accent. |
| `--no-provenance` | off | Skips the timestamped `provenance/provenance-<timestamp>.json` record of the per-line prosody, the LLM reply and the text transformations. |

`--text` is split into one utterance per line. Both a real newline and a literal
`\n` are treated as breaks, so `--text "one\ntwo"` works even though a shell does
not expand `\n` inside double quotes; `$'one\ntwo'` also works.

## Subtitles only (`dub.py subtitles`)

Subtitles are independent of dubbing: this command runs ASR and translation but
never loads a TTS model. It accepts the same `--video`/`--audio`, `--target`,
`--source`, `--output-dir`, `--device` and `--whisper-model` flags, plus:

| Option | Default | What it achieves |
|---|---|---|
| `--subtitles-mode MODE` | `soft` | `soft` muxes a toggleable track; `burn` renders the text into the frames (re-encodes); `off` writes the SRT only. |
| `--max-chars N` | `42` | Longest a cue's text may be before it is split. |
| `--force` | off | Ignores the cached transcript and recognises again. |

When the target language equals the detected source language, translation is
skipped. Outputs live in `<output>/<stem>-subtitles/`.

---

Next: [Outputs and caching](outputs-and-caching.md), or
[Troubleshooting](troubleshooting.md).
