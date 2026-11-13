# Troubleshooting

Symptoms, causes and fixes, plus how the logs work and what the project does not
attempt yet. The rows below name CLI flags; the app exposes the same option in
the form and shows the same message in the job's log.

## The browser app

| Symptom | Cause and fix |
|---|---|
| `dub-server: command not found` | The `web` extra is not installed. Activate the virtualenv, then `pip install -e ".[web]"`. |
| `http://127.0.0.1:8000` answers but the page is blank or 404 | `web/dist` is missing. Build it once with `cd web && npm install && npm run build`, then restart `dub-server`. |
| Edits to `web/src` have no effect | The server serves the built bundle. Re-run `npm run build` and hard-refresh the tab; or run `npm run dev` (port 5173, proxying `/api`). |
| Edits to `server/` or `pipeline/` have no effect | Restart `dub-server`. If the package was installed with `pip install .` instead of `pip install -e ".[web]"`, the server is running the frozen copy in `site-packages`. |
| `Address already in use` on start | Another process owns port 8000. Stop it, or set `DUB_PORT` (and open the new port). |
| The app is unreachable from another machine | It binds `127.0.0.1` on purpose. Set `DUB_HOST=0.0.0.0` only behind authentication — there is none built in. |
| An upload is rejected | Only media types and the size cap the server enforces are accepted; the file must go through the upload field. Paths outside `input/`, `output/` and the data directory are refused by design. |
| A job stays `QUEUED` | One heavy job runs at a time; the queue is bounded and excess work is rejected. Cancel or clear the running job on **Activity**, and check that `DUB_DATA_DIR` is writable. |
| A job fails immediately | Read its log in the activity panel: the message is the same one the CLI prints, so the rows in the next section apply. A `422` means the request was rejected before it was queued (wrong combination of options). |
| Live progress stops updating | The SSE stream dropped and the page is reconnecting with backoff; plain HTTP state and **Cancel** keep working. Refresh to resync. |
| `Could not stream the source file` | The job's input is outside the directories the app may read. Re-upload it through the app. |
| Where are the jobs and uploads stored? | SQLite, uploads and job metadata live in `DUB_DATA_DIR` (default `data/`); outputs in `DUB_OUTPUT_DIR` (default `output/`). To reset the app state, stop the server and remove `data/` — the output folders are separate and are not touched. |
| The app cannot download a model | Same causes as `dub models --download`: disk space, network, or a gated checkpoint (`hf auth login` or `HF_TOKEN`). The job log names the failing checkpoint. |
| A form field is greyed out | The selected backend cannot honour it; the reason is printed under the field. `GET /api/backends` publishes the same table. |
| **Submit** is disabled with a message | The options contradict each other (source equals target, a reference clip without its transcript, a borrowed accent with no clip, accent matching outside native mode). The message names the fix. |

## The command line and the pipeline

| Symptom | Cause and fix |
|---|---|
| `Missing model: 'whisper_large_v3' …` | Run `python dub.py models --download all` |
| `Required binary 'ffmpeg' was not found on PATH` | `sudo apt-get install -y ffmpeg` (audio-only jobs still work) |
| `Could not read audio from … Format not recognised` | The file is a video container. Pass it with `--video`; `--audio` is for audio-only files (or extract the track first) |
| `… contains no audio stream, so there is nothing to transcribe` | The file is video-only. Verify with `ffprobe -v error -show_entries stream=codec_type file.mp4`; mux an audio track in first |
| `CUDA is unusable for speech recognition … libcublas.so.12` | Informational — ASR fell back to CPU. Install `nvidia-cublas-cu12 nvidia-cudnn-cu12` for GPU ASR (Setup §3) |
| `argument --device: Unknown device 'vulkan0' …` | Vulkan is not a backend: ASR runs on CTranslate2 (CUDA or CPU only) and the other stages on PyTorch, which has no Vulkan device. NVIDIA: `--device cuda`; AMD: install the ROCm build of PyTorch, then `--device cuda` or `--device rocm:0`; Intel: `--device xpu`; otherwise `--device cpu`. Run `python dub.py devices` to see every card and the exact flag |
| `dub.py devices` shows `none` on a machine with a GPU | torch is a CPU build, or the accelerator's PyTorch wheel is missing. Check the `torch` line of `dub info`: AMD needs a ROCm wheel, Intel an XPU wheel, NVIDIA a CUDA wheel. Install the matching build and re-run `dub devices` |
| `No speech detected in the input` | No audible speech, or the wrong audio track. Check with `ffprobe` |
| `Translation response was not valid JSON` | Lower `LLM_CHUNK_SIZE` in `.env`, or use a model better at JSON |
| `Translation request to … failed` | The LLM server isn't reachable — check `LLM_BASE_URL` |
| `--ref-audio requires --ref-text` | Add the exact transcript, or drop both flags to auto-pick a donor |
| `--accent-audio only applies to --accent-mode reference-accent` | Add that flag, or remove `--accent-audio` |
| `--accent-mode reference-accent requires --accent-audio` | Add the native-target clip, or name a saved voice with `--accent-voice`, or use another mode |
| `--accent-audio requires --accent-text` | Transcribe the accent clip; the codes cannot be aligned without it. A saved voice carries both, so `--accent-voice` avoids this entirely |
| `--accent-voice cannot be combined with --accent-audio or --accent-text` | Drop one: a saved voice already supplies both the clip and its transcript |
| `--accent-voice implies --accent-mode reference-accent` | Drop the other `--accent-mode`, or drop `--accent-voice` |
| `Voice 'x' has no transcript in …` | Re-save it with `--clone-text`, or pass `--accent-audio` and `--accent-text` |
| `--accent-voice 'x' is a French voice, so it cannot supply a native English accent` | Use a voice in the target language, or `--accent-mode preserve` to keep the speaker's own accent |
| `does not support voice cloning` | Pointed at a CustomVoice/VoiceDesign checkpoint; use `tts_base` |
| `--style-instruct is only valid with --backend customvoice` | That flag is the CustomVoice delivery instruction. Other backends take `--prosody`/`--emotion`, and `--design` already accepts a free-form prompt |
| `--style-plan is only valid with --backend customvoice` | The LLM-written verbatim instruction is CustomVoice-only. Other backends use `--prosody-plan llm` for structured prosody |
| `--style-plan-scope only applies with --style-plan llm` | Add `--style-plan llm`, or drop `--style-plan-scope` |
| `fish-speech is not installed` | Create `.venv-fish` (see [Getting started](getting-started.md#fish-speech-s2-pro-optional-expressive-speech)) or set `FISH_PYTHON`; weights: `python dub.py models --download tts_fish` |
| `the CosyVoice worker is missing the Python module '…'` | A CosyVoice runtime dependency is absent. Install the full list in [Getting started](getting-started.md#cosyvoice-3-optional-expressive-speech). If the module is `pkg_resources`, run `pip install "setuptools<81"` |
| `--backend fish ... has no --prosody/--emotion/--prosody-plan channel` | Fish takes its delivery from `[tag]` cues in `--text`: write them yourself, or pass `--fish-tag-mode llm` and let the model write them |
| `Fish Speech S2-Pro needs about 24 GB of VRAM` | That message means the bf16 weights were requested (`--fish-quantize off`) or the int8 path could not start. Drop the flag to use the int8 default, retry with `--device cpu` (slow), or use a larger GPU |
| `unsupported operation: more than one element of the written-to tensor refers to a single memory location` | torch is older than 2.5 while transformers 4.57 builds the sliding-window attention mask with an in-place write onto an expanded tensor. It aborts Qwen3-TTS synthesis (the `blend`/`rate-probe` stages may still pass). Install `torch==2.5.1` and `torchaudio==2.5.1` as pinned in [Requirements](../requirements.txt); `--backend fish` and the other expressive backends are unaffected |
| `int8 Fish Speech needs torchao` | `.venv-fish/bin/pip install torchao==0.13.0`, or pass `--fish-quantize off` (needs ~24 GB) |
| `fish-speech 3.0.0 requires pyaudio, which is not installed` | Expected after the PortAudio-free install: the worker never opens a microphone. Ignore it, or install `portaudio19-dev` and reinstall fish-speech without `--no-deps` |
| `A Fish emotion seems to have no effect` | Fish reads `[tag]` cues from the text; the checkpoint documents a common set but accepts free-form descriptions too (`[tender]`, `[whisper in small voice]`). If a line has no tag at all, nothing changes its delivery |
| `Fish inline plan changed the words of a line; keeping the original` | The planner rewrote something other than tag placement (a synonym, a dropped word, a translation), so that line is spoken **without tags**. Missing spaces after punctuation (`modestes.Et`) are not a word change and are tolerated. If it keeps happening, shorten the lines or pass `--fish-tag-mode keep` and write the tags yourself |
| `Unsupported language 'xx'` | Use one of the ten supported languages |
| `No saved voice named …` | List the library with `python dub.py voices --list`, or create the voice first |
| `--add needs --lang …` | Say which language the clip is in (`--lang fr`), or pass `--transcribe` and let the ASR detect it |
| `--transcribe and --clone-text are alternatives` | Pass one or the other: `--transcribe` derives the text, `--clone-text` supplies it |
| `No speech was recognised in …` | The ASR heard nothing usable (silence, music, very short clip). Pass `--clone-text` with the exact words instead |
| `No usable qwen card for this model; 1 existing card(s) are kept.` | Informational: the voice was encoded by a different checkpoint (or by an older card format). That card stays, a new one is written beside it, and switching back needs no re-encode |
| A voice encodes on every run | No card matches *this* model and *this* clip + transcript (the log says `No usable qwen card…`), the card is unreadable, or `VOICES_PROMPT_CACHE=0` is set. `Loaded the stored prompt` means it was reused |
| A voice folder has several `prompts/qwen-*.npz` cards | Expected: one per model configuration. Each is ~30 KB and only the one matching the current weights is used |
| A stored card is not used by `--backend fish` | Cards hold one backend's encoding, and are Qwen-only for now. Fish (and CosyVoice3/Chatterbox) always read `voice.wav`, and the log says so |
| A voice added with `--add` does not sound like the speaker | Without a transcript it clones from the speaker embedding alone. Add `--clone-text` (or `--transcribe`) for ICL, and use a clip with only that one speaker, no music and 3–30 s of speech |
| The clone drifts away from the donor's voice | Check the transcript: ICL aligns the speech codes against it, so a paraphrased or corrected text yields a different voice. Run `voices … --verify` to have the ASR compare the two |
| CUDA out of memory | `--device cpu`, or free the GPU (a large local LLM often occupies it). On a multi-GPU box, `--device cuda:1` pins the run to the other card |
| On a long video, repeatedly | See [Memory on a long video](performance.md#memory-on-a-long-video): check `report.json` → `tts.oom_retries` and `tts.ceiling_shrinks`. The pipeline budgets each batch by decoded audio and releases cached blocks when free memory falls below 3 GB; if the counters are still non-zero, lower `--tts-batch`, or use a shorter reference clip — it is charged once per line |
| An unexpected amount of VRAM is in use | `--device cuda:1` now genuinely uses the second GPU; the log line and `report.json` → `device` both report the card that was used |
| `dtype fp16 is not usable on cpu (cpu); using fp32 instead` | Informational, not an error: half-precision on a CPU is slow and patchy, so float32 is used. Drop `--dtype` to let the capability decide |
| The GPU is not being used at all | Run `python dub.py info` — it reports the accelerator, the device it would pick, and the precision. If it says `cpu`, torch either sees no GPU or the wrong wheel is installed (see Setup §2 and §3) |
| Speech recognition is on the CPU when a GPU exists | CTranslate2 links against a specific CUDA major. Run `python dub.py info` and check the `asr device` line; Setup §3 has the fix |
| Dub sounds rushed or slow | Check `report.json` → `clamped_segments`; tighten the translation |
| Some words are cut off | `report.json` → `worst_trimmed_lines` names them (`id`, `trimmed`, `needed_ratio`); `needed_below_floor_segments` counts them. A `needed_ratio` below `compress_floor` means the translation was longer than even a 0.75 compression can hold, so the wording, not the tempo, is what to change: check `prefit` (did the pre-synthesis check run, and how much wording did it keep?) and `--length-budget`. Lowering `--min-stretch` further buys room at the cost of sounding hurried |
| A line lost a lot of words despite a low `trimmed_seconds_total` | Look at `worst_line_deleted_ratio` and `lines_over_10pct_deleted`: damage is usually concentrated in one or two lines, not spread evenly |
| Lines sound slow or have gaps | `report.json` → `deficit_seconds_total`: the translation is shorter than its slot, i.e. the word ceiling was conservative. Check `delivery_wps` (the probe's measurement) against `speech_rate_wps` (what the voice actually delivered): if the probe reads well under the delivery, raise `--budget-speedup`. See also [Reference pacing](voices.md#reference-pacing) |
| A rewrite did not help | Check `report.json` → `refine` (the measured pass) and → `prefit` (the pre-synthesis one). `attempted: false` means no line overflowed, or the translator cannot rewrite; `improved: 0` means the rewrites did not fit either, and the pre-fit check was skipped or its candidates were not shorter |
| `delivery_wps` looks wrong for my video | It is measured from one 40-word probe render per voice, so a very expressive or very short-spoken video can differ from `speech_rate_wps` (what the lines actually delivered). Delete `<job>/03a_rate.json` to re-probe, or set `--budget-speedup`/`--min-stretch` yourself and leave the budget out of it with `--length-budget off` |
| Dub drifts out of sync | Up to `--max-overrun` seconds per line is deliberate: it keeps the last word of a line that would otherwise lose it. Check `shifted_seconds_max` — if it is over the limit, the source speech genuinely overlaps itself, and `--max-overrun 0` restores exact sync at the cost of the line tails |
| Dub sounds quiet | Check `loudness_delta_vs_source`; `0` means it matches the original. Force a level with `--loudness lufs --lufs -16` |
| Music/ambience is missing | Install `demucs` (`pip install demucs`); without it the background cannot be separated and is discarded |
| Music fights the voice | Raise `--duck-db` (default 6 dB) |
| `Missing model: 'tts_voice_design' …` | The default `native` mode needs it to generate the accent anchor. Run `python dub.py models --download tts_voice_design` (or `--download all`) |
| `--accent-match only applies to --accent-mode native` | Use `--accent-mode native` (the default), or drop `--accent-match` |
| The dub still has the speaker's accent | The default `native` mode already borrows a generated anchor. If it is not enough, try `--accent-match gender`, then `--accent-mode speaker-only`, or supply your own clip with `--accent-mode reference-accent` |
| Still sounds foreign in `reference-accent` mode | Check `--accent-text` matches the clip word for word — mismatched text misaligns the borrowed codes. A longer, more natural accent clip also helps |
| The voice changes part-way through the video | Only possible in `reference-accent` with `--accent-blend single-stage`, where every line is free to follow the accent codes or the speaker embedding. The default `two-stage` removes that choice — each voice's lines are cloned from that voice's own `reference_blend[_<speaker>].wav`, so the voice cannot change between lines. Listen to that clip on its own to confirm stage 1 picked up the speaker's timbre; if it did not, raise `--blend-takes` or try a different `--accent-audio`. To measure it rather than judge it, see [Auditing a dub's voice](outputs-and-caching.md#auditing-the-voice-line-by-line) |
| A voice changes *between speakers* rather than mid-line | Expected, unless the two voices are the same person: each speaker is cloned from their own donor. If the count is wrong, see the speaker-threshold rows above |
| The borrowed accent dominates — the speaker's own voice is barely there | The blend was built too close to the accent voice. Check `report.json` → `donor.blend.donor_ratio`: a low value means it never got far enough to the speaker. Raise `--blend-iters`, or lower `--blend-ratio` if you want more of the accent |
| The blend reached a donor ratio of only … | Informational: the target was not met within `--blend-iters` passes, and the best pass was used anyway. Raise `--blend-iters` or lower `--blend-ratio` |
| `--accent-blend only applies to --accent-mode reference-accent` | Add that flag, or drop `--accent-blend` |
| `--blend-takes only applies to --accent-mode reference-accent` | Add that flag, or drop `--blend-takes` |
| `--blend-iters only applies to --accent-mode reference-accent` | Add that flag, or drop `--blend-iters` |
| `--blend-ratio only applies to --accent-mode reference-accent` | Add that flag, or drop `--blend-ratio` |
| Every line is dubbed in one voice, but the video has several speakers | Diarization is off or unavailable. Check the log for `Diarization (none)` — it means `--diarizer none` was set, or `funasr` is not installed. Install it (`pip install -r requirements-dev.txt`), then re-run. The detection happens before synthesis, so a cached job needs `--force` to pick up a newly installed backend |
| One person is being split into two voices | Raise `--speaker-threshold` (e.g. `0.6`); a distance-based threshold is genuinely material-dependent. `02b_speakers.json` shows how the segments were labelled. (Overlap detection is supervised by those labels but does not use this threshold.) |
| A line sounds like the wrong person, or you can hear two people on it | `report.json` → `overlap` counts segments that hold a second voice; `02c_overlap.json` lists their ids. By default such a line is dubbed in its dominant speaker's voice — run with `--overlap separate` (needs the optional `speechbrain`) to split it and dub both voices. Detection is deliberately conservative (a 2.5 s run that fits another speaker better), so short interruptions are not reported |
| Two people are merged into one voice | Lower `--speaker-threshold` (e.g. `0.3`), and check `--min-speaker-seconds` is not folding the quieter speaker away. `--diarizer none` confirms the rest of the pipeline is unaffected |
| A quiet speaker's lines are in the wrong voice | Their cluster was probably shorter than `--min-speaker-seconds` and got folded into the nearest voice. Lower it, at the cost of cloning from a shorter window |
| `funasr is not installed, so every line will be dubbed in one voice` | Informational fallback, not an error: the job still runs. Install `funasr` for multi-speaker dubbing, or pass `--diarizer none` to make the intent explicit |
| The dub is rushed, or words are missing | Check `accelerated_segments` and `needed_ratio_min`: the voice is being sped up because the written line is longer than the window holds. Raise `--budget-speedup` only if `deficit_seconds_total` is also high; otherwise the lines are simply long for their slots. See [When a translation is too long for its slot](dub-quality.md#when-a-translation-is-too-long-for-its-slot) |
| `audit_qc.py` reports `SKIP timbre — no accent clip to measure against` | Expected for a `preserve` job: nothing was borrowed, so "did the line follow the accent voice?" has no answer. Pass `--accent <clip>` to force one. The other three gates still report, and the exit code stays 0 |
| The timbre audit fails while the GPUs are busy | Pass `--device cpu` to `audit_qc.py` or `audit_timbre.py`. The measurement is slower there but identical |

Logs are written to stderr and include a job id, and are mirrored to
`<job folder>/run.log`. That file is **appended** to, never replaced: one input
is normally dubbed many times while its options are tuned, so every run keeps its
lines and opens with a `Run started` line to mark the boundary between them. Once
it passes `--log-max-bytes` (2 MiB by default) it is renamed to
`run-<YYYYmmdd-HHMMSS>.log` and a fresh `run.log` begins — at a run boundary, so a
single run is never split. `--log-file PATH` puts the copy somewhere else,
`--no-log-file` writes none, and `LOG_FILE` sets the default path. Set
`LOG_JSON=1` for one JSON object per line, on the terminal and in the file alike.

`dub.py say` accepts the same logging flags. Its log defaults to
`output/say/<name>/say.log` for the expressive backends and
`output/say/<name>.log` for Qwen, and records the backend, the resolved model, the
prosody plan, per-line render times and a total.

**Reading the timings.** A stage line looks like
`Stage synthesize: done in 461.2s`. With `-v`, per-batch and per-task lines add
their own `in 3.42s`, and the expressive backends report the checkpoint load
separately from generation (`worker load 10.35s, generate 8.16s`). In JSON mode
every timed record carries `seconds`, stage records also carry `stage`, and a
completed dub repeats the same numbers in `report.json` → `stage_seconds`.
Third-party HTTP and model loggers stay at WARNING even under `-v`, so the debug
output is about this pipeline rather than its dependencies.

**CosyVoice sounds like it has an English accent, or words are added/garbled.**
A neutral line now uses `inference_cross_lingual` when the donor speaks another
language than the target and `--accent-mode` is not `preserve`. That drops the
reference speech codes, so the model uses its own target-language phonetics
instead of cloning the donor's accent. Lines with an emotion/pace/volume
instruction still use the zero-shot path: prosody on a non-English line pairs the
instruction with the reference transcript, adds a closing language clause
(`请用法语说这句话。`), uses Chinese volume wording instead of the English clause,
and applies pace by time-stretching rather than as an instruction — each of those
otherwise destabilises French. Those instructed lines may keep some source
accent. Give the voice a transcript (a saved voice has one; `--clone-text` sets
one) and pass `--lang`; without a transcript the backend falls back to
`inference_instruct2`, which is less stable. For a fully native target accent on
every line, use `--expressive-accent blend` with `--accent-mode native` or
`reference-accent`.

**Chatterbox or Fish still has the source accent.**
Both clone a single reference clip. Without `--expressive-accent blend` that clip
is the source-language donor, so the source accent comes with it. Fish has no
language parameter at all; Chatterbox receives `language_id` but still conditions
on the reference speech. Use `--expressive-accent blend` with
`--accent-mode native` or `reference-accent` to build a target-language reference
with Qwen Base first, or pass `--accent-mode preserve` to acknowledge the
fallback. `-v` logs which reference each backend used.

**A prosody value seems to have no effect.** CosyVoice only has trained emotion
instructs for `happy`, `sad` and `angry`; `calm` deliberately adds no clause, and
`excited` is mapped onto happy. Chatterbox has a single `exaggeration` scalar, so
`emotion` only changes intensity, and `breath`/`pause`/`laughter` are dropped.
CustomVoice renders `emotion`/`pace`/`volume` into one instruction sentence and
drops `breath`/`pause`/`laughter`; pass `--style-instruct` to direct the delivery
yourself, or `--style-plan llm` to have the model write that instruction.
Pass `-v` to see the mode each CosyVoice line used (`zero_shot`,
`cross_lingual`, `instruct2`), the exact instruction sent, and — with
`--prosody-plan llm` or `--style-plan llm` — every LLM request and reply
(including the whole-passage overview and each chunk).

---

## Notes and limitations

- **Voice cloning requires consent.** Clone only voices you have permission to
  use. Voice is biometric data under GDPR, and synthetic-dubbing disclosure is
  increasingly required by platforms. A consent gate is planned for P6.
- **Licence**: this project is Apache-2.0 (see `LICENSE`).
- **Hardware portability is deliberate, and tested without the hardware.** The
  dtype follows the GPU's compute capability rather than its name (bfloat16 is
  unsupported on Pascal and Volta, and choosing it unconditionally was the most
  likely "works for me, not for you" failure). `--device cuda:N` is honoured
  exactly, so a second GPU can be pinned. Metal and XPU are usable but never
  chosen automatically, because speech recognition runs on CTranslate2. The rules
  are covered by `tests/test_device.py` against fake hardware profiles, so a
  regression is caught on a CPU-only CI runner.
- **Model licences**: Qwen3-TTS and Whisper weights are Apache-2.0 / MIT;
  CosyVoice3 is Apache-2.0; Chatterbox is MIT; **Fish S2-Pro is
  non-commercial**. The default path does not use Fish. See
  [Model licenses](model-licenses.md).
- **Python**: 3.10–3.12. `numpy==1.26.4` has no 3.13 wheels, so installation on
  3.13 fails until the pin is updated.
- **Multi-speaker dubbing attributes speakers per segment**, not per word, so
  overlapping speech is shown as a flag on the segment. By default such a line is
  dubbed in the voice of its dominant speaker and reported (`report.json` →
  `overlap`, `02c_overlap.json`); `--overlap separate` splits the two voices apart
  with SepFormer, re-transcribes each and dubs both. The speaker count comes from
  clustering rather than being declared. Both are tracked in `PLAN.md`.
- **Known gaps**, all tracked in `PLAN.md`: lip sync, multi-language source
  videos, and iterative length control in the translation loop. Background
  preservation, native-accent borrowing and multi-speaker dubbing are
  implemented (see [Accent modes](voices.md#accent-modes) and
  [Dubbing a video with several speakers](dub-quality.md#dubbing-a-video-with-several-speakers)).
- **Native accent is generated, not downloaded.** The default `native` mode
  synthesises a short anchor per language with the VoiceDesign checkpoint, cached
  under `models/anchors/`, so no voice is shipped per language and
  `reference-accent` with `--accent-audio` remains for choosing one yourself.
  `--accent-match` fits the anchor's rate (and optionally gender) to the source
  voice. There is no native-accent-free default path: to keep the source accent,
  opt in with `--accent-mode preserve`.
- **Room matching is off by default and adds no reverb unless asked.**
  `--room match` adds a room-tone floor and a gentler duck, both derived from the
  separated speech stem. A dry, close voice-over is the correct choice for narrated
  content, which is why it stays the default. The reverb tail is a separate, opt-in
  step (`--room-wet-db -20`): an earlier default of −12 dB was rejected on a real
  listen as an audible echo. `--no-background` removes the stem, so room matching
  then degrades to a dry dub and says so (`report.json` → `room.applied: false`).
- **Changing any setting re-runs the whole job** — the cache fingerprint is
  per-job, not per-stage. Fine for a CLI; worth refining before P6.

---

Back to the [docs index](README.md), or the [project README](../README.md).
