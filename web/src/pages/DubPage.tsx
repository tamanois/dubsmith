import React from "react";
import { api } from "../api";
import { backendOptions, capabilitiesOf, useBackends } from "../capabilities";
import type { Device, Job, VoiceInfo } from "../types";
import {
  ACCENT_BLEND_OPTIONS,
  ACCENT_MATCH,
  ACCENT_MATCH_OPTIONS,
  ACCENT_MODE,
  ACCENT_MODE_OPTIONS,
  DEFAULTS,
  DEVICE_OPTIONS,
  DIARIZER_OPTIONS,
  DTYPE_OPTIONS,
  EXPRESSIVE_ACCENT,
  EXPRESSIVE_ACCENT_OPTIONS,
  FISH_QUANTIZE_OPTIONS,
  LANGUAGE_OPTIONS,
  LENGTH_BUDGET_OPTIONS,
  LOUDNESS,
  LOUDNESS_OPTIONS,
  OVERLAP_MODE_OPTIONS,
  OVERLAP_SEPARATOR_OPTIONS,
  PROSODY_PLAN,
  PROSODY_PLAN_OPTIONS,
  ROOM,
  ROOM_OPTIONS,
  SEPARATOR_OPTIONS,
  SUBTITLES_MODE,
  SUBTITLES_MODE_OPTIONS,
  TTS_BACKEND,
} from "../options";
import {
  Badge,
  Button,
  Card,
  Disclosure,
  ErrorNote,
  Field,
  Input,
  Section,
  Select,
  Spinner,
  Toggle,
  num,
} from "../ui";
import { UploadField } from "../components/UploadField";
import { PresetPanel } from "../components/PresetPanel";
import { RecentInputs } from "../components/RecentInputs";

const VIDEO_EXT = [".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"];

type CapabilityFlag =
  | "prosody"
  | "expressive"
  | "fish_tuning"
  | "chatterbox_tuning";

function text(value: unknown, fallback = ""): string {
  return value === undefined || value === null ? fallback : String(value);
}

function flag(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

export function DubPage({
  onJobStarted,
  initialJob,
}: {
  onJobStarted: (id: string) => void;
  initialJob?: Job | null;
}) {
  const [mode, setMode] = React.useState<"dub" | "subtitles">("dub");
  const [inputPath, setInputPath] = React.useState("");
  const [inputName, setInputName] = React.useState("");
  const [target, setTarget] = React.useState<string>(DEFAULTS.language);
  const [source, setSource] = React.useState("");
  const [voice, setVoice] = React.useState("");
  const [accentVoice, setAccentVoice] = React.useState("");
  const [refAudio, setRefAudio] = React.useState("");
  const [refText, setRefText] = React.useState("");
  const [accentAudio, setAccentAudio] = React.useState("");
  const [accentText, setAccentText] = React.useState("");
  const [accentMode, setAccentMode] = React.useState<string>(DEFAULTS.accentMode);
  const [accentMatch, setAccentMatch] = React.useState<string>(DEFAULTS.accentMatch);
  const [backend, setBackend] = React.useState<string>(DEFAULTS.ttsBackend);
  const [prosodyPlan, setProsodyPlan] = React.useState<string>(DEFAULTS.prosodyPlan);
  const [device, setDevice] = React.useState<string>(DEFAULTS.device);
  const [dtype, setDtype] = React.useState<string>(DEFAULTS.dtype);
  const [devices, setDevices] = React.useState("");
  const [ttsBatch, setTtsBatch] = React.useState("8");
  const [whisperModel, setWhisperModel] = React.useState<string>(
    DEFAULTS.whisperModel,
  );
  const [llmModel, setLlmModel] = React.useState("");
  const [subtitlesMode, setSubtitlesMode] = React.useState<string>(
    DEFAULTS.subtitlesMode,
  );
  const [keepSubtitles, setKeepSubtitles] = React.useState(true);
  const [keepBackground, setKeepBackground] = React.useState(true);
  const [separator, setSeparator] = React.useState<string>(DEFAULTS.separator);
  const [duckDb, setDuckDb] = React.useState("6");
  const [backgroundGain, setBackgroundGain] = React.useState("0");
  const [loudness, setLoudness] = React.useState<string>(DEFAULTS.loudness);
  const [lufs, setLufs] = React.useState("-16");
  const [room, setRoom] = React.useState<string>(DEFAULTS.room);
  const [roomRt60, setRoomRt60] = React.useState("");
  const [roomWetDb, setRoomWetDb] = React.useState("0");
  const [roomDuckDb, setRoomDuckDb] = React.useState("3");
  const [roomToneDb, setRoomToneDb] = React.useState("-30");
  const [minStretch, setMinStretch] = React.useState("0.75");
  const [maxStretch, setMaxStretch] = React.useState("1.18");
  const [maxOverrun, setMaxOverrun] = React.useState("0.25");
  const [refine, setRefine] = React.useState(true);
  const [refineMaxPasses, setRefineMaxPasses] = React.useState("1");
  const [refineThreshold, setRefineThreshold] = React.useState("0.05");
  const [lengthBudget, setLengthBudget] = React.useState<string>(
    DEFAULTS.lengthBudget,
  );
  const [budgetSpeedup, setBudgetSpeedup] = React.useState("0.85");
  const [noSpeechThreshold, setNoSpeechThreshold] = React.useState("0.6");
  const [maxChars, setMaxChars] = React.useState("42");
  const [diarizer, setDiarizer] = React.useState<string>(DEFAULTS.diarizer);
  const [speakerThreshold, setSpeakerThreshold] = React.useState("");
  const [minSpeakerSeconds, setMinSpeakerSeconds] = React.useState("20");
  const [overlap, setOverlap] = React.useState<string>(DEFAULTS.overlapMode);
  const [overlapSeparator, setOverlapSeparator] = React.useState<string>(
    DEFAULTS.overlapSeparator,
  );
  const [expressiveAccent, setExpressiveAccent] = React.useState<string>(
    DEFAULTS.expressiveAccent,
  );
  const [accentBlend, setAccentBlend] = React.useState("");
  const [blendTakes, setBlendTakes] = React.useState("");
  const [blendIters, setBlendIters] = React.useState("");
  const [blendRatio, setBlendRatio] = React.useState("");
  const [chatterboxExaggeration, setChatterboxExaggeration] = React.useState("");
  const [chatterboxCfg, setChatterboxCfg] = React.useState("");
  const [chatterboxTemp, setChatterboxTemp] = React.useState("");
  const [fishTemp, setFishTemp] = React.useState("");
  const [fishTopP, setFishTopP] = React.useState("");
  const [fishQuantize, setFishQuantize] = React.useState<string>(
    DEFAULTS.fishQuantize,
  );
  const [force, setForce] = React.useState(false);
  const [provenance, setProvenance] = React.useState(true);
  const [savedVoices, setSavedVoices] = React.useState<VoiceInfo[]>([]);
  const [availableDevices, setAvailableDevices] = React.useState<Device[]>([]);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    api.voices({ pageSize: 200 }).then((data) => setSavedVoices(data.voices)).catch(() => undefined);
    api.devices().then((data) => setAvailableDevices(data.devices)).catch(() => undefined);
    api
      .settings()
      .then((data) => {
        if (data.default_device) setDevice(data.default_device);
        if (data.default_dtype) setDtype(data.default_dtype);
      })
      .catch(() => undefined);
  }, []);

  const applyConfig = React.useCallback((config: Record<string, unknown>) => {
    const input = config.video ?? config.audio;
    if (typeof input === "string") {
      setInputPath(input);
      setInputName(input.split(/[\\/]/).pop() ?? input);
    }
    setTarget(text(config.target_language, DEFAULTS.language));
    setSource(text(config.source_language));
    setDevice(text(config.device, DEFAULTS.device));
    setDtype(text(config.dtype, DEFAULTS.dtype));
    setDevices(
      Array.isArray(config.devices) ? (config.devices as string[]).join(",") : "",
    );
    setWhisperModel(text(config.whisper_model, DEFAULTS.whisperModel));
    setLlmModel(text(config.llm_model));
    setTtsBatch(text(config.tts_batch, "8"));
    setBackend(text(config.tts_backend, DEFAULTS.ttsBackend));
    setProsodyPlan(text(config.prosody_plan, DEFAULTS.prosodyPlan));
    setChatterboxExaggeration(text(config.chatterbox_exaggeration));
    setChatterboxCfg(text(config.chatterbox_cfg_weight));
    setChatterboxTemp(text(config.chatterbox_temperature));
    setFishTemp(text(config.fish_temperature));
    setFishTopP(text(config.fish_top_p));
    setFishQuantize(text(config.fish_quantize, DEFAULTS.fishQuantize));
    setMinStretch(text(config.min_stretch, "0.75"));
    setMaxStretch(text(config.max_stretch, "1.18"));
    setMaxOverrun(text(config.max_overrun, "0.25"));
    setLoudness(text(config.loudness_target, DEFAULTS.loudness));
    setLufs(text(config.target_lufs, "-16"));
    setKeepBackground(flag(config.keep_background, true));
    setSeparator(text(config.separator, DEFAULTS.separator));
    setDuckDb(text(config.duck_db, "6"));
    setBackgroundGain(text(config.background_gain_db, "0"));
    setRoom(text(config.room, DEFAULTS.room));
    setRoomRt60(text(config.room_rt60));
    setRoomWetDb(text(config.room_wet_db, "0"));
    setRoomDuckDb(text(config.room_duck_db, "3"));
    setRoomToneDb(text(config.room_tone_gain_db, "-30"));
    setAccentMode(text(config.accent_mode, DEFAULTS.accentMode));
    setAccentMatch(text(config.accent_match, DEFAULTS.accentMatch));
    setAccentBlend(text(config.accent_blend));
    setBlendTakes(text(config.blend_takes));
    setBlendIters(text(config.blend_iters));
    setBlendRatio(text(config.blend_ratio));
    setAccentAudio(text(config.accent_audio));
    setAccentText(text(config.accent_text));
    setExpressiveAccent(text(config.expressive_accent, DEFAULTS.expressiveAccent));
    setDiarizer(text(config.diarizer, DEFAULTS.diarizer));
    setSpeakerThreshold(text(config.speaker_threshold));
    setMinSpeakerSeconds(text(config.min_speaker_seconds, "20"));
    setOverlap(text(config.overlap_mode, DEFAULTS.overlapMode));
    setOverlapSeparator(text(config.overlap_separator, DEFAULTS.overlapSeparator));
    setRefine(flag(config.refine, true));
    setRefineMaxPasses(text(config.refine_max_passes, "1"));
    setRefineThreshold(text(config.refine_threshold, "0.05"));
    setLengthBudget(text(config.length_budget, DEFAULTS.lengthBudget));
    setBudgetSpeedup(text(config.budget_speedup, "0.85"));
    setNoSpeechThreshold(text(config.no_speech_threshold, "0.6"));
    setKeepSubtitles(flag(config.keep_subtitles, true));
    setSubtitlesMode(text(config.subtitles_mode, DEFAULTS.subtitlesMode));
    setMaxChars(text(config.max_chars, "42"));
    setForce(flag(config.force, false));
    setProvenance(flag(config.provenance, true));
    setRefAudio(text(config.ref_audio));
    setRefText(text(config.ref_text));
  }, []);

  // Load a past job's options for editing. Every field the form submits is read
  // back from the stored config, so Edit is a true prefill rather than a guess.
  React.useEffect(() => {
    if (!initialJob) return;
    setMode(initialJob.kind === "subtitles" ? "subtitles" : "dub");
    const params = (initialJob.params ?? {}) as Record<string, unknown>;
    applyConfig((params.config ?? {}) as Record<string, unknown>);
    // The saved voices live next to the config, not inside it.
    setVoice(text(params.voice));
    setAccentVoice(text(params.accent_voice));
  }, [initialJob, applyConfig]);

  const backends = useBackends("dub");
  const caps = capabilitiesOf(backends, backend);
  // Unknown capabilities => permissive; the server remains the authority.
  const allows = (feature: CapabilityFlag) => (caps ? caps[feature] : true);
  const reasonFor = (feature: CapabilityFlag) =>
    caps && !caps[feature] ? caps.unsupported?.[feature] : undefined;

  // Combinations the server would reject. Shown inline and used to block submit,
  // so an impossible request is never sent.
  const blockers: string[] = [];
  if (mode === "dub") {
    if (source && source === target) {
      blockers.push("Source and target language must differ.");
    }
    if (accentMode === ACCENT_MODE.preserve && !voice && refAudio && !refText.trim()) {
      blockers.push(
        "Add the reference clip's transcript, or choose another accent mode.",
      );
    }
    if (accentMode === ACCENT_MODE.referenceAccent && !accentVoice && !accentAudio) {
      blockers.push(
        "Borrowed accent needs an accent clip or a saved accent voice.",
      );
    }
    if (
      accentMode === ACCENT_MODE.referenceAccent &&
      !accentVoice &&
      accentAudio &&
      !accentText.trim()
    ) {
      blockers.push("Add the accent clip's transcript.");
    }
  }

  // The saved accent voice *is* the borrowed accent's clip, so it only means
  // something in that mode. Sending it alongside another mode would silently
  // switch the run to reference-accent, which the server rejects outright.
  const accentVoiceFor =
    accentMode === ACCENT_MODE.referenceAccent ? accentVoice || undefined : undefined;

  function chooseInput(path: string, name: string) {
    setInputPath(path);
    setInputName(name);
  }

  function isVideo(name: string): boolean {
    return VIDEO_EXT.some((ext) => name.toLowerCase().endsWith(ext));
  }

  function buildDubConfig(): Record<string, unknown> {
    const config: Record<string, unknown> = {
      target_language: target,
      source_language: source || undefined,
      output_dir: "output",
      device,
      devices: devices
        ? devices.split(",").map((item) => item.trim()).filter(Boolean)
        : undefined,
      dtype,
      whisper_model: whisperModel,
      llm_model: llmModel || undefined,
      tts_batch: num(ttsBatch),
      tts_backend: backend,
      prosody_plan: allows("prosody") ? prosodyPlan : PROSODY_PLAN.off,
      chatterbox_exaggeration: allows("chatterbox_tuning")
        ? num(chatterboxExaggeration)
        : undefined,
      chatterbox_cfg_weight: allows("chatterbox_tuning") ? num(chatterboxCfg) : undefined,
      chatterbox_temperature: allows("chatterbox_tuning")
        ? num(chatterboxTemp)
        : undefined,
      fish_temperature: allows("fish_tuning") ? num(fishTemp) : undefined,
      fish_top_p: allows("fish_tuning") ? num(fishTopP) : undefined,
      fish_quantize: allows("fish_tuning") ? fishQuantize : undefined,
      min_stretch: num(minStretch),
      max_stretch: num(maxStretch),
      max_overrun: num(maxOverrun),
      loudness_target: loudness,
      target_lufs: loudness === LOUDNESS.lufs ? num(lufs) : undefined,
      keep_background: keepBackground,
      separator,
      duck_db: num(duckDb),
      background_gain_db: num(backgroundGain),
      room: keepBackground ? room : ROOM.off,
      room_rt60: room === "match" ? num(roomRt60) : undefined,
      room_wet_db: room === "match" ? num(roomWetDb) : undefined,
      room_duck_db: room === "match" ? num(roomDuckDb) : undefined,
      room_tone_gain_db: room === "match" ? num(roomToneDb) : undefined,
      accent_mode: accentMode,
      accent_match: accentMode === ACCENT_MODE.native ? accentMatch : ACCENT_MATCH.off,
      accent_blend:
        accentMode === ACCENT_MODE.native || accentMode === ACCENT_MODE.referenceAccent
          ? accentBlend || undefined
          : undefined,
      blend_takes:
        accentMode === ACCENT_MODE.native || accentMode === ACCENT_MODE.referenceAccent
          ? num(blendTakes)
          : undefined,
      blend_iters:
        accentMode === ACCENT_MODE.native || accentMode === ACCENT_MODE.referenceAccent
          ? num(blendIters)
          : undefined,
      blend_ratio:
        accentMode === ACCENT_MODE.native || accentMode === ACCENT_MODE.referenceAccent
          ? num(blendRatio)
          : undefined,
      accent_audio:
        accentMode === ACCENT_MODE.referenceAccent && !accentVoice
          ? accentAudio || undefined
          : undefined,
      accent_text:
        accentMode === ACCENT_MODE.referenceAccent && !accentVoice
          ? accentText || undefined
          : undefined,
      expressive_accent: allows("expressive") ? expressiveAccent : EXPRESSIVE_ACCENT.donor,
      diarizer,
      speaker_threshold: num(speakerThreshold),
      min_speaker_seconds: num(minSpeakerSeconds),
      overlap_mode: overlap,
      overlap_separator: overlapSeparator,
      refine,
      refine_max_passes: num(refineMaxPasses),
      refine_threshold: num(refineThreshold),
      length_budget: lengthBudget,
      budget_speedup: num(budgetSpeedup),
      no_speech_threshold: num(noSpeechThreshold),
      keep_subtitles: keepSubtitles,
      subtitles_mode: keepSubtitles ? subtitlesMode : SUBTITLES_MODE.off,
      force,
      provenance,
    };
    if (isVideo(inputName)) config.video = inputPath;
    else config.audio = inputPath;
    if (refAudio && !voice) config.ref_audio = refAudio;
    if (refText && !voice) config.ref_text = refText;
    return config;
  }

  function buildPresetPayload(): Record<string, unknown> {
    const config = { ...buildDubConfig() };
    // A preset carries the options, not the input file it happened to use.
    delete config.video;
    delete config.audio;
    return {
      config,
      voice: voice || undefined,
      accent_voice: accentVoiceFor,
    };
  }

  function applyPreset(payload: Record<string, unknown>) {
    setMode("dub");
    const config = payload.config;
    if (config && typeof config === "object") {
      applyConfig(config as Record<string, unknown>);
    }
    setVoice(text(payload.voice));
    setAccentVoice(text(payload.accent_voice));
  }

  async function submit() {
    if (mode === "dub" && blockers.length) {
      setError(blockers[0]);
      return;
    }
    if (!inputPath) {
      setError("Choose an input video or audio file first.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      if (mode === "subtitles") {
        const config: Record<string, unknown> = {
          target_language: target,
          source_language: source || undefined,
          device,
          whisper_model: whisperModel,
          llm_model: llmModel || undefined,
          subtitles_mode: subtitlesMode,
          max_chars: num(maxChars),
          no_speech_threshold: num(noSpeechThreshold),
          force,
        };
        if (isVideo(inputName)) config.video = inputPath;
        else config.audio = inputPath;
        const job = await api.submitSubtitles(config);
        onJobStarted(job.job_id);
      } else {
        const job = await api.submitDub(
          {
            config: buildDubConfig(),
            voice: voice || undefined,
            accent_voice: accentVoiceFor,
          },
          "auto",
        );
        onJobStarted(job.job_id);
      }
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not start the job");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <Section
        title="Dubbing studio"
        subtitle="Turn a video into another language while keeping each speaker's voice."
        right={
          <div className="flex rounded-lg border border-white/10 bg-ink-850 p-1 text-xs">
            {(["dub", "subtitles"] as const).map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => setMode(value)}
                className={
                  mode === value
                    ? "rounded-md bg-accent-500 px-3 py-1.5 font-medium text-white"
                    : "rounded-md px-3 py-1.5 text-slate-400 hover:text-white"
                }
              >
                {value === "dub" ? "Dub" : "Subtitles only"}
              </button>
            ))}
          </div>
        }
      >
        <Card className="space-y-5 p-5">
          <div className="grid gap-4 md:grid-cols-2">
            <UploadField
              label="Input file"
              tooltip="The video or audio to dub. It is uploaded once and kept under the app data directory so later jobs can reuse it."
              hint="Video or audio. It is uploaded and kept under the app data directory."
              accept=".mp4,.mkv,.mov,.webm,.avi,.m4v,.mp3,.wav,.flac,.m4a,.aac,.ogg,.opus"
              value={inputPath}
              onUploaded={chooseInput}
            />
            <div className="grid gap-4 sm:grid-cols-2">
              <Field
                label="Target language"
                tooltip="The language the speakers will speak after dubbing."
              >
                <Select
                  value={target}
                  onChange={(event) => setTarget(event.target.value)}
                  options={LANGUAGE_OPTIONS}
                />
              </Field>
              <Field
                label="Source language"
                tooltip="The language spoken in the input. Leave on Auto-detect to let ASR identify it."
                hint={
                  source && source === target
                    ? "Source and target must differ."
                    : "Auto detects when left empty."
                }
              >
                <Select
                  value={source}
                  onChange={(event) => setSource(event.target.value)}
                  options={[
                    { value: "", label: "Auto-detect" },
                    ...LANGUAGE_OPTIONS,
                  ]}
                />
              </Field>
            </div>
          </div>

          <RecentInputs
            onPick={chooseInput}
            current={inputPath ? { path: inputPath, name: inputName } : null}
            accept=".mp4,.mkv,.mov,.webm,.avi,.m4v,.mp3,.wav,.flac,.m4a,.aac,.ogg,.opus"
          />

          {mode === "subtitles" ? (
            <div className="grid gap-4 md:grid-cols-3">
              <Field
                label="Subtitle mode"
                tooltip="How target subtitles reach the video: a toggleable track, burned into the frames, or a sidecar SRT only."
              >
                <Select
                  value={subtitlesMode}
                  onChange={(event) => setSubtitlesMode(event.target.value)}
                  options={SUBTITLES_MODE_OPTIONS}
                />
              </Field>
              <Field
                label="Max characters per cue"
                tooltip="Longer lines are split into shorter cues. Smaller values read faster but add more cues."
              >
                <Input value={maxChars} onChange={(event) => setMaxChars(event.target.value)} />
              </Field>
              <Field
                label="Whisper model"
                tooltip="ASR checkpoint used to transcribe. large-v3 is the most accurate."
              >
                <Input value={whisperModel} onChange={(event) => setWhisperModel(event.target.value)} />
              </Field>
              <Field
                label="Device"
                tooltip="Hardware to run on. Auto picks the best available GPU."
              >
                <Select
                  value={device}
                  onChange={(event) => setDevice(event.target.value)}
                  options={[
                    ...DEVICE_OPTIONS,
                    ...availableDevices.map((item) => ({
                      value: item.label,
                      label: `${item.label} — ${item.name}`,
                    })),
                  ]}
                />
              </Field>
              <Field
                label="Precision"
                tooltip="Numeric precision. Auto uses bfloat16/float16 when the GPU supports it."
              >
                <Select
                  value={dtype}
                  onChange={(event) => setDtype(event.target.value)}
                  options={DTYPE_OPTIONS}
                />
              </Field>
            </div>
          ) : null}

          {mode === "dub" ? (
            <>
              {/* Basic: the few choices most people actually make. */}
              <div className="grid gap-4 md:grid-cols-2">
                <Field
                  label="TTS backend"
                  tooltip="Voice engine. Qwen Base clones cleanly; CosyVoice3/Chatterbox add emotion control; Fish reads [tags] in the text."
                >
                  <Select
                    value={backend}
                    onChange={(event) => setBackend(event.target.value)}
                    options={backendOptions(backends, "dub")}
                  />
                </Field>
                <Field
                  label="Prosody plan"
                  tooltip="Who decides emotion and pace. 'Ask the LLM' writes it per line; 'Auto' plans when an LLM is reachable and falls back to neutral."
                  hint={
                    allows("prosody")
                      ? "Auto falls back to neutral when no LLM is reachable."
                      : reasonFor("prosody")
                  }
                >
                  <Select
                    value={prosodyPlan}
                    disabled={!allows("prosody")}
                    onChange={(event) => setProsodyPlan(event.target.value)}
                    options={PROSODY_PLAN_OPTIONS}
                  />
                </Field>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <Field
                  label="Saved voice"
                  tooltip="Clone one of your library voices instead of the source speaker."
                  hint="Leave empty to clone the source speaker."
                >
                  <Select
                    value={voice}
                    onChange={(event) => setVoice(event.target.value)}
                    options={[
                      { value: "", label: "Clone the source speaker" },
                      ...savedVoices.map((item) => ({
                        value: item.label,
                        label: `${item.label} (${item.language})`,
                      })),
                    ]}
                  />
                </Field>
                <Field
                  label="Accent mode"
                  tooltip="How target-language colour is added. The native anchor blends a target accent; preserve keeps the speaker's accent."
                >
                  <Select
                    value={accentMode}
                    onChange={(event) => setAccentMode(event.target.value)}
                    options={ACCENT_MODE_OPTIONS}
                  />
                </Field>
                <Field
                  label="Saved accent voice"
                  tooltip="A target-language voice used for pronunciation and accent colour, never for timbre."
                  hint={
                    accentMode === ACCENT_MODE.referenceAccent
                      ? "A target-language voice for pronunciation."
                      : "Only used when borrowing a reference accent."
                  }
                >
                  <Select
                    value={accentVoice}
                    disabled={accentMode !== ACCENT_MODE.referenceAccent}
                    onChange={(event) => setAccentVoice(event.target.value)}
                    options={[
                      { value: "", label: "None" },
                      ...savedVoices.map((item) => ({
                        value: item.label,
                        label: `${item.label} (${item.language})`,
                      })),
                    ]}
                  />
                </Field>
                <Field
                  label="Expressive accent"
                  tooltip="For expressive backends: clone the donor's delivery, or blend a native accent into it."
                  hint={allows("expressive") ? undefined : reasonFor("expressive")}
                >
                  <Select
                    value={expressiveAccent}
                    disabled={!allows("expressive")}
                    onChange={(event) => setExpressiveAccent(event.target.value)}
                    options={EXPRESSIVE_ACCENT_OPTIONS}
                  />
                </Field>
                <UploadField
                  label="Reference clip"
                  tooltip="Optional clip to clone instead of the source speaker."
                  hint="Optional: clone this voice instead of the source speaker."
                  accept=".wav,.mp3,.flac,.m4a,.ogg,.opus"
                  value={refAudio}
                  onUploaded={(path) => setRefAudio(path)}
                />
                <Field
                  label="Reference transcript"
                  tooltip="The exact words spoken in the reference clip; required in preserve mode and improves similarity."
                  hint="Required with a reference clip in preserve mode."
                >
                  <Input value={refText} onChange={(event) => setRefText(event.target.value)} />
                </Field>
                <UploadField
                  label="Accent clip"
                  tooltip="A target-language sample to borrow pronunciation from. Used only with reference-accent."
                  hint="Used only with reference-accent."
                  accept=".wav,.mp3,.flac,.m4a,.ogg,.opus"
                  value={accentAudio}
                  onUploaded={(path) => setAccentAudio(path)}
                />
                <Field
                  label="Accent transcript"
                  tooltip="The exact words spoken in the accent clip; it aligns the speech codes."
                >
                  <Input value={accentText} onChange={(event) => setAccentText(event.target.value)} />
                </Field>
              </div>

              <Disclosure title="Advanced options" hint="Fine-tuning most projects never need">
                <Field
                  label="Accent match"
                  tooltip="Match the speaker's rate (and gender) to the target anchor. Native mode only."
                  hint={
                    accentMode === ACCENT_MODE.native
                      ? "Only with the native anchor mode."
                      : "Available only in the native accent mode."
                  }
                >
                  <Select
                    value={accentMatch}
                    disabled={accentMode !== ACCENT_MODE.native}
                    onChange={(event) => setAccentMatch(event.target.value)}
                    options={ACCENT_MATCH_OPTIONS}
                  />
                </Field>

                <Disclosure title="Accent blend detail" hint="How the native target accent is built">
                  <div className="grid gap-4 md:grid-cols-4">
                    <Field label="Blend mode">
                      <Select
                        value={accentBlend}
                        onChange={(event) => setAccentBlend(event.target.value)}
                        options={[
                          { value: "", label: "Default (two-stage)" },
                          ...ACCENT_BLEND_OPTIONS,
                        ]}
                      />
                    </Field>
                    <Field label="Blend takes">
                      <Input value={blendTakes} onChange={(event) => setBlendTakes(event.target.value)} placeholder="3" />
                    </Field>
                    <Field label="Blend iterations">
                      <Input value={blendIters} onChange={(event) => setBlendIters(event.target.value)} placeholder="5" />
                    </Field>
                    <Field label="Blend ratio">
                      <Input value={blendRatio} onChange={(event) => setBlendRatio(event.target.value)} placeholder="0.85" />
                    </Field>
                  </div>
                </Disclosure>

                {backend === TTS_BACKEND.chatterbox ? (
                  <div className="grid gap-4 md:grid-cols-3">
                    <Field label="Exaggeration" tooltip="How theatrical the delivery is; blank lets the plan decide.">
                      <Input value={chatterboxExaggeration} onChange={(event) => setChatterboxExaggeration(event.target.value)} placeholder="derived" />
                    </Field>
                    <Field label="CFG weight" tooltip="Classifier-free guidance strength; higher stays closer to the prompt.">
                      <Input value={chatterboxCfg} onChange={(event) => setChatterboxCfg(event.target.value)} placeholder="0.5" />
                    </Field>
                    <Field label="Temperature" tooltip="Sampling randomness; lower is steadier, higher is more varied.">
                      <Input value={chatterboxTemp} onChange={(event) => setChatterboxTemp(event.target.value)} placeholder="0.8" />
                    </Field>
                  </div>
                ) : null}
                {backend === TTS_BACKEND.fish ? (
                  <div className="grid gap-4 md:grid-cols-3">
                    <Field label="Temperature" tooltip="Sampling randomness; lower is steadier, higher is more varied.">
                      <Input value={fishTemp} onChange={(event) => setFishTemp(event.target.value)} placeholder="0.7" />
                    </Field>
                    <Field label="Top-p" tooltip="Nucleus sampling cut-off; lower restricts to more likely tokens.">
                      <Input value={fishTopP} onChange={(event) => setFishTopP(event.target.value)} placeholder="0.7" />
                    </Field>
                    <Field label="Quantization" tooltip="int8 fits smaller GPUs; bf16 needs more memory but is a little more faithful.">
                      <Select
                        value={fishQuantize}
                        onChange={(event) => setFishQuantize(event.target.value)}
                        options={FISH_QUANTIZE_OPTIONS}
                      />
                    </Field>
                  </div>
                ) : null}

                <Disclosure title="Audio and subtitles" hint="Background, loudness, room and captions">
                  <div className="grid gap-4 md:grid-cols-2">
                    <Toggle
                      label="Keep music and ambience"
                      hint="Separate and duck the original bed under the dub."
                      checked={keepBackground}
                      onChange={setKeepBackground}
                    />
                    <Toggle
                      label="Write subtitles"
                      hint="Turn off to skip the SRT files."
                      checked={keepSubtitles}
                      onChange={setKeepSubtitles}
                    />
                    <Field label="Separator" tooltip="Stem splitter for music and ambience. Auto uses Demucs when installed.">
                      <Select
                        value={separator}
                        onChange={(event) => setSeparator(event.target.value)}
                        options={SEPARATOR_OPTIONS}
                      />
                    </Field>
                    <Field label="Subtitle mode" tooltip="A toggleable track, burned into the frames, or a sidecar SRT only.">
                      <Select
                        value={subtitlesMode}
                        onChange={(event) => setSubtitlesMode(event.target.value)}
                        options={SUBTITLES_MODE_OPTIONS}
                      />
                    </Field>
                    <Field label="Duck under dialogue (dB)" tooltip="How far the background bed drops under the dub, in decibels.">
                      <Input value={duckDb} onChange={(event) => setDuckDb(event.target.value)} />
                    </Field>
                    <Field label="Background gain (dB)" tooltip="Overall level change applied to the separated background.">
                      <Input value={backgroundGain} onChange={(event) => setBackgroundGain(event.target.value)} />
                    </Field>
                    <Field label="Room matching" tooltip="Add a short reverb that matches the source room to the dry dub.">
                      <Select
                        value={room}
                        onChange={(event) => setRoom(event.target.value)}
                        options={ROOM_OPTIONS}
                      />
                    </Field>
                    <Field label="Loudness" tooltip="Match the source loudness, or normalise to an absolute LUFS target.">
                      <Select
                        value={loudness}
                        onChange={(event) => setLoudness(event.target.value)}
                        options={LOUDNESS_OPTIONS}
                      />
                    </Field>
                    <Field label="LUFS target" tooltip="Integrated loudness target used with absolute loudness." hint="Only with absolute loudness.">
                      <Input value={lufs} onChange={(event) => setLufs(event.target.value)} />
                    </Field>
                  </div>
                  <Disclosure title="Room matching detail" hint="Only used when room matching is on">
                    <div className="grid gap-4 md:grid-cols-4">
                      <Field label="RT60 (s)" hint="Blank measures it.">
                        <Input value={roomRt60} onChange={(event) => setRoomRt60(event.target.value)} />
                      </Field>
                      <Field label="Wet level (dB)">
                        <Input value={roomWetDb} onChange={(event) => setRoomWetDb(event.target.value)} />
                      </Field>
                      <Field label="Room duck (dB)">
                        <Input value={roomDuckDb} onChange={(event) => setRoomDuckDb(event.target.value)} />
                      </Field>
                      <Field label="Room tone (dB)">
                        <Input value={roomToneDb} onChange={(event) => setRoomToneDb(event.target.value)} />
                      </Field>
                    </div>
                  </Disclosure>
                </Disclosure>

                <Disclosure title="Timing, speakers and repair" hint="Sync, diarization and length repair">
                  <div className="grid gap-4 md:grid-cols-3">
                    <Field label="Min stretch" tooltip="The most a line may be slowed when fitting its time slot (0.75 = 75%).">
                      <Input value={minStretch} onChange={(event) => setMinStretch(event.target.value)} />
                    </Field>
                    <Field label="Max stretch" tooltip="The most a line may be sped up when fitting its time slot (1.18 = 118%).">
                      <Input value={maxStretch} onChange={(event) => setMaxStretch(event.target.value)} />
                    </Field>
                    <Field label="Max overrun (s)" tooltip="How far a line may spill past its slot before it is repaired.">
                      <Input value={maxOverrun} onChange={(event) => setMaxOverrun(event.target.value)} />
                    </Field>
                    <Field label="Diarizer" tooltip="Split the audio by speaker, or treat everything as one voice.">
                      <Select
                        value={diarizer}
                        onChange={(event) => setDiarizer(event.target.value)}
                        options={DIARIZER_OPTIONS}
                      />
                    </Field>
                    <Field label="Speaker threshold" tooltip="Clustering strictness for speaker turns; higher splits less eagerly.">
                      <Input value={speakerThreshold} onChange={(event) => setSpeakerThreshold(event.target.value)} placeholder="0.7" />
                    </Field>
                    <Field label="Min speaker seconds" tooltip="Speakers shorter than this are merged into a neighbour.">
                      <Input value={minSpeakerSeconds} onChange={(event) => setMinSpeakerSeconds(event.target.value)} />
                    </Field>
                    <Field label="Overlap" tooltip="Keep the dominant voice, or separate and dub both overlapping speakers.">
                      <Select
                        value={overlap}
                        onChange={(event) => setOverlap(event.target.value)}
                        options={OVERLAP_MODE_OPTIONS}
                      />
                    </Field>
                    <Field label="Overlap separator" tooltip="Model used to split overlapping speech; Auto picks SepFormer when installed.">
                      <Select
                        value={overlapSeparator}
                        onChange={(event) => setOverlapSeparator(event.target.value)}
                        options={OVERLAP_SEPARATOR_OPTIONS}
                      />
                    </Field>
                    <Field label="Length budget" tooltip="Cap translated length by word count, or by duration only.">
                      <Select
                        value={lengthBudget}
                        onChange={(event) => setLengthBudget(event.target.value)}
                        options={LENGTH_BUDGET_OPTIONS}
                      />
                    </Field>
                    <Field label="Budget speedup" tooltip="Target speaking-rate multiplier used when a line is over its word budget.">
                      <Input value={budgetSpeedup} onChange={(event) => setBudgetSpeedup(event.target.value)} />
                    </Field>
                    <Field label="Refine passes" tooltip="How many LLM rewrite passes fix overlong lines.">
                      <Input value={refineMaxPasses} onChange={(event) => setRefineMaxPasses(event.target.value)} />
                    </Field>
                    <Field label="Refine threshold (s)" tooltip="Overrun beyond this triggers a rewrite pass.">
                      <Input value={refineThreshold} onChange={(event) => setRefineThreshold(event.target.value)} />
                    </Field>
                    <Field label="No-speech threshold" tooltip="ASR segments below this confidence are dropped." hint="ASR segments the model distrusts are dropped.">
                      <Input value={noSpeechThreshold} onChange={(event) => setNoSpeechThreshold(event.target.value)} />
                    </Field>
                    <Toggle
                      label="Repair overlong lines"
                      hint="Reword lines that would otherwise lose words."
                      checked={refine}
                      onChange={setRefine}
                    />
                    <Toggle
                      label="Write provenance"
                      hint="Per-line prosody and text transformations."
                      checked={provenance}
                      onChange={setProvenance}
                    />
                    <Toggle
                      label="Ignore caches"
                      hint="Rebuild every stage."
                      checked={force}
                      onChange={setForce}
                    />
                  </div>
                </Disclosure>

                <Disclosure title="Models and hardware" hint="Precision, batch and per-op devices">
                  <div className="grid gap-4 md:grid-cols-3">
                    <Field label="Device" tooltip="Hardware target. Auto picks the best available GPU.">
                      <Select
                        value={device}
                        onChange={(event) => setDevice(event.target.value)}
                        options={[
                          ...DEVICE_OPTIONS,
                          ...availableDevices.map((item) => ({
                            value: item.label,
                            label: `${item.label} — ${item.name}`,
                          })),
                        ]}
                      />
                    </Field>
                    <Field label="Precision" tooltip="Numeric precision. Auto uses bfloat16/float16 when supported.">
                      <Select
                        value={dtype}
                        onChange={(event) => setDtype(event.target.value)}
                        options={DTYPE_OPTIONS}
                      />
                    </Field>
                    <Field label="Fan-out devices" tooltip="Extra GPUs to spread synthesis over, each with its own worker." hint="Comma-separated, e.g. cuda:0,cuda:1">
                      <Input value={devices} onChange={(event) => setDevices(event.target.value)} placeholder="unset" />
                    </Field>
                    <Field label="TTS batch" tooltip="Lines synthesised per model call; higher is faster but uses more memory.">
                      <Input value={ttsBatch} onChange={(event) => setTtsBatch(event.target.value)} />
                    </Field>
                    <Field label="Whisper model" tooltip="ASR checkpoint used to transcribe; large-v3 is the most accurate.">
                      <Input value={whisperModel} onChange={(event) => setWhisperModel(event.target.value)} />
                    </Field>
                    <Field label="LLM model override" tooltip="Use a different OpenAI-compatible model for this run only.">
                      <Input value={llmModel} onChange={(event) => setLlmModel(event.target.value)} placeholder="from settings" />
                    </Field>
                  </div>
                </Disclosure>
              </Disclosure>

              <Disclosure title="Presets" hint="Save and reuse these options">
                <PresetPanel
                  kind="dub"
                  buildPayload={buildPresetPayload}
                  onApply={applyPreset}
                />
              </Disclosure>
            </>
          ) : null}


          {blockers.length ? <ErrorNote message={blockers.join(" ")} /> : null}
          {error ? <ErrorNote message={error} /> : null}

          <div className="flex items-center justify-between gap-3 border-t border-white/5 pt-4">
            <div className="text-xs text-slate-500">
              {inputName ? `Input: ${inputName}` : "No input selected"}
            </div>
            <Button
              onClick={submit}
              disabled={busy || !inputPath || blockers.length > 0}
            >
              {busy ? <Spinner /> : null}
              {mode === "dub" ? "Start dubbing" : "Start subtitles"}
            </Button>
          </div>
        </Card>
      </Section>
    </div>
  );
}
