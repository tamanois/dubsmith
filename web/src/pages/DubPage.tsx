import React from "react";
import { api } from "../api";
import type { Device, Job, VoiceInfo } from "../types";
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

const LANGUAGES = [
  "French",
  "English",
  "German",
  "Italian",
  "Portuguese",
  "Spanish",
  "Japanese",
  "Korean",
  "Chinese",
  "Russian",
];

const VIDEO_EXT = [".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"];

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
  const [target, setTarget] = React.useState("French");
  const [source, setSource] = React.useState("");
  const [voice, setVoice] = React.useState("");
  const [accentVoice, setAccentVoice] = React.useState("");
  const [refAudio, setRefAudio] = React.useState("");
  const [refText, setRefText] = React.useState("");
  const [accentAudio, setAccentAudio] = React.useState("");
  const [accentText, setAccentText] = React.useState("");
  const [accentMode, setAccentMode] = React.useState("native");
  const [accentMatch, setAccentMatch] = React.useState("off");
  const [backend, setBackend] = React.useState("qwen");
  const [prosodyPlan, setProsodyPlan] = React.useState("auto");
  const [device, setDevice] = React.useState("auto");
  const [dtype, setDtype] = React.useState("auto");
  const [devices, setDevices] = React.useState("");
  const [ttsBatch, setTtsBatch] = React.useState("8");
  const [whisperModel, setWhisperModel] = React.useState("large-v3");
  const [llmModel, setLlmModel] = React.useState("");
  const [subtitlesMode, setSubtitlesMode] = React.useState("soft");
  const [keepSubtitles, setKeepSubtitles] = React.useState(true);
  const [keepBackground, setKeepBackground] = React.useState(true);
  const [separator, setSeparator] = React.useState("auto");
  const [duckDb, setDuckDb] = React.useState("6");
  const [backgroundGain, setBackgroundGain] = React.useState("0");
  const [loudness, setLoudness] = React.useState("match-source");
  const [lufs, setLufs] = React.useState("-16");
  const [room, setRoom] = React.useState("off");
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
  const [lengthBudget, setLengthBudget] = React.useState("words");
  const [budgetSpeedup, setBudgetSpeedup] = React.useState("0.85");
  const [noSpeechThreshold, setNoSpeechThreshold] = React.useState("0.6");
  const [maxChars, setMaxChars] = React.useState("42");
  const [diarizer, setDiarizer] = React.useState("auto");
  const [speakerThreshold, setSpeakerThreshold] = React.useState("");
  const [minSpeakerSeconds, setMinSpeakerSeconds] = React.useState("20");
  const [overlap, setOverlap] = React.useState("dominant");
  const [overlapSeparator, setOverlapSeparator] = React.useState("auto");
  const [expressiveAccent, setExpressiveAccent] = React.useState("donor");
  const [accentBlend, setAccentBlend] = React.useState("");
  const [blendTakes, setBlendTakes] = React.useState("");
  const [blendIters, setBlendIters] = React.useState("");
  const [blendRatio, setBlendRatio] = React.useState("");
  const [chatterboxExaggeration, setChatterboxExaggeration] = React.useState("");
  const [chatterboxCfg, setChatterboxCfg] = React.useState("");
  const [chatterboxTemp, setChatterboxTemp] = React.useState("");
  const [fishTemp, setFishTemp] = React.useState("");
  const [fishTopP, setFishTopP] = React.useState("");
  const [fishQuantize, setFishQuantize] = React.useState("int8");
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
    setTarget(text(config.target_language, "French"));
    setSource(text(config.source_language));
    setDevice(text(config.device, "auto"));
    setDtype(text(config.dtype, "auto"));
    setDevices(
      Array.isArray(config.devices) ? (config.devices as string[]).join(",") : "",
    );
    setWhisperModel(text(config.whisper_model, "large-v3"));
    setLlmModel(text(config.llm_model));
    setTtsBatch(text(config.tts_batch, "8"));
    setBackend(text(config.tts_backend, "qwen"));
    setProsodyPlan(text(config.prosody_plan, "auto"));
    setChatterboxExaggeration(text(config.chatterbox_exaggeration));
    setChatterboxCfg(text(config.chatterbox_cfg_weight));
    setChatterboxTemp(text(config.chatterbox_temperature));
    setFishTemp(text(config.fish_temperature));
    setFishTopP(text(config.fish_top_p));
    setFishQuantize(text(config.fish_quantize, "int8"));
    setMinStretch(text(config.min_stretch, "0.75"));
    setMaxStretch(text(config.max_stretch, "1.18"));
    setMaxOverrun(text(config.max_overrun, "0.25"));
    setLoudness(text(config.loudness_target, "match-source"));
    setLufs(text(config.target_lufs, "-16"));
    setKeepBackground(flag(config.keep_background, true));
    setSeparator(text(config.separator, "auto"));
    setDuckDb(text(config.duck_db, "6"));
    setBackgroundGain(text(config.background_gain_db, "0"));
    setRoom(text(config.room, "off"));
    setRoomRt60(text(config.room_rt60));
    setRoomWetDb(text(config.room_wet_db, "0"));
    setRoomDuckDb(text(config.room_duck_db, "3"));
    setRoomToneDb(text(config.room_tone_gain_db, "-30"));
    setAccentMode(text(config.accent_mode, "native"));
    setAccentMatch(text(config.accent_match, "off"));
    setAccentBlend(text(config.accent_blend));
    setBlendTakes(text(config.blend_takes));
    setBlendIters(text(config.blend_iters));
    setBlendRatio(text(config.blend_ratio));
    setAccentAudio(text(config.accent_audio));
    setAccentText(text(config.accent_text));
    setExpressiveAccent(text(config.expressive_accent, "donor"));
    setDiarizer(text(config.diarizer, "auto"));
    setSpeakerThreshold(text(config.speaker_threshold));
    setMinSpeakerSeconds(text(config.min_speaker_seconds, "20"));
    setOverlap(text(config.overlap_mode, "dominant"));
    setOverlapSeparator(text(config.overlap_separator, "auto"));
    setRefine(flag(config.refine, true));
    setRefineMaxPasses(text(config.refine_max_passes, "1"));
    setRefineThreshold(text(config.refine_threshold, "0.05"));
    setLengthBudget(text(config.length_budget, "words"));
    setBudgetSpeedup(text(config.budget_speedup, "0.85"));
    setNoSpeechThreshold(text(config.no_speech_threshold, "0.6"));
    setKeepSubtitles(flag(config.keep_subtitles, true));
    setSubtitlesMode(text(config.subtitles_mode, "soft"));
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
    applyConfig((initialJob.params?.config ?? {}) as Record<string, unknown>);
  }, [initialJob, applyConfig]);

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
      prosody_plan: prosodyPlan,
      chatterbox_exaggeration: num(chatterboxExaggeration),
      chatterbox_cfg_weight: num(chatterboxCfg),
      chatterbox_temperature: num(chatterboxTemp),
      fish_temperature: num(fishTemp),
      fish_top_p: num(fishTopP),
      fish_quantize: backend === "fish" ? fishQuantize : undefined,
      min_stretch: num(minStretch),
      max_stretch: num(maxStretch),
      max_overrun: num(maxOverrun),
      loudness_target: loudness,
      target_lufs: loudness === "lufs" ? num(lufs) : undefined,
      keep_background: keepBackground,
      separator,
      duck_db: num(duckDb),
      background_gain_db: num(backgroundGain),
      room: keepBackground ? room : "off",
      room_rt60: room === "match" ? num(roomRt60) : undefined,
      room_wet_db: room === "match" ? num(roomWetDb) : undefined,
      room_duck_db: room === "match" ? num(roomDuckDb) : undefined,
      room_tone_gain_db: room === "match" ? num(roomToneDb) : undefined,
      accent_mode: accentMode,
      accent_match: accentMode === "native" ? accentMatch : "off",
      accent_blend:
        accentMode === "native" || accentMode === "reference-accent"
          ? accentBlend || undefined
          : undefined,
      blend_takes:
        accentMode === "native" || accentMode === "reference-accent"
          ? num(blendTakes)
          : undefined,
      blend_iters:
        accentMode === "native" || accentMode === "reference-accent"
          ? num(blendIters)
          : undefined,
      blend_ratio:
        accentMode === "native" || accentMode === "reference-accent"
          ? num(blendRatio)
          : undefined,
      accent_audio:
        accentMode === "reference-accent" && !accentVoice
          ? accentAudio || undefined
          : undefined,
      accent_text:
        accentMode === "reference-accent" && !accentVoice
          ? accentText || undefined
          : undefined,
      expressive_accent: backend === "qwen" ? "donor" : expressiveAccent,
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
      subtitles_mode: keepSubtitles ? subtitlesMode : "off",
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
      accent_voice: accentVoice || undefined,
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
            accent_voice: accentVoice || undefined,
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
              hint="Video or audio. It is uploaded and kept under the app data directory."
              accept=".mp4,.mkv,.mov,.webm,.avi,.m4v,.mp3,.wav,.flac,.m4a,.aac,.ogg,.opus"
              onUploaded={chooseInput}
            />
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Target language">
                <Select
                  value={target}
                  onChange={(event) => setTarget(event.target.value)}
                  options={LANGUAGES.map((language) => ({
                    value: language,
                    label: language,
                  }))}
                />
              </Field>
              <Field label="Source language" hint="Auto detects when left empty.">
                <Select
                  value={source}
                  onChange={(event) => setSource(event.target.value)}
                  options={[
                    { value: "", label: "Auto-detect" },
                    ...LANGUAGES.map((language) => ({
                      value: language,
                      label: language,
                    })),
                  ]}
                />
              </Field>
            </div>
          </div>

          <RecentInputs
            onPick={chooseInput}
            current={inputPath ? { path: inputPath, name: inputName } : null}
          />

          {mode === "dub" ? (
            <Disclosure title="Presets" hint="Save and reuse these options">
              <PresetPanel
                kind="dub"
                buildPayload={buildPresetPayload}
                onApply={applyPreset}
              />
            </Disclosure>
          ) : null}

          {mode === "subtitles" ? (
            <div className="grid gap-4 md:grid-cols-3">
              <Field label="Subtitle mode">
                <Select
                  value={subtitlesMode}
                  onChange={(event) => setSubtitlesMode(event.target.value)}
                  options={[
                    { value: "soft", label: "Soft track (toggleable)" },
                    { value: "burn", label: "Burn into the video" },
                    { value: "off", label: "Sidecar only" },
                  ]}
                />
              </Field>
              <Field label="Max characters per cue">
                <Input value={maxChars} onChange={(event) => setMaxChars(event.target.value)} />
              </Field>
              <Field label="Whisper model">
                <Input value={whisperModel} onChange={(event) => setWhisperModel(event.target.value)} />
              </Field>
              <Field label="Device">
                <Select
                  value={device}
                  onChange={(event) => setDevice(event.target.value)}
                  options={[
                    { value: "auto", label: "Auto" },
                    { value: "cpu", label: "CPU" },
                    ...availableDevices.map((item) => ({
                      value: item.label,
                      label: `${item.label} — ${item.name}`,
                    })),
                  ]}
                />
              </Field>
              <Field label="Precision">
                <Select
                  value={dtype}
                  onChange={(event) => setDtype(event.target.value)}
                  options={[
                    { value: "auto", label: "Auto" },
                    { value: "bf16", label: "bfloat16" },
                    { value: "fp16", label: "float16" },
                    { value: "fp32", label: "float32" },
                  ]}
                />
              </Field>
            </div>
          ) : null}

          {mode === "dub" ? (
            <>
              <Disclosure title="Voice and accent" hint="Who speaks, and with which accent">
                <div className="grid gap-4 md:grid-cols-2">
                  <Field label="Saved voice" hint="Leave empty to clone the source speaker.">
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
                  <Field label="Accent mode">
                    <Select
                      value={accentMode}
                      onChange={(event) => setAccentMode(event.target.value)}
                      options={[
                        { value: "native", label: "Native target accent (anchor)" },
                        { value: "preserve", label: "Keep the speaker's accent" },
                        { value: "speaker-only", label: "Speaker timbre only" },
                        { value: "reference-accent", label: "Borrow a reference accent" },
                      ]}
                    />
                  </Field>
                  <Field label="Accent match" hint="Only with the native anchor mode.">
                    <Select
                      value={accentMatch}
                      onChange={(event) => setAccentMatch(event.target.value)}
                      options={[
                        { value: "off", label: "Off" },
                        { value: "rate", label: "Match speaking rate" },
                        { value: "gender", label: "Match rate and gender" },
                      ]}
                    />
                  </Field>
                  <Field label="Saved accent voice" hint="A target-language voice for pronunciation.">
                    <Select
                      value={accentVoice}
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
                  <UploadField
                    label="Reference clip"
                    hint="Optional: clone this voice instead of the source speaker."
                    accept=".wav,.mp3,.flac,.m4a,.ogg,.opus"
                    onUploaded={(path) => setRefAudio(path)}
                  />
                  <Field label="Reference transcript" hint="Required with a reference clip in preserve mode.">
                    <Input value={refText} onChange={(event) => setRefText(event.target.value)} />
                  </Field>
                  <UploadField
                    label="Accent clip"
                    hint="Used only with reference-accent."
                    accept=".wav,.mp3,.flac,.m4a,.ogg,.opus"
                    onUploaded={(path) => setAccentAudio(path)}
                  />
                  <Field label="Accent transcript">
                    <Input value={accentText} onChange={(event) => setAccentText(event.target.value)} />
                  </Field>
                </div>
                <Disclosure title="Accent blend detail" hint="How the native target accent is built">
                  <div className="grid gap-4 md:grid-cols-4">
                    <Field label="Blend mode">
                      <Select
                        value={accentBlend}
                        onChange={(event) => setAccentBlend(event.target.value)}
                        options={[
                          { value: "", label: "Default (two-stage)" },
                          { value: "two-stage", label: "Two-stage" },
                          { value: "single-stage", label: "Single-stage" },
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
              </Disclosure>

              <Disclosure title="Backend and prosody" hint="Emotion, pace and expression">
                <div className="grid gap-4 md:grid-cols-2">
                  <Field label="TTS backend">
                    <Select
                      value={backend}
                      onChange={(event) => setBackend(event.target.value)}
                      options={[
                        { value: "qwen", label: "Qwen Base (cloning, no prosody)" },
                        { value: "cosyvoice3", label: "CosyVoice3 (emotion + tokens)" },
                        { value: "chatterbox", label: "Chatterbox (expression intensity)" },
                        { value: "fish", label: "Fish S2-Pro (inline tags)" },
                      ]}
                    />
                  </Field>
                  <Field
                    label="Prosody plan"
                    hint={backend === "qwen" ? "Ignored by the Qwen Base backend." : "Auto falls back to neutral when no LLM is reachable."}
                  >
                    <Select
                      value={prosodyPlan}
                      onChange={(event) => setProsodyPlan(event.target.value)}
                      options={[
                        { value: "auto", label: "Auto" },
                        { value: "llm", label: "Ask the LLM" },
                        { value: "off", label: "Neutral" },
                      ]}
                    />
                  </Field>
                  <Field label="Expressive accent">
                    <Select
                      value={expressiveAccent}
                      onChange={(event) => setExpressiveAccent(event.target.value)}
                      options={[
                        { value: "donor", label: "Clone the donor" },
                        { value: "blend", label: "Blend a native accent" },
                      ]}
                    />
                  </Field>
                </div>
                {backend === "chatterbox" ? (
                  <div className="grid gap-4 md:grid-cols-3">
                    <Field label="Exaggeration">
                      <Input value={chatterboxExaggeration} onChange={(event) => setChatterboxExaggeration(event.target.value)} placeholder="derived" />
                    </Field>
                    <Field label="CFG weight">
                      <Input value={chatterboxCfg} onChange={(event) => setChatterboxCfg(event.target.value)} placeholder="0.5" />
                    </Field>
                    <Field label="Temperature">
                      <Input value={chatterboxTemp} onChange={(event) => setChatterboxTemp(event.target.value)} placeholder="0.8" />
                    </Field>
                  </div>
                ) : null}
                {backend === "fish" ? (
                  <div className="grid gap-4 md:grid-cols-3">
                    <Field label="Temperature">
                      <Input value={fishTemp} onChange={(event) => setFishTemp(event.target.value)} placeholder="0.7" />
                    </Field>
                    <Field label="Top-p">
                      <Input value={fishTopP} onChange={(event) => setFishTopP(event.target.value)} placeholder="0.7" />
                    </Field>
                    <Field label="Quantization">
                      <Select
                        value={fishQuantize}
                        onChange={(event) => setFishQuantize(event.target.value)}
                        options={[
                          { value: "int8", label: "int8 (12 GB cards)" },
                          { value: "off", label: "bf16 (~24 GB)" },
                        ]}
                      />
                    </Field>
                  </div>
                ) : null}
              </Disclosure>

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
                  <Field label="Separator">
                    <Select
                      value={separator}
                      onChange={(event) => setSeparator(event.target.value)}
                      options={[
                        { value: "auto", label: "Auto" },
                        { value: "demucs", label: "Demucs" },
                        { value: "none", label: "None" },
                      ]}
                    />
                  </Field>
                  <Field label="Subtitle mode">
                    <Select
                      value={subtitlesMode}
                      onChange={(event) => setSubtitlesMode(event.target.value)}
                      options={[
                        { value: "soft", label: "Soft track (toggleable)" },
                        { value: "burn", label: "Burn into the video" },
                        { value: "off", label: "Sidecar only" },
                      ]}
                    />
                  </Field>
                  <Field label="Duck under dialogue (dB)">
                    <Input value={duckDb} onChange={(event) => setDuckDb(event.target.value)} />
                  </Field>
                  <Field label="Background gain (dB)">
                    <Input value={backgroundGain} onChange={(event) => setBackgroundGain(event.target.value)} />
                  </Field>
                  <Field label="Room matching">
                    <Select
                      value={room}
                      onChange={(event) => setRoom(event.target.value)}
                      options={[
                        { value: "off", label: "Dry voice-over" },
                        { value: "match", label: "Match the source room" },
                      ]}
                    />
                  </Field>
                  <Field label="Loudness">
                    <Select
                      value={loudness}
                      onChange={(event) => setLoudness(event.target.value)}
                      options={[
                        { value: "match-source", label: "Match the source" },
                        { value: "lufs", label: "Absolute LUFS" },
                      ]}
                    />
                  </Field>
                  <Field label="LUFS target" hint="Only with absolute loudness.">
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
                  <Field label="Min stretch">
                    <Input value={minStretch} onChange={(event) => setMinStretch(event.target.value)} />
                  </Field>
                  <Field label="Max stretch">
                    <Input value={maxStretch} onChange={(event) => setMaxStretch(event.target.value)} />
                  </Field>
                  <Field label="Max overrun (s)">
                    <Input value={maxOverrun} onChange={(event) => setMaxOverrun(event.target.value)} />
                  </Field>
                  <Field label="Diarizer">
                    <Select
                      value={diarizer}
                      onChange={(event) => setDiarizer(event.target.value)}
                      options={[
                        { value: "auto", label: "Separate speakers" },
                        { value: "none", label: "Single speaker" },
                      ]}
                    />
                  </Field>
                  <Field label="Speaker threshold">
                    <Input value={speakerThreshold} onChange={(event) => setSpeakerThreshold(event.target.value)} placeholder="0.7" />
                  </Field>
                  <Field label="Min speaker seconds">
                    <Input value={minSpeakerSeconds} onChange={(event) => setMinSpeakerSeconds(event.target.value)} />
                  </Field>
                  <Field label="Overlap">
                    <Select
                      value={overlap}
                      onChange={(event) => setOverlap(event.target.value)}
                      options={[
                        { value: "dominant", label: "Dominant speaker" },
                        { value: "separate", label: "Separate both voices" },
                      ]}
                    />
                  </Field>
                  <Field label="Overlap separator">
                    <Select
                      value={overlapSeparator}
                      onChange={(event) => setOverlapSeparator(event.target.value)}
                      options={[
                        { value: "auto", label: "Auto" },
                        { value: "sepformer", label: "SepFormer" },
                        { value: "none", label: "None" },
                      ]}
                    />
                  </Field>
                  <Field label="Length budget">
                    <Select
                      value={lengthBudget}
                      onChange={(event) => setLengthBudget(event.target.value)}
                      options={[
                        { value: "words", label: "Word budget" },
                        { value: "off", label: "Duration only" },
                      ]}
                    />
                  </Field>
                  <Field label="Budget speedup">
                    <Input value={budgetSpeedup} onChange={(event) => setBudgetSpeedup(event.target.value)} />
                  </Field>
                  <Field label="Refine passes">
                    <Input value={refineMaxPasses} onChange={(event) => setRefineMaxPasses(event.target.value)} />
                  </Field>
                  <Field label="Refine threshold (s)">
                    <Input value={refineThreshold} onChange={(event) => setRefineThreshold(event.target.value)} />
                  </Field>
                  <Field label="No-speech threshold" hint="ASR segments the model distrusts are dropped.">
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
                  <Field label="Device">
                    <Select
                      value={device}
                      onChange={(event) => setDevice(event.target.value)}
                      options={[
                        { value: "auto", label: "Auto" },
                        { value: "cpu", label: "CPU" },
                        ...availableDevices.map((item) => ({
                          value: item.label,
                          label: `${item.label} — ${item.name}`,
                        })),
                      ]}
                    />
                  </Field>
                  <Field label="Precision">
                    <Select
                      value={dtype}
                      onChange={(event) => setDtype(event.target.value)}
                      options={[
                        { value: "auto", label: "Auto" },
                        { value: "bf16", label: "bfloat16" },
                        { value: "fp16", label: "float16" },
                        { value: "fp32", label: "float32" },
                      ]}
                    />
                  </Field>
                  <Field label="Fan-out devices" hint="Comma-separated, e.g. cuda:0,cuda:1">
                    <Input value={devices} onChange={(event) => setDevices(event.target.value)} placeholder="unset" />
                  </Field>
                  <Field label="TTS batch">
                    <Input value={ttsBatch} onChange={(event) => setTtsBatch(event.target.value)} />
                  </Field>
                  <Field label="Whisper model">
                    <Input value={whisperModel} onChange={(event) => setWhisperModel(event.target.value)} />
                  </Field>
                  <Field label="LLM model override">
                    <Input value={llmModel} onChange={(event) => setLlmModel(event.target.value)} placeholder="from settings" />
                  </Field>
                </div>
              </Disclosure>
            </>
          ) : null}

          {error ? <ErrorNote message={error} /> : null}

          <div className="flex items-center justify-between gap-3 border-t border-white/5 pt-4">
            <div className="text-xs text-slate-500">
              {inputName ? `Input: ${inputName}` : "No input selected"}
            </div>
            <Button onClick={submit} disabled={busy || !inputPath}>
              {busy ? <Spinner /> : null}
              {mode === "dub" ? "Start dubbing" : "Start subtitles"}
            </Button>
          </div>
        </Card>
      </Section>
    </div>
  );
}
