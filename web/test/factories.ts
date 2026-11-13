import type {
  BackendCapabilities,
  Device,
  DeviceReport,
  FileInfo,
  Job,
  JobEvent,
  ModelInfo,
  OutputFile,
  SettingsInfo,
  VoiceInfo,
} from "../src/types";

export function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: "job-1",
    kind: "dub",
    state: "COMPLETED",
    stage: "done",
    params: {},
    result: {},
    error: null,
    created: 1_700_000_000,
    started: 1_700_000_010,
    finished: 1_700_000_020,
    events: [],
    ...overrides,
  };
}

export function makeJobEvent(overrides: Partial<JobEvent> = {}): JobEvent {
  return {
    id: 1,
    job_id: "job-1",
    ts: 1_700_000_001,
    kind: "log",
    message: "A log line",
    data: null,
    ...overrides,
  };
}

export function makeVoice(overrides: Partial<VoiceInfo> = {}): VoiceInfo {
  return {
    label: "warm-fr",
    language: "French",
    source: "design",
    seconds: 12.5,
    gender: "f",
    rate_band: "normal",
    instruct: "a warm French voice",
    has_transcript: true,
    cards: 1,
    ...overrides,
  };
}

export function makeModel(overrides: Partial<ModelInfo> = {}): ModelInfo {
  return {
    name: "tts_base",
    repo_id: "Qwen/Qwen3-TTS",
    description: "Voice cloning base model",
    license: "Apache-2.0",
    required: true,
    present: true,
    ...overrides,
  };
}

export function makeDevice(overrides: Partial<Device> = {}): Device {
  return {
    label: "cuda:0",
    kind: "cuda",
    name: "NVIDIA GeForce RTX 3060",
    total_memory: 12 * 1024 ** 3,
    supports_bf16: true,
    supports_fp16: true,
    asr_device: "cuda",
    tts_dtype: "bf16",
    ...overrides,
  };
}

export function makeDeviceReport(overrides: Partial<DeviceReport> = {}): DeviceReport {
  return {
    torch: "2.5.1 (CUDA 12.4)",
    devices: [makeDevice()],
    note: "Vulkan is not a backend here.",
    ...overrides,
  };
}

export function makeSettings(overrides: Partial<SettingsInfo> = {}): SettingsInfo {
  return {
    llm_base_url: "http://127.0.0.1:8080/v1",
    llm_model: "local-model",
    llm_timeout_s: 180,
    llm_chunk_size: 40,
    llm_plan_chunk_size: 10,
    loudness_target: "match-source",
    target_lufs: -16,
    keep_background: true,
    separator: "auto",
    duck_db: 6,
    models_dir: "models",
    voices_dir: "voices",
    log_level: "INFO",
    log_json: false,
    llm_api_key_set: false,
    default_device: "auto",
    default_dtype: "auto",
    theme: "system",
    editable: [],
    ...overrides,
  };
}

export function makeFile(overrides: Partial<FileInfo> = {}): FileInfo {
  return {
    id: "file-1",
    original_name: "clip.mp4",
    stored_name: "file-1__clip.mp4",
    path: "data/uploads/file-1__clip.mp4",
    size: 2048,
    kind: "input",
    created: 1_700_000_000,
    ...overrides,
  };
}

export function makeOutput(overrides: Partial<OutputFile> = {}): OutputFile {
  return {
    path: "/output/run/subs_target.srt",
    relative: "run/subs_target.srt",
    name: "subs_target.srt",
    size: 128,
    created: 1_700_000_000,
    ...overrides,
  };
}

export function makeCapabilities(
  overrides: Partial<BackendCapabilities> = {},
): BackendCapabilities {
  // Defaults describe Qwen Base: a saved-voice cloning backend with no controls
  // beyond the voice itself.
  return {
    id: "qwen",
    label: "Qwen3-TTS Base",
    prosody: false,
    style: false,
    inline_tags: false,
    fish_tuning: false,
    chatterbox_tuning: false,
    speaker: false,
    sources: ["voice", "clone", "design"],
    save_as: true,
    expressive: false,
    unsupported: {
      prosody: "--prosody is unavailable with the Qwen Base backend.",
      expressive: "Only expressive backends have an accent style.",
    },
    ...overrides,
  };
}

/** A payload that mirrors the real capability table closely enough to gate the UI. */
export function makeBackends(): {
  say: BackendCapabilities[];
  dub: BackendCapabilities[];
} {
  const expressive = {
    prosody: true,
    expressive: true,
    unsupported: {} as Record<string, string>,
  };
  return {
    say: [
      makeCapabilities(),
      makeCapabilities({
        id: "customvoice",
        label: "Qwen3 CustomVoice",
        prosody: true,
        style: true,
        speaker: true,
        sources: [],
        unsupported: {},
      }),
      makeCapabilities({ id: "cosyvoice3", label: "CosyVoice3", ...expressive }),
      makeCapabilities({
        id: "chatterbox",
        label: "Chatterbox",
        chatterbox_tuning: true,
        ...expressive,
      }),
      makeCapabilities({
        id: "fish",
        label: "Fish Speech",
        inline_tags: true,
        fish_tuning: true,
        save_as: false,
        unsupported: { save_as: "Fish cannot save a voice from a one-off run." },
      }),
    ],
    dub: [
      makeCapabilities(),
      makeCapabilities({ id: "cosyvoice3", label: "CosyVoice3", ...expressive }),
      makeCapabilities({
        id: "chatterbox",
        label: "Chatterbox",
        chatterbox_tuning: true,
        ...expressive,
      }),
      makeCapabilities({
        id: "fish",
        label: "Fish Speech",
        inline_tags: true,
        fish_tuning: true,
        ...expressive,
      }),
    ],
  };
}
