export type JobState =
  | "QUEUED"
  | "PROCESSING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED";

export interface JobEvent {
  id: number;
  job_id: string;
  ts: number;
  kind: string;
  message?: string | null;
  data?: Record<string, unknown> | null;
}

export interface Job {
  id: string;
  kind: string;
  state: JobState;
  stage?: string | null;
  params: Record<string, unknown>;
  result?: Record<string, unknown> | null;
  error?: string | null;
  created: number;
  started?: number | null;
  finished?: number | null;
  events?: JobEvent[];
}

export interface Device {
  label: string;
  kind: string;
  name: string;
  total_memory?: number | null;
  supports_bf16: boolean;
  supports_fp16: boolean;
  asr_device?: string;
  tts_dtype?: string;
}

export interface DeviceReport {
  torch: string;
  devices: Device[];
  note: string;
}

export interface ModelInfo {
  name: string;
  repo_id: string;
  description: string;
  license: string;
  required: boolean;
  present: boolean;
}

export interface VoiceInfo {
  label: string;
  language: string;
  source: string;
  seconds: number;
  gender: string;
  rate_band: string;
  instruct?: string | null;
  has_transcript: boolean;
  cards: number;
}

export interface OutputFile {
  path: string;
  relative: string;
  name: string;
  size: number;
  created: number;
}

export interface FileInfo {
  id: string;
  original_name: string;
  stored_name: string;
  path: string;
  size: number;
  kind: string;
  created: number;
}

export interface SettingsInfo {
  llm_base_url: string;
  llm_model: string;
  llm_timeout_s: number;
  llm_chunk_size: number;
  llm_plan_chunk_size: number;
  loudness_target: string;
  target_lufs: number;
  keep_background: boolean;
  separator: string;
  duck_db: number;
  models_dir: string;
  voices_dir: string;
  log_level: string;
  log_json: boolean;
  llm_api_key_set: boolean;
  default_device: string;
  default_dtype: string;
  editable: string[];
  [key: string]: unknown;
}
