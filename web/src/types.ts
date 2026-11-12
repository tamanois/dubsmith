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
  title?: string;
  input_label?: string | null;
  output_label?: string | null;
  job_dir?: string | null;
}

/** A page of a list endpoint, alongside its page of items. */
export interface PageInfo {
  total: number;
  page: number;
  page_size: number;
}

export interface Preset {
  id: string;
  kind: string;
  name: string;
  payload: Record<string, unknown>;
  created: number;
}

export interface Favorite {
  path: string;
  label?: string | null;
  kind: string;
  created: number;
}

export interface Usage {
  sizes: Record<string, number>;
  total: number;
  disk_free: number;
  disk_total: number;
}

export interface SettingsExport {
  version: number;
  settings: Record<string, unknown>;
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

/** One run folder, represented by its main artifact. */
export interface OutputRun {
  job_dir: string;
  job_id?: string | null;
  title: string;
  input_label?: string | null;
  created: number;
  primary: { path: string; name: string; size: number; created: number };
  file_count: number;
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
  theme: string;
  editable: string[];
  [key: string]: unknown;
}
