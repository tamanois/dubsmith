import type {
  DeviceReport,
  FileInfo,
  Job,
  JobEvent,
  ModelInfo,
  OutputFile,
  SettingsInfo,
  VoiceInfo,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) detail = payload.detail;
    } catch {
      /* keep the status message */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export const api = {
  devices: () => request<DeviceReport>("/api/devices"),
  packages: () =>
    request<{
      optional_packages: { name: string; available: boolean; purpose: string }[];
      system_binaries: { name: string; path: string | null; purpose: string }[];
    }>("/api/packages"),
  models: () => request<{ models: ModelInfo[] }>("/api/models"),
  downloadModels: (names: string[], force = false) =>
    request<{ job_id: string; state: string }>("/api/models/download", {
      method: "POST",
      body: JSON.stringify({ names, force }),
    }),
  voices: () => request<{ voices: VoiceInfo[] }>("/api/voices"),
  designVoice: (body: Record<string, unknown>) =>
    request<{ job_id: string; state: string }>("/api/voices/design", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  cloneVoice: (body: Record<string, unknown>) =>
    request<{ job_id: string; state: string }>("/api/voices/clone", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  addVoice: (body: Record<string, unknown>) =>
    request<{ job_id: string; state: string }>("/api/voices/add", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deleteVoice: (label: string) =>
    request<{ removed: string }>(`/api/voices/${encodeURIComponent(label)}`, {
      method: "DELETE",
    }),
  anchors: () =>
    request<{ anchors: { key: string; path: string; seconds: number }[] }>(
      "/api/anchors",
    ),
  generateAnchors: (languages: string[], force = false) =>
    request<{ job_id: string; state: string }>("/api/anchors/generate", {
      method: "POST",
      body: JSON.stringify({ languages, force }),
    }),
  submitDub: (config: Record<string, unknown>, stretcher = "auto") =>
    request<{ job_id: string; state: string }>(
      `/api/jobs/dub?stretcher=${encodeURIComponent(stretcher)}`,
      { method: "POST", body: JSON.stringify(config) },
    ),
  submitSay: (spec: Record<string, unknown>) =>
    request<{ job_id: string; state: string }>("/api/jobs/say", {
      method: "POST",
      body: JSON.stringify(spec),
    }),
  submitSubtitles: (config: Record<string, unknown>) =>
    request<{ job_id: string; state: string }>("/api/jobs/subtitles", {
      method: "POST",
      body: JSON.stringify(config),
    }),
  jobs: () => request<{ jobs: Job[] }>("/api/jobs?limit=40"),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  cancelJob: (id: string) =>
    request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  files: () => request<{ files: FileInfo[] }>("/api/files"),
  deleteFile: (id: string) =>
    request<{ deleted: string }>(`/api/files/${id}`, { method: "DELETE" }),
  outputs: () =>
    request<{ output_root: string; files: OutputFile[] }>("/api/outputs"),
  deleteOutput: (path: string) =>
    request<{ deleted: string }>(
      `/api/outputs?path=${encodeURIComponent(path)}`,
      { method: "DELETE" },
    ),
  outputUrl: (path: string) =>
    `${BASE}/api/outputs/download?path=${encodeURIComponent(path)}`,
  settings: () => request<SettingsInfo>("/api/settings"),
  patchSettings: (values: Record<string, unknown>) =>
    request<SettingsInfo>("/api/settings", {
      method: "PATCH",
      body: JSON.stringify({ values }),
    }),
  upload: async (file: File): Promise<{ id: string; name: string; path: string; size: number }> => {
    const form = new FormData();
    form.append("file", file);
    const response = await fetch(`${BASE}/api/files/upload`, {
      method: "POST",
      body: form,
    });
    if (!response.ok) {
      const payload = (await response.json().catch(() => ({}))) as {
        detail?: string;
      };
      throw new Error(payload.detail ?? "Upload failed");
    }
    return response.json();
  },
  fileUrl: (id: string) => `${BASE}/api/files/${id}/download`,
};

const EVENT_NAMES = [
  "state",
  "log",
  "stage_start",
  "stage_end",
  "stage_cached",
  "stage_failed",
  "download",
  "error",
  "job",
];

export function subscribeJob(
  id: string,
  onEvent: (event: JobEvent) => void,
  onJob: (job: Job) => void,
  onError?: (message: string) => void,
): () => void {
  const source = new EventSource(`${BASE}/api/jobs/${id}/events`);
  for (const name of EVENT_NAMES) {
    source.addEventListener(name, (raw) => {
      const event = raw as MessageEvent<string>;
      try {
        const data = JSON.parse(event.data) as JobEvent | Job;
        if (name === "job") {
          onJob(data as Job);
          source.close();
        } else {
          onEvent(data as JobEvent);
        }
      } catch {
        /* ignore malformed frames */
      }
    });
  }
  source.onerror = () => {
    onError?.("Lost the live connection; the job may still be running.");
    source.close();
  };
  return () => source.close();
}
