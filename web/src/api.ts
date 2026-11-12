import type {
  DeviceReport,
  Favorite,
  FileInfo,
  Job,
  JobEvent,
  ModelInfo,
  OutputFile,
  OutputRun,
  PageInfo,
  Preset,
  SettingsExport,
  SettingsInfo,
  Usage,
  VoiceInfo,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

function query(
  params: Record<string, string | number | boolean | undefined>,
): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

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
  deleteModel: (name: string) =>
    request<{ name: string; freed_bytes: number; warning?: string | null }>(
      `/api/models/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    ),
  usage: () => request<Usage>("/api/usage"),
  downloadModels: (names: string[], force = false) =>
    request<{ job_id: string; state: string }>("/api/models/download", {
      method: "POST",
      body: JSON.stringify({ names, force }),
    }),
  voices: (params: { page?: number; pageSize?: number } = {}) =>
    request<{ voices: VoiceInfo[] } & PageInfo>(
      `/api/voices${query({ page: params.page, page_size: params.pageSize })}`,
    ),
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
  jobs: (
    params: { page?: number; pageSize?: number; state?: string; q?: string } = {},
  ) =>
    request<{ jobs: Job[] } & PageInfo>(
      `/api/jobs${query({
        page: params.page,
        page_size: params.pageSize,
        state: params.state,
        q: params.q,
      })}`,
    ),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  jobInputUrl: (id: string) => `${BASE}/api/jobs/${id}/input`,
  retryJob: (id: string) =>
    request<{ job_id: string; state: string }>(`/api/jobs/${id}/retry`, {
      method: "POST",
    }),
  cancelJob: (id: string) =>
    request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  clearJobs: (ids: string[]) =>
    request<{ removed: number }>("/api/jobs/clear", {
      method: "POST",
      body: JSON.stringify({ ids }),
    }),
  files: (params: { page?: number; pageSize?: number } = {}) =>
    request<{ files: FileInfo[] } & PageInfo>(
      `/api/files${query({ page: params.page, page_size: params.pageSize })}`,
    ),
  deleteFile: (id: string) =>
    request<{ deleted: string }>(`/api/files/${id}`, { method: "DELETE" }),
  deleteFiles: (ids: string[]) =>
    request<{ removed: string[] }>("/api/files/delete", {
      method: "POST",
      body: JSON.stringify({ ids }),
    }),
  outputs: (params: { page?: number; pageSize?: number } = {}) =>
    request<{ output_root: string; runs: OutputRun[] } & PageInfo>(
      `/api/outputs${query({ page: params.page, page_size: params.pageSize })}`,
    ),
  outputFiles: (jobDir: string) =>
    request<{ job_dir: string; files: OutputFile[] }>(
      `/api/outputs/files${query({ job_dir: jobDir })}`,
    ),
  deleteOutput: (path: string) =>
    request<{ deleted: string }>(
      `/api/outputs?path=${encodeURIComponent(path)}`,
      { method: "DELETE" },
    ),
  deleteOutputs: (paths: string[]) =>
    request<{ removed: string[] }>("/api/outputs/delete", {
      method: "POST",
      body: JSON.stringify({ paths }),
    }),
  outputUrl: (path: string) =>
    `${BASE}/api/outputs/download?path=${encodeURIComponent(path)}`,
  outputInlineUrl: (path: string) =>
    `${BASE}/api/outputs/download?path=${encodeURIComponent(path)}&inline=1`,
  presets: (kind?: string) =>
    request<{ presets: Preset[] }>(`/api/presets${query({ kind })}`),
  createPreset: (body: {
    kind: string;
    name: string;
    payload: Record<string, unknown>;
  }) =>
    request<{ preset: Preset }>("/api/presets", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deletePreset: (id: string) =>
    request<{ deleted: string }>(`/api/presets/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),
  favorites: () => request<{ favorites: Favorite[] }>("/api/favorites"),
  addFavorite: (body: { path: string; label?: string; kind?: string }) =>
    request<{ favorite: Favorite }>("/api/favorites", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deleteFavorite: (path: string) =>
    request<{ deleted: string }>(
      `/api/favorites${query({ path })}`,
      { method: "DELETE" },
    ),
  settings: () => request<SettingsInfo>("/api/settings"),
  exportSettings: () => request<SettingsExport>("/api/settings/export"),
  importSettings: (document: SettingsExport) =>
    request<SettingsInfo>("/api/settings/import", {
      method: "POST",
      body: JSON.stringify(document),
    }),
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
  fileInlineUrl: (id: string) => `${BASE}/api/files/${id}/download?inline=1`,
  voiceAudioUrl: (label: string) =>
    `${BASE}/api/voices/${encodeURIComponent(label)}/audio`,
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
  let lastEventId = 0;
  let closed = false;
  let retries = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let source: EventSource | null = null;

  const connect = () => {
    if (closed) return;
    const current = new EventSource(
      `${BASE}/api/jobs/${id}/events?after=${lastEventId}`,
    );
    source = current;

    for (const name of EVENT_NAMES) {
      current.addEventListener(name, (raw) => {
        const event = raw as MessageEvent<string>;
        try {
          const data = JSON.parse(event.data) as JobEvent | Job;
          if (name === "job") {
            closed = true;
            onJob(data as Job);
            current.close();
          } else {
            lastEventId = Math.max(lastEventId, (data as JobEvent).id ?? 0);
            onEvent(data as JobEvent);
          }
        } catch {
          /* ignore malformed frames */
        }
      });
    }

    // A dropped stream must not take the cancel button with it: the panel keeps
    // working from HTTP state while this reconnects with backoff, resuming from
    // the last event id so no log line or stage change is missed.
    current.onerror = () => {
      onError?.("Lost the live connection; the job may still be running.");
      current.close();
      if (closed) return;
      retries += 1;
      const delay = Math.min(1000 * 2 ** (retries - 1), 15000);
      timer = setTimeout(connect, delay);
    };
  };

  connect();

  return () => {
    closed = true;
    if (timer !== null) clearTimeout(timer);
    source?.close();
  };
}
