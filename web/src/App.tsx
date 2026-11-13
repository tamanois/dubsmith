import React from "react";
import { api } from "./api";
import { DefaultsBar } from "./components/DefaultsBar";
import { JobPanel } from "./components/JobPanel";
import { Onboarding } from "./components/Onboarding";
import { ThemeToggle } from "./components/ThemeToggle";
import { DubPage } from "./pages/DubPage";
import { DevicesPage } from "./pages/DevicesPage";
import { FilesPage } from "./pages/FilesPage";
import { JobsPage } from "./pages/JobsPage";
import { ModelsPage } from "./pages/ModelsPage";
import { SayPage } from "./pages/SayPage";
import { SettingsPage } from "./pages/SettingsPage";
import { VoicesPage } from "./pages/VoicesPage";
import type { Device, Job, SettingsInfo } from "./types";
import { cn } from "./ui";

type Page =
  | "activity"
  | "dub"
  | "say"
  | "voices"
  | "models"
  | "devices"
  | "files"
  | "settings";

const NAV: { id: Page; label: string; glyph: string; hint: string }[] = [
  { id: "activity", label: "Activity", glyph: "◷", hint: "Jobs and live logs" },
  { id: "dub", label: "Dub", glyph: "▶", hint: "Translate a video" },
  { id: "say", label: "Say", glyph: "✎", hint: "Generate speech" },
  { id: "voices", label: "Voices", glyph: "◉", hint: "Voice library" },
  { id: "models", label: "Models", glyph: "▦", hint: "Weights" },
  { id: "devices", label: "Devices", glyph: "▤", hint: "Hardware and tools" },
  { id: "files", label: "Files", glyph: "▣", hint: "Inputs and outputs" },
  { id: "settings", label: "Settings", glyph: "⚙", hint: "Defaults" },
];

type Toast = { message: string; tone: "info" | "success" | "error" };

const PAGE_IDS: Page[] = [
  "activity",
  "dub",
  "say",
  "voices",
  "models",
  "devices",
  "files",
  "settings",
];

/** Read the page from the URL hash, so every screen has a shareable address. */
function pageFromHash(): Page {
  if (typeof window === "undefined") return "activity";
  const raw = window.location.hash.replace(/^#\/?/, "").split("?")[0];
  return (PAGE_IDS as string[]).includes(raw) ? (raw as Page) : "activity";
}

function resolveTheme(theme: string): "light" | "dark" {
  if (theme === "light" || theme === "dark") return theme;
  if (typeof window.matchMedia === "function") {
    return window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark";
  }
  return "dark";
}

// The completion watcher runs fast only while something is running; a fully
// idle app (or a background tab) must not keep querying the API.
const ACTIVE_POLL_MS = 2500;
const IDLE_POLL_MS = 15000;

export default function App() {
  const [page, setPage] = React.useState<Page>(() => pageFromHash());
  const [activeJobId, setActiveJobId] = React.useState<string | null>(null);
  const [toast, setToast] = React.useState<Toast | null>(null);
  const [settings, setSettings] = React.useState<SettingsInfo | null>(null);
  const [devices, setDevices] = React.useState<Device[]>([]);
  const [theme, setTheme] = React.useState<string>("system");
  const [editJob, setEditJob] = React.useState<Job | null>(null);
  // Previous state per job, so a completion is announced exactly once.
  const seen = React.useRef<Map<string, string>>(new Map());

  // Keep the visible page and the URL hash in sync in both directions, so a
  // deep link and the browser back/forward buttons both work.
  React.useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const navigate = React.useCallback((next: Page) => {
    const target = `#/${next}`;
    if (window.location.hash !== target) window.location.hash = target;
    setPage(next);
  }, []);

  React.useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 4000);
    return () => window.clearTimeout(timer);
  }, [toast]);

  React.useEffect(() => {
    api
      .settings()
      .then((data) => {
        setSettings(data);
        setTheme(data.theme || "system");
      })
      .catch(() => undefined);
    api
      .devices()
      .then((data) => setDevices(data.devices ?? []))
      .catch(() => undefined);
  }, []);

  // Apply the theme to <html>; "system" follows the OS preference live.
  React.useEffect(() => {
    const root = document.documentElement;
    const apply = () => root.classList.toggle("light", resolveTheme(theme) === "light");
    apply();
    if (theme !== "system" || typeof window.matchMedia !== "function") return;
    const media = window.matchMedia("(prefers-color-scheme: light)");
    media.addEventListener?.("change", apply);
    return () => media.removeEventListener?.("change", apply);
  }, [theme]);

  // Announce background jobs finishing, wherever the user is in the app.
  React.useEffect(() => {
    let alive = true;
    let inFlight = false;
    let timer: number | undefined;

    const schedule = (delay: number) => {
      if (!alive) return;
      window.clearTimeout(timer);
      timer = window.setTimeout(() => void check(), delay);
    };

    const check = async () => {
      if (!alive || inFlight) return;
      if (document.hidden) {
        schedule(IDLE_POLL_MS);
        return;
      }
      inFlight = true;
      let active = false;
      try {
        const data = await api.jobs({ pageSize: 50 });
        if (!alive) return;
        for (const job of data.jobs) {
          const before = seen.current.get(job.id);
          const terminal =
            job.state === "COMPLETED" ||
            job.state === "FAILED" ||
            job.state === "CANCELLED";
          if (!terminal) active = true;
          if (before && before !== job.state && terminal) {
            const label = job.title ?? job.kind;
            setToast(
              job.state === "COMPLETED"
                ? { message: `${label} finished.`, tone: "success" }
                : { message: `${label} ${job.state.toLowerCase()}.`, tone: "error" },
            );
          }
          seen.current.set(job.id, job.state);
        }
      } catch {
        /* a polling failure must never break the UI */
      } finally {
        inFlight = false;
      }
      schedule(active ? ACTIVE_POLL_MS : IDLE_POLL_MS);
    };

    // Returning to the tab refreshes immediately instead of waiting for the idle
    // tick, so a job that finished in the background is announced right away.
    const onVisible = () => {
      if (!document.hidden) void check();
    };

    check();
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      alive = false;
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  function updateSettings(patch: Record<string, unknown>) {
    setSettings((current) => (current ? { ...current, ...patch } : current));
    api
      .patchSettings(patch)
      .then((data) => setSettings(data))
      .catch((exc: unknown) =>
        setToast({
          message: exc instanceof Error ? exc.message : "Could not save the default",
          tone: "error",
        }),
      );
  }

  function changeTheme(next: string) {
    setTheme(next);
    updateSettings({ theme: next });
  }

  function startJob(id: string) {
    setEditJob(null);
    setActiveJobId(id);
    navigate("activity");
    setToast({
      message: "Job accepted. Live progress is shown in the activity panel.",
      tone: "info",
    });
  }

  function editJobFrom(kind: string, job: Job) {
    setEditJob(job);
    setActiveJobId(null);
    navigate(
      kind === "dub" || kind === "subtitles"
        ? "dub"
        : kind === "say"
          ? "say"
          : "voices",
    );
    setToast({ message: "Loaded the job's options for editing.", tone: "info" });
  }

  const current = NAV.find((item) => item.id === page);

  return (
    <div className="flex min-h-screen">
      <aside className="hidden w-64 shrink-0 border-r border-white/5 bg-ink-900/60 px-4 py-6 lg:flex lg:flex-col">
        <div className="px-2">
          <div className="text-lg font-semibold tracking-tight text-white">Dubsmith</div>
          <div className="mt-0.5 text-xs text-slate-500">Local dubbing studio</div>
        </div>
        <nav className="mt-8 space-y-1">
          {NAV.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                navigate(item.id);
                setEditJob(null);
              }}
              className={cn(
                "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition",
                page === item.id
                  ? "bg-accent-500/15 text-white"
                  : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-200",
              )}
            >
              <span
                className={cn(
                  "grid h-7 w-7 place-items-center rounded-md border text-xs",
                  page === item.id
                    ? "border-accent-500/40 bg-accent-500/15 text-accent-300"
                    : "border-white/10 bg-ink-800/60 text-slate-500",
                )}
              >
                {item.glyph}
              </span>
              <span className="min-w-0">
                <span className="block font-medium">{item.label}</span>
                <span className="block truncate text-[11px] text-slate-500">{item.hint}</span>
              </span>
            </button>
          ))}
        </nav>
        <div className="mt-auto rounded-xl border border-white/5 bg-ink-850/60 p-3 text-[11px] leading-relaxed text-slate-500">
          Processing stays local. Voice prints are biometric data: do not share the
          <span className="text-slate-400"> voices/</span> directory.
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 border-b border-white/5 bg-ink-950/80 px-5 py-4 backdrop-blur lg:px-8">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-2 lg:hidden">
                <select
                  aria-label="Page"
                  value={page}
                  onChange={(event) => {
                    navigate(event.target.value as Page);
                    setEditJob(null);
                  }}
                  className="rounded-lg border border-white/10 bg-ink-800 px-2 py-1 text-sm text-slate-200"
                >
                  {NAV.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </div>
              <h1 className="text-xl font-semibold tracking-tight text-white">
                {current?.label}
              </h1>
              <p className="mt-0.5 text-xs text-slate-500">{current?.hint}</p>
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
              <DefaultsBar
                devices={devices}
                device={settings?.default_device ?? "auto"}
                dtype={settings?.default_dtype ?? "auto"}
                onChange={updateSettings}
              />
              <ThemeToggle theme={theme} onChange={changeTheme} />
              <span className="hidden rounded-full border border-emerald-400/20 bg-emerald-400/10 px-2.5 py-1 text-emerald-300 sm:inline">
                local
              </span>
              {activeJobId ? (
                <button
                  type="button"
                  onClick={() => setActiveJobId(activeJobId)}
                  className="rounded-full border border-white/10 px-3 py-1 text-slate-300 hover:border-white/20"
                >
                  current job
                </button>
              ) : null}
            </div>
          </div>
        </header>

        <main className="flex-1 px-5 py-6 lg:px-8 lg:py-8">
          <div className="mx-auto w-full max-w-6xl">
            {page === "activity" ? (
              <JobsPage
                onOpenJob={setActiveJobId}
                activeJobId={activeJobId}
                onOpenFiles={() => navigate("files")}
              />
            ) : null}
            {page === "dub" ? (
              <DubPage onJobStarted={startJob} initialJob={editJob} />
            ) : null}
            {page === "say" ? (
              <SayPage onJobStarted={startJob} initialJob={editJob} />
            ) : null}
            {page === "voices" ? <VoicesPage onJobStarted={startJob} /> : null}
            {page === "models" ? <ModelsPage onJobStarted={startJob} /> : null}
            {page === "devices" ? <DevicesPage /> : null}
            {page === "files" ? <FilesPage /> : null}
            {page === "settings" ? <SettingsPage /> : null}
          </div>
        </main>
      </div>

      {activeJobId ? (
        <JobPanel
          key={activeJobId}
          jobId={activeJobId}
          onClose={() => setActiveJobId(null)}
          onJobStarted={startJob}
          onEdit={editJobFrom}
          onOpenFiles={() => {
            setActiveJobId(null);
            navigate("files");
          }}
        />
      ) : null}

      {toast ? (
        <div
          className={cn(
            "fixed bottom-5 left-1/2 z-50 -translate-x-1/2 rounded-full border px-4 py-2 text-sm shadow-glow",
            toast.tone === "success"
              ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-200"
              : toast.tone === "error"
                ? "border-rose-400/30 bg-rose-400/10 text-rose-200"
                : "border-accent-500/30 bg-ink-850 text-slate-200",
          )}
        >
          {toast.message}
        </div>
      ) : null}

      <Onboarding />
    </div>
  );
}
