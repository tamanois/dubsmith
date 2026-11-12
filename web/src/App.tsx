import React from "react";
import { JobPanel } from "./components/JobPanel";
import { DubPage } from "./pages/DubPage";
import { DevicesPage } from "./pages/DevicesPage";
import { FilesPage } from "./pages/FilesPage";
import { JobsPage } from "./pages/JobsPage";
import { ModelsPage } from "./pages/ModelsPage";
import { SayPage } from "./pages/SayPage";
import { SettingsPage } from "./pages/SettingsPage";
import { VoicesPage } from "./pages/VoicesPage";
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

export default function App() {
  const [page, setPage] = React.useState<Page>("activity");
  const [activeJobId, setActiveJobId] = React.useState<string | null>(null);
  const [toast, setToast] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 4000);
    return () => window.clearTimeout(timer);
  }, [toast]);

  function startJob(id: string) {
    setActiveJobId(id);
    setPage("activity");
    setToast("Job accepted. Live progress is shown in the activity panel.");
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
              onClick={() => setPage(item.id)}
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
                  value={page}
                  onChange={(event) => setPage(event.target.value as Page)}
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
            <div className="flex items-center gap-2 text-xs text-slate-500">
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
              <JobsPage onOpenJob={setActiveJobId} activeJobId={activeJobId} />
            ) : null}
            {page === "dub" ? <DubPage onJobStarted={startJob} /> : null}
            {page === "say" ? <SayPage onJobStarted={startJob} /> : null}
            {page === "voices" ? <VoicesPage onJobStarted={startJob} /> : null}
            {page === "models" ? <ModelsPage onJobStarted={startJob} /> : null}
            {page === "devices" ? <DevicesPage /> : null}
            {page === "files" ? <FilesPage /> : null}
            {page === "settings" ? <SettingsPage /> : null}
          </div>
        </main>
      </div>

      {activeJobId ? (
        <JobPanel jobId={activeJobId} onClose={() => setActiveJobId(null)} />
      ) : null}

      {toast ? (
        <div className="fixed bottom-5 left-1/2 z-50 -translate-x-1/2 rounded-full border border-accent-500/30 bg-ink-850 px-4 py-2 text-sm text-slate-200 shadow-glow">
          {toast}
        </div>
      ) : null}
    </div>
  );
}
