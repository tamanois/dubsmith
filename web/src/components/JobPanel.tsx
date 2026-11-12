import React from "react";
import { api, subscribeJob } from "../api";
import type { Job, JobEvent } from "../types";
import { Badge, Button, Card, ErrorNote, Spinner, formatSeconds } from "../ui";

export function JobPanel({
  jobId,
  onClose,
}: {
  jobId: string;
  onClose: () => void;
}) {
  const [job, setJob] = React.useState<Job | null>(null);
  const [logs, setLogs] = React.useState<JobEvent[]>([]);
  const [open, setOpen] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let alive = true;
    setJob(null);
    setLogs([]);
    setError(null);
    api
      .job(jobId)
      .then((row) => {
        if (!alive) return;
        setJob(row);
        setLogs(row.events ?? []);
      })
      .catch((exc: unknown) =>
        setError(exc instanceof Error ? exc.message : "Could not load the job"),
      );
    const close = subscribeJob(
      jobId,
      (event) => {
        setLogs((current) => [...current, event].slice(-300));
      },
      (finished) => setJob(finished),
      (message) => setError(message),
    );
    return () => {
      alive = false;
      close();
    };
  }, [jobId]);

  const active = job?.state === "QUEUED" || job?.state === "PROCESSING";
  const terminal = job && !active;

  async function cancel() {
    try {
      await api.cancelJob(jobId);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not cancel");
    }
  }

  const result = (job?.result ?? {}) as Record<string, unknown>;
  const resultPaths = Object.entries(result).filter(
    ([key, value]) =>
      typeof value === "string" &&
      ["job_dir", "dub_track", "dubbed_video", "target_srt", "output", "lines_dir"].includes(key),
  );

  return (
    <Card className="fixed bottom-5 right-5 z-40 w-[380px] max-w-[calc(100vw-2rem)] overflow-hidden">
      <div className="flex items-center justify-between gap-2 border-b border-white/5 px-4 py-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-semibold text-white">
              {job?.kind ?? "Job"}
            </span>
            {job ? <Badge state={job.state} /> : null}
          </div>
          <div className="mt-0.5 text-xs text-slate-500">
            {job?.stage ? `Stage: ${job.stage}` : jobId.slice(0, 10)}
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded-md px-2 py-1 text-slate-400 hover:bg-white/5 hover:text-white"
          aria-label="Close job panel"
        >
          ✕
        </button>
      </div>

      <div className="space-y-3 px-4 py-3">
        {error ? <ErrorNote message={error} /> : null}
        {active ? (
          <div className="flex items-center gap-2 text-sm text-slate-300">
            <Spinner />
            <span>Working…</span>
          </div>
        ) : null}
        {job?.error ? <ErrorNote message={job.error} /> : null}
        {terminal && !job?.error && job?.state === "CANCELLED" ? (
          <div className="text-sm text-amber-300">Cancelled.</div>
        ) : null}
        {terminal && !job?.error && job?.state === "COMPLETED" ? (
          <div className="text-sm text-emerald-300">Finished successfully.</div>
        ) : null}
        {resultPaths.length ? (
          <div className="space-y-1 rounded-lg border border-white/5 bg-ink-850/60 p-2 text-xs">
            {resultPaths.map(([key, value]) => (
              <div key={key} className="flex gap-2">
                <span className="w-24 shrink-0 text-slate-500">{key}</span>
                <span className="break-all text-slate-300">{String(value)}</span>
              </div>
            ))}
            {result.stage_seconds ? (
              <div className="mt-1 text-slate-500">
                Stages:{" "}
                {Object.entries(result.stage_seconds as Record<string, number>)
                  .map(([name, seconds]) => `${name} ${formatSeconds(seconds)}`)
                  .join(", ")}
              </div>
            ) : null}
          </div>
        ) : null}

        <div className="flex items-center justify-between gap-2">
          <button
            type="button"
            onClick={() => setOpen((value) => !value)}
            className="text-xs text-accent-400 hover:text-accent-300"
          >
            {open ? "Hide log" : `Show log (${logs.length})`}
          </button>
          {active ? (
            <Button variant="danger" onClick={cancel} className="px-3 py-1 text-xs">
              Cancel
            </Button>
          ) : null}
        </div>

        {open ? (
          <div className="max-h-56 overflow-auto rounded-lg border border-white/5 bg-black/30 p-2 font-mono text-[11px] leading-relaxed text-slate-400">
            {logs.length === 0 ? (
              <div className="text-slate-600">No log lines yet.</div>
            ) : (
              logs.map((line, index) => (
                <div key={`${line.id}-${index}`} className="whitespace-pre-wrap break-words">
                  {line.kind === "log" ? (
                    <span className="text-slate-400">{line.message}</span>
                  ) : line.kind === "error" ? (
                    <span className="text-rose-300">{line.message}</span>
                  ) : (
                    <span className="text-sky-300">{line.message || line.kind}</span>
                  )}
                </div>
              ))
            )}
          </div>
        ) : null}
      </div>
    </Card>
  );
}
