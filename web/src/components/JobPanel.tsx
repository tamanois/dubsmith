import React from "react";
import { api, subscribeJob } from "../api";
import type { Job, JobEvent } from "../types";
import { Badge, Button, Card, ErrorNote, Spinner, cn, formatSeconds } from "../ui";
import { CompareDialog } from "./CompareDialog";

export function JobPanel({
  jobId,
  onClose,
  onJobStarted,
  onEdit,
}: {
  jobId: string;
  onClose: () => void;
  onJobStarted?: (id: string) => void;
  onEdit?: (kind: string, job: Job) => void;
}) {
  const [job, setJob] = React.useState<Job | null>(null);
  const [logs, setLogs] = React.useState<JobEvent[]>([]);
  const [open, setOpen] = React.useState(false);
  const [wide, setWide] = React.useState(false);
  const [retrying, setRetrying] = React.useState(false);
  const [comparing, setComparing] = React.useState(false);
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
      const row = await api.cancelJob(jobId);
      // The cancel endpoint is plain HTTP, so it still works after the live
      // stream drops. Reflecting the returned row keeps the panel truthful even
      // when the SSE connection is gone and no terminal event will arrive.
      if (row && row.state) setJob(row);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not cancel");
    }
  }

  async function retry() {
    setRetrying(true);
    setError(null);
    try {
      const next = await api.retryJob(jobId);
      if (onJobStarted) onJobStarted(next.job_id);
      else onClose();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not retry");
    } finally {
      setRetrying(false);
    }
  }

  const result = (job?.result ?? {}) as Record<string, unknown>;
  const resultPaths = Object.entries(result).filter(
    ([key, value]) =>
      typeof value === "string" &&
      ["job_dir", "dub_track", "dubbed_video", "target_srt", "output", "lines_dir"].includes(key),
  );
  const canCompare =
    Boolean(terminal) &&
    (typeof result.dubbed_video === "string" || typeof result.dub_track === "string");

  return (
    <>
      <Card
        className={cn(
          "fixed bottom-5 right-5 z-40 max-w-[calc(100vw-2rem)] overflow-hidden transition-all",
          wide ? "w-[720px]" : "w-[380px]",
        )}
      >
      <div className="flex items-center justify-between gap-2 border-b border-white/5 px-4 py-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-semibold text-white">
              {job?.title ?? job?.kind ?? "Job"}
            </span>
            {job ? <Badge state={job.state} /> : null}
          </div>
          <div className="mt-0.5 truncate text-xs text-slate-500">
            {job?.input_label ? `${job.input_label}` : jobId.slice(0, 10)}
            {job?.output_label ? ` → ${job.output_label}` : ""}
            {job?.stage ? ` · ${job.stage}` : ""}
          </div>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setWide((value) => !value)}
            aria-label={wide ? "Shrink job panel" : "Expand job panel"}
            aria-expanded={wide}
            className="rounded-md px-2 py-1 text-slate-400 hover:bg-white/5 hover:text-white"
          >
            {wide ? "⤡" : "⤢"}
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md px-2 py-1 text-slate-400 hover:bg-white/5 hover:text-white"
            aria-label="Close job panel"
          >
            ✕
          </button>
        </div>
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
          <div className="flex items-center gap-2">
            {canCompare ? (
              <Button
                variant="ghost"
                onClick={() => setComparing(true)}
                className="px-3 py-1 text-xs"
              >
                Compare
              </Button>
            ) : null}
            {terminal ? (
              <Button
                variant="ghost"
                onClick={retry}
                disabled={retrying}
                className="px-3 py-1 text-xs"
              >
                {retrying ? <Spinner /> : null}
                Retry
              </Button>
            ) : null}
            {terminal && onEdit && job ? (
              <Button
                variant="ghost"
                onClick={() => onEdit(job.kind, job)}
                className="px-3 py-1 text-xs"
              >
                Edit
              </Button>
            ) : null}
            {active ? (
              <Button variant="danger" onClick={cancel} className="px-3 py-1 text-xs">
                Cancel
              </Button>
            ) : null}
          </div>
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

      {comparing && job ? (
        <CompareDialog job={job} onClose={() => setComparing(false)} />
      ) : null}
    </>
  );
}
