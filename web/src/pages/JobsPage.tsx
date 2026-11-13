import React from "react";
import { api } from "../api";
import type { Job } from "../types";
import {
  Badge,
  Button,
  Card,
  ConfirmDialog,
  Empty,
  ErrorNote,
  Input,
  Pagination,
  Section,
  Select,
  formatDate,
  formatSeconds,
} from "../ui";

const PAGE_SIZE = 20;
// Refresh quickly only while a job is running; an idle list does not need a
// request every few seconds, and a background tab needs none at all.
const ACTIVE_POLL_MS = 2500;
const IDLE_POLL_MS = 15000;
const STATE_OPTIONS = [
  { value: "", label: "All states" },
  { value: "QUEUED", label: "Queued" },
  { value: "PROCESSING", label: "Processing" },
  { value: "COMPLETED", label: "Completed" },
  { value: "FAILED", label: "Failed" },
  { value: "CANCELLED", label: "Cancelled" },
];

function isActive(job: Job): boolean {
  return job.state === "QUEUED" || job.state === "PROCESSING";
}

/** The one deliverable a finished job produced, in preference order. */
function primaryPath(job: Job): string | null {
  const result = (job.result ?? {}) as Record<string, unknown>;
  for (const key of [
    "dubbed_video",
    "burned_video",
    "output_video",
    "dub_track",
    "target_srt",
    "output",
  ]) {
    const value = result[key];
    if (typeof value === "string" && value) return value;
  }
  return null;
}

export function JobsPage({
  onOpenJob,
  activeJobId,
  onOpenFiles,
}: {
  onOpenJob: (id: string) => void;
  activeJobId: string | null;
  onOpenFiles?: () => void;
}) {
  const [jobs, setJobs] = React.useState<Job[]>([]);
  const [total, setTotal] = React.useState(0);
  const [page, setPage] = React.useState(1);
  const [stateFilter, setStateFilter] = React.useState("");
  const [queryInput, setQueryInput] = React.useState("");
  const [query, setQuery] = React.useState("");
  const [selected, setSelected] = React.useState<Set<string>>(new Set());
  const [confirming, setConfirming] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [tick, setTick] = React.useState(0);

  // Debounce the search so typing does not fire a request per keystroke.
  React.useEffect(() => {
    const timer = window.setTimeout(() => {
      setQuery(queryInput.trim());
      setPage(1);
    }, 300);
    return () => window.clearTimeout(timer);
  }, [queryInput]);

  // Load the current page, then keep it fresh: fast while a job runs, slow when
  // everything is finished, silent while the tab is hidden.
  React.useEffect(() => {
    let alive = true;
    let inFlight = false;
    let timer: number | undefined;

    const schedule = (delay: number) => {
      if (!alive) return;
      window.clearTimeout(timer);
      timer = window.setTimeout(() => void load(), delay);
    };

    const load = async () => {
      if (!alive || inFlight) return;
      if (document.hidden) {
        schedule(IDLE_POLL_MS);
        return;
      }
      inFlight = true;
      let active = false;
      try {
        const data = await api.jobs({
          page,
          pageSize: PAGE_SIZE,
          state: stateFilter || undefined,
          q: query || undefined,
        });
        if (!alive) return;
        setJobs(data.jobs);
        setTotal(Number(data.total) || 0);
        setError(null);
        active = data.jobs.some(isActive);
      } catch (exc: unknown) {
        if (!alive) return;
        setError(exc instanceof Error ? exc.message : "Could not load jobs");
      } finally {
        inFlight = false;
      }
      schedule(active ? ACTIVE_POLL_MS : IDLE_POLL_MS);
    };

    const onVisible = () => {
      if (!document.hidden) void load();
    };

    load();
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      alive = false;
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [page, stateFilter, query, tick]);

  // Force an immediate refresh after a mutation without waiting for the tick.
  const refresh = React.useCallback(() => {
    setTick((value) => value + 1);
  }, []);

  const selectable = jobs.filter((job) => !isActive(job));
  const allSelected =
    selectable.length > 0 && selectable.every((job) => selected.has(job.id));

  function toggle(id: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleAll() {
    setSelected(allSelected ? new Set() : new Set(selectable.map((job) => job.id)));
  }

  async function clearSelected() {
    const ids = [...selected];
    if (!ids.length) return;
    setBusy(true);
    setError(null);
    try {
      await api.clearJobs(ids);
      setSelected(new Set());
      setConfirming(false);
      refresh();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not clear the jobs");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Section
      title="Activity"
      subtitle="Every accepted job is durable and cancellable. Finished jobs can be retried or edited."
      right={
        <div className="flex gap-2">
          <Input
            aria-label="Search jobs"
            placeholder="Search input or id"
            value={queryInput}
            onChange={(event) => setQueryInput(event.target.value)}
            className="w-44 py-1 text-xs"
          />
          <Select
            aria-label="Filter by state"
            value={stateFilter}
            onChange={(event) => {
              setStateFilter(event.target.value);
              setPage(1);
              setSelected(new Set());
            }}
            options={STATE_OPTIONS}
            className="w-36 py-1 text-xs"
          />
          <Button variant="ghost" onClick={refresh}>
            Refresh
          </Button>
        </div>
      }
    >
      <div className="space-y-4">
        {error ? <ErrorNote message={error} /> : null}
        {selected.size > 0 ? (
          <div className="flex items-center justify-between gap-3 rounded-lg border border-white/10 bg-ink-850/60 px-3 py-2 text-xs text-slate-400">
            <span>{selected.size} selected</span>
            <div className="flex gap-2">
              {allSelected ? null : (
                <Button variant="ghost" className="px-2 py-1 text-xs" onClick={toggleAll}>
                  Select all
                </Button>
              )}
              <Button
                variant="danger"
                className="px-2 py-1 text-xs"
                onClick={() => setConfirming(true)}
              >
                Clear selected
              </Button>
            </div>
          </div>
        ) : null}
        {jobs.length === 0 ? (
          <Empty
            text={
              query || stateFilter
                ? "No jobs match this filter."
                : "No jobs yet. Start a dub, a voice clone or a speech generation."
            }
            action={
              query || stateFilter ? null : (
                <Button variant="ghost" onClick={refresh}>
                  Check again
                </Button>
              )
            }
          />
        ) : (
          <Card className="overflow-hidden">
            <div className="flex items-center gap-3 border-b border-white/5 px-4 py-2 text-[11px] uppercase tracking-wide text-slate-500">
              <input
                type="checkbox"
                aria-label="Select all jobs"
                checked={allSelected}
                disabled={selectable.length === 0}
                onChange={toggleAll}
              />
              <span className="flex-1">Job</span>
            </div>
            <div className="divide-y divide-white/5">
              {jobs.map((job) => {
                const stage = job.stage ? ` · ${job.stage}` : "";
                const duration =
                  job.started && job.finished ? job.finished - job.started : null;
                const output = primaryPath(job);
                // The output name leads; the operation type is the subtitle.
                const primaryLabel =
                  job.output_label ?? job.input_label ?? job.title ?? job.kind;
                return (
                  <div
                    key={job.id}
                    data-testid={`job-row-${job.id}`}
                    className={
                      "flex items-center gap-3 px-4 py-3 " +
                      (activeJobId === job.id ? "bg-accent-500/5" : "")
                    }
                  >
                    <input
                      type="checkbox"
                      aria-label={`Select ${job.title ?? job.kind}`}
                      checked={selected.has(job.id)}
                      disabled={isActive(job)}
                      onChange={() => toggle(job.id)}
                    />
                    <button
                      type="button"
                      onClick={() => onOpenJob(job.id)}
                      className="min-w-0 flex-1 text-left"
                    >
                      <div className="flex items-center gap-2">
                        <span className="truncate text-sm font-medium text-slate-200">
                          {primaryLabel}
                        </span>
                        <Badge state={job.state} />
                      </div>
                      <div className="mt-0.5 truncate text-xs text-slate-500">
                        <span>{job.title ?? job.kind}</span>
                        <span>{` · ${job.input_label ?? job.id.slice(0, 12)}`}</span>
                        <span>{` · ${formatDate(job.created)}`}</span>
                        {duration ? <span>{` · ${formatSeconds(duration)}`}</span> : null}
                        {stage ? <span>{stage}</span> : null}
                      </div>
                      {job.error ? (
                        <div className="mt-1 truncate text-xs text-rose-300">{job.error}</div>
                      ) : null}
                    </button>
                    {output ? (
                      <>
                        <a
                          href={api.outputUrl(output)}
                          className="rounded-lg border border-white/10 px-2 py-1 text-xs text-slate-300 hover:border-white/20"
                        >
                          Download
                        </a>
                        {onOpenFiles ? (
                          <button
                            type="button"
                            onClick={onOpenFiles}
                            className="rounded-lg border border-white/10 px-2 py-1 text-xs text-slate-300 hover:border-white/20"
                          >
                            Files
                          </button>
                        ) : null}
                      </>
                    ) : (
                      <span className="text-xs text-slate-500">Open</span>
                    )}
                  </div>
                );
              })}
            </div>
          </Card>
        )}
        <Pagination
          page={page}
          pageSize={PAGE_SIZE}
          total={total}
          onPage={(next) => {
            setPage(next);
            setSelected(new Set());
          }}
        />
      </div>

      <ConfirmDialog
        open={confirming}
        title="Clear selected jobs"
        message={`Delete ${selected.size} finished job record${
          selected.size === 1 ? "" : "s"
        }? Logs and job history are removed; output files on disk are kept.`}
        confirmLabel="Clear jobs"
        danger
        busy={busy}
        onConfirm={clearSelected}
        onCancel={() => setConfirming(false)}
      />
    </Section>
  );
}
