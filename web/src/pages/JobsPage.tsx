import React from "react";
import { api } from "../api";
import type { Job } from "../types";
import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorNote,
  Section,
  formatDate,
  formatSeconds,
} from "../ui";

export function JobsPage({
  onOpenJob,
  activeJobId,
}: {
  onOpenJob: (id: string) => void;
  activeJobId: string | null;
}) {
  const [jobs, setJobs] = React.useState<Job[]>([]);
  const [error, setError] = React.useState<string | null>(null);

  const refresh = React.useCallback(() => {
    api
      .jobs()
      .then((data) => setJobs(data.jobs))
      .catch((exc: unknown) =>
        setError(exc instanceof Error ? exc.message : "Could not load jobs"),
      );
  }, []);

  React.useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 2500);
    return () => window.clearInterval(timer);
  }, [refresh]);

  return (
    <Section
      title="Activity"
      subtitle="Every accepted job is durable, cancellable and keeps its own log."
      right={<Button variant="ghost" onClick={refresh}>Refresh</Button>}
    >
      <div className="space-y-4">
        {error ? <ErrorNote message={error} /> : null}
        {jobs.length === 0 ? (
          <Empty text="No jobs yet. Start a dub, a voice clone or a speech generation." />
        ) : (
          <Card className="overflow-hidden">
            <div className="divide-y divide-white/5">
              {jobs.map((job) => {
                const stage = job.stage ? ` · ${job.stage}` : "";
                const duration =
                  job.started && job.finished ? job.finished - job.started : null;
                return (
                  <button
                    key={job.id}
                    type="button"
                    onClick={() => onOpenJob(job.id)}
                    className={
                      "flex w-full items-center gap-3 px-4 py-3 text-left transition hover:bg-white/[0.03] " +
                      (activeJobId === job.id ? "bg-accent-500/5" : "")
                    }
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="truncate text-sm font-medium text-slate-200">
                          {job.kind}
                        </span>
                        <Badge state={job.state} />
                      </div>
                      <div className="mt-0.5 truncate text-xs text-slate-500">
                        {job.id.slice(0, 12)}
                        {stage} · {formatDate(job.created)}
                        {duration ? ` · ${formatSeconds(duration)}` : ""}
                      </div>
                      {job.error ? (
                        <div className="mt-1 truncate text-xs text-rose-300">{job.error}</div>
                      ) : null}
                    </div>
                    <span className="text-xs text-slate-500">Open</span>
                  </button>
                );
              })}
            </div>
          </Card>
        )}
      </div>
    </Section>
  );
}
