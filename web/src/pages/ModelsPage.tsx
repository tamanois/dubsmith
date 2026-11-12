import React from "react";
import { api } from "../api";
import type { ModelInfo } from "../types";
import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorNote,
  Section,
  Spinner,
} from "../ui";

export function ModelsPage({ onJobStarted }: { onJobStarted: (id: string) => void }) {
  const [models, setModels] = React.useState<ModelInfo[]>([]);
  const [busy, setBusy] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  const refresh = React.useCallback(() => {
    api
      .models()
      .then((data) => setModels(data.models))
      .catch((exc: unknown) =>
        setError(exc instanceof Error ? exc.message : "Could not load models"),
      );
  }, []);

  React.useEffect(refresh, [refresh]);

  async function download(names: string[]) {
    setBusy(names.join(","));
    setError(null);
    try {
      const job = await api.downloadModels(names);
      onJobStarted(job.job_id);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not start the download");
    } finally {
      setBusy(null);
    }
  }

  const present = models.filter((model) => model.present).length;

  return (
    <Section
      title="Models"
      subtitle={`${present} of ${models.length} checkpoints available locally. Weights are never committed.`}
      right={
        <div className="flex gap-2">
          <Button variant="ghost" onClick={refresh}>Refresh</Button>
          <Button disabled={busy !== null} onClick={() => download(["all"])}>
            {busy === "all" ? <Spinner /> : null}
            Download MVP set
          </Button>
        </div>
      }
    >
      <div className="space-y-4">
        {error ? <ErrorNote message={error} /> : null}
        {models.length === 0 ? <Empty text="No models reported yet." /> : null}
        <div className="grid gap-3 lg:grid-cols-2">
          {models.map((model) => (
            <Card key={model.name} className="p-4">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-white">{model.name}</span>
                    {model.required ? (
                      <span className="rounded-full border border-accent-500/30 bg-accent-500/10 px-2 py-0.5 text-[10px] uppercase tracking-wide text-accent-300">
                        required
                      </span>
                    ) : null}
                    <Badge state={model.present ? "COMPLETED" : "QUEUED"} />
                  </div>
                  <div className="mt-1 text-xs text-slate-400">{model.description}</div>
                  <div className="mt-1 break-all text-xs text-slate-500">{model.repo_id}</div>
                  <div className="mt-1 text-xs text-slate-500">License: {model.license}</div>
                </div>
                <Button
                  variant="ghost"
                  className="px-3 py-1 text-xs"
                  disabled={busy !== null}
                  onClick={() => download([model.name])}
                >
                  {busy === model.name ? <Spinner /> : null}
                  {model.present ? "Re-download" : "Download"}
                </Button>
              </div>
            </Card>
          ))}
        </div>
        <div className="flex justify-end">
          <Button variant="ghost" disabled={busy !== null} onClick={() => download(["everything"])}>
            {busy === "everything" ? <Spinner /> : null}
            Download everything (optional backends)
          </Button>
        </div>
      </div>
    </Section>
  );
}
