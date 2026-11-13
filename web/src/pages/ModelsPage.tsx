import React from "react";
import { api } from "../api";
import type { ModelInfo } from "../types";
import {
  Badge,
  Button,
  Card,
  ConfirmDialog,
  Empty,
  ErrorNote,
  Section,
  Spinner,
  formatBytes,
} from "../ui";

export function ModelsPage({ onJobStarted }: { onJobStarted: (id: string) => void }) {
  const [models, setModels] = React.useState<ModelInfo[]>([]);
  const [busy, setBusy] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [notice, setNotice] = React.useState<string | null>(null);
  const [pending, setPending] = React.useState<ModelInfo | null>(null);
  const [deleting, setDeleting] = React.useState(false);

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

  async function remove() {
    if (!pending) return;
    setDeleting(true);
    setError(null);
    setNotice(null);
    try {
      const result = await api.deleteModel(pending.name);
      setNotice(
        `Deleted ${pending.name}; freed ${formatBytes(result.freed_bytes)}.` +
          (result.warning ? ` ${result.warning}` : ""),
      );
      setPending(null);
      refresh();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not delete the model");
    } finally {
      setDeleting(false);
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
        {notice ? (
          <div className="rounded-lg border border-emerald-400/25 bg-emerald-400/10 px-3 py-2 text-sm text-emerald-200">
            {notice}
          </div>
        ) : null}
        {models.length === 0 ? (
          <Empty
            text="No models reported yet."
            action={
              <Button variant="ghost" onClick={refresh}>
                Reload models
              </Button>
            }
          />
        ) : null}
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
                <div className="flex flex-col items-end gap-2">
                  <Button
                    variant="ghost"
                    className="px-3 py-1 text-xs"
                    disabled={busy !== null || deleting}
                    onClick={() => download([model.name])}
                  >
                    {busy === model.name ? <Spinner /> : null}
                    {model.present ? "Re-download" : "Download"}
                  </Button>
                  {model.present ? (
                    <Button
                      variant="danger"
                      className="px-3 py-1 text-xs"
                      disabled={busy !== null || deleting}
                      onClick={() => setPending(model)}
                    >
                      Delete
                    </Button>
                  ) : null}
                </div>
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

      <ConfirmDialog
        open={pending !== null}
        title={`Delete ${pending?.name ?? "model"}`}
        message={
          `This removes the checkpoint from disk and frees its space. It can be ` +
          `downloaded again later.` +
          (pending?.required ? " This model is required by the MVP." : "")
        }
        confirmLabel="Delete model"
        danger
        busy={deleting}
        onConfirm={remove}
        onCancel={() => setPending(null)}
      />
    </Section>
  );
}
