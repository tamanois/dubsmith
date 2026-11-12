import React from "react";
import { api } from "../api";
import type { FileInfo, OutputFile, OutputRun } from "../types";
import {
  Button,
  Card,
  ConfirmDialog,
  CopyButton,
  Empty,
  ErrorNote,
  MediaPreview,
  Pagination,
  Section,
  Spinner,
  formatBytes,
  formatDate,
  isPlayable,
} from "../ui";
import { UploadField } from "../components/UploadField";

const PAGE_SIZE = 20;

type Pending = { title: string; message: string; run: () => Promise<void> };

export function FilesPage() {
  const [files, setFiles] = React.useState<FileInfo[]>([]);
  const [fileTotal, setFileTotal] = React.useState(0);
  const [filePage, setFilePage] = React.useState(1);
  const [runs, setRuns] = React.useState<OutputRun[]>([]);
  const [runTotal, setRunTotal] = React.useState(0);
  const [runPage, setRunPage] = React.useState(1);
  const [details, setDetails] = React.useState<{
    jobDir: string;
    files: OutputFile[];
  } | null>(null);
  const [selectedFiles, setSelectedFiles] = React.useState<Set<string>>(new Set());
  const [preview, setPreview] = React.useState<{ src: string; name: string } | null>(null);
  const [pending, setPending] = React.useState<Pending | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const refresh = React.useCallback(() => {
    Promise.all([
      api.files({ page: filePage, pageSize: PAGE_SIZE }),
      api.outputs({ page: runPage, pageSize: PAGE_SIZE }),
    ])
      .then(([uploads, generated]) => {
        setFiles(uploads.files);
        setFileTotal(Number(uploads.total) || 0);
        setRuns(generated.runs);
        setRunTotal(Number(generated.total) || 0);
      })
      .catch((exc: unknown) =>
        setError(exc instanceof Error ? exc.message : "Could not load files"),
      );
  }, [filePage, runPage]);

  React.useEffect(refresh, [refresh]);

  function toggle(
    setter: React.Dispatch<React.SetStateAction<Set<string>>>,
    id: string,
  ) {
    setter((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function play(src: string, name: string) {
    if (!isPlayable(name)) return;
    setPreview((current) => (current?.src === src ? null : { src, name }));
  }

  async function toggleDetails(jobDir: string) {
    if (details?.jobDir === jobDir) {
      setDetails(null);
      return;
    }
    setError(null);
    try {
      const data = await api.outputFiles(jobDir);
      setDetails({ jobDir, files: data.files });
    } catch (exc) {
      setError(
        exc instanceof Error ? exc.message : "Could not load the run details",
      );
    }
  }

  async function confirmRun() {
    if (!pending) return;
    setBusy(true);
    setError(null);
    try {
      await pending.run();
      setPending(null);
      setSelectedFiles(new Set());
      setDetails(null);
      setPreview(null);
      refresh();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not delete");
    } finally {
      setBusy(false);
    }
  }

  function deleteUpload(item: FileInfo) {
    setPending({
      title: `Delete ${item.original_name}`,
      message: "The uploaded file is removed from disk. Jobs that used it keep their outputs.",
      run: async () => {
        await api.deleteFile(item.id);
      },
    });
  }

  function deleteRunPrimary(run: OutputRun) {
    setPending({
      title: `Delete ${run.primary.name}`,
      message:
        "Remove the main output of this run? The other artifacts stay on disk; open Details to manage them.",
      run: async () => {
        await api.deleteOutput(run.primary.path);
      },
    });
  }

  function deleteDetailFile(file: OutputFile) {
    setPending({
      title: `Delete ${file.name}`,
      message: `Remove ${file.relative} from this run? This cannot be undone.`,
      run: async () => {
        await api.deleteOutput(file.path);
        if (details) {
          const data = await api.outputFiles(details.jobDir);
          setDetails({ jobDir: details.jobDir, files: data.files });
        }
      },
    });
  }

  function deleteSelectedUploads() {
    const ids = [...selectedFiles];
    if (!ids.length) return;
    setPending({
      title: `Delete ${ids.length} upload${ids.length === 1 ? "" : "s"}`,
      message: "The selected uploads are removed from disk.",
      run: async () => {
        await api.deleteFiles(ids);
      },
    });
  }

  return (
    <Section
      title="Files"
      subtitle="Uploaded inputs and generated artifacts. Outputs live in unique run folders, so nothing is overwritten."
      right={
        <Button variant="ghost" onClick={refresh}>
          {busy ? <Spinner /> : null}
          Refresh
        </Button>
      }
    >
      <div className="space-y-4">
        {error ? <ErrorNote message={error} /> : null}
        <Card className="p-5">
          <UploadField
            label="Upload a media file"
            hint="Videos and audio. The server stores it under data/uploads."
            accept=".mp4,.mkv,.mov,.webm,.avi,.m4v,.mp3,.wav,.flac,.m4a,.aac,.ogg,.opus"
            onUploaded={() => refresh()}
          />
        </Card>

        {preview ? (
          <Card className="space-y-2 p-4">
            <div className="flex items-center justify-between gap-2 text-xs text-slate-400">
              <span className="truncate">{preview.name}</span>
              <button
                type="button"
                onClick={() => setPreview(null)}
                className="text-slate-500 hover:text-slate-300"
              >
                Close player
              </button>
            </div>
            <MediaPreview src={preview.src} name={preview.name} />
          </Card>
        ) : null}

        <div className="text-sm font-semibold text-white">Your uploads</div>
        {files.length === 0 ? (
          <Empty text="No uploads yet. Add a video or audio clip above." />
        ) : (
          <>
            {selectedFiles.size > 0 ? (
              <div className="flex items-center justify-between gap-3 rounded-lg border border-white/10 bg-ink-850/60 px-3 py-2 text-xs text-slate-400">
                <span>{selectedFiles.size} selected</span>
                <Button variant="danger" className="px-2 py-1 text-xs" onClick={deleteSelectedUploads}>
                  Delete selected
                </Button>
              </div>
            ) : null}
            <Card className="overflow-hidden">
              <div className="divide-y divide-white/5">
                {files.map((item) => {
                  const src = api.fileInlineUrl(item.id);
                  const playing = preview?.src === src;
                  return (
                    <div key={item.id} className="flex flex-wrap items-center gap-3 px-4 py-3">
                      <input
                        type="checkbox"
                        aria-label={`Select ${item.original_name}`}
                        checked={selectedFiles.has(item.id)}
                        onChange={() => toggle(setSelectedFiles, item.id)}
                      />
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm text-slate-200">{item.original_name}</div>
                        <div className="text-xs text-slate-500">
                          {item.kind} · {formatBytes(item.size)} · {formatDate(item.created)}
                        </div>
                      </div>
                      {isPlayable(item.original_name) ? (
                        <Button
                          variant="ghost"
                          className="px-2 py-1 text-xs"
                          onClick={() => play(src, item.original_name)}
                        >
                          {playing ? "Hide" : "Play"}
                        </Button>
                      ) : null}
                      <CopyButton value={item.path} label={`Copy path of ${item.original_name}`} />
                      <a
                        href={api.fileUrl(item.id)}
                        className="rounded-lg border border-white/10 px-3 py-1 text-xs text-slate-300 hover:border-white/20"
                      >
                        Download
                      </a>
                      <Button
                        variant="danger"
                        className="px-2 py-1 text-xs"
                        aria-label={`Delete ${item.original_name}`}
                        onClick={() => deleteUpload(item)}
                      >
                        Delete
                      </Button>
                    </div>
                  );
                })}
              </div>
            </Card>
            <Pagination
              page={filePage}
              pageSize={PAGE_SIZE}
              total={fileTotal}
              onPage={(next) => {
                setFilePage(next);
                setSelectedFiles(new Set());
              }}
            />
          </>
        )}

        <div className="pt-2 text-sm font-semibold text-white">Generated outputs</div>
        {runs.length === 0 ? (
          <Empty
            text="No outputs yet. Each finished job appears here with its main file."
            action={
              <Button variant="ghost" onClick={refresh}>
                Reload outputs
              </Button>
            }
          />
        ) : (
          <>
            <Card className="overflow-hidden">
              <div className="divide-y divide-white/5">
                {runs.map((run) => {
                  const src = api.outputInlineUrl(run.primary.path);
                  const playing = preview?.src === src;
                  const playable = isPlayable(run.primary.name);
                  const open = details?.jobDir === run.job_dir;
                  return (
                    <div key={run.job_dir} className="px-4 py-3">
                      <div className="flex flex-wrap items-center gap-3">
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-sm text-slate-200">
                            {run.title}
                          </div>
                          <div className="truncate text-xs text-slate-500">
                            {run.input_label ? `${run.input_label} · ` : ""}
                            {run.primary.name} · {formatBytes(run.primary.size)} ·{" "}
                            {formatDate(run.created)}
                            {run.file_count > 1
                              ? ` · ${run.file_count} files`
                              : ""}
                          </div>
                        </div>
                        {playable ? (
                          <Button
                            variant="ghost"
                            className="px-2 py-1 text-xs"
                            onClick={() => play(src, run.primary.name)}
                          >
                            {playing ? "Hide" : "Play"}
                          </Button>
                        ) : null}
                        <CopyButton
                          value={run.primary.path}
                          label={`Copy path of ${run.primary.name}`}
                        />
                        <a
                          href={api.outputUrl(run.primary.path)}
                          className="rounded-lg border border-white/10 px-3 py-1 text-xs text-slate-300 hover:border-white/20"
                        >
                          Download
                        </a>
                        <Button
                          variant="ghost"
                          className="px-2 py-1 text-xs"
                          aria-expanded={open}
                          onClick={() => toggleDetails(run.job_dir)}
                        >
                          {open ? "Hide details" : `Details (${run.file_count})`}
                        </Button>
                        <Button
                          variant="danger"
                          className="px-2 py-1 text-xs"
                          aria-label={`Delete ${run.primary.name}`}
                          onClick={() => deleteRunPrimary(run)}
                        >
                          Delete
                        </Button>
                      </div>
                      {details && details.jobDir === run.job_dir ? (
                        <div className="mt-3 space-y-1 rounded-lg border border-white/5 bg-ink-850/60 p-2 text-xs">
                          {details.files.map((file) => (
                            <div
                              key={file.path}
                              className="flex flex-wrap items-center gap-2"
                            >
                              <span className="min-w-0 flex-1 truncate text-slate-400">
                                {file.relative}
                              </span>
                              <span className="text-slate-600">
                                {formatBytes(file.size)}
                              </span>
                              {isPlayable(file.name) ? (
                                <button
                                  type="button"
                                  onClick={() =>
                                    play(api.outputInlineUrl(file.path), file.name)
                                  }
                                  className="text-slate-400 hover:text-slate-200"
                                >
                                  Play
                                </button>
                              ) : null}
                              <a
                                href={api.outputUrl(file.path)}
                                className="rounded border border-white/10 px-2 py-0.5 text-slate-300 hover:border-white/20"
                              >
                                Download
                              </a>
                              <button
                                type="button"
                                aria-label={`Delete ${file.relative}`}
                                onClick={() => deleteDetailFile(file)}
                                className="text-rose-300 hover:text-rose-200"
                              >
                                Delete
                              </button>
                            </div>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            </Card>
            <Pagination
              page={runPage}
              pageSize={PAGE_SIZE}
              total={runTotal}
              onPage={(next) => {
                setRunPage(next);
                setDetails(null);
              }}
            />
          </>
        )}
      </div>

      <ConfirmDialog
        open={pending !== null}
        title={pending?.title ?? "Delete"}
        message={pending?.message ?? ""}
        confirmLabel="Delete"
        danger
        busy={busy}
        onConfirm={confirmRun}
        onCancel={() => setPending(null)}
      />
    </Section>
  );
}
