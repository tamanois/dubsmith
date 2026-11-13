import React from "react";
import { api } from "../api";
import type { FileInfo } from "../types";
import {
  Button,
  Empty,
  ErrorNote,
  MediaPreview,
  Pagination,
  Spinner,
  formatBytes,
  formatDate,
  isPlayable,
} from "../ui";

const PAGE_SIZE = 12;

function suffixAllows(name: string, accept?: string): boolean {
  if (!accept) return true;
  const allowed = accept
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
  if (!allowed.length) return true;
  const lower = name.toLowerCase();
  return allowed.some((ext) => lower.endsWith(ext));
}

/**
 * Browse every uploaded file, not just the handful of recents.
 *
 * The list is server-paginated, so it stays correct no matter how many inputs the
 * user has uploaded. Selecting a row hands the stored path back to the caller.
 */
export function FilePickerDialog({
  open,
  title,
  accept,
  onPick,
  onClose,
}: {
  open: boolean;
  title: string;
  accept?: string;
  onPick: (path: string, name: string) => void;
  onClose: () => void;
}) {
  const [files, setFiles] = React.useState<FileInfo[]>([]);
  const [total, setTotal] = React.useState(0);
  const [page, setPage] = React.useState(1);
  const [preview, setPreview] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!open) return;
    let alive = true;
    setLoading(true);
    setError(null);
    api
      .files({ page, pageSize: PAGE_SIZE })
      .then((data) => {
        if (!alive) return;
        setFiles(data.files);
        setTotal(Number(data.total) || 0);
      })
      .catch((exc: unknown) => {
        if (!alive) return;
        setError(exc instanceof Error ? exc.message : "Could not load uploads");
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [open, page]);

  // Reset the page and any preview whenever the dialog is reopened.
  React.useEffect(() => {
    if (!open) {
      setPage(1);
      setPreview(null);
    }
  }, [open]);

  if (!open) return null;

  const visible = files.filter((item) => suffixAllows(item.original_name, accept));

  function choose(item: FileInfo) {
    onPick(item.path, item.original_name);
    onClose();
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={title}
      className="fixed inset-0 z-50 grid place-items-center bg-black/50 p-4"
    >
      <div className="flex max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-white/10 bg-ink-900 shadow-glow">
        <div className="flex items-center justify-between gap-3 border-b border-white/5 px-5 py-3">
          <div className="min-w-0">
            <h3 className="text-base font-semibold text-white">{title}</h3>
            <p className="text-xs text-slate-500">Pick any file you have uploaded.</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close file picker"
            className="rounded-md px-2 py-1 text-slate-400 hover:bg-white/5 hover:text-white"
          >
            ✕
          </button>
        </div>

        <div className="min-h-0 flex-1 space-y-3 overflow-auto px-5 py-4">
          {error ? <ErrorNote message={error} /> : null}
          {loading && files.length === 0 ? (
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <Spinner />
              Loading uploads…
            </div>
          ) : null}
          {!loading && visible.length === 0 && !error ? (
            <Empty text="No matching uploads yet. Upload a file from the Files page." />
          ) : null}
          <div className="space-y-2">
            {visible.map((item) => {
              const src = api.fileInlineUrl(item.id);
              const playing = preview === src;
              return (
                <div
                  key={item.id}
                  className="rounded-xl border border-white/10 bg-ink-850/60 px-3 py-2"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm text-slate-200">
                        {item.original_name}
                      </div>
                      <div className="text-xs text-slate-500">
                        {item.kind} · {formatBytes(item.size)} · {formatDate(item.created)}
                      </div>
                    </div>
                    {isPlayable(item.original_name) ? (
                      <Button
                        variant="ghost"
                        className="px-2 py-1 text-xs"
                        onClick={() => setPreview((current) => (current === src ? null : src))}
                      >
                        {playing ? "Hide" : "Play"}
                      </Button>
                    ) : null}
                    <Button className="px-3 py-1 text-xs" onClick={() => choose(item)}>
                      Use
                    </Button>
                  </div>
                  {playing ? (
                    <div className="mt-2">
                      <MediaPreview src={src} name={item.original_name} autoPlay />
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        </div>

        <div className="border-t border-white/5 px-5 py-3">
          <Pagination
            page={page}
            pageSize={PAGE_SIZE}
            total={total}
            onPage={(next) => {
              setPreview(null);
              setPage(next);
            }}
          />
        </div>
      </div>
    </div>
  );
}
