import React from "react";
import { api } from "../api";
import type { FileInfo, OutputFile } from "../types";
import {
  Button,
  Card,
  Empty,
  ErrorNote,
  Section,
  Spinner,
  formatBytes,
  formatDate,
} from "../ui";
import { UploadField } from "../components/UploadField";

export function FilesPage() {
  const [files, setFiles] = React.useState<FileInfo[]>([]);
  const [outputs, setOutputs] = React.useState<OutputFile[]>([]);
  const [error, setError] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);

  const refresh = React.useCallback(() => {
    Promise.all([api.files(), api.outputs()])
      .then(([uploads, generated]) => {
        setFiles(uploads.files);
        setOutputs(generated.files);
      })
      .catch((exc: unknown) =>
        setError(exc instanceof Error ? exc.message : "Could not load files"),
      );
  }, []);

  React.useEffect(refresh, [refresh]);

  async function remove(item: FileInfo) {
    if (!window.confirm(`Delete "${item.original_name}"?`)) return;
    setBusy(true);
    setError(null);
    try {
      await api.deleteFile(item.id);
      refresh();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not delete the file");
    } finally {
      setBusy(false);
    }
  }

  async function removeOutput(item: OutputFile) {
    if (!window.confirm(`Delete output "${item.relative}"?`)) return;
    setBusy(true);
    setError(null);
    try {
      await api.deleteOutput(item.path);
      refresh();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not delete the output");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Section
      title="Files"
      subtitle="Uploaded inputs and generated artifacts. Outputs live in unique run folders, so nothing is overwritten."
      right={<Button variant="ghost" onClick={refresh}>{busy ? <Spinner /> : null}Refresh</Button>}
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
        <div className="text-sm font-semibold text-white">Your uploads</div>
        {files.length === 0 ? (
          <Empty text="No uploads yet. Add a video or audio clip above." />
        ) : (
          <Card className="overflow-hidden">
            <div className="divide-y divide-white/5">
              {files.map((item) => (
                <div key={item.id} className="flex items-center gap-3 px-4 py-3">
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm text-slate-200">{item.original_name}</div>
                    <div className="text-xs text-slate-500">
                      {item.kind} · {formatBytes(item.size)} · {formatDate(item.created)}
                    </div>
                  </div>
                  <a
                    href={api.fileUrl(item.id)}
                    className="rounded-lg border border-white/10 px-3 py-1 text-xs text-slate-300 hover:border-white/20"
                  >
                    Download
                  </a>
                  <Button variant="danger" className="px-2 py-1 text-xs" onClick={() => remove(item)}>
                    Delete
                  </Button>
                </div>
              ))}
            </div>
          </Card>
        )}

        <div className="pt-2 text-sm font-semibold text-white">Generated outputs</div>
        {outputs.length === 0 ? (
          <Empty text="No outputs yet. Finished jobs write into unique run folders." />
        ) : (
          <Card className="overflow-hidden">
            <div className="divide-y divide-white/5">
              {outputs.map((item) => (
                <div key={item.path} className="flex items-center gap-3 px-4 py-3">
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm text-slate-200">{item.relative}</div>
                    <div className="text-xs text-slate-500">
                      {formatBytes(item.size)} · {formatDate(item.created)}
                    </div>
                  </div>
                  <a
                    href={api.outputUrl(item.path)}
                    className="rounded-lg border border-white/10 px-3 py-1 text-xs text-slate-300 hover:border-white/20"
                  >
                    Download
                  </a>
                  <Button variant="danger" className="px-2 py-1 text-xs" onClick={() => removeOutput(item)}>
                    Delete
                  </Button>
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>
    </Section>
  );
}
