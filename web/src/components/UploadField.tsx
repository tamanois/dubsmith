import React from "react";
import { api } from "../api";
import { ErrorNote, Field, Spinner } from "../ui";

export function UploadField({
  label,
  hint,
  accept,
  onUploaded,
}: {
  label: string;
  hint?: string;
  accept?: string;
  onUploaded: (path: string, name: string) => void;
}) {
  const [busy, setBusy] = React.useState(false);
  const [name, setName] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  async function pick(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const uploaded = await api.upload(file);
      setName(uploaded.name);
      onUploaded(uploaded.path, uploaded.name);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Field label={label} hint={hint}>
      <label className="flex cursor-pointer items-center gap-3 rounded-lg border border-dashed border-white/15 bg-ink-800/40 px-3 py-3 text-sm text-slate-300 transition hover:border-accent-500/50">
        {busy ? <Spinner /> : <span className="text-lg">↑</span>}
        <span className="min-w-0 flex-1 truncate">
          {busy ? "Uploading…" : name ?? "Choose a file"}
        </span>
        <input
          type="file"
          accept={accept}
          aria-label={label}
          onChange={pick}
          className="hidden"
          disabled={busy}
        />
      </label>
      {error ? <ErrorNote message={error} /> : null}
    </Field>
  );
}
