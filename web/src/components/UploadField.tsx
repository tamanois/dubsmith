import React from "react";
import { api } from "../api";
import { ErrorNote, Field, Spinner } from "../ui";

export function UploadField({
  label,
  hint,
  tooltip,
  accept,
  value,
  onUploaded,
}: {
  label: string;
  hint?: string;
  tooltip?: string;
  accept?: string;
  /** A path chosen elsewhere (a picked upload or a job loaded for editing). */
  value?: string;
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

  // Show a name chosen outside the file input (a restored job, a picked upload),
  // so editing an existing job never looks like an empty field.
  const shown = name ?? (value ? value.split(/[\\/]/).pop() ?? value : null);

  return (
    <Field label={label} hint={hint} tooltip={tooltip}>
      <label className="flex cursor-pointer items-center gap-3 rounded-lg border border-dashed border-white/15 bg-ink-800/40 px-3 py-3 text-sm text-slate-300 transition hover:border-accent-500/50">
        {busy ? <Spinner /> : <span className="text-lg">↑</span>}
        <span className="min-w-0 flex-1 truncate">
          {busy ? "Uploading…" : shown ?? "Choose a file"}
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