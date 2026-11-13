import React from "react";
import { api } from "../api";
import type { Preset } from "../types";
import { Button, ConfirmDialog, ErrorNote, Field, Input, Spinner } from "../ui";

/**
 * Saved configurations for one form. A preset stores the options (not the input
 * files), so it can be applied to a new clip with one click.
 */
export function PresetPanel({
  kind,
  buildPayload,
  onApply,
}: {
  kind: string;
  buildPayload: () => Record<string, unknown>;
  onApply: (payload: Record<string, unknown>) => void;
}) {
  const [presets, setPresets] = React.useState<Preset[]>([]);
  const [name, setName] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [pending, setPending] = React.useState<Preset | null>(null);

  const load = React.useCallback(() => {
    api
      .presets(kind)
      .then((data) => setPresets(data.presets))
      .catch((exc: unknown) =>
        setError(exc instanceof Error ? exc.message : "Could not load presets"),
      );
  }, [kind]);

  React.useEffect(load, [load]);

  async function save() {
    if (!name.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await api.createPreset({
        kind,
        name: name.trim(),
        payload: buildPayload(),
      });
      setName("");
      load();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not save the preset");
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (!pending) return;
    setBusy(true);
    setError(null);
    try {
      await api.deletePreset(pending.id);
      setPending(null);
      load();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not delete the preset");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-2">
        <Field label="Save current options as a preset" className="min-w-48 flex-1">
          <Input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="e.g. Fast French"
          />
        </Field>
        <Button variant="ghost" onClick={save} disabled={busy || !name.trim()}>
          {busy ? <Spinner /> : null}
          Save preset
        </Button>
      </div>
      {error ? <ErrorNote message={error} /> : null}
      {presets.length === 0 ? (
        <div className="text-xs text-slate-500">No presets saved for this form yet.</div>
      ) : (
        <div className="flex flex-wrap gap-2">
          {presets.map((preset) => (
            <div
              key={preset.id}
              className="flex items-center gap-1 rounded-lg border border-white/10 bg-ink-850/60 px-2 py-1 text-xs"
            >
              <button
                type="button"
                onClick={() => onApply(preset.payload)}
                className="text-slate-200 hover:text-white"
              >
                Apply {preset.name}
              </button>
              <button
                type="button"
                aria-label={`Delete preset ${preset.name}`}
                onClick={() => setPending(preset)}
                className="text-slate-500 hover:text-rose-300"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}

      <ConfirmDialog
        open={pending !== null}
        title={`Delete preset ${pending?.name ?? ""}`}
        message="The saved options are removed. Your current form is not changed."
        confirmLabel="Delete preset"
        danger
        busy={busy}
        onConfirm={remove}
        onCancel={() => setPending(null)}
      />
    </div>
  );
}
