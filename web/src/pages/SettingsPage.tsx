import React from "react";
import { api } from "../api";
import type { SettingsInfo } from "../types";
import {
  Button,
  Card,
  ErrorNote,
  Field,
  Input,
  Section,
  Select,
  Spinner,
  Toggle,
  num,
} from "../ui";

export function SettingsPage() {
  const [settings, setSettings] = React.useState<SettingsInfo | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [saved, setSaved] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    api
      .settings()
      .then(setSettings)
      .catch((exc: unknown) =>
        setError(exc instanceof Error ? exc.message : "Could not load settings"),
      );
  }, []);

  function update<K extends keyof SettingsInfo>(key: K, value: SettingsInfo[K]) {
    setSettings((current) => (current ? { ...current, [key]: value } : current));
    setSaved(false);
  }

  async function save() {
    if (!settings) return;
    setBusy(true);
    setError(null);
    try {
      const values: Record<string, unknown> = {
        llm_base_url: settings.llm_base_url,
        llm_model: settings.llm_model,
        llm_timeout_s: num(String(settings.llm_timeout_s)),
        llm_chunk_size: num(String(settings.llm_chunk_size)),
        llm_plan_chunk_size: num(String(settings.llm_plan_chunk_size)),
        loudness_target: settings.loudness_target,
        target_lufs: num(String(settings.target_lufs)),
        keep_background: settings.keep_background,
        separator: settings.separator,
        duck_db: num(String(settings.duck_db)),
        models_dir: settings.models_dir,
        voices_dir: settings.voices_dir,
        log_level: settings.log_level,
        log_json: settings.log_json,
        default_device: settings.default_device,
        default_dtype: settings.default_dtype,
      };
      setSettings(await api.patchSettings(values));
      setSaved(true);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not save settings");
    } finally {
      setBusy(false);
    }
  }

  if (!settings) {
    return (
      <Section title="Settings" subtitle="Defaults used by every new job.">
        {error ? <ErrorNote message={error} /> : <div className="text-sm text-slate-500">Loading…</div>}
      </Section>
    );
  }

  return (
    <Section
      title="Settings"
      subtitle="Non-secret defaults. API keys stay in the environment and are never written to SQLite."
      right={
        <div className="flex items-center gap-2">
          {saved ? <span className="text-xs text-emerald-300">Saved</span> : null}
          <Button onClick={save} disabled={busy}>
            {busy ? <Spinner /> : null}
            Save settings
          </Button>
        </div>
      }
    >
      <div className="space-y-4">
        {error ? <ErrorNote message={error} /> : null}
        <Card className="space-y-5 p-5">
          <div className="text-sm font-semibold text-white">Language model</div>
          <div className="grid gap-4 md:grid-cols-2">
            <Field label="Base URL" hint="OpenAI-compatible endpoint used for translation and planning.">
              <Input value={settings.llm_base_url} onChange={(event) => update("llm_base_url", event.target.value)} />
            </Field>
            <Field label="Model">
              <Input value={settings.llm_model} onChange={(event) => update("llm_model", event.target.value)} />
            </Field>
            <Field label="Timeout (seconds)">
              <Input value={String(settings.llm_timeout_s)} onChange={(event) => update("llm_timeout_s", Number(event.target.value))} />
            </Field>
            <Field label="Translation chunk size">
              <Input value={String(settings.llm_chunk_size)} onChange={(event) => update("llm_chunk_size", Number(event.target.value))} />
            </Field>
            <Field label="Planning chunk size">
              <Input value={String(settings.llm_plan_chunk_size)} onChange={(event) => update("llm_plan_chunk_size", Number(event.target.value))} />
            </Field>
            <div className="rounded-lg border border-white/5 bg-ink-850/60 p-3 text-xs text-slate-400">
              API key: {settings.llm_api_key_set ? "configured in the environment" : "not set (LLM calls may fail)"}
            </div>
          </div>
        </Card>

        <Card className="space-y-5 p-5">
          <div className="text-sm font-semibold text-white">Audio defaults</div>
          <div className="grid gap-4 md:grid-cols-2">
            <Field label="Loudness policy">
              <Select
                value={settings.loudness_target}
                onChange={(event) => update("loudness_target", event.target.value)}
                options={[
                  { value: "match-source", label: "Match the source" },
                  { value: "lufs", label: "Absolute LUFS" },
                ]}
              />
            </Field>
            <Field label="LUFS target">
              <Input value={String(settings.target_lufs)} onChange={(event) => update("target_lufs", Number(event.target.value))} />
            </Field>
            <Field label="Background separator">
              <Select
                value={settings.separator}
                onChange={(event) => update("separator", event.target.value)}
                options={[
                  { value: "auto", label: "Auto" },
                  { value: "demucs", label: "Demucs" },
                  { value: "none", label: "None" },
                ]}
              />
            </Field>
            <Field label="Duck depth (dB)">
              <Input value={String(settings.duck_db)} onChange={(event) => update("duck_db", Number(event.target.value))} />
            </Field>
            <Toggle
              label="Keep music and ambience by default"
              checked={settings.keep_background}
              onChange={(value) => update("keep_background", value)}
            />
          </div>
        </Card>

        <Card className="space-y-5 p-5">
          <div className="text-sm font-semibold text-white">Storage and logs</div>
          <div className="grid gap-4 md:grid-cols-2">
            <Field label="Models directory">
              <Input value={settings.models_dir} onChange={(event) => update("models_dir", event.target.value)} />
            </Field>
            <Field label="Voices directory">
              <Input value={settings.voices_dir} onChange={(event) => update("voices_dir", event.target.value)} />
            </Field>
            <Field label="Log level">
              <Select
                value={settings.log_level}
                onChange={(event) => update("log_level", event.target.value)}
                options={[
                  { value: "INFO", label: "INFO" },
                  { value: "DEBUG", label: "DEBUG" },
                  { value: "WARNING", label: "WARNING" },
                  { value: "ERROR", label: "ERROR" },
                ]}
              />
            </Field>
            <Field label="Default device">
              <Select
                value={settings.default_device}
                onChange={(event) => update("default_device", event.target.value)}
                options={[
                  { value: "auto", label: "Auto" },
                  { value: "cpu", label: "CPU" },
                  { value: "cuda:0", label: "cuda:0" },
                  { value: "cuda:1", label: "cuda:1" },
                ]}
              />
            </Field>
            <Field label="Default precision">
              <Select
                value={settings.default_dtype}
                onChange={(event) => update("default_dtype", event.target.value)}
                options={[
                  { value: "auto", label: "Auto" },
                  { value: "bf16", label: "bfloat16" },
                  { value: "fp16", label: "float16" },
                  { value: "fp32", label: "float32" },
                ]}
              />
            </Field>
            <Toggle
              label="JSON logs"
              checked={settings.log_json}
              onChange={(value) => update("log_json", value)}
            />
          </div>
        </Card>
      </div>
    </Section>
  );
}
