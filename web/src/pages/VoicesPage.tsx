import React from "react";
import { api } from "../api";
import type { VoiceInfo } from "../types";
import {
  Button,
  Card,
  Disclosure,
  Empty,
  ErrorNote,
  Field,
  Input,
  Section,
  Select,
  Spinner,
} from "../ui";
import { UploadField } from "../components/UploadField";

const LANGUAGES = [
  "French",
  "English",
  "German",
  "Italian",
  "Portuguese",
  "Spanish",
  "Japanese",
  "Korean",
  "Chinese",
  "Russian",
];

export function VoicesPage({ onJobStarted }: { onJobStarted: (id: string) => void }) {
  const [voices, setVoices] = React.useState<VoiceInfo[]>([]);
  const [error, setError] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);

  const [prompt, setPrompt] = React.useState("");
  const [language, setLanguage] = React.useState("French");
  const [label, setLabel] = React.useState("");
  const [speak, setSpeak] = React.useState("");
  const [clip, setClip] = React.useState("");
  const [cloneText, setCloneText] = React.useState("");
  const [transcribe, setTranscribe] = React.useState(true);
  const [verify, setVerify] = React.useState(false);
  const [force, setForce] = React.useState(false);
  const [device, setDevice] = React.useState("auto");
  const [dtype, setDtype] = React.useState("auto");

  const refresh = React.useCallback(() => {
    api
      .voices()
      .then((data) => setVoices(data.voices))
      .catch((exc: unknown) =>
        setError(exc instanceof Error ? exc.message : "Could not load voices"),
      );
  }, []);

  React.useEffect(refresh, [refresh]);

  async function run(action: () => Promise<{ job_id: string }>) {
    setBusy(true);
    setError(null);
    try {
      const job = await action();
      onJobStarted(job.job_id);
      setPrompt("");
      setLabel("");
      setSpeak("");
      setClip("");
      setCloneText("");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not start the job");
    } finally {
      setBusy(false);
    }
  }

  async function remove(item: VoiceInfo) {
    if (!window.confirm(`Delete voice "${item.label}"?`)) return;
    setError(null);
    try {
      await api.deleteVoice(item.label);
      refresh();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not delete the voice");
    }
  }

  return (
    <Section
      title="Voice library"
      subtitle="Reference clips, transcripts and per-model encodings, ready to reuse."
      right={<Button variant="ghost" onClick={refresh}>Refresh</Button>}
    >
      <div className="space-y-4">
        {error ? <ErrorNote message={error} /> : null}
        {voices.length === 0 ? (
          <Empty text="No voices yet. Design one from a description, clone a clip, or register an existing recording." />
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {voices.map((item) => (
              <Card key={item.label} className="p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-semibold text-white">{item.label}</div>
                    <div className="mt-1 text-xs text-slate-400">
                      {item.language} · {item.source} · {item.seconds.toFixed(1)}s ·{" "}
                      {item.cards} card(s)
                    </div>
                    <div className="mt-1 text-xs text-slate-500">
                      {item.has_transcript ? "transcript stored" : "no transcript"} ·{" "}
                      {item.gender} / {item.rate_band}
                    </div>
                    {item.instruct ? (
                      <div className="mt-2 line-clamp-2 text-xs text-slate-400">
                        “{item.instruct}”
                      </div>
                    ) : null}
                  </div>
                  <Button variant="danger" className="px-2 py-1 text-xs" onClick={() => remove(item)}>
                    Delete
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        )}

        <Disclosure title="Design a voice" hint="From a text description">
          <div className="grid gap-4 md:grid-cols-2">
            <Field label="Description">
              <Input value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="a warm female French voice, slow and clear" />
            </Field>
            <Field label="Language">
              <Select
                value={language}
                onChange={(event) => setLanguage(event.target.value)}
                options={LANGUAGES.map((item) => ({ value: item, label: item }))}
              />
            </Field>
            <Field label="Label" hint="Optional; derived from the description otherwise.">
              <Input value={label} onChange={(event) => setLabel(event.target.value)} />
            </Field>
            <Field label="Audition text" hint="Optional clip rendered with the new voice.">
              <Input value={speak} onChange={(event) => setSpeak(event.target.value)} />
            </Field>
            <Field label="Device">
              <Select
                value={device}
                onChange={(event) => setDevice(event.target.value)}
                options={[
                  { value: "auto", label: "Auto" },
                  { value: "cpu", label: "CPU" },
                  { value: "cuda:0", label: "cuda:0" },
                  { value: "cuda:1", label: "cuda:1" },
                ]}
              />
            </Field>
            <Field label="Precision">
              <Select
                value={dtype}
                onChange={(event) => setDtype(event.target.value)}
                options={[
                  { value: "auto", label: "Auto" },
                  { value: "bf16", label: "bfloat16" },
                  { value: "fp16", label: "float16" },
                  { value: "fp32", label: "float32" },
                ]}
              />
            </Field>
          </div>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input type="checkbox" checked={force} onChange={(event) => setForce(event.target.checked)} />
            Overwrite an existing voice with the same label
          </label>
          <Button
            disabled={busy || !prompt.trim()}
            onClick={() =>
              run(() =>
                api.designVoice({
                  prompt,
                  language,
                  label: label || undefined,
                  speak: speak || undefined,
                  device,
                  dtype,
                  force,
                }),
              )
            }
          >
            {busy ? <Spinner /> : null}
            Design voice
          </Button>
        </Disclosure>

        <Disclosure title="Clone a clip" hint="Encode a donated recording once">
          <div className="grid gap-4 md:grid-cols-2">
            <UploadField
              label="Reference clip"
              accept=".wav,.mp3,.flac,.m4a,.ogg,.opus"
              onUploaded={(path) => setClip(path)}
            />
            <Field label="Transcript" hint="Leave empty to transcribe it locally.">
              <Input value={cloneText} onChange={(event) => setCloneText(event.target.value)} />
            </Field>
            <Field label="Language">
              <Select
                value={language}
                onChange={(event) => setLanguage(event.target.value)}
                options={LANGUAGES.map((item) => ({ value: item, label: item }))}
              />
            </Field>
            <Field label="Label">
              <Input value={label} onChange={(event) => setLabel(event.target.value)} />
            </Field>
          </div>
          <div className="flex flex-wrap gap-4 text-sm text-slate-300">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={transcribe}
                onChange={(event) => setTranscribe(event.target.checked)}
              />
              Transcribe the clip instead of using the text above
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={verify}
                onChange={(event) => setVerify(event.target.checked)}
              />
              Verify the transcript against the audio
            </label>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={force} onChange={(event) => setForce(event.target.checked)} />
              Overwrite an existing label
            </label>
          </div>
          <Button
            disabled={busy || !clip}
            onClick={() =>
              run(() =>
                api.cloneVoice({
                  audio: clip,
                  clone_text: cloneText || undefined,
                  transcribe,
                  verify,
                  language,
                  label: label || undefined,
                  device,
                  dtype,
                  force,
                }),
              )
            }
          >
            {busy ? <Spinner /> : null}
            Clone voice
          </Button>
        </Disclosure>

        <Disclosure title="Register an existing clip" hint="No model load, no encoding">
          <div className="grid gap-4 md:grid-cols-2">
            <UploadField
              label="Clip"
              accept=".wav,.mp3,.flac,.m4a,.ogg,.opus"
              onUploaded={(path) => setClip(path)}
            />
            <Field label="Transcript" hint="Optional, but improves cloning.">
              <Input value={cloneText} onChange={(event) => setCloneText(event.target.value)} />
            </Field>
            <Field label="Language">
              <Select
                value={language}
                onChange={(event) => setLanguage(event.target.value)}
                options={LANGUAGES.map((item) => ({ value: item, label: item }))}
              />
            </Field>
            <Field label="Label">
              <Input value={label} onChange={(event) => setLabel(event.target.value)} />
            </Field>
          </div>
          <div className="flex flex-wrap gap-4 text-sm text-slate-300">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={transcribe}
                onChange={(event) => setTranscribe(event.target.checked)}
              />
              Transcribe the clip
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={verify}
                onChange={(event) => setVerify(event.target.checked)}
              />
              Verify the transcript
            </label>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={force} onChange={(event) => setForce(event.target.checked)} />
              Overwrite an existing label
            </label>
          </div>
          <Button
            disabled={busy || !clip}
            onClick={() =>
              run(() =>
                api.addVoice({
                  audio: clip,
                  clone_text: cloneText || undefined,
                  transcribe,
                  verify,
                  language,
                  label: label || undefined,
                  device,
                  force,
                }),
              )
            }
          >
            {busy ? <Spinner /> : null}
            Register voice
          </Button>
        </Disclosure>
      </div>
    </Section>
  );
}
