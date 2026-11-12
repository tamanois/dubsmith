import React from "react";
import { api } from "../api";
import type { VoiceInfo } from "../types";
import {
  Button,
  Card,
  Disclosure,
  ErrorNote,
  Field,
  Input,
  Section,
  Select,
  Spinner,
  Textarea,
  Toggle,
  list,
  num,
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

function perLine(value: string): string[] | undefined {
  const lines = value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.length ? lines : undefined;
}

export function SayPage({ onJobStarted }: { onJobStarted: (id: string) => void }) {
  const [text, setText] = React.useState("");
  const [backend, setBackend] = React.useState("qwen");
  const [source, setSource] = React.useState<"voice" | "design" | "clone" | "speaker">("voice");
  const [voice, setVoice] = React.useState("");
  const [design, setDesign] = React.useState("");
  const [clone, setClone] = React.useState("");
  const [cloneName, setCloneName] = React.useState("");
  const [cloneText, setCloneText] = React.useState("");
  const [speaker, setSpeaker] = React.useState("");
  const [language, setLanguage] = React.useState("French");
  const [saveAs, setSaveAs] = React.useState("");
  const [prosody, setProsody] = React.useState("");
  const [emotions, setEmotions] = React.useState("");
  const [prosodyPlan, setProsodyPlan] = React.useState("off");
  const [styleInstructs, setStyleInstructs] = React.useState("");
  const [stylePlan, setStylePlan] = React.useState("off");
  const [stylePlanScope, setStylePlanScope] = React.useState("per-line");
  const [planChunk, setPlanChunk] = React.useState("");
  const [chatterboxExaggeration, setChatterboxExaggeration] = React.useState("");
  const [chatterboxCfg, setChatterboxCfg] = React.useState("");
  const [chatterboxTemp, setChatterboxTemp] = React.useState("");
  const [fishTemp, setFishTemp] = React.useState("");
  const [fishTopP, setFishTopP] = React.useState("");
  const [fishQuantize, setFishQuantize] = React.useState("int8");
  const [fishTagMode, setFishTagMode] = React.useState("keep");
  const [device, setDevice] = React.useState("auto");
  const [dtype, setDtype] = React.useState("auto");
  const [force, setForce] = React.useState(false);
  const [provenance, setProvenance] = React.useState(true);
  const [voices, setVoices] = React.useState<VoiceInfo[]>([]);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    api.voices().then((data) => setVoices(data.voices)).catch(() => undefined);
    api
      .settings()
      .then((data) => {
        if (data.default_device) setDevice(data.default_device);
        if (data.default_dtype) setDtype(data.default_dtype);
      })
      .catch(() => undefined);
  }, []);

  React.useEffect(() => {
    if (backend === "customvoice") setSource("speaker");
    else if (source === "speaker") setSource("voice");
  }, [backend]);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      const spec: Record<string, unknown> = {
        text,
        backend,
        language: language || undefined,
        save_as: saveAs || undefined,
        device,
        dtype,
        force,
        provenance,
        prosody: perLine(prosody),
        emotions: list(emotions),
        prosody_plan: prosodyPlan,
        style_instructs: perLine(styleInstructs),
        style_plan: stylePlan,
        style_plan_scope: stylePlanScope,
        plan_chunk: num(planChunk),
        chatterbox_exaggeration: num(chatterboxExaggeration),
        chatterbox_cfg_weight: num(chatterboxCfg),
        chatterbox_temperature: num(chatterboxTemp),
        fish_temperature: num(fishTemp),
        fish_top_p: num(fishTopP),
        fish_quantize: backend === "fish" ? fishQuantize : undefined,
        fish_tag_mode: fishTagMode,
      };
      if (source === "voice") spec.voice = voice || undefined;
      if (source === "design") spec.design = design || undefined;
      if (source === "clone") {
        spec.clone = clone || undefined;
        spec.clone_text = cloneText || undefined;
      }
      if (source === "speaker") spec.speaker = speaker || undefined;
      const job = await api.submitSay(spec);
      onJobStarted(job.job_id);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not start the job");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Section
      title="Voice generation"
      subtitle="Speak text with a saved, designed or cloned voice — with per-line prosody."
    >
      <Card className="space-y-5 p-5">
        <Field label="Text" hint="One line becomes one utterance. Blank lines are ignored.">
          <Textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder={"Bonsoir.\nVoici une seconde ligne, plus calme."}
          />
        </Field>

        <div className="grid gap-4 md:grid-cols-2">
          <Field label="Backend">
            <Select
              value={backend}
              onChange={(event) => setBackend(event.target.value)}
              options={[
                { value: "qwen", label: "Qwen Base (cloning)" },
                { value: "customvoice", label: "Qwen CustomVoice (preset speaker)" },
                { value: "cosyvoice3", label: "CosyVoice3" },
                { value: "chatterbox", label: "Chatterbox" },
                { value: "fish", label: "Fish S2-Pro" },
              ]}
            />
          </Field>
          <Field label="Language">
            <Select
              value={language}
              onChange={(event) => setLanguage(event.target.value)}
              options={LANGUAGES.map((item) => ({ value: item, label: item }))}
            />
          </Field>
        </div>

        {backend !== "customvoice" ? (
          <div className="space-y-4">
            <Field label="Voice source">
              <Select
                value={source}
                onChange={(event) => setSource(event.target.value as typeof source)}
                options={[
                  { value: "voice", label: "Saved voice" },
                  { value: "design", label: "Describe a new voice" },
                  { value: "clone", label: "Clone a clip" },
                ]}
              />
            </Field>
            {source === "voice" ? (
              <Field label="Saved voice">
                <Select
                  value={voice}
                  onChange={(event) => setVoice(event.target.value)}
                  options={[
                    { value: "", label: "Choose a voice" },
                    ...voices.map((item) => ({
                      value: item.label,
                      label: `${item.label} (${item.language})`,
                    })),
                  ]}
                />
              </Field>
            ) : null}
            {source === "design" ? (
              <Field label="Voice description">
                <Input
                  value={design}
                  onChange={(event) => setDesign(event.target.value)}
                  placeholder="a warm female French voice, slow and clear"
                />
              </Field>
            ) : null}
            {source === "clone" ? (
              <div className="grid gap-4 md:grid-cols-2">
                <UploadField
                  label="Reference clip"
                  accept=".wav,.mp3,.flac,.m4a,.ogg,.opus"
                  onUploaded={(path, name) => {
                    setClone(path);
                    setCloneName(name);
                  }}
                />
                <Field label="Clip transcript" hint="Optional; improves similarity and accent.">
                  <Input value={cloneText} onChange={(event) => setCloneText(event.target.value)} />
                </Field>
              </div>
            ) : null}
          </div>
        ) : (
          <Field label="Preset speaker" hint="e.g. Vivian or Ryan.">
            <Input value={speaker} onChange={(event) => setSpeaker(event.target.value)} />
          </Field>
        )}

        <Field label="Save as voice" hint="Optional: store an inline design or clone in the library.">
          <Input value={saveAs} onChange={(event) => setSaveAs(event.target.value)} placeholder="leave empty to keep it one-off" />
        </Field>

        <Disclosure title="Prosody" hint="Emotion, pace, volume and paralinguistic cues">
          <div className="grid gap-4 md:grid-cols-2">
            <Field label="Prosody plan">
              <Select
                value={prosodyPlan}
                onChange={(event) => setProsodyPlan(event.target.value)}
                options={[
                  { value: "off", label: "Use the values below" },
                  { value: "llm", label: "Ask the LLM to plan" },
                ]}
              />
            </Field>
            <Field label="Emotion shorthand" hint="Comma-separated, applied to every line.">
              <Input value={emotions} onChange={(event) => setEmotions(event.target.value)} placeholder="calm, warm" />
            </Field>
          </div>
          <Field
            label="Per-line prosody"
            hint="One spec per line: emotion=happy, pace=fast, volume=loud, breath, pause, laughter, exaggeration=0.8"
          >
            <Textarea
              value={prosody}
              onChange={(event) => setProsody(event.target.value)}
              placeholder={"emotion=calm, pace=slow\nemotion=excited, breath"}
            />
          </Field>
          {backend === "customvoice" ? (
            <>
              <Field label="Verbatim style instructions" hint="One per line; wins over prosody.">
                <Textarea
                  value={styleInstructs}
                  onChange={(event) => setStyleInstructs(event.target.value)}
                  placeholder="Speak with restrained anger, then fade to exhaustion."
                />
              </Field>
              <div className="grid gap-4 md:grid-cols-2">
                <Field label="Style plan">
                  <Select
                    value={stylePlan}
                    onChange={(event) => setStylePlan(event.target.value)}
                    options={[
                      { value: "off", label: "Use the instructions above" },
                      { value: "llm", label: "Ask the LLM to write them" },
                    ]}
                  />
                </Field>
                <Field label="Style plan scope">
                  <Select
                    value={stylePlanScope}
                    onChange={(event) => setStylePlanScope(event.target.value)}
                    options={[
                      { value: "per-line", label: "Per line" },
                      { value: "whole", label: "One for the whole text" },
                    ]}
                  />
                </Field>
              </div>
            </>
          ) : null}
          {backend === "fish" ? (
            <Field label="Fish tag mode">
              <Select
                value={fishTagMode}
                onChange={(event) => setFishTagMode(event.target.value)}
                options={[
                  { value: "keep", label: "Keep the tags I wrote" },
                  { value: "llm", label: "Let the LLM add tags" },
                ]}
              />
            </Field>
          ) : null}
          <Field label="Planning chunk size" hint="Lines per LLM planning request.">
            <Input value={planChunk} onChange={(event) => setPlanChunk(event.target.value)} placeholder="from settings" />
          </Field>
        </Disclosure>

        <Disclosure title="Backend tuning and hardware" hint="Expression, sampling and precision">
          {backend === "chatterbox" ? (
            <div className="grid gap-4 md:grid-cols-3">
              <Field label="Exaggeration">
                <Input value={chatterboxExaggeration} onChange={(event) => setChatterboxExaggeration(event.target.value)} />
              </Field>
              <Field label="CFG weight">
                <Input value={chatterboxCfg} onChange={(event) => setChatterboxCfg(event.target.value)} />
              </Field>
              <Field label="Temperature">
                <Input value={chatterboxTemp} onChange={(event) => setChatterboxTemp(event.target.value)} />
              </Field>
            </div>
          ) : null}
          {backend === "fish" ? (
            <div className="grid gap-4 md:grid-cols-3">
              <Field label="Temperature">
                <Input value={fishTemp} onChange={(event) => setFishTemp(event.target.value)} />
              </Field>
              <Field label="Top-p">
                <Input value={fishTopP} onChange={(event) => setFishTopP(event.target.value)} />
              </Field>
              <Field label="Quantization">
                <Select
                  value={fishQuantize}
                  onChange={(event) => setFishQuantize(event.target.value)}
                  options={[
                    { value: "int8", label: "int8" },
                    { value: "off", label: "bf16" },
                  ]}
                />
              </Field>
            </div>
          ) : null}
          <div className="grid gap-4 md:grid-cols-2">
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
            <Toggle label="Ignore caches" checked={force} onChange={setForce} />
            <Toggle label="Write provenance" checked={provenance} onChange={setProvenance} />
          </div>
        </Disclosure>

        {error ? <ErrorNote message={error} /> : null}
        {cloneName ? <div className="text-xs text-slate-500">Clip: {cloneName}</div> : null}

        <div className="flex justify-end border-t border-white/5 pt-4">
          <Button onClick={submit} disabled={busy || !text.trim()}>
            {busy ? <Spinner /> : null}
            Generate speech
          </Button>
        </div>
      </Card>
    </Section>
  );
}
