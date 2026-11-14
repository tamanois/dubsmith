import React from "react";
import { api } from "../api";
import { backendOptions, capabilitiesOf, useBackends } from "../capabilities";
import type { Job, VoiceInfo } from "../types";
import {
  DEFAULTS,
  DEVICE_OPTIONS,
  DTYPE_OPTIONS,
  FISH_QUANTIZE_OPTIONS,
  FISH_TAG_MODE,
  FISH_TAG_MODE_OPTIONS,
  LANGUAGE_OPTIONS,
  SAY_BACKEND,
  SAY_PROSODY_PLAN,
  SAY_PROSODY_PLAN_OPTIONS,
  STYLE_PLAN,
  STYLE_PLAN_OPTIONS,
  STYLE_PLAN_SCOPE,
  STYLE_PLAN_SCOPE_OPTIONS,
} from "../options";
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
import { PresetPanel } from "../components/PresetPanel";
import { RecentInputs } from "../components/RecentInputs";

function perLine(value: string): string[] | undefined {
  const lines = value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.length ? lines : undefined;
}

function asText(value: unknown, fallback = ""): string {
  return value === undefined || value === null ? fallback : String(value);
}

function joined(value: unknown, separator: string): string {
  return Array.isArray(value) ? value.map(String).join(separator) : "";
}

type CapabilityFlag =
  | "prosody"
  | "style"
  | "fish_tuning"
  | "chatterbox_tuning"
  | "save_as";

export function SayPage({
  onJobStarted,
  initialJob,
}: {
  onJobStarted: (id: string) => void;
  initialJob?: Job | null;
}) {
  const [text, setText] = React.useState("");
  const [backend, setBackend] = React.useState<string>(DEFAULTS.sayBackend);
  const [source, setSource] = React.useState<"voice" | "design" | "clone" | "speaker">("voice");
  const [voice, setVoice] = React.useState("");
  const [design, setDesign] = React.useState("");
  const [clone, setClone] = React.useState("");
  const [cloneName, setCloneName] = React.useState("");
  const [cloneText, setCloneText] = React.useState("");
  const [speaker, setSpeaker] = React.useState("");
  const [language, setLanguage] = React.useState<string>(DEFAULTS.language);
  const [saveAs, setSaveAs] = React.useState("");
  const [prosody, setProsody] = React.useState("");
  const [emotions, setEmotions] = React.useState("");
  const [prosodyPlan, setProsodyPlan] = React.useState<string>(
    DEFAULTS.sayProsodyPlan,
  );
  const [styleInstructs, setStyleInstructs] = React.useState("");
  const [stylePlan, setStylePlan] = React.useState<string>(DEFAULTS.stylePlan);
  const [stylePlanScope, setStylePlanScope] = React.useState<string>(
    DEFAULTS.stylePlanScope,
  );
  const [planChunk, setPlanChunk] = React.useState("");
  const [chatterboxExaggeration, setChatterboxExaggeration] = React.useState("");
  const [chatterboxCfg, setChatterboxCfg] = React.useState("");
  const [chatterboxTemp, setChatterboxTemp] = React.useState("");
  const [fishTemp, setFishTemp] = React.useState("");
  const [fishTopP, setFishTopP] = React.useState("");
  const [fishQuantize, setFishQuantize] = React.useState<string>(
    DEFAULTS.fishQuantize,
  );
  const [fishTagMode, setFishTagMode] = React.useState<string>(DEFAULTS.fishTagMode);
  const [device, setDevice] = React.useState<string>(DEFAULTS.device);
  const [dtype, setDtype] = React.useState<string>(DEFAULTS.dtype);
  const [force, setForce] = React.useState(false);
  const [provenance, setProvenance] = React.useState(true);
  const [voices, setVoices] = React.useState<VoiceInfo[]>([]);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    api.voices({ pageSize: 200 }).then((data) => setVoices(data.voices)).catch(() => undefined);
    api
      .settings()
      .then((data) => {
        if (data.default_device) setDevice(data.default_device);
        if (data.default_dtype) setDtype(data.default_dtype);
      })
      .catch(() => undefined);
  }, []);

  const backends = useBackends("say");
  const caps = capabilitiesOf(backends, backend);
  // Unknown capabilities => permissive: the server is still the authority, and a
  // metadata hiccup must never block every form.
  const allows = (feature: CapabilityFlag) => (caps ? caps[feature] : true);
  const reasonFor = (feature: CapabilityFlag) =>
    caps && !caps[feature] ? caps.unsupported?.[feature] : undefined;

  const applySpec = React.useCallback((spec: Record<string, unknown>) => {
    setText(asText(spec.text));
    setBackend(asText(spec.backend, DEFAULTS.sayBackend));
    setLanguage(asText(spec.language, DEFAULTS.language));
    setSaveAs(asText(spec.save_as));
    setDevice(asText(spec.device, DEFAULTS.device));
    setDtype(asText(spec.dtype, DEFAULTS.dtype));
    setForce(Boolean(spec.force));
    setProvenance(spec.provenance === undefined ? true : Boolean(spec.provenance));
    setProsody(joined(spec.prosody, "\n"));
    setEmotions(joined(spec.emotions, ", "));
    setProsodyPlan(asText(spec.prosody_plan, DEFAULTS.sayProsodyPlan));
    setStyleInstructs(joined(spec.style_instructs, "\n"));
    setStylePlan(asText(spec.style_plan, DEFAULTS.stylePlan));
    setStylePlanScope(asText(spec.style_plan_scope, DEFAULTS.stylePlanScope));
    setPlanChunk(asText(spec.plan_chunk));
    setChatterboxExaggeration(asText(spec.chatterbox_exaggeration));
    setChatterboxCfg(asText(spec.chatterbox_cfg_weight));
    setChatterboxTemp(asText(spec.chatterbox_temperature));
    setFishTemp(asText(spec.fish_temperature));
    setFishTopP(asText(spec.fish_top_p));
    setFishQuantize(asText(spec.fish_quantize, DEFAULTS.fishQuantize));
    setFishTagMode(asText(spec.fish_tag_mode, DEFAULTS.fishTagMode));

    if (spec.speaker) {
      setSource("speaker");
      setSpeaker(asText(spec.speaker));
    } else if (spec.design) {
      setSource("design");
      setDesign(asText(spec.design));
    } else if (spec.clone) {
      setSource("clone");
      setClone(asText(spec.clone));
      setCloneName(asText(spec.clone).split(/[\\/]/).pop() ?? "");
      setCloneText(asText(spec.clone_text));
    } else {
      setSource("voice");
      setVoice(asText(spec.voice));
    }
  }, []);

  // Prefill from a past job so "Edit" opens the same options, not a blank form.
  React.useEffect(() => {
    if (!initialJob) return;
    applySpec((initialJob.params?.spec ?? {}) as Record<string, unknown>);
  }, [initialJob, applySpec]);

  React.useEffect(() => {
    if (backend === SAY_BACKEND.customvoice) setSource("speaker");
    else if (source === "speaker") setSource("voice");
  }, [backend]);

  // A backend change must not leave a stale value behind for a control it does
  // not support; the form shows the truth, so the payload should too.
  React.useEffect(() => {
    if (!caps) return;
    if (!caps.prosody) {
      setProsody("");
      setEmotions("");
      setProsodyPlan(SAY_PROSODY_PLAN.off);
    }
    if (!caps.style) {
      setStyleInstructs("");
      setStylePlan(STYLE_PLAN.off);
      setStylePlanScope(STYLE_PLAN_SCOPE.perLine);
    }
    if (!caps.fish_tuning) {
      setFishTemp("");
      setFishTopP("");
      setFishTagMode(FISH_TAG_MODE.keep);
    }
    if (!caps.chatterbox_tuning) {
      setChatterboxExaggeration("");
      setChatterboxCfg("");
      setChatterboxTemp("");
    }
    if (!caps.save_as) setSaveAs("");
  }, [caps]);

  // Combinations the pipeline rejects. Shown inline and used to block submit, so
  // an impossible request is never sent and never becomes a failed job.
  const blockers: string[] = [];
  if (source === "voice" && !voice) {
    blockers.push("Choose a saved voice, or switch the voice source.");
  }
  if (source === "design" && !design.trim()) {
    blockers.push("Describe the voice to design.");
  }
  if (source === "clone" && !clone) {
    blockers.push("Add a reference clip to clone.");
  }
  if (source === "speaker" && !speaker.trim()) {
    blockers.push("Pick or type a preset speaker, e.g. Vivian.");
  }
  if (source === "speaker" && (!language || language === "Auto")) {
    blockers.push("CustomVoice needs an explicit language.");
  }

  function buildSpec(): Record<string, unknown> {
    const spec: Record<string, unknown> = {
      text,
      backend,
      language: language || undefined,
      save_as: allows("save_as") ? saveAs || undefined : undefined,
      device,
      dtype,
      force,
      provenance,
      // Strip anything the selected backend cannot honour. This is defence in
      // depth: a preset or a restored edit can carry a value the controls no
      // longer show.
      prosody: allows("prosody") ? perLine(prosody) : undefined,
      emotions: allows("prosody") ? list(emotions) : undefined,
      prosody_plan: allows("prosody") ? prosodyPlan : SAY_PROSODY_PLAN.off,
      style_instructs: allows("style") ? perLine(styleInstructs) : undefined,
      style_plan: allows("style") ? stylePlan : STYLE_PLAN.off,
      style_plan_scope: allows("style") ? stylePlanScope : STYLE_PLAN_SCOPE.perLine,
      plan_chunk: num(planChunk),
      chatterbox_exaggeration: allows("chatterbox_tuning")
        ? num(chatterboxExaggeration)
        : undefined,
      chatterbox_cfg_weight: allows("chatterbox_tuning") ? num(chatterboxCfg) : undefined,
      chatterbox_temperature: allows("chatterbox_tuning")
        ? num(chatterboxTemp)
        : undefined,
      fish_temperature: allows("fish_tuning") ? num(fishTemp) : undefined,
      fish_top_p: allows("fish_tuning") ? num(fishTopP) : undefined,
      fish_quantize: allows("fish_tuning") ? fishQuantize : undefined,
      fish_tag_mode: allows("fish_tuning") ? fishTagMode : FISH_TAG_MODE.keep,
    };
    if (source === "voice") spec.voice = voice || undefined;
    if (source === "design") spec.design = design || undefined;
    if (source === "clone") {
      spec.clone = clone || undefined;
      spec.clone_text = cloneText || undefined;
    }
    if (source === "speaker") spec.speaker = speaker || undefined;
    return spec;
  }

  function buildPresetPayload(): Record<string, unknown> {
    const spec = { ...buildSpec() };
    // The spoken text and the donated clip are content, not reusable options.
    delete spec.text;
    delete spec.clone;
    return spec;
  }

  function applyPreset(payload: Record<string, unknown>) {
    // Keep whatever the user has typed; a preset only changes the options.
    applySpec({ ...payload, text });
  }

  async function submit() {
    if (blockers.length) {
      setError(blockers[0]);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const job = await api.submitSay(buildSpec());
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
        <Field
          label="Text"
          tooltip="The words to speak. Each non-empty line becomes one utterance, so line breaks control prosody and timing."
          hint="One line becomes one utterance. Blank lines are ignored."
        >
          <Textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder={"Bonsoir.\nVoici une seconde ligne, plus calme."}
          />
        </Field>

        <div className="grid gap-4 md:grid-cols-2">
          <Field
            label="Backend"
            tooltip="Voice engine. Qwen Base clones; CustomVoice uses a preset speaker; the others add emotion controls."
          >
            <Select
              value={backend}
              onChange={(event) => setBackend(event.target.value)}
              options={backendOptions(backends, "say")}
            />
          </Field>
            <Field label="Language">
            <Select
              value={language}
              onChange={(event) => setLanguage(event.target.value)}
              options={LANGUAGE_OPTIONS}
            />
          </Field>
        </div>

        {backend !== SAY_BACKEND.customvoice ? (
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
                  tooltip="The clip whose voice is cloned. A clean, single-speaker recording works best."
                  accept=".wav,.mp3,.flac,.m4a,.ogg,.opus"
                  value={clone}
                  onUploaded={(path, name) => {
                    setClone(path);
                    setCloneName(name);
                  }}
                />
                <Field
                  label="Clip transcript"
                  tooltip="The exact words in the clip; it improves similarity and accent."
                  hint="Optional; improves similarity and accent."
                >
                  <Input value={cloneText} onChange={(event) => setCloneText(event.target.value)} />
                </Field>
              </div>
            ) : null}
            {source === "clone" ? (
              <RecentInputs
                onPick={(path, name) => {
                  setClone(path);
                  setCloneName(name);
                }}
                current={clone ? { path: clone, name: cloneName || clone } : null}
                accept=".wav,.mp3,.flac,.m4a,.ogg,.opus"
              />
            ) : null}
          </div>
        ) : (
          <Field label="Preset speaker" hint="e.g. Vivian or Ryan.">
            <Input value={speaker} onChange={(event) => setSpeaker(event.target.value)} />
          </Field>
        )}

        <Field
          label="Save as voice"
          tooltip="Store an inline design or clone in the library for later use."
          hint={
            allows("save_as")
              ? "Optional: store an inline design or clone in the library."
              : reasonFor("save_as")
          }
        >
          <Input
            value={saveAs}
            disabled={!allows("save_as")}
            onChange={(event) => setSaveAs(event.target.value)}
            placeholder="leave empty to keep it one-off"
          />
        </Field>

        {/* Basic: emotion and pacing, which most people tweak. */}
        <div className="grid gap-4 md:grid-cols-2">
          <Field
            label="Prosody plan"
            tooltip="Who decides emotion and pace. 'Use the values below' keeps your input; 'Ask the LLM' plans it per line."
            hint={allows("prosody") ? undefined : reasonFor("prosody")}
          >
            <Select
              value={prosodyPlan}
              disabled={!allows("prosody")}
              onChange={(event) => setProsodyPlan(event.target.value)}
              options={SAY_PROSODY_PLAN_OPTIONS}
            />
          </Field>
          <Field
            label="Emotion shorthand"
            tooltip="Comma-separated emotions applied to every line, e.g. calm, warm."
            hint={
              allows("prosody")
                ? "Comma-separated, applied to every line."
                : reasonFor("prosody")
            }
          >
            <Input
              value={emotions}
              disabled={!allows("prosody")}
              onChange={(event) => setEmotions(event.target.value)}
              placeholder="calm, warm"
            />
          </Field>
        </div>
        <Field
          label="Per-line prosody"
          tooltip="One spec per line, overriding the shorthand for that utterance."
          hint={
            allows("prosody")
              ? "One spec per line: emotion=happy, pace=fast, volume=loud, breath, pause, laughter, exaggeration=0.8"
              : reasonFor("prosody")
          }
        >
          <Textarea
            value={prosody}
            disabled={!allows("prosody")}
            onChange={(event) => setProsody(event.target.value)}
            placeholder={"emotion=calm, pace=slow\nemotion=excited, breath"}
          />
        </Field>

        <Disclosure title="Advanced options" hint="Style planning, sampling and hardware">
          {backend === SAY_BACKEND.customvoice ? (
            <>
              <Field
                label="Verbatim style instructions"
                tooltip="Exact style directions for the preset speaker; they win over the prosody above."
                hint="One per line; wins over prosody."
              >
                <Textarea
                  value={styleInstructs}
                  onChange={(event) => setStyleInstructs(event.target.value)}
                  placeholder="Speak with restrained anger, then fade to exhaustion."
                />
              </Field>
              <div className="grid gap-4 md:grid-cols-2">
                <Field label="Style plan" tooltip="Keep your instructions, or let the LLM write them.">
                  <Select
                    value={stylePlan}
                    onChange={(event) => setStylePlan(event.target.value)}
                    options={STYLE_PLAN_OPTIONS}
                  />
                </Field>
                <Field label="Style plan scope" tooltip="One style per line, or a single style for the whole text.">
                  <Select
                    value={stylePlanScope}
                    onChange={(event) => setStylePlanScope(event.target.value)}
                    options={STYLE_PLAN_SCOPE_OPTIONS}
                  />
                </Field>
              </div>
            </>
          ) : null}
          {backend === SAY_BACKEND.fish ? (
            <Field label="Fish tag mode" tooltip="Keep the [tags] you wrote, or let the LLM add them.">
              <Select
                value={fishTagMode}
                onChange={(event) => setFishTagMode(event.target.value)}
                options={FISH_TAG_MODE_OPTIONS}
              />
            </Field>
          ) : null}
          <Field
            label="Planning chunk size"
            tooltip="How many lines go into one LLM planning request. Larger is fewer calls but less precise."
            hint="Lines per LLM planning request."
          >
            <Input value={planChunk} onChange={(event) => setPlanChunk(event.target.value)} placeholder="from settings" />
          </Field>
          {backend === SAY_BACKEND.chatterbox ? (
            <div className="grid gap-4 md:grid-cols-3">
              <Field label="Exaggeration" tooltip="How theatrical the delivery is.">
                <Input value={chatterboxExaggeration} onChange={(event) => setChatterboxExaggeration(event.target.value)} />
              </Field>
              <Field label="CFG weight" tooltip="Guidance strength; higher stays closer to the prompt.">
                <Input value={chatterboxCfg} onChange={(event) => setChatterboxCfg(event.target.value)} />
              </Field>
              <Field label="Temperature" tooltip="Sampling randomness; lower is steadier.">
                <Input value={chatterboxTemp} onChange={(event) => setChatterboxTemp(event.target.value)} />
              </Field>
            </div>
          ) : null}
          {backend === SAY_BACKEND.fish ? (
            <div className="grid gap-4 md:grid-cols-3">
              <Field label="Temperature" tooltip="Sampling randomness; lower is steadier.">
                <Input value={fishTemp} onChange={(event) => setFishTemp(event.target.value)} />
              </Field>
              <Field label="Top-p" tooltip="Nucleus sampling cut-off; lower restricts to more likely tokens.">
                <Input value={fishTopP} onChange={(event) => setFishTopP(event.target.value)} />
              </Field>
              <Field label="Quantization" tooltip="int8 fits smaller GPUs; bf16 needs more memory but is a little more faithful.">
                <Select
                  value={fishQuantize}
                  onChange={(event) => setFishQuantize(event.target.value)}
                  options={FISH_QUANTIZE_OPTIONS}
                />
              </Field>
            </div>
          ) : null}
          <div className="grid gap-4 md:grid-cols-2">
            <Field label="Device" tooltip="Hardware target. Auto picks the best available GPU.">
              <Select
                value={device}
                onChange={(event) => setDevice(event.target.value)}
                options={[
                  ...DEVICE_OPTIONS,
                  { value: "cuda:0", label: "cuda:0" },
                  { value: "cuda:1", label: "cuda:1" },
                ]}
              />
            </Field>
            <Field label="Precision" tooltip="Numeric precision. Auto uses bfloat16/float16 when supported.">
              <Select
                value={dtype}
                onChange={(event) => setDtype(event.target.value)}
                options={DTYPE_OPTIONS}
              />
            </Field>
            <Toggle label="Ignore caches" checked={force} onChange={setForce} />
            <Toggle label="Write provenance" checked={provenance} onChange={setProvenance} />
          </div>
        </Disclosure>

        <Disclosure title="Presets" hint="Save and reuse these options">
          <PresetPanel kind="say" buildPayload={buildPresetPayload} onApply={applyPreset} />
        </Disclosure>

        {blockers.length ? <ErrorNote message={blockers.join(" ")} /> : null}
        {error ? <ErrorNote message={error} /> : null}
        {cloneName ? <div className="text-xs text-slate-500">Clip: {cloneName}</div> : null}

        <div className="flex justify-end border-t border-white/5 pt-4">
          <Button onClick={submit} disabled={busy || !text.trim() || blockers.length > 0}>
            {busy ? <Spinner /> : null}
            Generate speech
          </Button>
        </div>
      </Card>
    </Section>
  );
}
