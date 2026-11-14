/**
 * The option values every form shares, declared once.
 *
 * This module mirrors `pipeline/options.py`, which owns the spelling of every
 * value the CLI and the API accept. A control whose option list disagrees with
 * the server is a request the server rejects after the user filled the whole
 * form in, so the forms must never invent a value.
 *
 * Each vocabulary is declared as a `const` object: one literal per value,
 * reachable by name so a comparison reads (`accentMode === ACCENT_MODE.preserve`)
 * and is checked by the compiler. Everything else is derived from it:
 *
 * - `*S` — the values as an array, in CLI order;
 * - the exported union type — what a handler may return;
 * - `*_OPTIONS` — `{ value, label }` for `<Select>`.
 *
 * Backend ids and labels are deliberately absent: they come from
 * `GET /api/backends` (`capabilities.ts`), which reads the Python capability
 * table, so the pipeline stays the only place a backend is declared. The id
 * lists below exist only as a fallback for a failed metadata fetch.
 */

export type Option = { value: string; label: string };

function options(values: readonly string[], labels: string[]): Option[] {
  return values.map((value, index) => ({ value, label: labels[index] ?? value }));
}

/* -------------------------------------------------------------------------- */
/* Languages                                                                  */
/* -------------------------------------------------------------------------- */

/** Languages the dubbing pipeline offers. `pipeline/config.py` accepts these. */
export const LANGUAGES = [
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
] as const;

export type Language = (typeof LANGUAGES)[number];

export const LANGUAGE_OPTIONS: Option[] = LANGUAGES.map((language) => ({
  value: language,
  label: language,
}));

/* -------------------------------------------------------------------------- */
/* Hardware                                                                   */
/* -------------------------------------------------------------------------- */

export const DTYPE = {
  auto: "auto",
  bf16: "bf16",
  fp16: "fp16",
  fp32: "fp32",
} as const;
export type Dtype = (typeof DTYPE)[keyof typeof DTYPE];
export const DTYPES = Object.values(DTYPE);

export const DTYPE_OPTIONS: Option[] = options(DTYPES, [
  "Auto",
  "bfloat16",
  "float16",
  "float32",
]);

/** Device spellings that are not a concrete card; the rest come from the API. */
export const DEVICE_AUTO = "auto";
export const DEVICE_CPU = "cpu";

export const DEVICE_OPTIONS: Option[] = [
  { value: DEVICE_AUTO, label: "Auto" },
  { value: DEVICE_CPU, label: "CPU" },
];

export const STRETCHER = {
  auto: "auto",
  rubberband: "rubberband",
  sox: "sox",
  librosa: "librosa",
  none: "none",
} as const;
export type Stretcher = (typeof STRETCHER)[keyof typeof STRETCHER];
export const STRETCHERS = Object.values(STRETCHER);

export const SEPARATOR = {
  auto: "auto",
  demucs: "demucs",
  none: "none",
} as const;
export type Separator = (typeof SEPARATOR)[keyof typeof SEPARATOR];
export const SEPARATORS = Object.values(SEPARATOR);

export const SEPARATOR_OPTIONS: Option[] = options(SEPARATORS, [
  "Auto",
  "Demucs",
  "None",
]);

export const LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] as const;
export type LogLevel = (typeof LOG_LEVELS)[number];

export const LOG_LEVEL_OPTIONS: Option[] = LOG_LEVELS.map((level) => ({
  value: level,
  label: level,
}));

/* -------------------------------------------------------------------------- */
/* Synthesis                                                                  */
/* -------------------------------------------------------------------------- */

export const PROSODY_PLAN = {
  auto: "auto",
  llm: "llm",
  off: "off",
} as const;
export type ProsodyPlan = (typeof PROSODY_PLAN)[keyof typeof PROSODY_PLAN];
export const PROSODY_PLANS = Object.values(PROSODY_PLAN);

export const PROSODY_PLAN_OPTIONS: Option[] = options(PROSODY_PLANS, [
  "Auto",
  "Ask the LLM",
  "Neutral",
]);

/** `say` plans explicitly, so it has no `auto`. */
export const SAY_PROSODY_PLAN = {
  off: "off",
  llm: "llm",
} as const;
export type SayProsodyPlan = (typeof SAY_PROSODY_PLAN)[keyof typeof SAY_PROSODY_PLAN];
export const SAY_PROSODY_PLANS = Object.values(SAY_PROSODY_PLAN);

export const SAY_PROSODY_PLAN_OPTIONS: Option[] = options(SAY_PROSODY_PLANS, [
  "Use the values below",
  "Ask the LLM to plan",
]);

export const STYLE_PLAN = {
  off: "off",
  llm: "llm",
} as const;
export type StylePlan = (typeof STYLE_PLAN)[keyof typeof STYLE_PLAN];
export const STYLE_PLANS = Object.values(STYLE_PLAN);

export const STYLE_PLAN_OPTIONS: Option[] = options(STYLE_PLANS, [
  "Use the instructions above",
  "Ask the LLM to write them",
]);

export const STYLE_PLAN_SCOPE = {
  perLine: "per-line",
  whole: "whole",
} as const;
export type StylePlanScope = (typeof STYLE_PLAN_SCOPE)[keyof typeof STYLE_PLAN_SCOPE];
export const STYLE_PLAN_SCOPES = Object.values(STYLE_PLAN_SCOPE);

export const STYLE_PLAN_SCOPE_OPTIONS: Option[] = options(STYLE_PLAN_SCOPES, [
  "Per line",
  "One for the whole text",
]);

export const FISH_QUANTIZE = {
  int8: "int8",
  off: "off",
} as const;
export type FishQuantize = (typeof FISH_QUANTIZE)[keyof typeof FISH_QUANTIZE];
export const FISH_QUANTIZES = Object.values(FISH_QUANTIZE);

export const FISH_QUANTIZE_OPTIONS: Option[] = options(FISH_QUANTIZES, [
  "int8 (fits a 12 GB card)",
  "bf16 (~24 GB)",
]);

export const FISH_TAG_MODE = {
  keep: "keep",
  llm: "llm",
} as const;
export type FishTagMode = (typeof FISH_TAG_MODE)[keyof typeof FISH_TAG_MODE];
export const FISH_TAG_MODES = Object.values(FISH_TAG_MODE);

export const FISH_TAG_MODE_OPTIONS: Option[] = options(FISH_TAG_MODES, [
  "Keep the tags I wrote",
  "Let the LLM add tags",
]);

/* -------------------------------------------------------------------------- */
/* Voice and accent                                                           */
/* -------------------------------------------------------------------------- */

export const ACCENT_MODE = {
  native: "native",
  preserve: "preserve",
  speakerOnly: "speaker-only",
  referenceAccent: "reference-accent",
} as const;
export type AccentMode = (typeof ACCENT_MODE)[keyof typeof ACCENT_MODE];
export const ACCENT_MODES = Object.values(ACCENT_MODE);

export const ACCENT_MODE_OPTIONS: Option[] = options(ACCENT_MODES, [
  "Native target accent (anchor)",
  "Keep the speaker's accent",
  "Speaker timbre only",
  "Borrow a reference accent",
]);

export const ACCENT_MATCH = {
  off: "off",
  rate: "rate",
  gender: "gender",
} as const;
export type AccentMatch = (typeof ACCENT_MATCH)[keyof typeof ACCENT_MATCH];
export const ACCENT_MATCHES = Object.values(ACCENT_MATCH);

export const ACCENT_MATCH_OPTIONS: Option[] = options(ACCENT_MATCHES, [
  "Off",
  "Match speaking rate",
  "Match rate and gender",
]);

export const ACCENT_BLEND = {
  twoStage: "two-stage",
  singleStage: "single-stage",
} as const;
export type AccentBlend = (typeof ACCENT_BLEND)[keyof typeof ACCENT_BLEND];
export const ACCENT_BLENDS = Object.values(ACCENT_BLEND);

export const ACCENT_BLEND_OPTIONS: Option[] = options(ACCENT_BLENDS, [
  "Two-stage",
  "Single-stage",
]);

export const EXPRESSIVE_ACCENT = {
  donor: "donor",
  blend: "blend",
} as const;
export type ExpressiveAccent =
  (typeof EXPRESSIVE_ACCENT)[keyof typeof EXPRESSIVE_ACCENT];
export const EXPRESSIVE_ACCENTS = Object.values(EXPRESSIVE_ACCENT);

export const EXPRESSIVE_ACCENT_OPTIONS: Option[] = options(EXPRESSIVE_ACCENTS, [
  "Clone the donor",
  "Blend a native accent",
]);

/* -------------------------------------------------------------------------- */
/* Audio bed, timing and speakers                                             */
/* -------------------------------------------------------------------------- */

export const LOUDNESS = {
  matchSource: "match-source",
  lufs: "lufs",
} as const;
export type LoudnessPolicy = (typeof LOUDNESS)[keyof typeof LOUDNESS];
export const LOUDNESS_POLICIES = Object.values(LOUDNESS);

export const LOUDNESS_OPTIONS: Option[] = options(LOUDNESS_POLICIES, [
  "Match the source",
  "Absolute LUFS",
]);

export const ROOM = {
  off: "off",
  match: "match",
} as const;
export type Room = (typeof ROOM)[keyof typeof ROOM];
export const ROOMS = Object.values(ROOM);

export const ROOM_OPTIONS: Option[] = options(ROOMS, [
  "Dry voice-over",
  "Match the source room",
]);

export const LENGTH_BUDGET = {
  words: "words",
  off: "off",
} as const;
export type LengthBudget = (typeof LENGTH_BUDGET)[keyof typeof LENGTH_BUDGET];
export const LENGTH_BUDGETS = Object.values(LENGTH_BUDGET);

export const LENGTH_BUDGET_OPTIONS: Option[] = options(LENGTH_BUDGETS, [
  "Word budget",
  "Duration only",
]);

export const DIARIZER = {
  auto: "auto",
  none: "none",
} as const;
export type Diarizer = (typeof DIARIZER)[keyof typeof DIARIZER];
export const DIARIZERS = Object.values(DIARIZER);

export const DIARIZER_OPTIONS: Option[] = options(DIARIZERS, [
  "Separate speakers",
  "Single speaker",
]);

export const OVERLAP_MODE = {
  dominant: "dominant",
  separate: "separate",
} as const;
export type OverlapMode = (typeof OVERLAP_MODE)[keyof typeof OVERLAP_MODE];
export const OVERLAP_MODES = Object.values(OVERLAP_MODE);

export const OVERLAP_MODE_OPTIONS: Option[] = options(OVERLAP_MODES, [
  "Dominant speaker",
  "Separate both voices",
]);

export const OVERLAP_SEPARATOR = {
  auto: "auto",
  sepformer: "sepformer",
  none: "none",
} as const;
export type OverlapSeparator =
  (typeof OVERLAP_SEPARATOR)[keyof typeof OVERLAP_SEPARATOR];
export const OVERLAP_SEPARATORS = Object.values(OVERLAP_SEPARATOR);

export const OVERLAP_SEPARATOR_OPTIONS: Option[] = options(OVERLAP_SEPARATORS, [
  "Auto",
  "SepFormer",
  "None",
]);

export const SUBTITLES_MODE = {
  soft: "soft",
  burn: "burn",
  off: "off",
} as const;
export type SubtitlesMode = (typeof SUBTITLES_MODE)[keyof typeof SUBTITLES_MODE];
export const SUBTITLES_MODES = Object.values(SUBTITLES_MODE);

export const SUBTITLES_MODE_OPTIONS: Option[] = options(SUBTITLES_MODES, [
  "Soft track (toggleable)",
  "Burn into the video",
  "Sidecar only",
]);

/* -------------------------------------------------------------------------- */
/* Backends                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Ids of the synthesisers a dub can use. Mirrors `TtsBackend` in
 * `pipeline/options.py`, which is still the authority: `GET /api/backends`
 * supplies both ids and labels, and these are used for comparisons and as the
 * fallback while that fetch is in flight, so a metadata hiccup cannot empty the
 * picker. CustomVoice is absent on purpose - it speaks a preset speaker and
 * cannot carry a source speaker's timbre into another language.
 */
export const TTS_BACKEND = {
  qwen: "qwen",
  cosyvoice3: "cosyvoice3",
  chatterbox: "chatterbox",
  fish: "fish",
} as const;
export type TtsBackend = (typeof TTS_BACKEND)[keyof typeof TTS_BACKEND];
export const TTS_BACKENDS = Object.values(TTS_BACKEND);

/** Ids `dub.py say` can synthesise with. Mirrors `SayBackend`. */
export const SAY_BACKEND = {
  qwen: "qwen",
  customvoice: "customvoice",
  cosyvoice3: "cosyvoice3",
  chatterbox: "chatterbox",
  fish: "fish",
} as const;
export type SayBackend = (typeof SAY_BACKEND)[keyof typeof SAY_BACKEND];
export const SAY_BACKENDS = Object.values(SAY_BACKEND);

/* -------------------------------------------------------------------------- */
/* Defaults                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * What a form starts with when nothing else applies. Mirrors the pydantic
 * defaults in `pipeline/config.py`: a new job left untouched must send exactly
 * what the CLI would.
 */
export const DEFAULTS = {
  language: LANGUAGES[0],
  device: DEVICE_AUTO,
  dtype: DTYPE.auto,
  whisperModel: "large-v3",
  ttsBackend: TTS_BACKEND.qwen,
  sayBackend: SAY_BACKEND.qwen,
  prosodyPlan: PROSODY_PLAN.auto,
  sayProsodyPlan: SAY_PROSODY_PLAN.off,
  stylePlan: STYLE_PLAN.off,
  stylePlanScope: STYLE_PLAN_SCOPE.perLine,
  fishQuantize: FISH_QUANTIZE.int8,
  fishTagMode: FISH_TAG_MODE.keep,
  accentMode: ACCENT_MODE.native,
  accentMatch: ACCENT_MATCH.off,
  accentBlend: ACCENT_BLEND.twoStage,
  expressiveAccent: EXPRESSIVE_ACCENT.donor,
  loudness: LOUDNESS.matchSource,
  room: ROOM.off,
  separator: SEPARATOR.auto,
  lengthBudget: LENGTH_BUDGET.words,
  diarizer: DIARIZER.auto,
  overlapMode: OVERLAP_MODE.dominant,
  overlapSeparator: OVERLAP_SEPARATOR.auto,
  subtitlesMode: SUBTITLES_MODE.soft,
  stretcher: STRETCHER.auto,
} as const;
