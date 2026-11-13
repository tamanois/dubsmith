/**
 * The forms' option lists must describe the values the server accepts.
 *
 * `options.ts` derives each `{ value, label }` array from the vocabulary itself,
 * so these tests assert the invariants a future hand edit could break: a label
 * per value, no duplicate values, defaults that are real members, and backend
 * fallbacks that match the ids the API can return.
 */
import { describe, expect, it } from "vitest";
import {
  ACCENT_BLEND_OPTIONS,
  ACCENT_BLENDS,
  ACCENT_MATCH_OPTIONS,
  ACCENT_MATCHES,
  ACCENT_MODE_OPTIONS,
  ACCENT_MODES,
  DEFAULTS,
  DEVICE_OPTIONS,
  DIARIZER_OPTIONS,
  DIARIZERS,
  DTYPE_OPTIONS,
  DTYPES,
  EXPRESSIVE_ACCENT_OPTIONS,
  EXPRESSIVE_ACCENTS,
  FISH_QUANTIZE_OPTIONS,
  FISH_QUANTIZES,
  FISH_TAG_MODE_OPTIONS,
  FISH_TAG_MODES,
  LANGUAGES,
  LENGTH_BUDGET_OPTIONS,
  LENGTH_BUDGETS,
  LOG_LEVELS,
  LOUDNESS_OPTIONS,
  LOUDNESS_POLICIES,
  OVERLAP_MODE_OPTIONS,
  OVERLAP_MODES,
  OVERLAP_SEPARATOR_OPTIONS,
  OVERLAP_SEPARATORS,
  PROSODY_PLAN_OPTIONS,
  PROSODY_PLANS,
  ROOM_OPTIONS,
  ROOMS,
  SAY_BACKENDS,
  SAY_PROSODY_PLAN_OPTIONS,
  SAY_PROSODY_PLANS,
  SEPARATOR_OPTIONS,
  SEPARATORS,
  STYLE_PLAN_OPTIONS,
  STYLE_PLAN_SCOPE_OPTIONS,
  STYLE_PLAN_SCOPES,
  STYLE_PLANS,
  SUBTITLES_MODE_OPTIONS,
  SUBTITLES_MODES,
  TTS_BACKENDS,
  type Option,
} from "../src/options";

const VOCABULARIES: { name: string; values: string[]; options: Option[] }[] = [
  { name: "accent blend", values: ACCENT_BLENDS, options: ACCENT_BLEND_OPTIONS },
  { name: "accent match", values: ACCENT_MATCHES, options: ACCENT_MATCH_OPTIONS },
  { name: "accent mode", values: ACCENT_MODES, options: ACCENT_MODE_OPTIONS },
  { name: "diarizer", values: DIARIZERS, options: DIARIZER_OPTIONS },
  { name: "dtype", values: DTYPES, options: DTYPE_OPTIONS },
  {
    name: "expressive accent",
    values: EXPRESSIVE_ACCENTS,
    options: EXPRESSIVE_ACCENT_OPTIONS,
  },
  { name: "fish quantize", values: FISH_QUANTIZES, options: FISH_QUANTIZE_OPTIONS },
  { name: "fish tag mode", values: FISH_TAG_MODES, options: FISH_TAG_MODE_OPTIONS },
  { name: "length budget", values: LENGTH_BUDGETS, options: LENGTH_BUDGET_OPTIONS },
  { name: "loudness", values: LOUDNESS_POLICIES, options: LOUDNESS_OPTIONS },
  { name: "overlap mode", values: OVERLAP_MODES, options: OVERLAP_MODE_OPTIONS },
  {
    name: "overlap separator",
    values: OVERLAP_SEPARATORS,
    options: OVERLAP_SEPARATOR_OPTIONS,
  },
  { name: "prosody plan", values: PROSODY_PLANS, options: PROSODY_PLAN_OPTIONS },
  { name: "room", values: ROOMS, options: ROOM_OPTIONS },
  {
    name: "say prosody plan",
    values: SAY_PROSODY_PLANS,
    options: SAY_PROSODY_PLAN_OPTIONS,
  },
  { name: "separator", values: SEPARATORS, options: SEPARATOR_OPTIONS },
  { name: "style plan", values: STYLE_PLANS, options: STYLE_PLAN_OPTIONS },
  {
    name: "style plan scope",
    values: STYLE_PLAN_SCOPES,
    options: STYLE_PLAN_SCOPE_OPTIONS,
  },
  { name: "subtitles mode", values: SUBTITLES_MODES, options: SUBTITLES_MODE_OPTIONS },
];

describe("option vocabularies", () => {
  it.each(VOCABULARIES)("$name exposes one labelled option per value", ({ values, options }) => {
    expect(options.map((option) => option.value)).toEqual(values);
    expect(options.every((option) => option.label.trim().length > 0)).toBe(true);
  });

  it.each(VOCABULARIES)("$name has unique values and labels", ({ values, options }) => {
    expect(new Set(values).size).toBe(values.length);
    expect(new Set(options.map((option) => option.label)).size).toBe(options.length);
  });

  it("declares every vocabulary in the order the CLI lists it", () => {
    expect(ACCENT_MODES).toEqual([
      "native",
      "preserve",
      "speaker-only",
      "reference-accent",
    ]);
    expect(SUBTITLES_MODES).toEqual(["soft", "burn", "off"]);
    expect(DTYPES).toEqual(["auto", "bf16", "fp16", "fp32"]);
  });

  it("describes devices that are not a concrete card", () => {
    expect(DEVICE_OPTIONS.map((option) => option.value)).toEqual(["auto", "cpu"]);
  });

  it("offers log levels quietest last", () => {
    expect(LOG_LEVELS).toEqual(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]);
  });
});

describe("languages", () => {
  it("lists each language once", () => {
    expect(new Set(LANGUAGES).size).toBe(LANGUAGES.length);
    expect(LANGUAGES).toContain("French");
  });
});

describe("form defaults", () => {
  it.each([
    ["dtype", DEFAULTS.dtype, DTYPES],
    ["prosody plan", DEFAULTS.prosodyPlan, PROSODY_PLANS],
    ["say prosody plan", DEFAULTS.sayProsodyPlan, SAY_PROSODY_PLANS],
    ["style plan", DEFAULTS.stylePlan, STYLE_PLANS],
    ["style plan scope", DEFAULTS.stylePlanScope, STYLE_PLAN_SCOPES],
    ["fish quantize", DEFAULTS.fishQuantize, FISH_QUANTIZES],
    ["fish tag mode", DEFAULTS.fishTagMode, FISH_TAG_MODES],
    ["accent mode", DEFAULTS.accentMode, ACCENT_MODES],
    ["accent match", DEFAULTS.accentMatch, ACCENT_MATCHES],
    ["expressive accent", DEFAULTS.expressiveAccent, EXPRESSIVE_ACCENTS],
    ["loudness", DEFAULTS.loudness, LOUDNESS_POLICIES],
    ["room", DEFAULTS.room, ROOMS],
    ["separator", DEFAULTS.separator, SEPARATORS],
    ["length budget", DEFAULTS.lengthBudget, LENGTH_BUDGETS],
    ["diarizer", DEFAULTS.diarizer, DIARIZERS],
    ["overlap mode", DEFAULTS.overlapMode, OVERLAP_MODES],
    ["overlap separator", DEFAULTS.overlapSeparator, OVERLAP_SEPARATORS],
    ["subtitles mode", DEFAULTS.subtitlesMode, SUBTITLES_MODES],
  ])("default %s is a declared value", (_name, value, values) => {
    expect(values).toContain(value);
  });

  it("starts a dub with the pipeline's own defaults", () => {
    expect(DEFAULTS.ttsBackend).toBe("qwen");
    expect(DEFAULTS.accentMode).toBe("native");
    expect(DEFAULTS.subtitlesMode).toBe("soft");
    expect(DEFAULTS.language).toBe(LANGUAGES[0]);
  });
});

describe("backend fallbacks", () => {
  it("lists dub backends once, without the preset-speaker backend", () => {
    expect(new Set(TTS_BACKENDS).size).toBe(TTS_BACKENDS.length);
    expect(TTS_BACKENDS).not.toContain("customvoice");
    expect(SAY_BACKENDS).toContain("customvoice");
  });

  it("offers every dub backend to say as well", () => {
    expect(TTS_BACKENDS.every((backend) => SAY_BACKENDS.includes(backend))).toBe(true);
  });
});
