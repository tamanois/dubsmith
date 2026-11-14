import React from "react";
import { api } from "./api";
import type { BackendCapabilities } from "./types";
import { SAY_BACKENDS, TTS_BACKENDS, type Option } from "./options";

export type BackendMap = Record<string, BackendCapabilities>;

type Groups = { say: BackendMap; dub: BackendMap };

// One fetch per page load, shared by every form. A failed fetch resolves to empty
// maps on purpose: the forms then stay permissive and the server stays the
// authority, so a metadata hiccup can never block all work.
let cache: Promise<Groups> | null = null;

function load(): Promise<Groups> {
  if (!cache) {
    cache = (async () => {
      try {
        const data = await api.backends();
        return {
          say: Object.fromEntries(
            (data.say ?? []).map((caps) => [caps.id, caps]),
          ),
          dub: Object.fromEntries(
            (data.dub ?? []).map((caps) => [caps.id, caps]),
          ),
        };
      } catch {
        return { say: {}, dub: {} };
      }
    })();
  }
  return cache;
}

/** Capabilities for one job kind, keyed by backend id. Empty until loaded. */
export function useBackends(kind: "say" | "dub"): BackendMap {
  const [map, setMap] = React.useState<BackendMap>({});
  React.useEffect(() => {
    let alive = true;
    load().then((groups) => {
      if (alive) setMap(groups[kind]);
    });
    return () => {
      alive = false;
    };
  }, [kind]);
  return map;
}

/** The selected backend's capabilities, or null while unknown (=> permissive). */
export function capabilitiesOf(
  map: BackendMap,
  backend: string,
): BackendCapabilities | null {
  return map[backend] ?? null;
}

/**
 * Picker options for one job kind, in the order the server lists them.
 *
 * The API owns the ids *and* the labels, so a renamed backend needs no frontend
 * change. Before the fetch resolves - or if it fails - the known ids are shown
 * with the id as the label: a metadata hiccup must not leave the picker empty,
 * because the server would still accept the job.
 */
export function backendOptions(map: BackendMap, kind: "say" | "dub"): Option[] {
  const options = Object.values(map).map((caps) => ({
    value: caps.id,
    label: caps.label,
  }));
  if (options.length) return options;
  const fallback = kind === "say" ? SAY_BACKENDS : TTS_BACKENDS;
  return fallback.map((id) => ({ value: id, label: id }));
}
