import React from "react";
import { api } from "./api";
import type { BackendCapabilities } from "./types";

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
