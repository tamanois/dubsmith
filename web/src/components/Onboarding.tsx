import React from "react";
import { Button } from "../ui";

const STORAGE_KEY = "dubsmith.onboarded";

/** A dismissible, one-time pointer to the features that are easy to miss. */
export function Onboarding() {
  const [visible, setVisible] = React.useState(false);

  React.useEffect(() => {
    try {
      if (!window.localStorage.getItem(STORAGE_KEY)) setVisible(true);
    } catch {
      /* storage may be unavailable (private mode); just stay hidden */
    }
  }, []);

  function dismiss() {
    setVisible(false);
    try {
      window.localStorage.setItem(STORAGE_KEY, "1");
    } catch {
      /* best effort */
    }
  }

  if (!visible) return null;

  return (
    <div
      role="dialog"
      aria-label="Getting started"
      className="fixed bottom-5 left-5 z-40 w-[340px] max-w-[calc(100vw-2rem)] rounded-2xl border border-white/10 bg-ink-900/95 p-4 shadow-glow"
    >
      <div className="text-sm font-semibold text-white">Welcome to Dubsmith</div>
      <ul className="mt-2 space-y-1 text-xs text-slate-400">
        <li>
          Pick the default device and precision in the header; every form uses it
          unless overridden.
        </li>
        <li>Finished jobs can be retried, edited or compared from the activity panel.</li>
        <li>Save a configuration as a preset and replay inputs from Recent / favorites.</li>
        <li>Lists are paginated, and every delete asks for confirmation.</li>
      </ul>
      <div className="mt-3 flex justify-end">
        <Button className="px-3 py-1 text-xs" onClick={dismiss}>
          Got it
        </Button>
      </div>
    </div>
  );
}
