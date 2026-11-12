import { api } from "../api";
import type { Job } from "../types";
import { Button, MediaPreview } from "../ui";

/**
 * Side-by-side playback of the file a job started from and the dub it produced,
 * so the result can be checked without leaving the app.
 */
export function CompareDialog({
  job,
  onClose,
}: {
  job: Job;
  onClose: () => void;
}) {
  const result = (job.result ?? {}) as Record<string, unknown>;
  const dubbed =
    typeof result.dubbed_video === "string"
      ? result.dubbed_video
      : typeof result.dub_track === "string"
        ? result.dub_track
        : null;
  const originalName = job.input_label ?? "original";

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Compare original and dub"
      className="fixed inset-0 z-50 grid place-items-center bg-black/50 p-4"
    >
      <div className="w-full max-w-4xl rounded-2xl border border-white/10 bg-ink-900 p-5 shadow-glow">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-base font-semibold text-white">Compare</h3>
          <Button variant="ghost" className="px-3 py-1 text-xs" onClick={onClose}>
            Close
          </Button>
        </div>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <div className="text-xs uppercase tracking-wide text-slate-500">
              Original
            </div>
            <MediaPreview src={api.jobInputUrl(job.id)} name={originalName} />
          </div>
          <div className="space-y-2">
            <div className="text-xs uppercase tracking-wide text-slate-500">Dub</div>
            {dubbed ? (
              <MediaPreview
                src={api.outputUrl(dubbed)}
                name={dubbed.split(/[\\/]/).pop() ?? "dub"}
              />
            ) : (
              <div className="text-sm text-slate-500">
                This job produced no playable dub artifact.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
