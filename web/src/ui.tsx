import React from "react";

export function cn(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

export function Card({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-white/10 bg-ink-900/80 shadow-glow backdrop-blur",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function Section({
  title,
  subtitle,
  right,
  children,
}: {
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-white">{title}</h2>
          {subtitle ? <p className="mt-1 text-sm text-slate-400">{subtitle}</p> : null}
        </div>
        {right}
      </div>
      {children}
    </section>
  );
}

export function Disclosure({
  title,
  hint,
  children,
  defaultOpen = false,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = React.useState(defaultOpen);
  return (
    <div className="overflow-hidden rounded-xl border border-white/10 bg-ink-850/60">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
      >
        <span>
          <span className="text-sm font-medium text-slate-200">{title}</span>
          {hint ? <span className="ml-2 text-xs text-slate-500">{hint}</span> : null}
        </span>
        <span
          className={cn(
            "text-slate-500 transition-transform",
            open ? "rotate-180" : "rotate-0",
          )}
        >
          ▾
        </span>
      </button>
      {open ? <div className="space-y-4 border-t border-white/5 px-4 py-4">{children}</div> : null}
    </div>
  );
}

/**
 * A hover/focus tooltip for a field label. The text lives in the DOM (hidden
 * until hover or keyboard focus) so it is discoverable to tests and assistive
 * technology instead of relying on the browser's native `title` delay.
 */
export function Tooltip({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex align-middle">
      <span
        aria-hidden="true"
        className="cursor-help text-[10px] leading-none text-slate-500 group-hover:text-slate-300"
      >
        ⓘ
      </span>
      <span
        role="tooltip"
        className="pointer-events-none absolute right-0 top-full z-30 mt-1 hidden w-56 whitespace-normal rounded-lg border border-white/10 bg-ink-900 p-2 text-[11px] font-normal normal-case leading-snug tracking-normal text-slate-300 shadow-glow group-hover:block group-focus-within:block"
      >
        {text}
      </span>
    </span>
  );
}

export function Field({
  label,
  hint,
  tooltip,
  children,
  className,
}: {
  label: string;
  hint?: string;
  tooltip?: string;
  children: React.ReactNode;
  className?: string;
}) {
  // The tooltip lives outside the <label> on purpose: label text is the
  // field's accessible name, and a tooltip must not become part of it.
  return (
    <div className={cn("relative space-y-1.5", className)}>
      <label className="block space-y-1.5">
        <span className="block text-xs font-medium uppercase tracking-wide text-slate-400">
          {label}
        </span>
        {children}
      </label>
      {tooltip ? (
        <span className="absolute right-0 top-0">
          <Tooltip text={tooltip} />
        </span>
      ) : null}
      {hint ? <span className="block text-xs text-slate-500">{hint}</span> : null}
    </div>
  );
}

const control =
  "w-full rounded-lg border border-white/10 bg-ink-800/80 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-accent-500 focus:ring-2 focus:ring-accent-500/20";

export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cn(control, props.className)} />;
}

export function Textarea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...props} className={cn(control, "min-h-28", props.className)} />;
}

export function Select({
  options,
  ...props
}: React.SelectHTMLAttributes<HTMLSelectElement> & {
  options: { value: string; label: string }[];
}) {
  return (
    <select {...props} className={cn(control, props.className)}>
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );
}

export function Toggle({
  label,
  hint,
  tooltip,
  checked,
  onChange,
}: {
  label: string;
  hint?: string;
  tooltip?: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <div className="flex cursor-pointer items-start gap-3">
      <button
        type="button"
        role="switch"
        aria-label={label}
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cn(
          "mt-0.5 h-5 w-9 rounded-full border transition",
          checked
            ? "border-accent-500 bg-accent-500/80"
            : "border-white/15 bg-ink-800",
        )}
      >
        <span
          className={cn(
            "block h-4 w-4 rounded-full bg-white transition-transform",
            checked ? "translate-x-4" : "translate-x-0.5",
          )}
        />
      </button>
      <span>
        <span className="flex items-center gap-1 text-sm text-slate-200">
          <span>{label}</span>
          {tooltip ? <Tooltip text={tooltip} /> : null}
        </span>
        {hint ? <span className="block text-xs text-slate-500">{hint}</span> : null}
      </span>
    </div>
  );
}

export function Button({
  variant = "primary",
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "ghost" | "danger";
}) {
  const styles = {
    primary:
      "bg-accent-500 text-white hover:bg-accent-400 disabled:bg-accent-500/40",
    ghost:
      "border border-white/10 bg-ink-800/60 text-slate-200 hover:border-white/20 hover:bg-ink-700/60",
    danger:
      "border border-rose-500/30 bg-rose-500/10 text-rose-200 hover:bg-rose-500/20",
  };
  return (
    <button
      {...props}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-60",
        styles[variant],
        className,
      )}
    />
  );
}

export function Spinner({ className }: { className?: string }) {
  // `currentColor` keeps the loader visible on both the dark and the light
  // surface, and inside coloured buttons (where the text is already white).
  return (
    <span
      className={cn(
        "inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent",
        className,
      )}
    />
  );
}

export function Badge({ state }: { state: string }) {
  const tone: Record<string, string> = {
    QUEUED: "border-amber-400/30 bg-amber-400/10 text-amber-200",
    PROCESSING: "border-sky-400/30 bg-sky-400/10 text-sky-200",
    COMPLETED: "border-emerald-400/30 bg-emerald-400/10 text-emerald-200",
    FAILED: "border-rose-400/30 bg-rose-400/10 text-rose-200",
    CANCELLED: "border-slate-400/30 bg-slate-400/10 text-slate-300",
  };
  return (
    <span
      className={cn(
        "rounded-full border px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide",
        tone[state] ?? tone.CANCELLED,
      )}
    >
      {state}
    </span>
  );
}

export function Empty({
  text,
  action,
}: {
  text: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-dashed border-white/10 px-6 py-10 text-center">
      <p className="text-sm text-slate-500">{text}</p>
      {action ? <div className="mt-4 flex justify-center">{action}</div> : null}
    </div>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-rose-500/25 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
      {message}
    </div>
  );
}

export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Delete",
  danger = false,
  busy = false,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  danger?: boolean;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  if (!open) return null;
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={title}
      className="fixed inset-0 z-50 grid place-items-center bg-black/50 p-4"
    >
      <div className="w-full max-w-md rounded-2xl border border-white/10 bg-ink-900 p-5 shadow-glow">
        <h3 className="text-base font-semibold text-white">{title}</h3>
        <p className="mt-2 text-sm text-slate-400">{message}</p>
        <div className="mt-5 flex justify-end gap-2">
          <Button variant="ghost" onClick={onCancel} disabled={busy}>
            Cancel
          </Button>
          <Button
            variant={danger ? "danger" : "primary"}
            onClick={onConfirm}
            disabled={busy}
          >
            {busy ? <Spinner /> : null}
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
}

export function Pagination({
  page,
  pageSize,
  total,
  onPage,
}: {
  page: number;
  pageSize: number;
  total: number;
  onPage: (page: number) => void;
}) {
  const safeTotal = Number.isFinite(total) ? Math.max(0, total) : 0;
  const size = Math.max(1, pageSize);
  const pages = Math.max(1, Math.ceil(safeTotal / size));
  if (safeTotal <= size) return null;
  return (
    <div className="flex items-center justify-between gap-3 text-xs text-slate-500">
      <span>
        {safeTotal} item{safeTotal === 1 ? "" : "s"}
      </span>
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          className="px-2 py-1 text-xs"
          disabled={page <= 1}
          onClick={() => onPage(page - 1)}
        >
          Previous
        </Button>
        <span>
          Page {page} of {pages}
        </span>
        <Button
          variant="ghost"
          className="px-2 py-1 text-xs"
          disabled={page >= pages}
          onClick={() => onPage(page + 1)}
        >
          Next
        </Button>
      </div>
    </div>
  );
}

export function CopyButton({
  value,
  label = "Copy path",
}: {
  value: string;
  label?: string;
}) {
  const [copied, setCopied] = React.useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard is unavailable (e.g. insecure origin); do nothing */
    }
  }
  return (
    <button
      type="button"
      onClick={copy}
      aria-label={label}
      className="rounded-lg border border-white/10 px-2 py-1 text-xs text-slate-400 transition hover:border-white/20 hover:text-slate-200"
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

const _VIDEO_EXTENSIONS = [".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"];
const _AUDIO_EXTENSIONS = [
  ".mp3",
  ".wav",
  ".flac",
  ".m4a",
  ".aac",
  ".ogg",
  ".opus",
];

export function isVideoFile(name: string): boolean {
  const lower = name.toLowerCase();
  return _VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

/** True when a browser can actually play this file inline. */
export function isPlayable(name: string): boolean {
  const lower = name.toLowerCase();
  return (
    isVideoFile(name) || _AUDIO_EXTENSIONS.some((ext) => lower.endsWith(ext))
  );
}

export function MediaPreview({
  src,
  name,
  autoPlay = false,
}: {
  src: string;
  name: string;
  autoPlay?: boolean;
}) {
  if (isVideoFile(name)) {
    return (
      <video
        key={src}
        controls
        autoPlay={autoPlay}
        playsInline
        preload="metadata"
        src={src}
        aria-label={`Play ${name}`}
        className="max-h-64 w-full rounded-lg border border-white/10 bg-black/30"
      />
    );
  }
  return (
    <audio
      key={src}
      controls
      autoPlay={autoPlay}
      preload="metadata"
      src={src}
      aria-label={`Play ${name}`}
      className="w-full"
    />
  );
}

export function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KiB`;
  if (size < 1024 ** 3) return `${(size / 1024 ** 2).toFixed(1)} MiB`;
  return `${(size / 1024 ** 3).toFixed(2)} GiB`;
}

export function formatSeconds(value?: number | null): string {
  if (value == null) return "—";
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}m ${seconds}s`;
}

export function formatDate(value?: number | null): string {
  if (!value) return "—";
  return new Date(value * 1000).toLocaleString();
}

export function num(value: string): number | undefined {
  if (value.trim() === "") return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export function list(value: string): string[] | undefined {
  const parts = value
    .split(/[\n,]/)
    .map((part) => part.trim())
    .filter(Boolean);
  return parts.length ? parts : undefined;
}
