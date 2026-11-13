import type { Device } from "../types";

const DTYPES = [
  { value: "auto", label: "auto" },
  { value: "bf16", label: "bfloat16" },
  { value: "fp16", label: "float16" },
  { value: "fp32", label: "float32" },
];

/**
 * Always-visible defaults for every new job. Kept in the header rather than on
 * each form: the per-job override still exists, but this is the fallback a user
 * should not have to go hunting for.
 */
export function DefaultsBar({
  devices,
  device,
  dtype,
  onChange,
}: {
  devices: Device[];
  device: string;
  dtype: string;
  onChange: (patch: { default_device?: string; default_dtype?: string }) => void;
}) {
  const deviceOptions = [
    { value: "auto", label: "auto" },
    { value: "cpu", label: "cpu" },
    ...devices.map((item) => ({
      value: item.label,
      label: `${item.label} — ${item.name}`,
    })),
  ];
  // A stored device may no longer be present; keep it selectable so the UI does
  // not silently change the user's saved preference.
  if (!deviceOptions.some((option) => option.value === device)) {
    deviceOptions.push({ value: device, label: device });
  }

  return (
    <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
      <span className="flex items-center gap-1">
        <span>Device</span>
        <select
          aria-label="Default device"
          value={device}
          onChange={(event) => onChange({ default_device: event.target.value })}
          className="rounded-md border border-white/10 bg-ink-850 px-2 py-1 text-slate-200"
        >
          {deviceOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </span>
      <span className="flex items-center gap-1">
        <span>Precision</span>
        <select
          aria-label="Default precision"
          value={dtype}
          onChange={(event) => onChange({ default_dtype: event.target.value })}
          className="rounded-md border border-white/10 bg-ink-850 px-2 py-1 text-slate-200"
        >
          {DTYPES.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </span>
    </div>
  );
}
