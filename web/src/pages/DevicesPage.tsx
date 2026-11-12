import React from "react";
import { api } from "../api";
import type { Device, DeviceReport } from "../types";
import {
  Button,
  Card,
  Empty,
  ErrorNote,
  Section,
  Spinner,
  formatBytes,
} from "../ui";

interface PackageRow {
  name: string;
  available: boolean;
  purpose: string;
}

interface BinaryRow {
  name: string;
  path: string | null;
  purpose: string;
}

export function DevicesPage() {
  const [report, setReport] = React.useState<DeviceReport | null>(null);
  const [packages, setPackages] = React.useState<PackageRow[]>([]);
  const [binaries, setBinaries] = React.useState<BinaryRow[]>([]);
  const [error, setError] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);

  const refresh = React.useCallback(() => {
    setBusy(true);
    Promise.all([api.devices(), api.packages()])
      .then(([deviceReport, packageReport]) => {
        setReport(deviceReport);
        setPackages(packageReport.optional_packages ?? []);
        setBinaries(packageReport.system_binaries ?? []);
      })
      .catch((exc: unknown) =>
        setError(exc instanceof Error ? exc.message : "Could not load devices"),
      )
      .finally(() => setBusy(false));
  }, []);

  React.useEffect(refresh, [refresh]);

  function deviceLine(device: Device) {
    return `${device.label} · ${device.name} · ${device.total_memory ? formatBytes(device.total_memory) : "memory unknown"} · bf16 ${device.supports_bf16 ? "yes" : "no"} · ASR ${device.asr_device ?? device.label} · TTS ${device.tts_dtype ?? "auto"}`;
  }

  return (
    <Section
      title="Devices and environment"
      subtitle="What this machine can run, and which optional pieces are installed."
      right={<Button variant="ghost" onClick={refresh}>{busy ? <Spinner /> : null}Refresh</Button>}
    >
      <div className="space-y-4">
        {error ? <ErrorNote message={error} /> : null}
        <Card className="p-5">
          <div className="mb-3 text-sm font-semibold text-white">Detected devices</div>
          {report?.devices.length ? (
            <div className="space-y-2">
              {report.devices.map((device) => (
                <div key={device.label} className="rounded-lg border border-white/5 bg-ink-850/60 px-3 py-2 text-xs text-slate-300">
                  {deviceLine(device)}
                </div>
              ))}
            </div>
          ) : (
            <Empty text="No GPU detected. Everything will run on the CPU." />
          )}
          <div className="mt-3 text-xs text-slate-500">
            torch {report?.torch ?? "unknown"} · {report?.note}
          </div>
        </Card>

        <div className="grid gap-4 lg:grid-cols-2">
          <Card className="p-5">
            <div className="mb-3 text-sm font-semibold text-white">Optional packages</div>
            <div className="space-y-2">
              {packages.map((item) => (
                <div key={item.name} className="flex items-start justify-between gap-3 text-xs">
                  <div>
                    <div className="font-medium text-slate-200">{item.name}</div>
                    <div className="text-slate-500">{item.purpose}</div>
                  </div>
                  <span className={item.available ? "text-emerald-300" : "text-rose-300"}>
                    {item.available ? "installed" : "missing"}
                  </span>
                </div>
              ))}
            </div>
          </Card>
          <Card className="p-5">
            <div className="mb-3 text-sm font-semibold text-white">System binaries</div>
            <div className="space-y-2">
              {binaries.map((item) => (
                <div key={item.name} className="flex items-start justify-between gap-3 text-xs">
                  <div>
                    <div className="font-medium text-slate-200">{item.name}</div>
                    <div className="text-slate-500">{item.purpose}</div>
                  </div>
                  <span className={item.path ? "text-emerald-300" : "text-rose-300"}>
                    {item.path ?? "missing"}
                  </span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </Section>
  );
}
