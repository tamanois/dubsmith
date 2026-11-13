import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DevicesPage } from "../src/pages/DevicesPage";
import { makeDevice, makeDeviceReport } from "./factories";

const devices = vi.fn();
const packages = vi.fn();
const usage = vi.fn();

vi.mock("../src/api", () => ({
  api: {
    devices: (...args: unknown[]) => devices(...args),
    packages: (...args: unknown[]) => packages(...args),
    usage: (...args: unknown[]) => usage(...args),
  },
}));

beforeEach(() => {
  devices.mockReset();
  packages.mockReset();
  usage.mockReset();
  devices.mockResolvedValue(makeDeviceReport());
  usage.mockResolvedValue({
    sizes: { models_dir: 1024 ** 3, voices_dir: 0, output: 2048, uploads: 0 },
    total: 1024 ** 3 + 2048,
    disk_free: 100 * 1024 ** 3,
    disk_total: 500 * 1024 ** 3,
  });
  packages.mockResolvedValue({
    optional_packages: [
      { name: "demucs", available: true, purpose: "background separation" },
      { name: "onnxruntime", available: false, purpose: "speaker diarization" },
    ],
    system_binaries: [
      { name: "ffmpeg", path: "/usr/bin/ffmpeg", purpose: "media decoding" },
      { name: "ffprobe", path: null, purpose: "media probing" },
    ],
  });
});

describe("DevicesPage", () => {
  it("renders detected devices, packages, and binaries", async () => {
    render(<DevicesPage />);

    expect(
      await screen.findByText(/cuda:0 · NVIDIA GeForce RTX 3060 · 12\.00 GiB/),
    ).toBeInTheDocument();
    expect(screen.getByText(/bf16 yes · ASR cuda · TTS bf16/)).toBeInTheDocument();
    expect(screen.getByText(/torch 2\.5\.1 \(CUDA 12\.4\)/)).toBeInTheDocument();
    expect(screen.getByText("demucs")).toBeInTheDocument();
    expect(screen.getByText("installed")).toBeInTheDocument();
    expect(screen.getAllByText("missing")).toHaveLength(2);
    expect(screen.getByText("/usr/bin/ffmpeg")).toBeInTheDocument();
    expect(screen.getByText("ffprobe")).toBeInTheDocument();
    expect(screen.getByText("Local storage")).toBeInTheDocument();
    expect(screen.getByText("Free on disk")).toBeInTheDocument();
  });

  it("explains when only the CPU is available", async () => {
    devices.mockResolvedValueOnce(
      makeDeviceReport({ devices: [], torch: "2.5.1 (CPU)", note: "No CUDA" }),
    );
    render(<DevicesPage />);
    expect(
      await screen.findByText("No GPU detected. Everything will run on the CPU."),
    ).toBeInTheDocument();
    expect(screen.getByText(/torch 2\.5\.1 \(CPU\)/)).toBeInTheDocument();
  });

  it("handles a device with unknown memory and no accelerator settings", async () => {
    devices.mockResolvedValueOnce(
      makeDeviceReport({
        devices: [
          makeDevice({
            label: "cpu",
            kind: "cpu",
            name: "CPU",
            total_memory: null,
            supports_bf16: false,
            asr_device: undefined,
            tts_dtype: undefined,
          }),
        ],
      }),
    );
    render(<DevicesPage />);
    expect(
      await screen.findByText(/cpu · CPU · memory unknown · bf16 no · ASR cpu · TTS auto/),
    ).toBeInTheDocument();
  });

  it("shows a load error but keeps the other panels", async () => {
    devices.mockRejectedValueOnce(new Error("device probe failed"));
    render(<DevicesPage />);
    expect(await screen.findByText(/device probe failed/)).toBeInTheDocument();
    // The storage panel still rendered from its own endpoint.
    expect(await screen.findByText("Local storage")).toBeInTheDocument();
  });

  it("refreshes both reports", async () => {
    render(<DevicesPage />);
    await screen.findByText(/cuda:0/);
    expect(devices).toHaveBeenCalledTimes(1);
    expect(packages).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: /Refresh/ }));
    expect(devices).toHaveBeenCalledTimes(2);
    expect(packages).toHaveBeenCalledTimes(2);
  });
});
