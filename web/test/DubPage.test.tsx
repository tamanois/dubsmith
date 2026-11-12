import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DubPage } from "../src/pages/DubPage";

const voices = vi.fn();
const devices = vi.fn();
const settings = vi.fn();

vi.mock("../src/api", () => ({
  api: {
    voices: () => voices(),
    devices: () => devices(),
    settings: () => settings(),
  },
}));

describe("DubPage subtitles-only mode", () => {
  beforeEach(() => {
    voices.mockResolvedValue({ voices: [] });
    devices.mockResolvedValue({ devices: [] });
    settings.mockResolvedValue({ default_device: "auto", default_dtype: "auto" });
  });

  it("offers a device selector so a busy GPU can be avoided", async () => {
    render(<DubPage onJobStarted={() => undefined} />);
    fireEvent.click(
      screen.getByRole("button", { name: "Subtitles only" }),
    );

    const device = await screen.findByLabelText(/^Device/);
    expect(device).toBeInTheDocument();
    expect(within(device).getByRole("option", { name: "CPU" })).toBeInTheDocument();
    await waitFor(() => expect(devices).toHaveBeenCalled());
  });
});
