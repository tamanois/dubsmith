import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PresetPanel } from "../src/components/PresetPanel";
import type { Preset } from "../src/types";

const presets = vi.fn();
const createPreset = vi.fn();
const deletePreset = vi.fn();

vi.mock("../src/api", () => ({
  api: {
    presets: (...args: unknown[]) => presets(...args),
    createPreset: (...args: unknown[]) => createPreset(...args),
    deletePreset: (...args: unknown[]) => deletePreset(...args),
  },
}));

function makePreset(overrides: Partial<Preset> = {}): Preset {
  return {
    id: "p1",
    kind: "dub",
    name: "Fast French",
    payload: { target_language: "French" },
    created: 1_700_000_000,
    ...overrides,
  };
}

beforeEach(() => {
  presets.mockReset();
  createPreset.mockReset();
  deletePreset.mockReset();
  presets.mockResolvedValue({ presets: [makePreset()] });
  createPreset.mockResolvedValue({ preset: makePreset() });
  deletePreset.mockResolvedValue({ deleted: "p1" });
});

describe("PresetPanel", () => {
  it("saves the current form as a preset", async () => {
    render(
      <PresetPanel
        kind="dub"
        buildPayload={() => ({ target_language: "German", device: "cpu" })}
        onApply={vi.fn()}
      />,
    );
    await screen.findByText("Apply Fast French");

    fireEvent.change(screen.getByPlaceholderText("e.g. Fast French"), {
      target: { value: "German CPU" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save preset" }));

    await waitFor(() =>
      expect(createPreset).toHaveBeenCalledWith({
        kind: "dub",
        name: "German CPU",
        payload: { target_language: "German", device: "cpu" },
      }),
    );
  });

  it("applies a saved preset", async () => {
    const onApply = vi.fn();
    render(
      <PresetPanel kind="dub" buildPayload={() => ({})} onApply={onApply} />,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Apply Fast French" }));
    expect(onApply).toHaveBeenCalledWith({ target_language: "French" });
  });

  it("deletes a preset after confirmation", async () => {
    render(
      <PresetPanel kind="dub" buildPayload={() => ({})} onApply={vi.fn()} />,
    );
    fireEvent.click(
      await screen.findByRole("button", { name: "Delete preset Fast French" }),
    );
    expect(
      screen.getByRole("dialog", { name: "Delete preset Fast French" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Delete preset" }));
    await waitFor(() => expect(deletePreset).toHaveBeenCalledWith("p1"));
  });

  it("cannot save without a name", async () => {
    render(
      <PresetPanel kind="dub" buildPayload={() => ({})} onApply={vi.fn()} />,
    );
    expect(
      await screen.findByRole("button", { name: "Save preset" }),
    ).toBeDisabled();
  });
});
