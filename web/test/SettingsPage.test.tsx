import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SettingsPage } from "../src/pages/SettingsPage";
import { makeSettings } from "./factories";

const settings = vi.fn();
const patchSettings = vi.fn();
const exportSettings = vi.fn();
const importSettings = vi.fn();

vi.mock("../src/api", () => ({
  api: {
    settings: (...args: unknown[]) => settings(...args),
    patchSettings: (...args: unknown[]) => patchSettings(...args),
    exportSettings: (...args: unknown[]) => exportSettings(...args),
    importSettings: (...args: unknown[]) => importSettings(...args),
  },
}));

beforeEach(() => {
  settings.mockReset();
  patchSettings.mockReset();
  exportSettings.mockReset();
  importSettings.mockReset();
  settings.mockResolvedValue(makeSettings());
  patchSettings.mockImplementation(async (values: Record<string, unknown>) =>
    makeSettings(values),
  );
});

describe("SettingsPage", () => {
  it("exports settings as a download", async () => {
    exportSettings.mockResolvedValue({ version: 1, settings: { duck_db: 6 } });
    const createObjectURL = vi.fn(() => "blob:settings");
    const revokeObjectURL = vi.fn();
    Object.assign(URL, { createObjectURL, revokeObjectURL });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);

    render(<SettingsPage />);
    await screen.findByDisplayValue("local-model");
    fireEvent.click(screen.getByRole("button", { name: "Export" }));

    await waitFor(() => expect(exportSettings).toHaveBeenCalled());
    expect(createObjectURL).toHaveBeenCalled();
    expect(click).toHaveBeenCalled();
  });

  it("opens a file picker to import settings", async () => {
    const click = vi
      .spyOn(HTMLInputElement.prototype, "click")
      .mockImplementation(() => undefined);

    render(<SettingsPage />);
    await screen.findByDisplayValue("local-model");
    fireEvent.click(screen.getByRole("button", { name: "Import" }));
    expect(click).toHaveBeenCalled();
  });

  it("loads current settings", async () => {
    render(<SettingsPage />);
    expect(await screen.findByDisplayValue("http://127.0.0.1:8080/v1")).toBeInTheDocument();
    expect(screen.getByDisplayValue("local-model")).toBeInTheDocument();
    expect(
      screen.getByText(/API key: not set \(LLM calls may fail\)/),
    ).toBeInTheDocument();
  });

  it("shows a load error", async () => {
    settings.mockRejectedValue(new Error("settings unavailable"));
    render(<SettingsPage />);
    expect(await screen.findByText("settings unavailable")).toBeInTheDocument();
  });

  it("saves edited values and shows confirmation", async () => {
    render(<SettingsPage />);
    const baseUrl = await screen.findByDisplayValue("http://127.0.0.1:8080/v1");
    fireEvent.change(baseUrl, { target: { value: "http://127.0.0.1:9000/v1" } });
    fireEvent.click(screen.getByRole("switch", { name: "JSON logs" }));
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));

    await waitFor(() =>
      expect(patchSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          llm_base_url: "http://127.0.0.1:9000/v1",
          log_json: true,
        }),
      ),
    );
    expect(await screen.findByText("Saved")).toBeInTheDocument();
  });

  it("clears the saved marker when a field changes again", async () => {
    render(<SettingsPage />);
    const model = await screen.findByDisplayValue("local-model");
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
    await screen.findByText("Saved");

    fireEvent.change(model, { target: { value: "another-model" } });
    expect(screen.queryByText("Saved")).not.toBeInTheDocument();
  });

  it("sends numeric values and device defaults", async () => {
    render(<SettingsPage />);
    const timeout = await screen.findByDisplayValue("180");
    fireEvent.change(timeout, { target: { value: "90" } });
    fireEvent.change(screen.getByLabelText(/Default device/), {
      target: { value: "cuda:1" },
    });
    fireEvent.change(screen.getByLabelText(/Default precision/), {
      target: { value: "fp16" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));

    await waitFor(() =>
      expect(patchSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          llm_timeout_s: 90,
          default_device: "cuda:1",
          default_dtype: "fp16",
        }),
      ),
    );
  });

  it("tracks the background toggle", async () => {
    render(<SettingsPage />);
    fireEvent.click(
      await screen.findByRole("switch", { name: "Keep music and ambience by default" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
    await waitFor(() =>
      expect(patchSettings).toHaveBeenCalledWith(
        expect.objectContaining({ keep_background: false }),
      ),
    );
  });

  it("disables saving while the request is in flight", async () => {
    let resolve!: (value: unknown) => void;
    patchSettings.mockReturnValue(new Promise((done) => (resolve = done)));
    render(<SettingsPage />);
    await screen.findByDisplayValue("local-model");
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));

    const save = screen.getByRole("button", { name: "Save settings" });
    await waitFor(() => expect(save).toBeDisabled());
    resolve(makeSettings());
    await waitFor(() => expect(save).not.toBeDisabled());
  });

  it("shows a save failure and stays editable", async () => {
    patchSettings.mockRejectedValue(new Error("invalid settings"));
    render(<SettingsPage />);
    await screen.findByDisplayValue("local-model");
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));

    expect(await screen.findByText("invalid settings")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save settings" })).not.toBeDisabled();
  });

  it("explains when an API key is already configured", async () => {
    settings.mockResolvedValueOnce(makeSettings({ llm_api_key_set: true }));
    render(<SettingsPage />);
    expect(
      await screen.findByText(/API key: configured in the environment/),
    ).toBeInTheDocument();
  });
});
