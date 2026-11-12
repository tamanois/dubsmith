import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ModelsPage } from "../src/pages/ModelsPage";
import { makeModel } from "./factories";

const models = vi.fn();
const downloadModels = vi.fn();
const deleteModel = vi.fn();

vi.mock("../src/api", () => ({
  api: {
    models: (...args: unknown[]) => models(...args),
    downloadModels: (...args: unknown[]) => downloadModels(...args),
    deleteModel: (...args: unknown[]) => deleteModel(...args),
  },
}));

beforeEach(() => {
  models.mockReset();
  downloadModels.mockReset();
  deleteModel.mockReset();
  models.mockResolvedValue({
    models: [
      makeModel({ name: "tts_base", required: true, present: true }),
      makeModel({ name: "asr", required: false, present: false }),
    ],
  });
  downloadModels.mockResolvedValue({ job_id: "job-9", state: "QUEUED" });
});

describe("ModelsPage", () => {
  it("lists models with availability and required badges", async () => {
    render(<ModelsPage onJobStarted={vi.fn()} />);
    expect(await screen.findByText("tts_base")).toBeInTheDocument();
    expect(screen.getByText("asr")).toBeInTheDocument();
    expect(screen.getByText(/1 of 2 checkpoints available locally/)).toBeInTheDocument();
    expect(screen.getByText("required")).toBeInTheDocument();
    expect(screen.getByText("COMPLETED")).toBeInTheDocument();
    expect(screen.getByText("QUEUED")).toBeInTheDocument();
  });

  it("downloads a single model and reports the job", async () => {
    const onJobStarted = vi.fn();
    render(<ModelsPage onJobStarted={onJobStarted} />);
    fireEvent.click(await screen.findByRole("button", { name: "Download" }));

    await waitFor(() => expect(downloadModels).toHaveBeenCalledWith(["asr"]));
    expect(onJobStarted).toHaveBeenCalledWith("job-9");
  });

  it("offers re-download for a present model", async () => {
    render(<ModelsPage onJobStarted={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Re-download" }));
    await waitFor(() =>
      expect(downloadModels).toHaveBeenCalledWith(["tts_base"]),
    );
  });

  it("deletes a present model after a danger confirmation", async () => {
    deleteModel.mockResolvedValue({
      name: "tts_base",
      freed_bytes: 2048,
      warning: null,
    });
    render(<ModelsPage onJobStarted={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete" }));
    expect(
      screen.getByRole("dialog", { name: "Delete tts_base" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Delete model" }));
    await waitFor(() => expect(deleteModel).toHaveBeenCalledWith("tts_base"));
    expect(await screen.findByText(/freed 2\.0 KiB/)).toBeInTheDocument();
  });

  it("downloads the MVP set and everything", async () => {
    render(<ModelsPage onJobStarted={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Download MVP set" }));
    await waitFor(() => expect(downloadModels).toHaveBeenCalledWith(["all"]));

    fireEvent.click(
      screen.getByRole("button", { name: "Download everything (optional backends)" }),
    );
    await waitFor(() => expect(downloadModels).toHaveBeenCalledWith(["everything"]));
  });

  it("disables downloads while a download is starting", async () => {
    let resolve!: (value: unknown) => void;
    downloadModels.mockReturnValue(new Promise((done) => (resolve = done)));
    render(<ModelsPage onJobStarted={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Download MVP set" }));

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Download everything (optional backends)" }),
      ).toBeDisabled(),
    );
    resolve({ job_id: "job-9", state: "QUEUED" });
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Download everything (optional backends)" }),
      ).not.toBeDisabled(),
    );
  });

  it("shows a download failure and does not report a job", async () => {
    downloadModels.mockRejectedValue(new Error("download unavailable"));
    const onJobStarted = vi.fn();
    render(<ModelsPage onJobStarted={onJobStarted} />);
    fireEvent.click(await screen.findByRole("button", { name: "Download" }));

    expect(await screen.findByText("download unavailable")).toBeInTheDocument();
    expect(onJobStarted).not.toHaveBeenCalled();
  });

  it("shows an empty state and a load error", async () => {
    models.mockResolvedValueOnce({ models: [] });
    const { unmount } = render(<ModelsPage onJobStarted={vi.fn()} />);
    expect(await screen.findByText("No models reported yet.")).toBeInTheDocument();
    unmount();

    models.mockRejectedValueOnce(new Error("registry offline"));
    render(<ModelsPage onJobStarted={vi.fn()} />);
    expect(await screen.findByText("registry offline")).toBeInTheDocument();
  });
});
