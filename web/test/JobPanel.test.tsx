import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { JobPanel } from "../src/components/JobPanel";
import { makeJob, makeJobEvent } from "./factories";

const job = vi.fn();
const cancelJob = vi.fn();
const retryJob = vi.fn();
const subscribeJob = vi.fn();
const unsubscribe = vi.fn();
let onEvent: ((event: ReturnType<typeof makeJobEvent>) => void) | undefined;
let onJob: ((finished: ReturnType<typeof makeJob>) => void) | undefined;
let onError: ((message: string) => void) | undefined;

vi.mock("../src/api", () => ({
  api: {
    job: (...args: unknown[]) => job(...args),
    cancelJob: (...args: unknown[]) => cancelJob(...args),
    retryJob: (...args: unknown[]) => retryJob(...args),
    jobInputUrl: (id: string) => `/api/jobs/${id}/input`,
    outputUrl: (path: string) =>
      `/api/outputs/download?path=${encodeURIComponent(path)}`,
  },
  subscribeJob: (...args: unknown[]) => subscribeJob(...args),
}));

function renderPanel(overrides: Partial<Parameters<typeof JobPanel>[0]> = {}) {
  const onClose = vi.fn();
  const view = render(<JobPanel jobId="job-1" onClose={onClose} {...overrides} />);
  return { ...view, onClose };
}

beforeEach(() => {
  job.mockReset();
  cancelJob.mockReset();
  retryJob.mockReset();
  subscribeJob.mockReset();
  unsubscribe.mockReset();
  onEvent = undefined;
  onJob = undefined;
  onError = undefined;
  subscribeJob.mockImplementation(
    (
      _id: string,
      event: typeof onEvent,
      finished: typeof onJob,
      error: typeof onError,
    ) => {
      onEvent = event;
      onJob = finished;
      onError = error;
      return unsubscribe;
    },
  );
});

describe("JobPanel terminal state", () => {
  it("says cancelled, not successfully finished, for a cancelled job", async () => {
    job.mockResolvedValue(makeJob({ state: "CANCELLED" }));
    renderPanel();

    expect(await screen.findByText("Cancelled.")).toBeInTheDocument();
    expect(screen.queryByText("Finished successfully.")).not.toBeInTheDocument();
  });

  it("still says successfully finished for a completed job", async () => {
    job.mockResolvedValue(makeJob({ state: "COMPLETED" }));
    renderPanel();

    expect(await screen.findByText("Finished successfully.")).toBeInTheDocument();
    expect(screen.queryByText("Cancelled.")).not.toBeInTheDocument();
  });

  it("shows the error and neither terminal success message for a failed job", async () => {
    job.mockResolvedValue(makeJob({ state: "FAILED", error: "boom" }));
    renderPanel();

    expect(await screen.findByText("boom")).toBeInTheDocument();
    expect(screen.queryByText("Finished successfully.")).not.toBeInTheDocument();
    expect(screen.queryByText("Cancelled.")).not.toBeInTheDocument();
  });
});

describe("JobPanel loading and errors", () => {
  it("shows a loading placeholder before the job arrives", () => {
    job.mockReturnValue(new Promise(() => undefined));
    renderPanel();
    expect(screen.getByText("Job")).toBeInTheDocument();
    expect(screen.getByText("job-1")).toBeInTheDocument();
  });

  it("shows a load error", async () => {
    job.mockRejectedValue(new Error("job not found"));
    renderPanel();
    expect(await screen.findByText("job not found")).toBeInTheDocument();
  });

  it("shows a live stream error", async () => {
    job.mockResolvedValue(makeJob({ state: "PROCESSING" }));
    renderPanel();
    await screen.findByText("Working…");
    act(() => onError?.("connection dropped"));
    expect(await screen.findByText("connection dropped")).toBeInTheDocument();
  });
});

describe("JobPanel logs", () => {
  it("appends live log lines and lets the user open and close the drawer", async () => {
    job.mockResolvedValue(makeJob({ state: "PROCESSING", events: [] }));
    renderPanel();

    const toggle = await screen.findByRole("button", { name: "Show log (0)" });
    fireEvent.click(toggle);
    expect(screen.getByText("No log lines yet.")).toBeInTheDocument();

    act(() => {
      onEvent?.(makeJobEvent({ id: 1, kind: "log", message: "first line" }));
      onEvent?.(makeJobEvent({ id: 2, kind: "error", message: "second line" }));
      onEvent?.(makeJobEvent({ id: 3, kind: "stage_end", message: "" }));
    });
    expect(screen.getByText("first line")).toBeInTheDocument();
    expect(screen.getByText("second line")).toBeInTheDocument();
    expect(screen.getByText("stage_end")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Hide log" }));
    expect(screen.queryByText("first line")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show log (3)" })).toBeInTheDocument();
  });

  it("keeps at most 300 log lines", async () => {
    job.mockResolvedValue(makeJob({ state: "PROCESSING", events: [] }));
    renderPanel();
    await screen.findByRole("button", { name: "Show log (0)" });

    act(() => {
      for (let index = 0; index < 305; index += 1) {
        onEvent?.(makeJobEvent({ id: index, message: `line ${index}` }));
      }
    });
    expect(screen.getByRole("button", { name: "Show log (300)" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show log (300)" }));
    expect(screen.getByText("line 304")).toBeInTheDocument();
    expect(screen.queryByText("line 0")).not.toBeInTheDocument();
  });
});

describe("JobPanel cancellation and completion", () => {
  it("offers cancel only while active and calls the API", async () => {
    job.mockResolvedValue(makeJob({ state: "QUEUED" }));
    renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(cancelJob).toHaveBeenCalledWith("job-1"));
  });

  it("does not offer cancel after the job finishes", async () => {
    job.mockResolvedValue(makeJob({ state: "COMPLETED" }));
    renderPanel();
    await screen.findByText("Finished successfully.");
    expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
  });

  it("shows a cancel failure", async () => {
    job.mockResolvedValue(makeJob({ state: "PROCESSING" }));
    cancelJob.mockRejectedValue(new Error("cannot cancel"));
    renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(await screen.findByText("cannot cancel")).toBeInTheDocument();
  });

  it("keeps cancel working after the live stream drops", async () => {
    job.mockResolvedValue(makeJob({ state: "PROCESSING" }));
    cancelJob.mockResolvedValue(makeJob({ state: "CANCELLED" }));
    renderPanel();
    await screen.findByText("Working…");

    act(() => onError?.("connection dropped"));
    expect(await screen.findByText("connection dropped")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(cancelJob).toHaveBeenCalledWith("job-1"));
    expect(await screen.findByText("Cancelled.")).toBeInTheDocument();
    expect(screen.queryByText("Working…")).not.toBeInTheDocument();
  });

  it("updates when the terminal job event arrives", async () => {
    job.mockResolvedValue(makeJob({ state: "PROCESSING" }));
    renderPanel();
    await screen.findByText("Working…");
    act(() => onJob?.(makeJob({ state: "COMPLETED" })));
    expect(await screen.findByText("Finished successfully.")).toBeInTheDocument();
    expect(screen.queryByText("Working…")).not.toBeInTheDocument();
  });
});

describe("JobPanel results and lifecycle", () => {
  it("renders only whitelisted result paths and stage durations", async () => {
    job.mockResolvedValue(
      makeJob({
        state: "COMPLETED",
        result: {
          job_dir: "/output/run",
          dub_track: "/output/run/dub.wav",
          secret: "/output/run/private.txt",
          stage_seconds: { asr: 1.234, translate: 65 },
        },
      }),
    );
    renderPanel();

    expect(await screen.findByText("/output/run")).toBeInTheDocument();
    expect(screen.getByText("/output/run/dub.wav")).toBeInTheDocument();
    expect(screen.queryByText("/output/run/private.txt")).not.toBeInTheDocument();
    expect(screen.getByText(/asr 1\.2s/)).toBeInTheDocument();
    expect(screen.getByText(/translate 1m 5s/)).toBeInTheDocument();
  });

  it("calls onClose and unsubscribes on unmount", async () => {
    job.mockResolvedValue(makeJob({ state: "COMPLETED" }));
    const { onClose, unmount } = renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: "Close job panel" }));
    expect(onClose).toHaveBeenCalled();

    unmount();
    expect(unsubscribe).toHaveBeenCalled();
  });
});

describe("JobPanel actions and labels", () => {
  it("shows a human title and the input and output labels", async () => {
    job.mockResolvedValue(
      makeJob({
        state: "COMPLETED",
        kind: "say",
        title: "Generate speech",
        input_label: "Bonjour le monde",
        output_label: "out.wav",
      }),
    );
    renderPanel();

    expect(await screen.findByText("Generate speech")).toBeInTheDocument();
    expect(screen.getByText(/Bonjour le monde/)).toBeInTheDocument();
    expect(screen.getByText(/→ out\.wav/)).toBeInTheDocument();
  });

  it("expands and shrinks the panel", async () => {
    job.mockResolvedValue(makeJob({ state: "COMPLETED" }));
    renderPanel();

    const expand = await screen.findByRole("button", { name: "Expand job panel" });
    expect(expand).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(expand);

    const shrink = screen.getByRole("button", { name: "Shrink job panel" });
    expect(shrink).toHaveAttribute("aria-expanded", "true");
  });

  it("retries a finished job and hands the new id to the app", async () => {
    job.mockResolvedValue(makeJob({ state: "FAILED", error: "boom" }));
    retryJob.mockResolvedValue({ job_id: "job-2", state: "QUEUED" });
    const onJobStarted = vi.fn();
    renderPanel({ onJobStarted });

    fireEvent.click(await screen.findByRole("button", { name: "Retry" }));

    await waitFor(() => expect(retryJob).toHaveBeenCalledWith("job-1"));
    await waitFor(() => expect(onJobStarted).toHaveBeenCalledWith("job-2"));
  });

  it("shows a retry failure", async () => {
    job.mockResolvedValue(makeJob({ state: "FAILED", error: "boom" }));
    retryJob.mockRejectedValue(new Error("cannot retry"));
    renderPanel();

    fireEvent.click(await screen.findByRole("button", { name: "Retry" }));
    expect(await screen.findByText("cannot retry")).toBeInTheDocument();
  });

  it("offers edit only when the app can receive the job", async () => {
    job.mockResolvedValue(makeJob({ state: "FAILED", kind: "dub", error: "boom" }));
    const onEdit = vi.fn();
    renderPanel({ onEdit });

    fireEvent.click(await screen.findByRole("button", { name: "Edit" }));
    expect(onEdit).toHaveBeenCalledWith("dub", expect.objectContaining({ id: "job-1" }));
  });

  it("does not offer actions while the job is still running", async () => {
    job.mockResolvedValue(makeJob({ state: "PROCESSING" }));
    const onEdit = vi.fn();
    renderPanel({ onEdit });
    await screen.findByText("Working…");

    expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit" })).not.toBeInTheDocument();
  });

  it("opens the compare view for a completed dub", async () => {
    job.mockResolvedValue(
      makeJob({
        state: "COMPLETED",
        input_label: "talk.mp4",
        result: { dubbed_video: "/output/run/video_dubbed.mp4" },
      }),
    );
    renderPanel();

    fireEvent.click(await screen.findByRole("button", { name: "Compare" }));
    expect(
      screen.getByRole("dialog", { name: "Compare original and dub" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Play talk.mp4")).toBeInTheDocument();
    expect(screen.getByLabelText("Play video_dubbed.mp4")).toBeInTheDocument();
  });
});
