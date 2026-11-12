import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { JobPanel } from "../src/components/JobPanel";

const job = vi.fn();
const cancelJob = vi.fn();
const unsubscribe = vi.fn();

vi.mock("../src/api", () => ({
  api: {
    job: (...args: unknown[]) => job(...args),
    cancelJob: (...args: unknown[]) => cancelJob(...args),
  },
  subscribeJob: () => unsubscribe,
}));

function jobState(state: string, error: string | null = null) {
  return {
    id: "job-1",
    kind: "dub",
    state,
    stage: "done",
    params: {},
    result: {},
    error,
    created: 0,
    events: [],
  };
}

describe("JobPanel terminal state", () => {
  beforeEach(() => {
    job.mockReset();
    cancelJob.mockReset();
  });

  it("says cancelled, not successfully finished, for a cancelled job", async () => {
    job.mockResolvedValue(jobState("CANCELLED"));
    render(<JobPanel jobId="job-1" onClose={() => undefined} />);

    expect(await screen.findByText("Cancelled.")).toBeInTheDocument();
    expect(screen.queryByText("Finished successfully.")).not.toBeInTheDocument();
  });

  it("still says successfully finished for a completed job", async () => {
    job.mockResolvedValue(jobState("COMPLETED"));
    render(<JobPanel jobId="job-1" onClose={() => undefined} />);

    expect(await screen.findByText("Finished successfully.")).toBeInTheDocument();
    expect(screen.queryByText("Cancelled.")).not.toBeInTheDocument();
  });

  it("shows the error and neither terminal success message for a failed job", async () => {
    job.mockResolvedValue(jobState("FAILED", "boom"));
    render(<JobPanel jobId="job-1" onClose={() => undefined} />);

    expect(await screen.findByText("boom")).toBeInTheDocument();
    expect(screen.queryByText("Finished successfully.")).not.toBeInTheDocument();
    expect(screen.queryByText("Cancelled.")).not.toBeInTheDocument();
  });
});
