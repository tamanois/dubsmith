import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { JobsPage } from "../src/pages/JobsPage";
import { makeJob } from "./factories";

const jobs = vi.fn();

vi.mock("../src/api", () => ({
  api: {
    jobs: (...args: unknown[]) => jobs(...args),
  },
}));

beforeEach(() => {
  jobs.mockReset();
  jobs.mockResolvedValue({ jobs: [] });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("JobsPage", () => {
  it("shows an empty state", async () => {
    render(<JobsPage onOpenJob={vi.fn()} activeJobId={null} />);
    expect(
      await screen.findByText(/No jobs yet/),
    ).toBeInTheDocument();
  });

  it("renders a job row and opens it", async () => {
    jobs.mockResolvedValue({
      jobs: [
        makeJob({
          id: "abcdef1234567890",
          kind: "dub",
          state: "FAILED",
          stage: "translate",
          error: "llm down",
          created: 1_700_000_000,
        }),
      ],
    });
    const onOpenJob = vi.fn();
    render(<JobsPage onOpenJob={onOpenJob} activeJobId={null} />);

    expect(await screen.findByText("dub")).toBeInTheDocument();
    expect(screen.getByText("FAILED")).toBeInTheDocument();
    expect(screen.getByText(/abcdef123456/)).toBeInTheDocument();
    expect(screen.getByText(/translate/)).toBeInTheDocument();
    expect(screen.getByText("llm down")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /dub/ }));
    expect(onOpenJob).toHaveBeenCalledWith("abcdef1234567890");
  });

  it("shows duration only when both timestamps exist", async () => {
    jobs.mockResolvedValue({
      jobs: [
        makeJob({ id: "dur-yes", started: 100, finished: 165 }),
        makeJob({ id: "dur-no", started: null, finished: 200 }),
      ],
    });
    render(<JobsPage onOpenJob={vi.fn()} activeJobId={null} />);

    expect(await screen.findByText(/1m 5s/)).toBeInTheDocument();
    expect(screen.getByText(/dur-yes/)).toBeInTheDocument();
    expect(screen.getByText(/dur-no/)).toBeInTheDocument();
    expect(screen.getByText(/dur-no/).textContent).not.toMatch(/1m 5s/);
  });

  it("marks the active job", async () => {
    jobs.mockResolvedValue({
      jobs: [
        makeJob({ id: "active-1" }),
        makeJob({ id: "other-1" }),
      ],
    });
    render(<JobsPage onOpenJob={vi.fn()} activeJobId="active-1" />);

    const active = await screen.findByRole("button", { name: /active-1/ });
    expect(active.className).toContain("bg-accent-500/5");
    const other = screen.getByRole("button", { name: /other-1/ });
    expect(other.className).not.toContain("bg-accent-500/5");
  });

  it("refreshes on demand", async () => {
    render(<JobsPage onOpenJob={vi.fn()} activeJobId={null} />);
    await screen.findByText(/No jobs yet/);
    expect(jobs).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
    expect(jobs).toHaveBeenCalledTimes(2);
  });

  it("shows an API error", async () => {
    jobs.mockRejectedValue(new Error("store unavailable"));
    render(<JobsPage onOpenJob={vi.fn()} activeJobId={null} />);
    expect(await screen.findByText("store unavailable")).toBeInTheDocument();
  });

  it("polls every 2500 ms and stops on unmount", async () => {
    vi.useFakeTimers();
    const { unmount } = render(<JobsPage onOpenJob={vi.fn()} activeJobId={null} />);
    await act(async () => undefined);
    expect(jobs).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(2500);
    });
    expect(jobs).toHaveBeenCalledTimes(2);

    unmount();
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(jobs).toHaveBeenCalledTimes(2);
  });
});
