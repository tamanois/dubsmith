import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { JobsPage } from "../src/pages/JobsPage";
import { makeJob } from "./factories";

const jobs = vi.fn();
const clearJobs = vi.fn();

vi.mock("../src/api", () => ({
  api: {
    jobs: (...args: unknown[]) => jobs(...args),
    clearJobs: (...args: unknown[]) => clearJobs(...args),
  },
}));

function pageOf(
  items: ReturnType<typeof makeJob>[],
  total = items.length,
  page = 1,
  pageSize = 20,
) {
  return { jobs: items, total, page, page_size: pageSize };
}

beforeEach(() => {
  jobs.mockReset();
  clearJobs.mockReset();
  jobs.mockResolvedValue(pageOf([]));
});

afterEach(() => {
  vi.useRealTimers();
  Object.defineProperty(document, "hidden", { configurable: true, value: false });
});

describe("JobsPage", () => {
  it("shows an empty state", async () => {
    render(<JobsPage onOpenJob={vi.fn()} activeJobId={null} />);
    expect(await screen.findByText(/No jobs yet/)).toBeInTheDocument();
  });

  it("renders a human title, labels and opens the job", async () => {
    jobs.mockResolvedValue(
      pageOf([
        makeJob({
          id: "abcdef1234567890",
          kind: "dub",
          title: "Dub video",
          input_label: "talk.mp4",
          output_label: "video_dubbed.mp4",
          state: "FAILED",
          error: "llm down",
          created: 1_700_000_000,
        }),
      ]),
    );
    const onOpenJob = vi.fn();
    render(<JobsPage onOpenJob={onOpenJob} activeJobId={null} />);

    expect(await screen.findByText("Dub video")).toBeInTheDocument();
    expect(screen.getByText("FAILED")).toBeInTheDocument();
    expect(screen.getByText(/talk\.mp4/)).toBeInTheDocument();
    expect(screen.getByText(/video_dubbed\.mp4/)).toBeInTheDocument();
    expect(screen.getByText("llm down")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Dub video/ }));
    expect(onOpenJob).toHaveBeenCalledWith("abcdef1234567890");
  });

  it("shows duration only when both timestamps exist", async () => {
    jobs.mockResolvedValue(
      pageOf([
        makeJob({ id: "dur-yes", started: 100, finished: 165 }),
        makeJob({ id: "dur-no", started: null, finished: 200 }),
      ]),
    );
    render(<JobsPage onOpenJob={vi.fn()} activeJobId={null} />);

    expect(await screen.findByText(/1m 5s/)).toBeInTheDocument();
    expect(screen.getByText(/dur-yes/)).toBeInTheDocument();
    expect(screen.getByText(/dur-no/).textContent).not.toMatch(/1m 5s/);
  });

  it("marks the active job", async () => {
    jobs.mockResolvedValue(
      pageOf([makeJob({ id: "active-1" }), makeJob({ id: "other-1" })]),
    );
    render(<JobsPage onOpenJob={vi.fn()} activeJobId="active-1" />);

    await screen.findByTestId("job-row-active-1");
    expect(screen.getByTestId("job-row-active-1").className).toContain(
      "bg-accent-500/5",
    );
    expect(screen.getByTestId("job-row-other-1").className).not.toContain(
      "bg-accent-500/5",
    );
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

  it("sends the state filter and the debounced search", async () => {
    render(<JobsPage onOpenJob={vi.fn()} activeJobId={null} />);
    await screen.findByText(/No jobs yet/);

    fireEvent.change(screen.getByRole("combobox", { name: "Filter by state" }), {
      target: { value: "FAILED" },
    });
    await waitFor(() =>
      expect(jobs).toHaveBeenLastCalledWith(
        expect.objectContaining({ state: "FAILED", page: 1 }),
      ),
    );

    fireEvent.change(screen.getByRole("textbox", { name: "Search jobs" }), {
      target: { value: "talk" },
    });
    await waitFor(() =>
      expect(jobs).toHaveBeenLastCalledWith(
        expect.objectContaining({ q: "talk", page: 1 }),
      ),
    );
  });

  it("paginates long lists", async () => {
    jobs.mockResolvedValue(pageOf([makeJob({ id: "one" })], 45));
    render(<JobsPage onOpenJob={vi.fn()} activeJobId={null} />);

    expect(await screen.findByText("Page 1 of 3")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() =>
      expect(jobs).toHaveBeenLastCalledWith(
        expect.objectContaining({ page: 2, pageSize: 20 }),
      ),
    );
  });

  it("clears selected terminal jobs after confirmation", async () => {
    jobs.mockResolvedValue(
      pageOf([makeJob({ id: "done-1", title: "Dub video", state: "COMPLETED" })]),
    );
    clearJobs.mockResolvedValue({ removed: 1 });
    render(<JobsPage onOpenJob={vi.fn()} activeJobId={null} />);

    fireEvent.click(await screen.findByRole("checkbox", { name: "Select Dub video" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear selected" }));
    expect(screen.getByRole("dialog", { name: "Clear selected jobs" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Clear jobs" }));
    await waitFor(() => expect(clearJobs).toHaveBeenCalledWith(["done-1"]));
  });

  it("does not let running jobs be selected", async () => {
    jobs.mockResolvedValue(
      pageOf([makeJob({ id: "run-1", title: "Dub video", state: "PROCESSING" })]),
    );
    render(<JobsPage onOpenJob={vi.fn()} activeJobId={null} />);

    expect(
      await screen.findByRole("checkbox", { name: "Select Dub video" }),
    ).toBeDisabled();
  });

  it("polls every 2500 ms while a job is running and stops on unmount", async () => {
    vi.useFakeTimers();
    jobs.mockResolvedValue(
      pageOf([
        makeJob({ id: "run-1", title: "Dub video", state: "PROCESSING" }),
      ]),
    );
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

  it("backs off to a slow poll when every job is finished", async () => {
    vi.useFakeTimers();
    const { unmount } = render(<JobsPage onOpenJob={vi.fn()} activeJobId={null} />);
    await act(async () => undefined);
    expect(jobs).toHaveBeenCalledTimes(1);

    // Empty list: no fast tick.
    await act(async () => {
      vi.advanceTimersByTime(2500);
    });
    expect(jobs).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(12500);
    });
    expect(jobs).toHaveBeenCalledTimes(2);
    unmount();
  });

  it("does not poll while the tab is hidden", async () => {
    vi.useFakeTimers();
    const { unmount } = render(<JobsPage onOpenJob={vi.fn()} activeJobId={null} />);
    await act(async () => undefined);
    expect(jobs).toHaveBeenCalledTimes(1);

    Object.defineProperty(document, "hidden", { configurable: true, value: true });
    await act(async () => {
      vi.advanceTimersByTime(30000);
    });
    expect(jobs).toHaveBeenCalledTimes(1);

    Object.defineProperty(document, "hidden", { configurable: true, value: false });
    await act(async () => {
      document.dispatchEvent(new Event("visibilitychange"));
    });
    expect(jobs).toHaveBeenCalledTimes(2);
    unmount();
  });
});
