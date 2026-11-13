import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../src/App";
import { makeDeviceReport, makeJob, makeSettings } from "./factories";

const apiMock = vi.hoisted(() => ({
  settings: vi.fn(),
  devices: vi.fn(),
  patchSettings: vi.fn(),
  jobs: vi.fn(),
}));

vi.mock("../src/api", () => ({
  api: apiMock,
  subscribeJob: vi.fn(() => () => undefined),
}));

vi.mock("../src/components/Onboarding", () => ({
  Onboarding: () => null,
}));

vi.mock("../src/components/JobPanel", () => ({
  JobPanel: ({ jobId, onClose }: { jobId: string; onClose: () => void }) => (
    <div>
      <span>Job panel {jobId}</span>
      <button type="button" onClick={onClose}>
        close panel
      </button>
    </div>
  ),
}));

vi.mock("../src/pages/JobsPage", () => ({
  JobsPage: ({
    onOpenJob,
    activeJobId,
  }: {
    onOpenJob: (id: string) => void;
    activeJobId: string | null;
  }) => (
    <div>
      <span>Jobs page</span>
      <span>active:{activeJobId ?? "none"}</span>
      <button type="button" onClick={() => onOpenJob("opened-job")}>
        open job
      </button>
    </div>
  ),
}));

vi.mock("../src/pages/DubPage", () => ({
  DubPage: ({ onJobStarted }: { onJobStarted: (id: string) => void }) => (
    <div>
      <span>Dub page</span>
      <button type="button" onClick={() => onJobStarted("job-42")}>
        start dub
      </button>
    </div>
  ),
}));

vi.mock("../src/pages/SayPage", () => ({
  SayPage: () => <div>Say page</div>,
}));

vi.mock("../src/pages/VoicesPage", () => ({
  VoicesPage: () => <div>Voices page</div>,
}));

vi.mock("../src/pages/ModelsPage", () => ({
  ModelsPage: () => <div>Models page</div>,
}));

vi.mock("../src/pages/DevicesPage", () => ({
  DevicesPage: () => <div>Devices page</div>,
}));

vi.mock("../src/pages/FilesPage", () => ({
  FilesPage: () => <div>Files page</div>,
}));

vi.mock("../src/pages/SettingsPage", () => ({
  SettingsPage: () => <div>Settings page</div>,
}));

beforeEach(() => {
  // The page now comes from the URL hash, so each test must start from a clean
  // address instead of inheriting the previous test's route.
  window.location.hash = "";
  apiMock.settings.mockReset();
  apiMock.devices.mockReset();
  apiMock.patchSettings.mockReset();
  apiMock.jobs.mockReset();
  apiMock.settings.mockResolvedValue(makeSettings());
  apiMock.devices.mockResolvedValue(makeDeviceReport());
  apiMock.patchSettings.mockImplementation(async (patch: Record<string, unknown>) => ({
    ...makeSettings(),
    ...patch,
  }));
  apiMock.jobs.mockResolvedValue({
    jobs: [],
    total: 0,
    page: 1,
    page_size: 50,
  });
});

afterEach(() => {
  vi.useRealTimers();
  setDocumentHidden(false);
});

function setDocumentHidden(hidden: boolean) {
  Object.defineProperty(document, "hidden", { configurable: true, value: hidden });
}

describe("App navigation", () => {
  it("opens on Activity with no job panel", () => {
    render(<App />);
    expect(screen.getByText("Jobs page")).toBeInTheDocument();
    expect(screen.queryByText(/Job panel/)).not.toBeInTheDocument();
    expect(screen.getByText("local")).toBeInTheDocument();
  });

  it("navigates every sidebar destination", () => {
    render(<App />);
    const destinations: [RegExp, string][] = [
      [/Dub/, "Dub page"],
      [/Say/, "Say page"],
      [/Voices/, "Voices page"],
      [/Models/, "Models page"],
      [/Devices/, "Devices page"],
      [/Files/, "Files page"],
      [/Settings/, "Settings page"],
      [/Activity/, "Jobs page"],
    ];
    for (const [name, content] of destinations) {
      fireEvent.click(screen.getByRole("button", { name }));
      expect(screen.getByText(content)).toBeInTheDocument();
    }
  });

  it("navigates with the compact page selector", () => {
    render(<App />);
    fireEvent.change(screen.getByRole("combobox", { name: "Page" }), {
      target: { value: "settings" },
    });
    expect(screen.getByText("Settings page")).toBeInTheDocument();
  });

  it("opens the page named in the URL hash", () => {
    window.location.hash = "#/files";
    render(<App />);
    expect(screen.getByText("Files page")).toBeInTheDocument();
  });

  it("writes the route to the URL hash when navigating", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: /Settings/ }));
    expect(window.location.hash).toBe("#/settings");
  });

  it("opens a job from the activity list", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "open job" }));
    expect(screen.getByText("Job panel opened-job")).toBeInTheDocument();
  });
});

describe("App job flow", () => {
  it("switches to Activity, opens the panel and shows a toast on job start", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: /Dub/ }));
    fireEvent.click(screen.getByRole("button", { name: "start dub" }));

    expect(screen.getByText("Jobs page")).toBeInTheDocument();
    expect(screen.getByText("active:job-42")).toBeInTheDocument();
    expect(screen.getByText("Job panel job-42")).toBeInTheDocument();
    expect(
      screen.getByText("Job accepted. Live progress is shown in the activity panel."),
    ).toBeInTheDocument();
  });

  it("dismisses the toast after four seconds", () => {
    vi.useFakeTimers();
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: /Dub/ }));
    fireEvent.click(screen.getByRole("button", { name: "start dub" }));
    expect(screen.getByText(/Job accepted/)).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(4000);
    });
    expect(screen.queryByText(/Job accepted/)).not.toBeInTheDocument();
  });

  it("closes the current job panel", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: /Dub/ }));
    fireEvent.click(screen.getByRole("button", { name: "start dub" }));

    fireEvent.click(screen.getByRole("button", { name: "close panel" }));
    expect(screen.queryByText(/Job panel/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "current job" })).not.toBeInTheDocument();
  });
});

describe("App preferences and notifications", () => {
  it("persists a theme change and applies the light class", async () => {
    render(<App />);
    await waitFor(() => expect(apiMock.settings).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: "Light" }));

    await waitFor(() =>
      expect(apiMock.patchSettings).toHaveBeenCalledWith({ theme: "light" }),
    );
    expect(document.documentElement.classList.contains("light")).toBe(true);
  });

  it("persists the default device from the header", async () => {
    render(<App />);
    await waitFor(() => expect(apiMock.devices).toHaveBeenCalled());

    const select = await screen.findByRole("combobox", { name: "Default device" });
    fireEvent.change(select, { target: { value: "cpu" } });

    await waitFor(() =>
      expect(apiMock.patchSettings).toHaveBeenCalledWith({ default_device: "cpu" }),
    );
  });

  it("announces a job finishing in the background", async () => {
    vi.useFakeTimers();
    apiMock.jobs.mockResolvedValueOnce({
      jobs: [makeJob({ id: "job-9", state: "PROCESSING", title: "Dub video" })],
      total: 1,
      page: 1,
      page_size: 50,
    });
    render(<App />);
    await act(async () => undefined);

    apiMock.jobs.mockResolvedValue({
      jobs: [makeJob({ id: "job-9", state: "COMPLETED", title: "Dub video" })],
      total: 1,
      page: 1,
      page_size: 50,
    });
    await act(async () => {
      vi.advanceTimersByTime(2500);
    });

    expect(screen.getByText("Dub video finished.")).toBeInTheDocument();
  });

  it("slows the completion watcher down when nothing is running", async () => {
    vi.useFakeTimers();
    apiMock.jobs.mockResolvedValue({
      jobs: [makeJob({ id: "job-1", state: "COMPLETED", title: "Dub video" })],
      total: 1,
      page: 1,
      page_size: 50,
    });
    render(<App />);
    await act(async () => undefined);
    expect(apiMock.jobs).toHaveBeenCalledTimes(1);

    // No active job, so the fast tick must not fire.
    await act(async () => {
      vi.advanceTimersByTime(2500);
    });
    expect(apiMock.jobs).toHaveBeenCalledTimes(1);

    // The idle tick still refreshes the view of finished jobs.
    await act(async () => {
      vi.advanceTimersByTime(12500);
    });
    expect(apiMock.jobs).toHaveBeenCalledTimes(2);
  });

  it("pauses the completion watcher while the tab is hidden", async () => {
    vi.useFakeTimers();
    apiMock.jobs.mockResolvedValue({ jobs: [], total: 0, page: 1, page_size: 50 });
    render(<App />);
    await act(async () => undefined);
    expect(apiMock.jobs).toHaveBeenCalledTimes(1);

    setDocumentHidden(true);
    await act(async () => {
      vi.advanceTimersByTime(15000);
    });
    expect(apiMock.jobs).toHaveBeenCalledTimes(1);

    setDocumentHidden(false);
    await act(async () => {
      document.dispatchEvent(new Event("visibilitychange"));
    });
    expect(apiMock.jobs).toHaveBeenCalledTimes(2);
  });
});
