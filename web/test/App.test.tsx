import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../src/App";

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

afterEach(() => {
  vi.useRealTimers();
});

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
    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "settings" },
    });
    expect(screen.getByText("Settings page")).toBeInTheDocument();
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
