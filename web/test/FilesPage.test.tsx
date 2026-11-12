import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FilesPage } from "../src/pages/FilesPage";
import type { OutputRun } from "../src/types";
import { makeFile } from "./factories";

const files = vi.fn();
const outputs = vi.fn();
const outputFiles = vi.fn();
const deleteFile = vi.fn();
const deleteFiles = vi.fn();
const deleteOutput = vi.fn();
const upload = vi.fn();

vi.mock("../src/api", () => ({
  api: {
    files: (...args: unknown[]) => files(...args),
    outputs: (...args: unknown[]) => outputs(...args),
    outputFiles: (...args: unknown[]) => outputFiles(...args),
    deleteFile: (...args: unknown[]) => deleteFile(...args),
    deleteFiles: (...args: unknown[]) => deleteFiles(...args),
    deleteOutput: (...args: unknown[]) => deleteOutput(...args),
    upload: (...args: unknown[]) => upload(...args),
    fileUrl: (id: string) => `/api/files/${id}/download`,
    fileInlineUrl: (id: string) => `/api/files/${id}/download?inline=1`,
    outputUrl: (path: string) => `/api/outputs/download?path=${encodeURIComponent(path)}`,
    outputInlineUrl: (path: string) =>
      `/api/outputs/download?path=${encodeURIComponent(path)}&inline=1`,
  },
}));

function filesPage(items: ReturnType<typeof makeFile>[], total = items.length) {
  return { files: items, total, page: 1, page_size: 20 };
}

function runOf(overrides: Partial<OutputRun> = {}): OutputRun {
  return {
    job_dir: "/output/talk__dub__French__abc",
    job_id: "job-1",
    title: "Dub video",
    input_label: "talk.mp4",
    created: 1_700_000_000,
    primary: {
      path: "/output/talk__dub__French__abc/dub.wav",
      name: "dub.wav",
      size: 4096,
      created: 1_700_000_000,
    },
    file_count: 3,
    ...overrides,
  };
}

function outputsPage(items: OutputRun[], total = items.length) {
  return { output_root: "/output", runs: items, total, page: 1, page_size: 20 };
}

beforeEach(() => {
  files.mockReset();
  outputs.mockReset();
  outputFiles.mockReset();
  deleteFile.mockReset();
  deleteFiles.mockReset();
  deleteOutput.mockReset();
  upload.mockReset();
  files.mockResolvedValue(filesPage([]));
  outputs.mockResolvedValue(outputsPage([]));
  outputFiles.mockResolvedValue({ job_dir: "/output/run", files: [] });
  deleteFile.mockResolvedValue({ deleted: "file-1" });
  deleteFiles.mockResolvedValue({ removed: ["file-1"] });
  deleteOutput.mockResolvedValue({ deleted: "/output/run/dub.wav" });
  upload.mockResolvedValue({ id: "2", name: "new.wav", path: "/tmp/new.wav", size: 3 });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("FilesPage", () => {
  it("shows both empty states", async () => {
    render(<FilesPage />);
    expect(
      await screen.findByText("No uploads yet. Add a video or audio clip above."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("No outputs yet. Each finished job appears here with its main file."),
    ).toBeInTheDocument();
  });

  it("lists uploads and one row per output run", async () => {
    files.mockResolvedValue(filesPage([makeFile()]));
    outputs.mockResolvedValue(outputsPage([runOf()]));
    render(<FilesPage />);

    expect(await screen.findByText("clip.mp4")).toBeInTheDocument();
    expect(screen.getByText(/input · 2\.0 KiB/)).toBeInTheDocument();
    expect(screen.getByText("Dub video")).toBeInTheDocument();
    expect(screen.getByText(/talk\.mp4 · dub\.wav/)).toBeInTheDocument();
    expect(screen.getByText(/3 files/)).toBeInTheDocument();

    const links = screen.getAllByRole("link", { name: "Download" });
    expect(links[0]).toHaveAttribute("href", "/api/files/file-1/download");
    expect(links[1]).toHaveAttribute(
      "href",
      "/api/outputs/download?path=%2Foutput%2Ftalk__dub__French__abc%2Fdub.wav",
    );
  });

  it("expands a run to show every artifact", async () => {
    outputs.mockResolvedValue(outputsPage([runOf()]));
    outputFiles.mockResolvedValue({
      job_dir: "/output/talk__dub__French__abc",
      files: [
        {
          path: "/output/talk__dub__French__abc/dub.wav",
          relative: "dub.wav",
          name: "dub.wav",
          size: 4096,
          created: 1_700_000_000,
        },
        {
          path: "/output/talk__dub__French__abc/report.json",
          relative: "report.json",
          name: "report.json",
          size: 128,
          created: 1_700_000_000,
        },
      ],
    });
    render(<FilesPage />);

    fireEvent.click(await screen.findByRole("button", { name: "Details (3)" }));

    await waitFor(() =>
      expect(outputFiles).toHaveBeenCalledWith("/output/talk__dub__French__abc"),
    );
    expect(await screen.findByText("report.json")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Hide details" })).toBeInTheDocument();
  });

  it("does not offer playback for an unsupported primary file", async () => {
    outputs.mockResolvedValue(
      outputsPage([
        runOf({
          primary: {
            path: "/output/run/report.json",
            name: "report.json",
            size: 10,
            created: 1_700_000_000,
          },
        }),
      ]),
    );
    render(<FilesPage />);
    await screen.findByText("Dub video");
    expect(screen.queryByRole("button", { name: "Play" })).not.toBeInTheDocument();
  });

  it("shows a load error", async () => {
    files.mockRejectedValue(new Error("uploads unavailable"));
    render(<FilesPage />);
    expect(await screen.findByText("uploads unavailable")).toBeInTheDocument();
  });

  it("refreshes on demand", async () => {
    render(<FilesPage />);
    await screen.findByText(/No uploads yet/);
    expect(files).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
    expect(files).toHaveBeenCalledTimes(2);
  });

  it("plays an upload inline", async () => {
    files.mockResolvedValue(filesPage([makeFile()]));
    render(<FilesPage />);
    await screen.findByText("clip.mp4");

    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    expect(await screen.findByLabelText("Play clip.mp4")).toBeInTheDocument();
  });

  it("copies an upload path", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    files.mockResolvedValue(filesPage([makeFile()]));
    render(<FilesPage />);
    await screen.findByText("clip.mp4");

    fireEvent.click(screen.getByRole("button", { name: "Copy path of clip.mp4" }));
    await waitFor(() =>
      expect(writeText).toHaveBeenCalledWith("data/uploads/file-1__clip.mp4"),
    );
    expect(await screen.findByText("Copied")).toBeInTheDocument();
  });

  it("deletes an upload after confirmation and refreshes", async () => {
    files.mockResolvedValue(filesPage([makeFile()]));
    render(<FilesPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete clip.mp4" }));
    expect(screen.getByRole("dialog", { name: "Delete clip.mp4" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(deleteFile).toHaveBeenCalledWith("file-1"));
    await waitFor(() => expect(files).toHaveBeenCalledTimes(2));
  });

  it("does not delete an upload when confirmation is cancelled", async () => {
    files.mockResolvedValue(filesPage([makeFile()]));
    render(<FilesPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete clip.mp4" }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(deleteFile).not.toHaveBeenCalled();
  });

  it("deletes the main output of a run after confirmation", async () => {
    outputs.mockResolvedValue(outputsPage([runOf()]));
    render(<FilesPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete dub.wav" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() =>
      expect(deleteOutput).toHaveBeenCalledWith(
        "/output/talk__dub__French__abc/dub.wav",
      ),
    );
  });

  it("shows a delete failure", async () => {
    files.mockResolvedValue(filesPage([makeFile()]));
    deleteFile.mockRejectedValue(new Error("cannot delete"));
    render(<FilesPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete clip.mp4" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(await screen.findByText("cannot delete")).toBeInTheDocument();
  });

  it("shows a delete failure for a run output", async () => {
    outputs.mockResolvedValue(outputsPage([runOf()]));
    deleteOutput.mockRejectedValue(new Error("output is locked"));
    render(<FilesPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete dub.wav" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(await screen.findByText("output is locked")).toBeInTheDocument();
  });

  it("deletes selected uploads in bulk", async () => {
    files.mockResolvedValue(filesPage([makeFile()]));
    render(<FilesPage />);
    fireEvent.click(await screen.findByRole("checkbox", { name: "Select clip.mp4" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete selected" }));
    expect(screen.getByRole("dialog", { name: "Delete 1 upload" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(deleteFiles).toHaveBeenCalledWith(["file-1"]));
  });

  it("paginates long run lists", async () => {
    outputs.mockResolvedValue(outputsPage([runOf()], 45));
    render(<FilesPage />);
    expect(await screen.findByText("Page 1 of 3")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() =>
      expect(outputs).toHaveBeenLastCalledWith({ page: 2, pageSize: 20 }),
    );
  });

  it("refreshes the lists after an upload", async () => {
    render(<FilesPage />);
    await screen.findByText(/No uploads yet/);
    const input = screen.getByLabelText("Upload a media file") as HTMLInputElement;
    fireEvent.change(input, {
      target: { files: [new File(["x"], "new.wav", { type: "audio/wav" })] },
    });

    await waitFor(() => expect(files).toHaveBeenCalledTimes(2));
    expect(upload).toHaveBeenCalled();
  });
});
