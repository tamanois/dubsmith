import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FilesPage } from "../src/pages/FilesPage";
import { makeFile, makeOutput } from "./factories";

const files = vi.fn();
const outputs = vi.fn();
const deleteFile = vi.fn();
const deleteOutput = vi.fn();
const upload = vi.fn();

vi.mock("../src/api", () => ({
  api: {
    files: (...args: unknown[]) => files(...args),
    outputs: (...args: unknown[]) => outputs(...args),
    deleteFile: (...args: unknown[]) => deleteFile(...args),
    deleteOutput: (...args: unknown[]) => deleteOutput(...args),
    upload: (...args: unknown[]) => upload(...args),
    fileUrl: (id: string) => `/api/files/${id}/download`,
    outputUrl: (path: string) => `/api/outputs/download?path=${encodeURIComponent(path)}`,
  },
}));

beforeEach(() => {
  files.mockReset();
  outputs.mockReset();
  deleteFile.mockReset();
  deleteOutput.mockReset();
  upload.mockReset();
  files.mockResolvedValue({ files: [] });
  outputs.mockResolvedValue({ output_root: "/output", files: [] });
  deleteFile.mockResolvedValue({ deleted: "file-1" });
  deleteOutput.mockResolvedValue({ deleted: "/output/run/subs_target.srt" });
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
      screen.getByText("No outputs yet. Finished jobs write into unique run folders."),
    ).toBeInTheDocument();
  });

  it("lists uploads and outputs with download links", async () => {
    files.mockResolvedValue({ files: [makeFile()] });
    outputs.mockResolvedValue({
      output_root: "/output",
      files: [makeOutput()],
    });
    render(<FilesPage />);

    expect(await screen.findByText("clip.mp4")).toBeInTheDocument();
    expect(screen.getByText(/input · 2\.0 KiB/)).toBeInTheDocument();
    expect(screen.getByText("run/subs_target.srt")).toBeInTheDocument();
    expect(screen.getByText(/128 B/)).toBeInTheDocument();

    const links = screen.getAllByRole("link", { name: "Download" });
    expect(links[0]).toHaveAttribute("href", "/api/files/file-1/download");
    expect(links[1]).toHaveAttribute(
      "href",
      "/api/outputs/download?path=%2Foutput%2Frun%2Fsubs_target.srt",
    );
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
    fireEvent.click(screen.getByRole("button", { name: /Refresh/ }));
    expect(files).toHaveBeenCalledTimes(2);
  });

  it("deletes an upload after confirmation and refreshes", async () => {
    files.mockResolvedValue({ files: [makeFile()] });
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<FilesPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete" }));

    await waitFor(() => expect(deleteFile).toHaveBeenCalledWith("file-1"));
    expect(confirm).toHaveBeenCalledWith('Delete "clip.mp4"?');
    await waitFor(() => expect(files).toHaveBeenCalledTimes(2));
  });

  it("does not delete an upload when confirmation is cancelled", async () => {
    files.mockResolvedValue({ files: [makeFile()] });
    vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<FilesPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete" }));
    expect(deleteFile).not.toHaveBeenCalled();
  });

  it("deletes a generated output after confirmation", async () => {
    outputs.mockResolvedValue({ output_root: "/output", files: [makeOutput()] });
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<FilesPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete" }));

    await waitFor(() =>
      expect(deleteOutput).toHaveBeenCalledWith("/output/run/subs_target.srt"),
    );
    expect(confirm).toHaveBeenCalledWith('Delete output "run/subs_target.srt"?');
  });

  it("shows a delete failure", async () => {
    files.mockResolvedValue({ files: [makeFile()] });
    vi.spyOn(window, "confirm").mockReturnValue(true);
    deleteFile.mockRejectedValue(new Error("cannot delete"));
    render(<FilesPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete" }));
    expect(await screen.findByText("cannot delete")).toBeInTheDocument();
  });

  it("shows a delete failure for an output", async () => {
    outputs.mockResolvedValue({ output_root: "/output", files: [makeOutput()] });
    vi.spyOn(window, "confirm").mockReturnValue(true);
    deleteOutput.mockRejectedValue(new Error("output is locked"));
    render(<FilesPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete" }));
    expect(await screen.findByText("output is locked")).toBeInTheDocument();
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
