import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { UploadField } from "../src/components/UploadField";

const upload = vi.fn();

vi.mock("../src/api", () => ({
  api: {
    upload: (...args: unknown[]) => upload(...args),
  },
}));

function chooseFile(name = "clip.mp4") {
  const input = screen.getByLabelText("Input file") as HTMLInputElement;
  fireEvent.change(input, {
    target: { files: [new File(["x"], name, { type: "video/mp4" })] },
  });
  return input;
}

beforeEach(() => {
  upload.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("UploadField", () => {
  it("does nothing when no file is selected", () => {
    render(<UploadField label="Input file" onUploaded={vi.fn()} />);
    const input = screen.getByLabelText("Input file") as HTMLInputElement;
    fireEvent.change(input, { target: { files: [] } });
    expect(upload).not.toHaveBeenCalled();
  });

  it("shows progress, then the file name, and calls onUploaded", async () => {
    let resolve!: (value: unknown) => void;
    upload.mockReturnValue(new Promise((done) => (resolve = done)));
    const onUploaded = vi.fn();
    render(<UploadField label="Input file" onUploaded={onUploaded} />);

    const input = chooseFile();
    expect(screen.getByText("Uploading…")).toBeInTheDocument();
    expect(input).toBeDisabled();

    resolve({ id: "1", name: "clip.mp4", path: "/tmp/clip.mp4", size: 1 });
    await waitFor(() =>
      expect(onUploaded).toHaveBeenCalledWith("/tmp/clip.mp4", "clip.mp4"),
    );
    expect(screen.getByText("clip.mp4")).toBeInTheDocument();
    expect(input).not.toBeDisabled();
  });

  it("shows an upload error and allows a retry", async () => {
    upload.mockRejectedValueOnce(new Error("upload denied"));
    const onUploaded = vi.fn();
    render(<UploadField label="Input file" onUploaded={onUploaded} />);

    chooseFile();
    expect(await screen.findByText("upload denied")).toBeInTheDocument();
    expect(onUploaded).not.toHaveBeenCalled();

    upload.mockResolvedValueOnce({
      id: "2",
      name: "second.mp4",
      path: "/tmp/second.mp4",
      size: 2,
    });
    chooseFile("second.mp4");
    await waitFor(() =>
      expect(onUploaded).toHaveBeenCalledWith("/tmp/second.mp4", "second.mp4"),
    );
    expect(screen.getByText("second.mp4")).toBeInTheDocument();
  });

  it("replaces the displayed name on a second upload", async () => {
    upload
      .mockResolvedValueOnce({ id: "1", name: "one.mp4", path: "/tmp/one.mp4", size: 1 })
      .mockResolvedValueOnce({ id: "2", name: "two.mp4", path: "/tmp/two.mp4", size: 2 });
    const onUploaded = vi.fn();
    render(<UploadField label="Input file" onUploaded={onUploaded} />);

    chooseFile("one.mp4");
    await waitFor(() => expect(screen.getByText("one.mp4")).toBeInTheDocument());
    chooseFile("two.mp4");
    await waitFor(() => expect(screen.getByText("two.mp4")).toBeInTheDocument());
    expect(onUploaded).toHaveBeenLastCalledWith("/tmp/two.mp4", "two.mp4");
  });
});
