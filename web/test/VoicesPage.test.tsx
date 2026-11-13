import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VoicesPage } from "../src/pages/VoicesPage";
import { makeVoice } from "./factories";

const voices = vi.fn();
const designVoice = vi.fn();
const cloneVoice = vi.fn();
const addVoice = vi.fn();
const deleteVoice = vi.fn();

vi.mock("../src/api", () => ({
  api: {
    voices: (...args: unknown[]) => voices(...args),
    designVoice: (...args: unknown[]) => designVoice(...args),
    cloneVoice: (...args: unknown[]) => cloneVoice(...args),
    addVoice: (...args: unknown[]) => addVoice(...args),
    deleteVoice: (...args: unknown[]) => deleteVoice(...args),
    voiceAudioUrl: (label: string) => `/api/voices/${label}/audio`,
  },
}));

vi.mock("../src/components/UploadField", () => ({
  UploadField: ({
    label,
    onUploaded,
  }: {
    label: string;
    onUploaded: (path: string, name: string) => void;
  }) => (
    <button type="button" onClick={() => onUploaded("/tmp/clip.wav", "clip.wav")}>
      upload:{label}
    </button>
  ),
}));

function openDisclosure(name: RegExp): HTMLElement {
  const button = screen.getByRole("button", { name });
  fireEvent.click(button);
  return button.parentElement as HTMLElement;
}

beforeEach(() => {
  voices.mockReset();
  designVoice.mockReset();
  cloneVoice.mockReset();
  addVoice.mockReset();
  deleteVoice.mockReset();
  voices.mockResolvedValue({ voices: [], total: 0, page: 1, page_size: 20 });
  for (const action of [designVoice, cloneVoice, addVoice]) {
    action.mockResolvedValue({ job_id: "job-voice", state: "QUEUED" });
  }
  deleteVoice.mockResolvedValue({ removed: "warm-fr" });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("VoicesPage library", () => {
  it("shows an empty state", async () => {
    render(<VoicesPage onJobStarted={vi.fn()} />);
    expect(await screen.findByText(/No voices yet/)).toBeInTheDocument();
  });

  it("renders voice metadata", async () => {
    voices.mockResolvedValue({
      voices: [
        makeVoice({ label: "warm-fr", instruct: "warm and slow" }),
        makeVoice({ label: "plain-en", has_transcript: false, instruct: null }),
      ],
      total: 2,
      page: 1,
      page_size: 20,
    });
    render(<VoicesPage onJobStarted={vi.fn()} />);

    expect(await screen.findByText("warm-fr")).toBeInTheDocument();
    expect(screen.getAllByText(/French · design · 12\.5s · 1 card/)).toHaveLength(2);
    expect(
      screen.getAllByText(/(transcript stored|no transcript) · f \/ normal/),
    ).toHaveLength(2);
    expect(screen.getByText(/warm and slow/)).toBeInTheDocument();
    expect(screen.getByText("plain-en")).toBeInTheDocument();
    expect(screen.getByText(/no transcript/)).toBeInTheDocument();
  });

  it("deletes a voice after confirmation and refreshes", async () => {
    voices.mockResolvedValue({
      voices: [makeVoice()],
      total: 1,
      page: 1,
      page_size: 20,
    });
    render(<VoicesPage onJobStarted={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete" }));
    expect(screen.getByRole("dialog", { name: "Delete warm-fr" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Delete voice" }));
    await waitFor(() => expect(deleteVoice).toHaveBeenCalledWith("warm-fr"));
    await waitFor(() => expect(voices).toHaveBeenCalledTimes(2));
  });

  it("keeps a voice when confirmation is cancelled", async () => {
    voices.mockResolvedValue({
      voices: [makeVoice()],
      total: 1,
      page: 1,
      page_size: 20,
    });
    render(<VoicesPage onJobStarted={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete" }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(deleteVoice).not.toHaveBeenCalled();
  });

  it("auditions a voice", async () => {
    voices.mockResolvedValue({
      voices: [makeVoice()],
      total: 1,
      page: 1,
      page_size: 20,
    });
    render(<VoicesPage onJobStarted={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Play" }));
    expect(await screen.findByLabelText("Play warm-fr")).toBeInTheDocument();
  });

  it("paginates the library", async () => {
    voices.mockResolvedValue({
      voices: [makeVoice()],
      total: 45,
      page: 1,
      page_size: 20,
    });
    render(<VoicesPage onJobStarted={vi.fn()} />);
    expect(await screen.findByText("Page 1 of 3")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() =>
      expect(voices).toHaveBeenLastCalledWith({ page: 2, pageSize: 20 }),
    );
  });

  it("shows a delete failure and a load error", async () => {
    voices.mockResolvedValueOnce({
      voices: [makeVoice()],
      total: 1,
      page: 1,
      page_size: 20,
    });
    deleteVoice.mockRejectedValueOnce(new Error("voice in use"));
    const { unmount } = render(<VoicesPage onJobStarted={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete voice" }));
    expect(await screen.findByText("voice in use")).toBeInTheDocument();
    unmount();

    voices.mockRejectedValueOnce(new Error("voices unreadable"));
    render(<VoicesPage onJobStarted={vi.fn()} />);
    expect(await screen.findByText("voices unreadable")).toBeInTheDocument();
  });
});

describe("VoicesPage design", () => {
  it("requires a description and sends the design payload", async () => {
    const onJobStarted = vi.fn();
    render(<VoicesPage onJobStarted={onJobStarted} />);
    const design = openDisclosure(/Design a voice/);
    const button = within(design).getByRole("button", { name: "Design voice" });
    expect(button).toBeDisabled();

    fireEvent.change(within(design).getByRole("textbox", { name: /Description/ }), {
      target: { value: "a warm French voice" },
    });
    fireEvent.change(within(design).getByRole("textbox", { name: /Label/ }), {
      target: { value: "warm-fr" },
    });
    fireEvent.change(within(design).getByRole("textbox", { name: /Audition text/ }), {
      target: { value: "Bonjour" },
    });
    fireEvent.change(within(design).getByLabelText(/Device/), {
      target: { value: "cuda:0" },
    });
    fireEvent.change(within(design).getByLabelText(/Precision/), {
      target: { value: "fp16" },
    });
    fireEvent.click(within(design).getByRole("checkbox", { name: /Overwrite/ }));
    fireEvent.click(button);

    await waitFor(() =>
      expect(designVoice).toHaveBeenCalledWith({
        prompt: "a warm French voice",
        language: "French",
        label: "warm-fr",
        speak: "Bonjour",
        device: "cuda:0",
        dtype: "fp16",
        force: true,
      }),
    );
    expect(onJobStarted).toHaveBeenCalledWith("job-voice");
    await waitFor(() => expect(button).toBeDisabled());
  });

  it("shows a design failure", async () => {
    designVoice.mockRejectedValueOnce(new Error("design failed"));
    render(<VoicesPage onJobStarted={vi.fn()} />);
    const design = openDisclosure(/Design a voice/);
    fireEvent.change(within(design).getByRole("textbox", { name: /Description/ }), {
      target: { value: "a voice" },
    });
    fireEvent.click(within(design).getByRole("button", { name: "Design voice" }));
    expect(await screen.findByText("design failed")).toBeInTheDocument();
  });
});

describe("VoicesPage clone and register", () => {
  it("requires a clip and sends the clone payload", async () => {
    const onJobStarted = vi.fn();
    render(<VoicesPage onJobStarted={onJobStarted} />);
    const clone = openDisclosure(/Clone a clip/);
    const button = within(clone).getByRole("button", { name: "Clone voice" });
    expect(button).toBeDisabled();

    fireEvent.click(within(clone).getByRole("button", { name: "upload:Reference clip" }));
    fireEvent.change(within(clone).getByRole("textbox", { name: /Transcript/ }), {
      target: { value: "Bonjour tout le monde" },
    });
    fireEvent.change(within(clone).getByRole("textbox", { name: /Label/ }), {
      target: { value: "deep-fr" },
    });
    fireEvent.change(within(clone).getByLabelText(/Language/), {
      target: { value: "German" },
    });
    fireEvent.click(within(clone).getByRole("checkbox", { name: /Verify/ }));
    fireEvent.click(within(clone).getByRole("checkbox", { name: /Overwrite/ }));
    fireEvent.click(button);

    await waitFor(() =>
      expect(cloneVoice).toHaveBeenCalledWith({
        audio: "/tmp/clip.wav",
        clone_text: "Bonjour tout le monde",
        transcribe: true,
        verify: true,
        language: "German",
        label: "deep-fr",
        device: "auto",
        dtype: "auto",
        force: true,
      }),
    );
    expect(onJobStarted).toHaveBeenCalledWith("job-voice");
  });

  it("sends the register payload without a precision field", async () => {
    render(<VoicesPage onJobStarted={vi.fn()} />);
    const register = openDisclosure(/Register an existing clip/);
    fireEvent.click(within(register).getByRole("button", { name: "upload:Clip" }));
    fireEvent.click(within(register).getByRole("button", { name: "Register voice" }));

    await waitFor(() =>
      expect(addVoice).toHaveBeenCalledWith({
        audio: "/tmp/clip.wav",
        clone_text: undefined,
        transcribe: true,
        verify: false,
        language: "French",
        label: undefined,
        device: "auto",
        force: false,
      }),
    );
  });

  it("disables every action while a voice job starts", async () => {
    let resolve!: (value: unknown) => void;
    designVoice.mockReturnValue(new Promise((done) => (resolve = done)));
    const onJobStarted = vi.fn();
    render(<VoicesPage onJobStarted={onJobStarted} />);
    const design = openDisclosure(/Design a voice/);
    const description = within(design).getByRole("textbox", { name: /Description/ });
    fireEvent.change(description, { target: { value: "a voice" } });
    fireEvent.click(within(design).getByRole("button", { name: "Design voice" }));

    const clone = openDisclosure(/Clone a clip/);
    fireEvent.click(within(clone).getByRole("button", { name: "upload:Reference clip" }));
    await waitFor(() =>
      expect(within(clone).getByRole("button", { name: "Clone voice" })).toBeDisabled(),
    );
    expect(within(design).getByRole("button", { name: "Design voice" })).toBeDisabled();

    resolve({ job_id: "job-voice", state: "QUEUED" });
    await waitFor(() => expect(onJobStarted).toHaveBeenCalledWith("job-voice"));
    expect(description).toHaveValue("");
  });
});
