import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SayPage } from "../src/pages/SayPage";
import { makeBackends, makeJob, makeSettings, makeVoice } from "./factories";

const voices = vi.fn();
const settings = vi.fn();
const backends = vi.fn();
const submitSay = vi.fn();
const upload = vi.hoisted(() => ({ path: "/tmp/clone.wav", name: "clone.wav" }));

vi.mock("../src/api", () => ({
  api: {
    voices: (...args: unknown[]) => voices(...args),
    settings: (...args: unknown[]) => settings(...args),
    backends: (...args: unknown[]) => backends(...args),
    submitSay: (...args: unknown[]) => submitSay(...args),
    files: async () => ({ files: [], total: 0, page: 1, page_size: 20 }),
    favorites: async () => ({ favorites: [] }),
    addFavorite: async () => ({ favorite: { path: "", kind: "input" } }),
    presets: async () => ({ presets: [] }),
    createPreset: vi.fn(),
    deletePreset: vi.fn(),
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
    <button type="button" onClick={() => onUploaded(upload.path, upload.name)}>
      upload:{label}
    </button>
  ),
}));

function openDisclosure(name: RegExp): HTMLElement {
  const button = screen.getByRole("button", { name });
  fireEvent.click(button);
  return button.parentElement as HTMLElement;
}

/** Open the collapsed "Advanced options" section and return its container. */
function openAdvanced(): HTMLElement {
  return openDisclosure(/Advanced options/);
}

async function renderPage() {
  const onJobStarted = vi.fn();
  render(<SayPage onJobStarted={onJobStarted} />);
  await waitFor(() => {
    expect(voices).toHaveBeenCalled();
    expect(settings).toHaveBeenCalled();
  });
  await act(async () => {
    await Promise.resolve();
  });
  return onJobStarted;
}

function typeText(value = "Bonjour.") {
  fireEvent.change(screen.getByLabelText(/^Text/), { target: { value } });
}

/** The form refuses to submit without a voice, so pick the saved one. */
function chooseVoice(label = "warm-fr") {
  fireEvent.change(screen.getByLabelText(/Saved voice/), { target: { value: label } });
}

function chooseSpeaker() {
  fireEvent.change(screen.getByLabelText(/Preset speaker/), {
    target: { value: "Vivian" },
  });
}

function spec() {
  return submitSay.mock.calls[0][0] as Record<string, unknown>;
}

beforeEach(() => {
  voices.mockReset();
  settings.mockReset();
  backends.mockReset();
  submitSay.mockReset();
  upload.path = "/tmp/clone.wav";
  upload.name = "clone.wav";
  backends.mockResolvedValue(makeBackends());
  voices.mockResolvedValue({
    voices: [makeVoice({ label: "warm-fr", language: "French" })],
  });
  settings.mockResolvedValue(makeSettings());
  submitSay.mockResolvedValue({ job_id: "job-say", state: "QUEUED" });
});

describe("SayPage basics", () => {
  it("requires text before submitting", async () => {
    await renderPage();
    chooseVoice();
    const button = screen.getByRole("button", { name: "Generate speech" });
    expect(button).toBeDisabled();
    typeText("   ");
    expect(button).toBeDisabled();
    typeText("Bonjour.");
    expect(button).not.toBeDisabled();
  });

  it("blocks a submission with no voice chosen", async () => {
    await renderPage();
    typeText("Bonjour.");
    const button = screen.getByRole("button", { name: "Generate speech" });

    expect(button).toBeDisabled();
    expect(
      screen.getByText("Choose a saved voice, or switch the voice source."),
    ).toBeInTheDocument();

    chooseVoice();
    expect(button).not.toBeDisabled();
  });

  it("sends a saved voice spec", async () => {
    const onJobStarted = await renderPage();
    typeText("Bonjour.");
    fireEvent.change(screen.getByLabelText(/^Backend/), { target: { value: "qwen" } });
    fireEvent.change(screen.getByLabelText(/^Language/), { target: { value: "German" } });
    fireEvent.change(screen.getByLabelText(/Saved voice/), {
      target: { value: "warm-fr" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate speech" }));

    await waitFor(() => expect(submitSay).toHaveBeenCalled());
    expect(spec()).toEqual(
      expect.objectContaining({
        text: "Bonjour.",
        backend: "qwen",
        language: "German",
        voice: "warm-fr",
        device: "auto",
        dtype: "auto",
        force: false,
        provenance: true,
        prosody_plan: "off",
        style_plan: "off",
        style_plan_scope: "per-line",
        fish_tag_mode: "keep",
      }),
    );
    expect(onJobStarted).toHaveBeenCalledWith("job-say");
  });

  it("sends a design spec", async () => {
    await renderPage();
    typeText("Bonsoir.");
    fireEvent.change(screen.getByLabelText(/Voice source/), {
      target: { value: "design" },
    });
    fireEvent.change(screen.getByLabelText(/Voice description/), {
      target: { value: "a warm French voice" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate speech" }));

    await waitFor(() => expect(submitSay).toHaveBeenCalled());
    expect(spec()).toEqual(
      expect.objectContaining({ design: "a warm French voice" }),
    );
    expect(spec()).not.toHaveProperty("voice");
    expect(spec()).not.toHaveProperty("clone");
  });

  it("sends a clone spec and shows the clip name", async () => {
    await renderPage();
    typeText("Bonsoir.");
    fireEvent.change(screen.getByLabelText(/Voice source/), {
      target: { value: "clone" },
    });
    fireEvent.click(screen.getByRole("button", { name: "upload:Reference clip" }));
    fireEvent.change(screen.getByLabelText(/Clip transcript/), {
      target: { value: "Bonjour tout le monde" },
    });
    expect(screen.getByText("Clip: clone.wav")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Generate speech" }));

    await waitFor(() => expect(submitSay).toHaveBeenCalled());
    expect(spec()).toEqual(
      expect.objectContaining({
        clone: "/tmp/clone.wav",
        clone_text: "Bonjour tout le monde",
      }),
    );
  });

  it("switches to a preset speaker for CustomVoice", async () => {
    await renderPage();
    typeText("Bonjour.");
    fireEvent.change(screen.getByLabelText(/^Backend/), {
      target: { value: "customvoice" },
    });
    expect(screen.queryByLabelText(/Voice source/)).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText(/Preset speaker/), {
      target: { value: "Vivian" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate speech" }));

    await waitFor(() => expect(submitSay).toHaveBeenCalled());
    expect(spec()).toEqual(
      expect.objectContaining({ backend: "customvoice", speaker: "Vivian" }),
    );
    expect(spec().voice).toBeUndefined();
  });

  it("returns to saved voices when leaving CustomVoice", async () => {
    await renderPage();
    fireEvent.change(screen.getByLabelText(/^Backend/), {
      target: { value: "customvoice" },
    });
    fireEvent.change(screen.getByLabelText(/^Backend/), { target: { value: "qwen" } });
    expect(await screen.findByLabelText(/Voice source/)).toHaveValue("voice");
  });
});

describe("SayPage prosody and backend tuning", () => {
  it("parses prosody, emotions and planning values", async () => {
    await renderPage();
    typeText("Bonjour.\nBonsoir.");
    // Qwen Base has no prosody channel, so use a backend that has one.
    fireEvent.change(screen.getByLabelText(/^Backend/), {
      target: { value: "chatterbox" },
    });
    chooseVoice();
    fireEvent.change(screen.getByLabelText(/Emotion shorthand/), {
      target: { value: "calm, warm" },
    });
    fireEvent.change(screen.getByLabelText(/Per-line prosody/), {
      target: { value: "emotion=calm\n\nemotion=excited" },
    });
    const advanced = openAdvanced();
    fireEvent.change(within(advanced).getByLabelText(/Planning chunk size/), {
      target: { value: "6" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate speech" }));

    await waitFor(() => expect(submitSay).toHaveBeenCalled());
    expect(spec()).toEqual(
      expect.objectContaining({
        backend: "chatterbox",
        emotions: ["calm", "warm"],
        prosody: ["emotion=calm", "emotion=excited"],
        plan_chunk: 6,
      }),
    );
  });

  it("sends CustomVoice style instructions", async () => {
    await renderPage();
    typeText("Bonjour.");
    fireEvent.change(screen.getByLabelText(/^Backend/), {
      target: { value: "customvoice" },
    });
    chooseSpeaker();
    const advanced = openAdvanced();
    fireEvent.change(within(advanced).getByLabelText(/Verbatim style instructions/), {
      target: { value: "Speak softly.\nThen fade out." },
    });
    fireEvent.change(within(advanced).getByLabelText(/^Style plan$/), {
      target: { value: "llm" },
    });
    fireEvent.change(within(advanced).getByLabelText(/Style plan scope/), {
      target: { value: "whole" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate speech" }));

    await waitFor(() => expect(submitSay).toHaveBeenCalled());
    expect(spec()).toEqual(
      expect.objectContaining({
        style_instructs: ["Speak softly.", "Then fade out."],
        style_plan: "llm",
        style_plan_scope: "whole",
      }),
    );
  });

  it("sends Chatterbox sampling fields", async () => {
    await renderPage();
    typeText("Bonjour.");
    fireEvent.change(screen.getByLabelText(/^Backend/), {
      target: { value: "chatterbox" },
    });
    chooseVoice();
    const advanced = openAdvanced();
    fireEvent.change(within(advanced).getByLabelText(/Exaggeration/), {
      target: { value: "0.9" },
    });
    fireEvent.change(within(advanced).getByLabelText(/CFG weight/), {
      target: { value: "0.5" },
    });
    fireEvent.change(within(advanced).getByLabelText(/Temperature/), {
      target: { value: "0.6" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate speech" }));

    await waitFor(() => expect(submitSay).toHaveBeenCalled());
    expect(spec()).toEqual(
      expect.objectContaining({
        chatterbox_exaggeration: 0.9,
        chatterbox_cfg_weight: 0.5,
        chatterbox_temperature: 0.6,
      }),
    );
    expect(spec().fish_quantize).toBeUndefined();
  });

  it("sends Fish sampling, quantization and tag mode", async () => {
    await renderPage();
    typeText("Bonjour.");
    fireEvent.change(screen.getByLabelText(/^Backend/), {
      target: { value: "fish" },
    });
    chooseVoice();
    const advanced = openAdvanced();
    fireEvent.change(within(advanced).getByLabelText(/Temperature/), {
      target: { value: "0.7" },
    });
    fireEvent.change(within(advanced).getByLabelText(/Top-p/), {
      target: { value: "0.8" },
    });
    fireEvent.change(within(advanced).getByLabelText(/Quantization/), {
      target: { value: "off" },
    });
    fireEvent.change(within(advanced).getByLabelText(/Fish tag mode/), {
      target: { value: "llm" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate speech" }));

    await waitFor(() => expect(submitSay).toHaveBeenCalled());
    expect(spec()).toEqual(
      expect.objectContaining({
        fish_temperature: 0.7,
        fish_top_p: 0.8,
        fish_quantize: "off",
        fish_tag_mode: "llm",
      }),
    );
  });

  it("applies settings defaults, hardware choices and toggles", async () => {
    settings.mockResolvedValue(
      makeSettings({ default_device: "cuda:1", default_dtype: "fp16" }),
    );
    await renderPage();
    typeText("Bonjour.");
    chooseVoice();
    const advanced = openAdvanced();
    expect(within(advanced).getByLabelText(/^Device/)).toHaveValue("cuda:1");
    expect(within(advanced).getByLabelText(/^Precision/)).toHaveValue("fp16");
    fireEvent.change(within(advanced).getByLabelText(/^Device/), {
      target: { value: "cpu" },
    });
    fireEvent.change(within(advanced).getByLabelText(/^Precision/), {
      target: { value: "fp32" },
    });
    fireEvent.click(within(advanced).getByRole("switch", { name: "Ignore caches" }));
    fireEvent.click(within(advanced).getByRole("switch", { name: "Write provenance" }));
    fireEvent.change(screen.getByLabelText(/Save as voice/), {
      target: { value: "kept-voice" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate speech" }));

    await waitFor(() => expect(submitSay).toHaveBeenCalled());
    expect(spec()).toEqual(
      expect.objectContaining({
        device: "cpu",
        dtype: "fp32",
        force: true,
        provenance: false,
        save_as: "kept-voice",
      }),
    );
  });
});

describe("SayPage capabilities", () => {
  it("disables the prosody controls on Qwen Base and says why", async () => {
    await renderPage();

    expect(screen.getByLabelText(/Prosody plan/)).toBeDisabled();
    expect(screen.getByLabelText(/Emotion shorthand/)).toBeDisabled();
    expect(screen.getByLabelText(/Per-line prosody/)).toBeDisabled();
    expect(
      screen.getAllByText(/--prosody is unavailable with the Qwen Base backend/)
        .length,
    ).toBeGreaterThan(0);
  });

  it("drops prosody values when the backend has no such channel", async () => {
    await renderPage();
    typeText("Bonjour.");
    fireEvent.change(screen.getByLabelText(/^Backend/), {
      target: { value: "chatterbox" },
    });
    chooseVoice();
    fireEvent.change(screen.getByLabelText(/Emotion shorthand/), {
      target: { value: "calm" },
    });
    fireEvent.change(screen.getByLabelText(/Per-line prosody/), {
      target: { value: "emotion=calm" },
    });
    fireEvent.change(screen.getByLabelText(/^Backend/), { target: { value: "qwen" } });

    expect(screen.getByLabelText(/Emotion shorthand/)).toHaveValue("");
    fireEvent.click(screen.getByRole("button", { name: "Generate speech" }));

    await waitFor(() => expect(submitSay).toHaveBeenCalled());
    expect(spec().prosody).toBeUndefined();
    expect(spec().emotions).toBeUndefined();
    expect(spec().prosody_plan).toBe("off");
  });

  it("disables Save as voice where the backend cannot save one", async () => {
    await renderPage();
    expect(screen.getByLabelText(/Save as voice/)).not.toBeDisabled();

    fireEvent.change(screen.getByLabelText(/^Backend/), { target: { value: "fish" } });
    expect(screen.getByLabelText(/Save as voice/)).toBeDisabled();
    expect(
      screen.getByText("Fish cannot save a voice from a one-off run."),
    ).toBeInTheDocument();
  });
});

describe("SayPage failures", () => {
  it("shows a submit failure and re-enables the button", async () => {
    submitSay.mockRejectedValue(new Error("no GPU memory"));
    await renderPage();
    typeText("Bonjour.");
    chooseVoice();
    fireEvent.click(screen.getByRole("button", { name: "Generate speech" }));

    expect(await screen.findByText("no GPU memory")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate speech" })).not.toBeDisabled();
  });

  it("disables the button while the request is in flight", async () => {
    let resolve!: (value: unknown) => void;
    submitSay.mockReturnValue(new Promise((done) => (resolve = done)));
    await renderPage();
    typeText("Bonjour.");
    chooseVoice();
    fireEvent.click(screen.getByRole("button", { name: "Generate speech" }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Generate speech" })).toBeDisabled(),
    );
    resolve({ job_id: "job-say", state: "QUEUED" });
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Generate speech" })).not.toBeDisabled(),
    );
  });
});

describe("SayPage editing", () => {
  it("prefills a stored speech spec", async () => {
    const job = makeJob({
      kind: "say",
      params: {
        spec: {
          text: "Bonsoir.",
          backend: "fish",
          language: "German",
          voice: "warm-fr",
          prosody_plan: "off",
        },
      },
    });
    render(<SayPage onJobStarted={vi.fn()} initialJob={job} />);

    await waitFor(() => expect(screen.getByLabelText(/^Text/)).toHaveValue("Bonsoir."));
    expect(screen.getByLabelText(/^Backend/)).toHaveValue("fish");
    expect(screen.getByLabelText(/^Language/)).toHaveValue("German");
    await waitFor(() =>
      expect(screen.getByLabelText(/Saved voice/)).toHaveValue("warm-fr"),
    );
  });
});
