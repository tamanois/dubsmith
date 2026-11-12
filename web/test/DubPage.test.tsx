import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DubPage } from "../src/pages/DubPage";
import { makeDevice, makeDeviceReport, makeSettings, makeVoice } from "./factories";

const voices = vi.fn();
const devices = vi.fn();
const settings = vi.fn();
const submitDub = vi.fn();
const submitSubtitles = vi.fn();
const upload = vi.hoisted(() => ({ path: "/tmp/clip.mp4", name: "clip.mp4" }));

vi.mock("../src/api", () => ({
  api: {
    voices: (...args: unknown[]) => voices(...args),
    devices: (...args: unknown[]) => devices(...args),
    settings: (...args: unknown[]) => settings(...args),
    submitDub: (...args: unknown[]) => submitDub(...args),
    submitSubtitles: (...args: unknown[]) => submitSubtitles(...args),
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

function openDisclosure(name: RegExp, root: HTMLElement = document.body): HTMLElement {
  const button = within(root).getByRole("button", { name });
  fireEvent.click(button);
  return button.parentElement as HTMLElement;
}

async function renderPage() {
  const onJobStarted = vi.fn();
  render(<DubPage onJobStarted={onJobStarted} />);
  await waitFor(() => {
    expect(voices).toHaveBeenCalled();
    expect(devices).toHaveBeenCalled();
    expect(settings).toHaveBeenCalled();
  });
  await act(async () => {
    await Promise.resolve();
  });
  return onJobStarted;
}

function chooseInput() {
  fireEvent.click(screen.getByRole("button", { name: "upload:Input file" }));
}

function dubConfig() {
  return submitDub.mock.calls[0][0] as { config: Record<string, unknown> };
}

beforeEach(() => {
  voices.mockReset();
  devices.mockReset();
  settings.mockReset();
  submitDub.mockReset();
  submitSubtitles.mockReset();
  upload.path = "/tmp/clip.mp4";
  upload.name = "clip.mp4";
  voices.mockResolvedValue({
    voices: [
      makeVoice({ label: "warm-fr", language: "French" }),
      makeVoice({ label: "warm-en", language: "English" }),
    ],
  });
  devices.mockResolvedValue(
    makeDeviceReport({
      devices: [
        makeDevice({ label: "cuda:0", name: "RTX 3060" }),
        makeDevice({ label: "cuda:1", name: "RTX 3060" }),
      ],
    }),
  );
  settings.mockResolvedValue(makeSettings());
  submitDub.mockResolvedValue({ job_id: "job-dub", state: "QUEUED" });
  submitSubtitles.mockResolvedValue({ job_id: "job-subs", state: "QUEUED" });
});

describe("DubPage subtitles mode", () => {
  async function openSubtitles() {
    fireEvent.click(screen.getByRole("button", { name: "Subtitles only" }));
    await screen.findByRole("button", { name: "Start subtitles" });
  }

  it("starts disabled until an input is chosen", async () => {
    await renderPage();
    expect(screen.getByRole("button", { name: "Start dubbing" })).toBeDisabled();
    await openSubtitles();
    expect(screen.getByRole("button", { name: "Start subtitles" })).toBeDisabled();
  });

  it("sends a video input with subtitle options", async () => {
    const onJobStarted = await renderPage();
    await openSubtitles();
    chooseInput();
    fireEvent.change(screen.getByLabelText(/Target language/), {
      target: { value: "German" },
    });
    fireEvent.change(screen.getByLabelText(/Source language/), {
      target: { value: "English" },
    });
    fireEvent.change(screen.getByLabelText(/^Device/), { target: { value: "cuda:0" } });
    fireEvent.change(screen.getByLabelText(/^Precision/), { target: { value: "fp16" } });
    fireEvent.change(screen.getByLabelText(/Max characters per cue/), {
      target: { value: "30" },
    });
    fireEvent.change(screen.getByLabelText(/Subtitle mode/), {
      target: { value: "burn" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start subtitles" }));

    await waitFor(() => expect(submitSubtitles).toHaveBeenCalled());
    expect(submitSubtitles.mock.calls[0][0]).toEqual({
      target_language: "German",
      source_language: "English",
      device: "cuda:0",
      whisper_model: "large-v3",
      llm_model: undefined,
      subtitles_mode: "burn",
      max_chars: 30,
      no_speech_threshold: 0.6,
      force: false,
      video: "/tmp/clip.mp4",
    });
    expect(onJobStarted).toHaveBeenCalledWith("job-subs");
  });

  it("sends an audio input in subtitles mode", async () => {
    await renderPage();
    await openSubtitles();
    upload.path = "/tmp/voice.mp3";
    upload.name = "voice.mp3";
    chooseInput();
    fireEvent.click(screen.getByRole("button", { name: "Start subtitles" }));

    await waitFor(() => expect(submitSubtitles).toHaveBeenCalled());
    expect(submitSubtitles.mock.calls[0][0]).toEqual(
      expect.objectContaining({ audio: "/tmp/voice.mp3" }),
    );
    expect(submitSubtitles.mock.calls[0][0]).not.toHaveProperty("video");
  });
});

describe("DubPage dubbing mode", () => {
  it("sends a video input and reports the job", async () => {
    const onJobStarted = await renderPage();
    chooseInput();
    fireEvent.click(screen.getByRole("button", { name: "Start dubbing" }));

    await waitFor(() => expect(submitDub).toHaveBeenCalled());
    const { config } = dubConfig();
    expect(config).toEqual(
      expect.objectContaining({
        video: "/tmp/clip.mp4",
        output_dir: "output",
        tts_backend: "qwen",
        device: "auto",
        dtype: "auto",
        tts_batch: 8,
        keep_background: true,
        keep_subtitles: true,
        subtitles_mode: "soft",
        overlap_mode: "dominant",
        refine: true,
        provenance: true,
      }),
    );
    expect(submitDub.mock.calls[0][1]).toBe("auto");
    expect(submitDub.mock.calls[0][0]).toEqual(
      expect.objectContaining({ voice: undefined, accent_voice: undefined }),
    );
    expect(onJobStarted).toHaveBeenCalledWith("job-dub");
  });

  it("uses a saved voice and accent voice", async () => {
    await renderPage();
    chooseInput();
    const voice = openDisclosure(/Voice and accent/);
    fireEvent.change(within(voice).getByLabelText(/Saved voice/), {
      target: { value: "warm-fr" },
    });
    fireEvent.change(within(voice).getByLabelText(/Saved accent voice/), {
      target: { value: "warm-en" },
    });
    fireEvent.click(within(voice).getByRole("button", { name: "upload:Reference clip" }));
    fireEvent.change(within(voice).getByLabelText(/Reference transcript/), {
      target: { value: "Bonjour" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start dubbing" }));

    await waitFor(() => expect(submitDub).toHaveBeenCalled());
    expect(submitDub.mock.calls[0][0]).toEqual(
      expect.objectContaining({ voice: "warm-fr", accent_voice: "warm-en" }),
    );
    expect(dubConfig().config.ref_audio).toBeUndefined();
    expect(dubConfig().config.ref_text).toBeUndefined();
  });

  it("sends a reference clip when no saved voice is chosen", async () => {
    await renderPage();
    chooseInput();
    const voice = openDisclosure(/Voice and accent/);
    upload.path = "/tmp/reference.wav";
    upload.name = "reference.wav";
    fireEvent.click(within(voice).getByRole("button", { name: "upload:Reference clip" }));
    fireEvent.change(within(voice).getByLabelText(/Reference transcript/), {
      target: { value: "Bonjour tout le monde" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start dubbing" }));

    await waitFor(() => expect(submitDub).toHaveBeenCalled());
    expect(dubConfig().config).toEqual(
      expect.objectContaining({
        ref_audio: "/tmp/reference.wav",
        ref_text: "Bonjour tout le monde",
      }),
    );
  });

  it("sends native accent matching and blend detail", async () => {
    await renderPage();
    chooseInput();
    const voice = openDisclosure(/Voice and accent/);
    fireEvent.change(within(voice).getByLabelText(/Accent match/), {
      target: { value: "rate" },
    });
    const blend = openDisclosure(/Accent blend detail/, voice);
    fireEvent.change(within(blend).getByLabelText(/Blend mode/), {
      target: { value: "two-stage" },
    });
    fireEvent.change(within(blend).getByLabelText(/Blend takes/), {
      target: { value: "3" },
    });
    fireEvent.change(within(blend).getByLabelText(/Blend iterations/), {
      target: { value: "5" },
    });
    fireEvent.change(within(blend).getByLabelText(/Blend ratio/), {
      target: { value: "0.85" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start dubbing" }));

    await waitFor(() => expect(submitDub).toHaveBeenCalled());
    expect(dubConfig().config).toEqual(
      expect.objectContaining({
        accent_mode: "native",
        accent_match: "rate",
        accent_blend: "two-stage",
        blend_takes: 3,
        blend_iters: 5,
        blend_ratio: 0.85,
      }),
    );
  });

  it("sends a reference accent clip only in reference-accent mode", async () => {
    await renderPage();
    chooseInput();
    const voice = openDisclosure(/Voice and accent/);
    fireEvent.change(within(voice).getByLabelText(/Accent mode/), {
      target: { value: "reference-accent" },
    });
    expect(within(voice).getByLabelText(/Accent match/)).toHaveValue("off");
    upload.path = "/tmp/accent.wav";
    upload.name = "accent.wav";
    fireEvent.click(within(voice).getByRole("button", { name: "upload:Accent clip" }));
    fireEvent.change(within(voice).getByLabelText(/Accent transcript/), {
      target: { value: "Guten Tag" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start dubbing" }));

    await waitFor(() => expect(submitDub).toHaveBeenCalled());
    expect(dubConfig().config).toEqual(
      expect.objectContaining({
        accent_mode: "reference-accent",
        accent_match: "off",
        accent_audio: "/tmp/accent.wav",
        accent_text: "Guten Tag",
      }),
    );
  });

  it("applies background, subtitle and loudness conditionals", async () => {
    await renderPage();
    chooseInput();
    const audio = openDisclosure(/Audio and subtitles/);
    fireEvent.click(within(audio).getByRole("switch", { name: "Keep music and ambience" }));
    fireEvent.click(within(audio).getByRole("switch", { name: "Write subtitles" }));
    fireEvent.change(within(audio).getByLabelText(/Separator/), {
      target: { value: "none" },
    });
    fireEvent.change(within(audio).getByLabelText(/Loudness/), {
      target: { value: "lufs" },
    });
    fireEvent.change(within(audio).getByLabelText(/LUFS target/), {
      target: { value: "-20" },
    });
    fireEvent.change(within(audio).getByLabelText(/Duck under dialogue/), {
      target: { value: "9" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start dubbing" }));

    await waitFor(() => expect(submitDub).toHaveBeenCalled());
    expect(dubConfig().config).toEqual(
      expect.objectContaining({
        keep_background: false,
        room: "off",
        keep_subtitles: false,
        subtitles_mode: "off",
        separator: "none",
        loudness_target: "lufs",
        target_lufs: -20,
        duck_db: 9,
      }),
    );
  });

  it("sends room matching detail only when room matching is on", async () => {
    await renderPage();
    chooseInput();
    const audio = openDisclosure(/Audio and subtitles/);
    fireEvent.change(within(audio).getByLabelText(/Room matching/), {
      target: { value: "match" },
    });
    const detail = openDisclosure(/Room matching detail/, audio);
    fireEvent.change(within(detail).getByLabelText(/RT60/), {
      target: { value: "0.4" },
    });
    fireEvent.change(within(detail).getByLabelText(/Wet level/), {
      target: { value: "-6" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start dubbing" }));

    await waitFor(() => expect(submitDub).toHaveBeenCalled());
    expect(dubConfig().config).toEqual(
      expect.objectContaining({
        room: "match",
        room_rt60: 0.4,
        room_wet_db: -6,
        room_duck_db: 3,
        room_tone_gain_db: -30,
      }),
    );
  });

  it("sends backend-specific sampling fields", async () => {
    await renderPage();
    chooseInput();
    const backend = openDisclosure(/Backend and prosody/);
    fireEvent.change(within(backend).getByLabelText(/TTS backend/), {
      target: { value: "chatterbox" },
    });
    fireEvent.change(within(backend).getByLabelText(/Exaggeration/), {
      target: { value: "0.8" },
    });
    fireEvent.change(within(backend).getByLabelText(/CFG weight/), {
      target: { value: "0.5" },
    });
    fireEvent.change(within(backend).getByLabelText(/Temperature/), {
      target: { value: "0.7" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start dubbing" }));

    await waitFor(() => expect(submitDub).toHaveBeenCalled());
    expect(dubConfig().config).toEqual(
      expect.objectContaining({
        tts_backend: "chatterbox",
        chatterbox_exaggeration: 0.8,
        chatterbox_cfg_weight: 0.5,
        chatterbox_temperature: 0.7,
      }),
    );
    expect(dubConfig().config.fish_quantize).toBeUndefined();
  });

  it("sends fish quantization only for the fish backend", async () => {
    await renderPage();
    chooseInput();
    const backend = openDisclosure(/Backend and prosody/);
    fireEvent.change(within(backend).getByLabelText(/TTS backend/), {
      target: { value: "fish" },
    });
    fireEvent.change(within(backend).getByLabelText(/Quantization/), {
      target: { value: "off" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start dubbing" }));

    await waitFor(() => expect(submitDub).toHaveBeenCalled());
    expect(dubConfig().config.fish_quantize).toBe("off");
    expect(dubConfig().config.expressive_accent).toBe("donor");
  });

  it("parses fan-out devices and applies settings defaults", async () => {
    settings.mockResolvedValue(
      makeSettings({ default_device: "cuda:1", default_dtype: "fp16" }),
    );
    await renderPage();
    chooseInput();
    const models = openDisclosure(/Models and hardware/);
    expect(within(models).getByLabelText(/^Device/)).toHaveValue("cuda:1");
    expect(within(models).getByLabelText(/^Precision/)).toHaveValue("fp16");
    fireEvent.change(within(models).getByLabelText(/Fan-out devices/), {
      target: { value: "cuda:0, cuda:1,," },
    });
    fireEvent.change(within(models).getByLabelText(/TTS batch/), {
      target: { value: "4" },
    });
    fireEvent.change(within(models).getByLabelText(/LLM model override/), {
      target: { value: "qwen2.5" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start dubbing" }));

    await waitFor(() => expect(submitDub).toHaveBeenCalled());
    expect(dubConfig().config).toEqual(
      expect.objectContaining({
        device: "cuda:1",
        dtype: "fp16",
        devices: ["cuda:0", "cuda:1"],
        tts_batch: 4,
        llm_model: "qwen2.5",
      }),
    );
  });

  it("sends timing and cache toggles", async () => {
    await renderPage();
    chooseInput();
    const timing = openDisclosure(/Timing, speakers and repair/);
    fireEvent.click(within(timing).getByRole("switch", { name: "Ignore caches" }));
    fireEvent.click(within(timing).getByRole("switch", { name: "Write provenance" }));
    fireEvent.click(within(timing).getByRole("switch", { name: "Repair overlong lines" }));
    fireEvent.click(screen.getByRole("button", { name: "Start dubbing" }));

    await waitFor(() => expect(submitDub).toHaveBeenCalled());
    expect(dubConfig().config).toEqual(
      expect.objectContaining({ force: true, provenance: false, refine: false }),
    );
  });

  it("shows a submit failure and re-enables the button", async () => {
    submitDub.mockRejectedValue(new Error("GPU busy"));
    await renderPage();
    chooseInput();
    fireEvent.click(screen.getByRole("button", { name: "Start dubbing" }));

    expect(await screen.findByText("GPU busy")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start dubbing" })).not.toBeDisabled();
  });

  it("disables submission while the request is in flight", async () => {
    let resolve!: (value: unknown) => void;
    submitDub.mockReturnValue(new Promise((done) => (resolve = done)));
    await renderPage();
    chooseInput();
    fireEvent.click(screen.getByRole("button", { name: "Start dubbing" }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Start dubbing" })).toBeDisabled(),
    );
    resolve({ job_id: "job-dub", state: "QUEUED" });
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Start dubbing" })).not.toBeDisabled(),
    );
  });
});
