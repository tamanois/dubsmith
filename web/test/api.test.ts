import { afterEach, describe, expect, it, vi } from "vitest";
import { api, subscribeJob } from "../src/api";
import { FakeEventSource } from "./fakeEventSource";

function jsonResponse(payload: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  FakeEventSource.reset();
});

describe("api request handling", () => {
  it("parses a successful JSON response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ models: [] })));
    await expect(api.models()).resolves.toEqual({ models: [] });
  });

  it("uses the server detail when a request fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "bad input" }, 400)),
    );
    await expect(api.models()).rejects.toThrow("bad input");
  });

  it("falls back to a status message when the error body is not JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 502,
        json: async () => {
          throw new Error("not json");
        },
      }),
    );
    await expect(api.models()).rejects.toThrow("Request failed (502)");
  });

  it("encodes download URLs", () => {
    expect(api.outputUrl("/output/a b/c d.srt")).toBe(
      "/api/outputs/download?path=%2Foutput%2Fa%20b%2Fc%20d.srt",
    );
    expect(api.fileUrl("file-1")).toBe("/api/files/file-1/download");
  });
});

describe("api upload", () => {
  it("returns the uploaded file payload", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ id: "1", name: "clip.wav", path: "/tmp/clip.wav", size: 4 }),
      ),
    );
    const result = await api.upload(new File(["x"], "clip.wav", { type: "audio/wav" }));
    expect(result).toEqual({ id: "1", name: "clip.wav", path: "/tmp/clip.wav", size: 4 });
  });

  it("surfaces an upload detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "too large" }, 413)),
    );
    await expect(api.upload(new File(["x"], "clip.wav"))).rejects.toThrow("too large");
  });

  it("uses a generic message when the upload body has no detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => {
          throw new Error("not json");
        },
      }),
    );
    await expect(api.upload(new File(["x"], "clip.wav"))).rejects.toThrow("Upload failed");
  });
});

describe("subscribeJob", () => {
  it("dispatches non-terminal events and closes on the final job", () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    const onEvent = vi.fn();
    const onJob = vi.fn();
    const onError = vi.fn();

    const unsubscribe = subscribeJob("job-1", onEvent, onJob, onError);
    const source = FakeEventSource.instances[0];
    source.emit("log", { id: 1, kind: "log", message: "hello" });
    source.emit("job", { id: "job-1", state: "COMPLETED" });

    expect(onEvent).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "log", message: "hello" }),
    );
    expect(onJob).toHaveBeenCalledWith(
      expect.objectContaining({ id: "job-1", state: "COMPLETED" }),
    );
    expect(source.close).toHaveBeenCalled();
    unsubscribe();
  });

  it("ignores malformed SSE frames", () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    const onEvent = vi.fn();
    subscribeJob("job-1", onEvent, vi.fn());
    const source = FakeEventSource.instances[0];
    source.emit("log", "{not json");
    expect(onEvent).not.toHaveBeenCalled();
  });

  it("reports a stream error and closes", () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    const onError = vi.fn();
    subscribeJob("job-1", vi.fn(), vi.fn(), onError);
    const source = FakeEventSource.instances[0];
    source.fail();
    expect(onError).toHaveBeenCalledWith(
      "Lost the live connection; the job may still be running.",
    );
    expect(source.close).toHaveBeenCalled();
  });

  it("closes the stream when unsubscribed", () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    const unsubscribe = subscribeJob("job-1", vi.fn(), vi.fn());
    const source = FakeEventSource.instances[0];
    unsubscribe();
    expect(source.close).toHaveBeenCalled();
  });
});
