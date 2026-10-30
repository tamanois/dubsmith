"""Tests for the IndexTTS2 subprocess bridge.

No model or GPU: the engine's worker is faked for the request/read path, and the
time-stretcher is faked for the duration test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline.errors import DependencyError, TTSError
from pipeline.index_tts import IndexTTSEngine, resolve_index_tts_device


def _fake_worker(rate: int = 24_000, seconds: float = 1.0):
    captured: list[dict] = []

    def run(self, request_path: Path, out_dir: Path) -> dict:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        captured.append(payload)
        files = []
        for index in range(len(payload["lines"])):
            path = out_dir / f"line_{index:03d}.wav"
            sf.write(
                str(path), np.zeros(int(rate * seconds), dtype=np.float32), rate
            )
            files.append(str(path))
        return {"sample_rate": rate, "files": files}

    return captured, run


def test_device_maps_to_cuda_or_cpu() -> None:
    assert resolve_index_tts_device("cpu") == "cpu"
    assert resolve_index_tts_device("cuda:1") == "cuda:1"
    assert resolve_index_tts_device("mps") == "cpu"


def test_synthesize_passes_the_emotion_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured, run = _fake_worker()
    monkeypatch.setattr(IndexTTSEngine, "_ensure_available", lambda self: None)
    monkeypatch.setattr(IndexTTSEngine, "_run_worker", run)

    emotion = tmp_path / "angry.wav"
    engine = IndexTTSEngine("models/indextts", device="cpu", emotion_audio=emotion,
                            emotion_alpha=0.7)
    out = engine.synthesize(
        ["Why didn't you tell me?"], reference=tmp_path / "voice.wav"
    )

    assert len(out) == 1
    assert engine.sample_rate == 24_000
    payload = captured[0]
    assert payload["lines"] == ["Why didn't you tell me?"]
    assert payload["emotion_audio"].endswith("angry.wav")
    assert payload["emotion_alpha"] == 0.7
    assert payload["use_fp16"] is True


def test_duration_stretches_each_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, run = _fake_worker(rate=24_000, seconds=1.0)
    monkeypatch.setattr(IndexTTSEngine, "_ensure_available", lambda self: None)
    monkeypatch.setattr(IndexTTSEngine, "_run_worker", run)

    ratios: list[float] = []

    class Recorder:
        def stretch(self, samples, sample_rate, ratio):
            ratios.append(round(float(ratio), 4))
            return samples

    monkeypatch.setattr(
        "pipeline.timing.detect_stretcher", lambda *a, **k: Recorder()
    )

    engine = IndexTTSEngine("m", device="cpu")
    engine.synthesize(
        ["one", "two"],
        reference=tmp_path / "v.wav",
        durations=[1.1, None],
    )

    assert ratios == [1.1]


def test_missing_package_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("INDEXTTS_PYTHON", raising=False)
    monkeypatch.setattr(IndexTTSEngine, "_project_root", lambda self: tmp_path)
    monkeypatch.setattr("pipeline.index_tts._module_available", lambda name: False)

    with pytest.raises(DependencyError, match="index-tts"):
        IndexTTSEngine("m", device="cpu").synthesize(
            ["a"], reference=tmp_path / "x.wav"
        )


def test_worker_failure_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Failed:
        returncode = 1
        stdout = ""
        stderr = "boom: model exploded"

    monkeypatch.setattr("pipeline.index_tts.subprocess.run", lambda *a, **k: Failed())
    monkeypatch.setattr(IndexTTSEngine, "_ensure_available", lambda self: None)

    with pytest.raises(TTSError, match="boom"):
        IndexTTSEngine("m", device="cpu").synthesize(
            ["a"], reference=tmp_path / "x.wav"
        )
