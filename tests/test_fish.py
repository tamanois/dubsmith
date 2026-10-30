"""Tests for the Fish Speech subprocess bridge.

No model or GPU: the engine's worker is faked for the request/read path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline.errors import DependencyError, TTSError
from pipeline.fish import FishEngine, resolve_fish_device
from pipeline.prosody import ProsodySpec


def _fake_worker(rate: int = 44_100, seconds: float = 0.25):
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
    assert resolve_fish_device("cpu") == "cpu"
    assert resolve_fish_device("cuda:0") == "cuda:0"
    assert resolve_fish_device("mps") == "cpu"


def test_synthesize_tags_lines_and_reads_wavs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured, run = _fake_worker()
    monkeypatch.setattr(FishEngine, "_ensure_available", lambda self: None)
    monkeypatch.setattr(FishEngine, "_run_worker", run)

    engine = FishEngine("models/fish", device="cpu")
    out = engine.synthesize(
        ["First one.", "Second one"],
        reference=tmp_path / "voice.wav",
        prosody=[
            ProsodySpec(emotion="excited"),
            ProsodySpec(emotion="whisper", pause=True),
        ],
        ref_text="reference words",
    )

    assert len(out) == 2
    assert all(wav.dtype == np.float32 and wav.ndim == 1 for wav in out)
    assert engine.sample_rate == 44_100
    payload = captured[0]
    assert payload["lines"] == [
        "[excited] First one.",
        "[whisper] Second one [pause]",
    ]
    assert payload["reference_text"] == "reference words"
    assert payload["device"] == "cpu"


def test_sampling_is_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured, run = _fake_worker()
    monkeypatch.setattr(FishEngine, "_ensure_available", lambda self: None)
    monkeypatch.setattr(FishEngine, "_run_worker", run)

    engine = FishEngine("m", device="cpu", temperature=0.5, top_p=0.8)
    engine.synthesize(["hi"], reference=tmp_path / "v.wav")

    payload = captured[0]
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.8


def test_keep_mode_sends_the_lines_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured, run = _fake_worker()
    monkeypatch.setattr(FishEngine, "_ensure_available", lambda self: None)
    monkeypatch.setattr(FishEngine, "_run_worker", run)

    engine = FishEngine("m", device="cpu")
    engine.synthesize(
        ["[whisper] Wait.", "[sad] Then go."],
        reference=tmp_path / "v.wav",
        tag_mode="keep",
    )

    assert captured[0]["lines"] == ["[whisper] Wait.", "[sad] Then go."]


def test_missing_package_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FISH_PYTHON", raising=False)
    monkeypatch.setattr(FishEngine, "_project_root", lambda self: tmp_path)
    monkeypatch.setattr("pipeline.fish._module_available", lambda name: False)

    with pytest.raises(DependencyError, match="fish-speech"):
        FishEngine("m", device="cpu").synthesize(
            ["a"], reference=tmp_path / "x.wav"
        )


def test_worker_failure_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Failed:
        returncode = 1
        stdout = ""
        stderr = "boom: model exploded"

    monkeypatch.setattr("pipeline.fish.subprocess.run", lambda *a, **k: Failed())
    monkeypatch.setattr(FishEngine, "_ensure_available", lambda self: None)

    with pytest.raises(TTSError, match="boom"):
        FishEngine("m", device="cpu").synthesize(
            ["a"], reference=tmp_path / "x.wav"
        )
