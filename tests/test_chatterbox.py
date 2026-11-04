"""Tests for the Chatterbox subprocess bridge.

No model or GPU: the engine's worker is faked for the request/read path, and the
worker itself is run against a tiny fake ``chatterbox`` package on ``PYTHONPATH``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline.chatterbox import ChatterboxEngine, resolve_chatterbox_device
from pipeline.errors import DependencyError, TTSError
from pipeline.prosody import ProsodySpec

ROOT = Path(__file__).resolve().parents[1]


def test_device_maps_to_cuda_or_cpu() -> None:
    assert resolve_chatterbox_device("cuda:1") == "cuda"
    assert resolve_chatterbox_device("cpu") == "cpu"


def test_synthesize_builds_the_request_and_reads_wavs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict] = []

    def fake_run(self, request_path: Path, out_dir: Path) -> dict:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        captured.append(payload)
        files = []
        for index in range(len(payload["lines"])):
            path = out_dir / f"line_{index:03d}.wav"
            sf.write(str(path), np.zeros(2400, dtype=np.float32), 24000)
            files.append(str(path))
        return {"sample_rate": 24000, "files": files}

    monkeypatch.setattr(ChatterboxEngine, "_ensure_available", lambda self: None)
    monkeypatch.setattr(ChatterboxEngine, "_run_worker", fake_run)

    engine = ChatterboxEngine("models/chatterbox", device="cpu")
    trace: list[dict] = []
    out = engine.synthesize(
        ["bonjour", "au revoir"],
        reference=tmp_path / "voice.wav",
        prosody=[ProsodySpec(emotion="happy"), ProsodySpec(emotion="sad")],
        language="fr",
        trace=trace,
    )

    assert len(out) == 2
    assert all(wav.dtype == np.float32 and wav.ndim == 1 for wav in out)
    assert engine.sample_rate == 24000
    payload = captured[0]
    assert payload["lines"] == ["bonjour", "au revoir"]
    assert payload["language"] == "fr"
    assert payload["exaggerations"][0] > payload["exaggerations"][1]
    assert payload["reference"].endswith("voice.wav")
    assert [item["channel"] for item in trace] == ["chatterbox", "chatterbox"]
    assert trace[0]["exaggeration"] == payload["exaggerations"][0]


def test_engine_overrides_exaggeration_and_passes_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict] = []

    def fake_run(self, request_path: Path, out_dir: Path) -> dict:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        captured.append(payload)
        files = []
        for index in range(len(payload["lines"])):
            path = out_dir / f"line_{index:03d}.wav"
            sf.write(str(path), np.zeros(2400, dtype=np.float32), 24000)
            files.append(str(path))
        return {"sample_rate": 24000, "files": files}

    monkeypatch.setattr(ChatterboxEngine, "_ensure_available", lambda self: None)
    monkeypatch.setattr(ChatterboxEngine, "_run_worker", fake_run)

    engine = ChatterboxEngine(
        "m", device="cpu", exaggeration=0.2, cfg_weight=0.6, temperature=0.7
    )
    engine.synthesize(
        ["bonjour", "salut"],
        reference=tmp_path / "voice.wav",
        prosody=[ProsodySpec(), ProsodySpec(emotion="happy")],
        language="fr",
    )

    payload = captured[0]
    assert payload["exaggerations"] == [0.2, 0.2]
    assert payload["cfg_weight"] == 0.6
    assert payload["temperature"] == 0.7


def test_missing_package_is_reported_by_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Isolate from a real .venv-chatterbox that may exist in the checkout.
    monkeypatch.delenv("CHATTERBOX_PYTHON", raising=False)
    monkeypatch.setattr(ChatterboxEngine, "_project_root", lambda self: tmp_path)
    monkeypatch.setattr("pipeline.chatterbox._module_available", lambda name: False)
    engine = ChatterboxEngine("m", device="cpu")

    with pytest.raises(DependencyError, match="chatterbox-tts|chatterbox"):
        engine.synthesize(["a"], reference=tmp_path / "x.wav")


def test_worker_failure_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Failed:
        returncode = 1
        stdout = ""
        stderr = "boom: model exploded"

    monkeypatch.setattr("pipeline.chatterbox.subprocess.run", lambda *a, **k: Failed())
    engine = ChatterboxEngine("m", device="cpu")

    with pytest.raises(TTSError, match="model exploded"):
        engine._run_worker(tmp_path / "request.json", tmp_path)


def _fake_package(tmp_path: Path) -> Path:
    package = tmp_path / "fake"
    (package / "chatterbox").mkdir(parents=True)
    (package / "chatterbox" / "__init__.py").write_text("", encoding="utf-8")
    (package / "chatterbox" / "mtl_tts.py").write_text(
        """
import numpy as np

class ChatterboxMultilingualTTS:
    sr = 24000
    def __init__(self, device="cpu"):
        self.device = device
    @classmethod
    def from_pretrained(cls, model_dir=None, device="cpu"):
        if model_dir is not None:
            raise AssertionError("local weights must go through from_local")
        return cls(device=device)
    @classmethod
    def from_local(cls, ckpt_dir, device):
        return cls(device=device)
    def generate(self, text, audio_prompt_path=None, exaggeration=0.5, language_id=None):
        assert audio_prompt_path is not None
        assert language_id in (None, "fr", "en")
        return np.zeros(1200, dtype=np.float32)
""",
        encoding="utf-8",
    )
    return package


def _run_worker(package: Path, tmp_path: Path, request: dict) -> dict:
    out_dir = tmp_path / "out"
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(package), str(ROOT)])
    completed = subprocess.run(
        [sys.executable, "-m", "pipeline.chatterbox_worker", str(request_path)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    return json.loads((out_dir / "result.json").read_text(encoding="utf-8"))


def test_worker_end_to_end_with_a_fake_package(tmp_path: Path) -> None:
    payload = _run_worker(
        _fake_package(tmp_path),
        tmp_path,
        {
            "model_dir": None,
            "reference": str(tmp_path / "ref.wav"),
            "lines": ["bonjour"],
            "exaggerations": [0.8],
            "language": "fr",
            "device": "cpu",
            "output_dir": str(tmp_path / "out"),
        },
    )

    assert payload["sample_rate"] == 24000
    assert len(payload["files"]) == 1
    assert Path(payload["files"][0]).is_file()


def test_worker_loads_the_managed_checkpoint_from_disk(tmp_path: Path) -> None:
    """`models --download tts_chatterbox` output must be used via from_local."""
    checkpoint = tmp_path / "chatterbox"
    checkpoint.mkdir()
    for name in (
        "ve.pt",
        "t3_mtl23ls_v2.safetensors",
        "s3gen.pt",
        "grapheme_mtl_merged_expanded_v1.json",
        "Cangjie5_TC.json",
    ):
        (checkpoint / name).write_bytes(b"")

    payload = _run_worker(
        _fake_package(tmp_path),
        tmp_path,
        {
            "model_dir": str(checkpoint),
            "reference": str(tmp_path / "ref.wav"),
            "lines": ["bonjour"],
            "exaggerations": [0.8],
            "language": "fr",
            "device": "cpu",
            "output_dir": str(tmp_path / "out"),
        },
    )

    assert len(payload["files"]) == 1
