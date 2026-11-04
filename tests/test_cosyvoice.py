"""Tests for the CosyVoice 3 subprocess bridge.

No model or GPU: the engine's worker is faked for the request/read path, and the
worker itself is run against a tiny fake ``cosyvoice`` package on ``PYTHONPATH``.
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

from pipeline.cosyvoice import CosyVoiceEngine
from pipeline.cosyvoice_worker import _normalize_instruct, NEUTRAL_INSTRUCT
from pipeline.errors import DependencyError, TTSError
from pipeline.prosody import ProsodySpec

ROOT = Path(__file__).resolve().parents[1]


def test_normalize_instruct_adds_the_delimiter_once() -> None:
    assert _normalize_instruct("calm") == "calm<|endofprompt|>"
    assert _normalize_instruct("calm<|endofprompt|>") == "calm<|endofprompt|>"
    assert _normalize_instruct("") == ""


def test_pythonpath_puts_an_override_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COSYVOICE_PYTHONPATH", str(tmp_path))

    paths = CosyVoiceEngine("m", device="cpu")._pythonpath()

    assert Path(paths[0]).resolve() == tmp_path.resolve()


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

    monkeypatch.setattr(CosyVoiceEngine, "_ensure_available", lambda self: None)
    monkeypatch.setattr(CosyVoiceEngine, "_run_worker", fake_run)

    engine = CosyVoiceEngine("models/Fun-CosyVoice3-0.5B", device="cpu")
    out = engine.synthesize(
        ["bonjour", "au revoir"],
        reference=tmp_path / "voice.wav",
        prosody=[ProsodySpec(emotion="happy"), ProsodySpec(emotion="sad")],
    )

    assert len(out) == 2
    assert all(wav.dtype == np.float32 and wav.ndim == 1 for wav in out)
    assert engine.sample_rate == 24000
    assert captured[0]["lines"] == ["bonjour", "au revoir"]
    assert captured[0]["reference"].endswith("voice.wav")
    assert all(
        instruct.startswith("You are a helpful assistant.")
        for instruct in captured[0]["instructs"]
    )


def test_synthesize_renders_structured_prosody(
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

    monkeypatch.setattr(CosyVoiceEngine, "_ensure_available", lambda self: None)
    monkeypatch.setattr(CosyVoiceEngine, "_run_worker", fake_run)

    engine = CosyVoiceEngine("m", device="cpu")
    trace: list[dict] = []
    engine.synthesize(
        ["Hello there. How are you?"],
        reference=tmp_path / "voice.wav",
        prosody=[ProsodySpec(emotion="happy", pace="fast", breath=True)],
        trace=trace,
    )

    payload = captured[0]
    assert "[breath]" in payload["lines"][0]
    assert payload["instructs"][0].startswith("You are a helpful assistant.")
    assert payload["instructs"][0].endswith("<|endofprompt|>")
    assert "开心" in payload["instructs"][0]
    assert trace[0]["channel"] == "cosyvoice"
    assert trace[0]["mode"] == "instruct2"
    assert trace[0]["instruct"] == payload["instructs"][0]


def _capturing_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[CosyVoiceEngine, list[dict]]:
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

    monkeypatch.setattr(CosyVoiceEngine, "_ensure_available", lambda self: None)
    monkeypatch.setattr(CosyVoiceEngine, "_run_worker", fake_run)
    return CosyVoiceEngine("m", device="cpu"), captured


def test_neutral_prosody_uses_zero_shot_with_the_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, captured = _capturing_engine(tmp_path, monkeypatch)
    engine.synthesize(
        ["Bonjour."],
        reference=tmp_path / "voice.wav",
        prosody=[ProsodySpec()],
        ref_text="Bonjour, je suis Delphine.",
    )

    payload = captured[0]
    assert payload["modes"] == ["zero_shot"]
    assert payload["prompts"][0] == (
        "You are a helpful assistant.<|endofprompt|>Bonjour, je suis Delphine."
    )
    assert payload["lines"] == ["Bonjour."]
    assert payload["text_frontend"] is False


def test_inline_tokens_use_cross_lingual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, captured = _capturing_engine(tmp_path, monkeypatch)
    engine.synthesize(
        ["Bonjour."],
        reference=tmp_path / "voice.wav",
        prosody=[ProsodySpec(breath=True)],
        ref_text="Bonjour.",
    )

    payload = captured[0]
    assert payload["modes"] == ["cross_lingual"]
    assert payload["lines"][0].startswith(
        "You are a helpful assistant.<|endofprompt|>"
    )


def test_synthesize_french_instruct_uses_zero_shot_with_the_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, captured = _capturing_engine(tmp_path, monkeypatch)
    engine.synthesize(
        ["Félicitations!"],
        reference=tmp_path / "voice.wav",
        prosody=[ProsodySpec(emotion="happy")],
        ref_text="Félicitations, vous avez gagné.",
        language="French",
    )

    payload = captured[0]
    assert payload["modes"] == ["zero_shot"]
    assert payload["instructs"] == [""]
    assert "请用法语说这句话" in payload["prompts"][0]
    assert payload["prompts"][0].endswith("Félicitations, vous avez gagné.")


def test_synthesize_french_instruct_without_transcript_keeps_instruct2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, captured = _capturing_engine(tmp_path, monkeypatch)
    engine.synthesize(
        ["Félicitations!"],
        reference=tmp_path / "voice.wav",
        prosody=[ProsodySpec(emotion="happy")],
        language="French",
    )

    payload = captured[0]
    assert payload["modes"] == ["instruct2"]
    assert "请用法语说这句话" in payload["instructs"][0]


def test_synthesize_applies_fast_pace_by_stretching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[float] = []

    class FakeStretcher:
        def stretch(self, samples, sample_rate, ratio):
            calls.append(ratio)
            return np.zeros(1200, dtype=np.float32)

    engine, captured = _capturing_engine(tmp_path, monkeypatch)
    engine._stretcher = FakeStretcher()
    out = engine.synthesize(
        ["Bonjour."],
        reference=tmp_path / "voice.wav",
        prosody=[ProsodySpec(emotion="happy", pace="fast")],
        ref_text="Bonjour.",
        language="French",
    )

    assert calls == [0.90]
    assert out[0].size == 1200
    assert captured[0]["modes"] == ["zero_shot"]
    assert "语速" not in captured[0]["prompts"][0]


def test_synthesize_does_not_stretch_english(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[float] = []

    class FakeStretcher:
        def stretch(self, samples, sample_rate, ratio):
            calls.append(ratio)
            return samples

    engine, captured = _capturing_engine(tmp_path, monkeypatch)
    engine._stretcher = FakeStretcher()
    engine.synthesize(
        ["Hello."],
        reference=tmp_path / "voice.wav",
        prosody=[ProsodySpec(pace="fast")],
        ref_text="Hello.",
        language="English",
    )

    assert calls == []
    assert "语速" in captured[0]["instructs"][0]


def test_worker_failure_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Failed:
        returncode = 1
        stdout = ""
        stderr = "boom: model exploded"

    monkeypatch.setattr("pipeline.cosyvoice.subprocess.run", lambda *a, **k: Failed())
    engine = CosyVoiceEngine("m", device="cpu")

    with pytest.raises(TTSError, match="model exploded"):
        engine._run_worker(tmp_path / "request.json", tmp_path)


def test_worker_missing_package_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Failed:
        returncode = 1
        stdout = ""
        stderr = "ModuleNotFoundError: No module named 'cosyvoice'"

    monkeypatch.setattr("pipeline.cosyvoice.subprocess.run", lambda *a, **k: Failed())
    engine = CosyVoiceEngine("m", device="cpu")

    with pytest.raises(DependencyError, match="cosyvoice"):
        engine._run_worker(tmp_path / "request.json", tmp_path)


def test_missing_package_is_reported_by_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(CosyVoiceEngine, "_project_root", lambda self: tmp_path)
    monkeypatch.setattr("pipeline.cosyvoice._module_available", lambda name: False)
    engine = CosyVoiceEngine("m", device="cpu")

    with pytest.raises(DependencyError, match="models --download tts_cosyvoice3"):
        engine.synthesize(["a"], reference=tmp_path / "x.wav")


def _fake_cosyvoice_package(tmp_path: Path) -> Path:
    """A tiny ``cosyvoice`` package exposing the worker's ``AutoModel`` entry."""
    package = tmp_path / "fake"
    (package / "cosyvoice" / "cli").mkdir(parents=True)
    (package / "cosyvoice" / "__init__.py").write_text("", encoding="utf-8")
    (package / "cosyvoice" / "cli" / "__init__.py").write_text("", encoding="utf-8")
    (package / "cosyvoice" / "cli" / "cosyvoice.py").write_text(
        """
import numpy as np

class CosyVoice3:
    sample_rate = 24000
    def __init__(self, model_dir, fp16=False):
        self.model_dir = model_dir
        self.fp16 = fp16
    def inference_instruct2(self, text, instruct, prompt, stream=False, **kwargs):
        assert instruct.endswith("<|endofprompt|>")
        yield {"tts_speech": np.zeros(1200, dtype=np.float32)}
    def inference_cross_lingual(self, text, prompt, stream=False, **kwargs):
        yield {"tts_speech": np.zeros(1200, dtype=np.float32)}
    def inference_zero_shot(self, text, prompt_text, prompt, stream=False, **kwargs):
        assert prompt_text
        yield {"tts_speech": np.zeros(1200, dtype=np.float32)}

def AutoModel(model_dir, fp16=False):
    return CosyVoice3(model_dir, fp16=fp16)
""",
        encoding="utf-8",
    )
    return package


def _run_fake_worker(package: Path, tmp_path: Path, request: dict) -> dict:
    out_dir = tmp_path / "out"
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(package), str(ROOT)])
    completed = subprocess.run(
        [sys.executable, "-m", "pipeline.cosyvoice_worker", str(request_path)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    return json.loads((out_dir / "result.json").read_text(encoding="utf-8"))


def test_worker_end_to_end_with_a_fake_package(tmp_path: Path) -> None:
    package = _fake_cosyvoice_package(tmp_path)
    payload = _run_fake_worker(
        package,
        tmp_path,
        {
            "model_dir": "fake-model",
            "reference": str(tmp_path / "ref.wav"),
            "lines": ["bonjour"],
            "instructs": ["calm"],
            "fp16": False,
            "output_dir": str(tmp_path / "out"),
        },
    )

    assert payload["sample_rate"] == 24000
    assert len(payload["files"]) == 1
    assert Path(payload["files"][0]).is_file()


def test_worker_fills_a_neutral_instruct_on_the_instruct_path(
    tmp_path: Path,
) -> None:
    """An empty instruct still needs the marker, so the worker supplies it."""
    package = _fake_cosyvoice_package(tmp_path)
    payload = _run_fake_worker(
        package,
        tmp_path,
        {
            "model_dir": "fake-model",
            "reference": str(tmp_path / "ref.wav"),
            "lines": ["bonjour"],
            "modes": ["instruct2"],
            "instructs": [""],
            "fp16": False,
            "output_dir": str(tmp_path / "out"),
        },
    )

    assert len(payload["files"]) == 1
    assert NEUTRAL_INSTRUCT.endswith("<|endofprompt|>")
