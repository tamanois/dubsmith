"""IndexTTS2 synthesis, isolated in a child process.

IndexTTS2 clones a speaker from one clip and takes **emotion from a separate
reference clip**, blended by ``emo_alpha``. It has no structured prosody channel,
so ``--prosody``/``--emotion`` do not apply.

The released IndexTTS2.0 does **not** implement precise duration control (the
project's README marks it "not yet enabled in this release"). ``--duration`` is
therefore honoured by time-stretching the generated line with the pipeline's own
stretcher (``pipeline.timing``), which is the same mechanism the dubbing pipeline
already uses to fit a line to its slot.

IndexTTS pins ``torch==2.8`` and ``transformers==4.52``, which conflict with the
Qwen checkpoint, so the model runs in :mod:`pipeline.index_tts_worker`.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import resolve_device
from .errors import DependencyError, TTSError

_LOGGER = logging.getLogger(__name__)

__all__ = ["IndexTTSEngine", "resolve_index_tts_device"]

#: Fallback when the checkpoint does not expose a rate; the worker reports its own.
SAMPLE_RATE = 22_050


def resolve_index_tts_device(device: str) -> str:
    """IndexTTS2 accepts an indexed CUDA device or the CPU."""
    try:
        resolved = str(resolve_device(device)).lower()
    except Exception:  # noqa: BLE001 - an unusable device is reported by the model
        return "cpu"
    return resolved if resolved.startswith("cuda") else "cpu"


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False


class IndexTTSEngine:
    """Synthesiser backed by IndexTTS2, run out of process."""

    def __init__(
        self,
        model_dir: Path | str,
        *,
        device: str = "auto",
        use_fp16: bool = True,
        emotion_audio: Path | str | None = None,
        emotion_alpha: float = 1.0,
        stretcher: str = "auto",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.use_fp16 = use_fp16
        self.emotion_audio = Path(emotion_audio) if emotion_audio else None
        self.emotion_alpha = float(emotion_alpha)
        self.stretcher_preference = stretcher
        self._sample_rate = SAMPLE_RATE
        _LOGGER.debug(
            "IndexTTS engine: model=%s device=%s emotion=%s alpha=%.2f",
            self.model_dir,
            self.device,
            self.emotion_audio,
            self.emotion_alpha,
        )

    @property
    def sample_rate(self) -> int:
        """Output sample rate, learned from the first successful synthesis."""
        return self._sample_rate

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _python_executable(self) -> str:
        """The interpreter the worker runs in (isolated env, like Chatterbox)."""
        override = os.environ.get("INDEXTTS_PYTHON")
        if override:
            return override
        candidate = self._project_root() / ".venv-indextts" / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
        return sys.executable

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self._project_root())
        return env

    def _ensure_available(self) -> None:
        """Whether a usable IndexTTS installation can be found."""
        if os.environ.get("INDEXTTS_PYTHON"):
            if Path(os.environ["INDEXTTS_PYTHON"]).is_file():
                return
            raise DependencyError(
                f"INDEXTTS_PYTHON points at {os.environ['INDEXTTS_PYTHON']}, "
                "which does not exist."
            )
        if (self._project_root() / ".venv-indextts" / "bin" / "python").is_file():
            return
        if _module_available("indextts"):
            return
        raise DependencyError(
            "index-tts is not installed. Create the isolated environment "
            "described in docs/voices.md (IndexTTS pins torch 2.8 / transformers "
            "4.52, which conflict with Qwen), or set INDEXTTS_PYTHON to an "
            "environment that has it. Weights: "
            "python dub.py models --download tts_index_tts2"
        )

    def _run_worker(self, request_path: Path, out_dir: Path) -> dict[str, Any]:
        """Run the child and return its ``result.json`` payload."""
        command = [
            self._python_executable(),
            "-m",
            "pipeline.index_tts_worker",
            str(request_path),
        ]
        started = time.perf_counter()
        try:
            completed = subprocess.run(  # noqa: S603 - argument list, shell disabled
                command,
                cwd=str(self._project_root()),
                env=self._worker_env(),
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:  # pragma: no cover - interpreter failure
            raise TTSError(f"Could not start the IndexTTS worker: {exc}") from exc
        elapsed = time.perf_counter() - started

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            if "No module named 'indextts'" in stderr:
                raise DependencyError(
                    "indextts could not be imported by the worker. Create the "
                    ".venv-indextts environment described in docs/voices.md, or "
                    "set INDEXTTS_PYTHON."
                )
            tail = "\n".join(stderr.splitlines()[-12:])
            raise TTSError(
                f"IndexTTS worker failed ({completed.returncode}) after "
                f"{elapsed:.1f}s:\n{tail}"
            )

        if completed.stderr.strip():
            _LOGGER.debug(
                "IndexTTS worker [%s] output:\n%s",
                self._python_executable(),
                completed.stderr.strip(),
            )

        result_path = out_dir / "result.json"
        if not result_path.is_file():
            raise TTSError("The IndexTTS worker produced no result manifest.")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TTSError("The IndexTTS worker returned an invalid result.")
        return payload

    def _fit_durations(
        self,
        arrays: list[np.ndarray],
        durations: Sequence[float | None] | None,
        sample_rate: int,
    ) -> list[np.ndarray]:
        """Time-stretch lines to their target seconds, when one is given.

        An explicit ``--duration`` is a user request, so the ratio window is
        wider than the dubbing pipeline's default ±18% (0.5-2.0). Beyond that a
        warning states the duration actually reached; extreme stretching would
        audibly damage the voice.
        """
        if not durations:
            return arrays
        from .timing import compute_ratio, detect_stretcher

        stretcher = None
        fitted: list[np.ndarray] = []
        for index, (samples, target) in enumerate(zip(arrays, durations)):
            if not target or target <= 0:
                fitted.append(samples)
                continue
            actual = samples.size / float(sample_rate)
            ratio = compute_ratio(
                float(target), actual, min_ratio=0.5, max_ratio=2.0
            )
            if abs(ratio - 1.0) < 1e-3:
                fitted.append(samples)
                continue
            achieved = actual * ratio
            if abs(achieved - float(target)) > max(0.05, 0.05 * float(target)):
                _LOGGER.warning(
                    "IndexTTS line %d: %.2fs -> %.2fs needs %.2fx; clamped to "
                    "%.2fx (reaches %.2fs).",
                    index + 1,
                    actual,
                    target,
                    float(target) / actual,
                    ratio,
                    achieved,
                )
            else:
                _LOGGER.debug(
                    "IndexTTS line %d: %.2fs -> %.2fs (ratio %.3f)",
                    index + 1,
                    actual,
                    target,
                    ratio,
                )
            if stretcher is None:
                stretcher = detect_stretcher(self.stretcher_preference)
            fitted.append(stretcher.stretch(samples, sample_rate, ratio))
        return fitted

    def synthesize(
        self,
        lines: Sequence[str],
        *,
        reference: Path | str,
        language: str | None = None,
        durations: Sequence[float | None] | None = None,
    ) -> list[np.ndarray]:
        """Synthesise each line in the reference voice.

        Args:
            lines: Lines to speak, one utterance each.
            reference: Clip of the voice whose timbre is cloned.
            language: Ignored: IndexTTS2 infers the language from the text.
            durations: Optional target seconds per line. The released IndexTTS2
                has no native duration control, so each line is time-stretched
                with the pipeline's stretcher.

        Returns:
            One mono float32 array per line, in order.
        """
        if not lines:
            return []

        self._ensure_available()

        _LOGGER.debug(
            "IndexTTS request: lines=%d reference=%s emotion=%s alpha=%.2f "
            "interpreter=%s",
            len(lines),
            reference,
            self.emotion_audio,
            self.emotion_alpha,
            self._python_executable(),
        )
        started = time.perf_counter()

        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="qwen-tts-indextts-") as tmp:
            out_dir = Path(tmp)
            request_path = out_dir / "request.json"
            request: dict[str, Any] = {
                "model_dir": str(self.model_dir),
                "reference": str(reference),
                "lines": list(lines),
                "device": resolve_index_tts_device(self.device),
                "use_fp16": bool(self.use_fp16),
                "verbose": _LOGGER.isEnabledFor(logging.DEBUG),
                "output_dir": str(out_dir),
            }
            if self.emotion_audio is not None:
                request["emotion_audio"] = str(self.emotion_audio)
                request["emotion_alpha"] = float(self.emotion_alpha)
            request_path.write_text(
                json.dumps(request, ensure_ascii=False), encoding="utf-8"
            )
            result = self._run_worker(request_path, out_dir)

            sample_rate = int(result.get("sample_rate", SAMPLE_RATE))
            arrays: list[np.ndarray] = []
            for path in result.get("files", []):
                data, rate = sf.read(str(path), dtype="float32", always_2d=False)
                if getattr(data, "ndim", 1) > 1:
                    data = data.mean(axis=-1)
                arrays.append(
                    np.ascontiguousarray(
                        np.asarray(data, dtype=np.float32).reshape(-1)
                    )
                )
                sample_rate = int(rate)
            if len(arrays) != len(lines):
                raise TTSError(
                    f"IndexTTS returned {len(arrays)} clip(s) for {len(lines)} "
                    "line(s)."
                )
            elapsed = time.perf_counter() - started
            arrays = self._fit_durations(arrays, durations, sample_rate)
            _LOGGER.info("IndexTTS2: %d line(s) in %.2fs", len(lines), elapsed)
            self._sample_rate = sample_rate
            return arrays
