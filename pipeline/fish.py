"""Fish Speech S2-Pro synthesis, isolated in a child process.

Fish Speech reads emotion and paralinguistics from **inline tags in the text**
(``[sad]``, ``[sigh]``, ``[whisper in small voice]``, ...) rather than from a
side-channel instruct, so the lines are sent to the model exactly as the caller
wrote or planned them. There is deliberately no prosody renderer here: the tag is
the interface.

Fish pins ``torch==2.8`` and ``transformers==5.19``, which cannot coexist with
the Qwen checkpoint's ``transformers==4.57``. The model therefore runs in
:mod:`pipeline.fish_worker`, like CosyVoice and Chatterbox.
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

__all__ = ["FishEngine", "resolve_fish_device"]

#: Fallback when the checkpoint does not expose a rate; the worker reports its own.
SAMPLE_RATE = 24_000


def resolve_fish_device(device: str) -> str:
    """Fish runs on one CUDA device or the CPU.

    For ``auto`` the plain ``"cuda"`` is returned, which resolves to ``cuda:0`` -
    the single card the int8 weights fit on. An explicit ``cuda:N`` is kept.
    """
    raw = str(device or "auto").strip().lower()
    try:
        resolved = str(resolve_device(device)).lower()
    except Exception:  # noqa: BLE001 - an unusable device is reported by the model
        return "cpu"
    if not resolved.startswith("cuda"):
        return "cpu"
    return "cuda" if raw in {"", "auto"} else resolved


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False


class FishEngine:
    """Synthesiser backed by Fish Speech S2-Pro, run out of process."""

    def __init__(
        self,
        model_dir: Path | str,
        *,
        device: str = "auto",
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        quantize: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        #: ``int8`` (default in the worker) fits a 12 GB card; ``off`` loads the
        #: bf16 weights Fish documents as needing ~24 GB.
        self.quantize = quantize
        self._sample_rate = SAMPLE_RATE
        _LOGGER.debug(
            "Fish engine: model=%s device=%s quantize=%s temperature=%s top_p=%s",
            self.model_dir,
            self.device,
            quantize,
            temperature,
            top_p,
        )

    @property
    def sample_rate(self) -> int:
        """Output sample rate, learned from the first successful synthesis."""
        return self._sample_rate

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _python_executable(self) -> str:
        """The interpreter the worker runs in (isolated env, like Chatterbox)."""
        override = os.environ.get("FISH_PYTHON")
        if override:
            return override
        candidate = self._project_root() / ".venv-fish" / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
        return sys.executable

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self._project_root())
        return env

    def _ensure_available(self) -> None:
        """Whether a usable Fish installation can be found."""
        if os.environ.get("FISH_PYTHON"):
            if Path(os.environ["FISH_PYTHON"]).is_file():
                return
            raise DependencyError(
                f"FISH_PYTHON points at {os.environ['FISH_PYTHON']}, which does "
                "not exist."
            )
        if (self._project_root() / ".venv-fish" / "bin" / "python").is_file():
            return
        if _module_available("fish_speech"):
            return
        raise DependencyError(
            "fish-speech is not installed. Create the isolated environment "
            "described in docs/voices.md (Fish pins torch 2.8 / transformers "
            "5.19, which conflict with Qwen), or set FISH_PYTHON to an "
            "environment that has it. Weights: "
            "python dub.py models --download tts_fish"
        )

    def _run_worker(self, request_path: Path, out_dir: Path) -> dict[str, Any]:
        """Run the child and return its ``result.json`` payload."""
        command = [
            self._python_executable(),
            "-m",
            "pipeline.fish_worker",
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
            raise TTSError(f"Could not start the Fish worker: {exc}") from exc
        elapsed = time.perf_counter() - started

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            if "No module named 'fish_speech'" in stderr:
                raise DependencyError(
                    "fish_speech could not be imported by the worker. Create the "
                    ".venv-fish environment described in docs/voices.md, or set "
                    "FISH_PYTHON."
                )
            if "CUDA out of memory" in stderr or "OutOfMemoryError" in stderr:
                raise TTSError(
                    "Fish Speech ran out of VRAM. The int8 default needs ~6 GB "
                    "resident (~11 GB peak) and --fish-quantize off needs ~24 "
                    "GB; retry with --device cpu (slower) or a larger card."
                )
            tail = "\n".join(stderr.splitlines()[-12:])
            raise TTSError(
                f"Fish worker failed ({completed.returncode}) after "
                f"{elapsed:.1f}s:\n{tail}"
            )

        if completed.stderr.strip():
            _LOGGER.debug(
                "Fish worker [%s] output:\n%s",
                self._python_executable(),
                completed.stderr.strip(),
            )

        result_path = out_dir / "result.json"
        if not result_path.is_file():
            raise TTSError("The Fish worker produced no result manifest.")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TTSError("The Fish worker returned an invalid result.")
        return payload

    def synthesize(
        self,
        lines: Sequence[str],
        *,
        reference: Path | str,
        ref_text: str | None = None,
        language: str | None = None,
        trace: list[dict[str, Any]] | None = None,
    ) -> list[np.ndarray]:
        """Synthesise each line in the reference voice, tags included.

        Fish reads delivery from ``[tag]`` cues inside the text and has no
        separate style channel, so the lines are sent exactly as given: the
        caller writes them (``--text``) or the planner does.

        Args:
            lines: Lines to speak, one utterance each, tags included.
            reference: Clip of the voice to clone.
            ref_text: Transcript of ``reference``; improves cloning fidelity.
            language: Ignored: Fish infers the language from the text.

        Returns:
            One mono float32 array per line, in order.
        """
        if not lines:
            return []

        tagged = [str(line) for line in lines]
        if trace is not None:
            for text in tagged:
                trace.append({"channel": "fish", "text": text})
        self._ensure_available()

        _LOGGER.debug(
            "Fish request: lines=%d reference=%s device=%s interpreter=%s",
            len(lines),
            reference,
            self.device,
            self._python_executable(),
        )
        for index, text in enumerate(tagged, start=1):
            _LOGGER.debug("  line %d: %r", index, text)
        started = time.perf_counter()

        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="qwen-tts-fish-") as tmp:
            out_dir = Path(tmp)
            request_path = out_dir / "request.json"
            request: dict[str, Any] = {
                "model_dir": str(self.model_dir),
                "reference": str(reference),
                "reference_text": (ref_text or "").strip(),
                "lines": tagged,
                "device": resolve_fish_device(self.device),
                "verbose": _LOGGER.isEnabledFor(logging.DEBUG),
                "output_dir": str(out_dir),
            }
            if self.temperature is not None:
                request["temperature"] = float(self.temperature)
            if self.top_p is not None:
                request["top_p"] = float(self.top_p)
            if self.max_new_tokens is not None:
                request["max_new_tokens"] = int(self.max_new_tokens)
            if self.quantize:
                request["quantize"] = str(self.quantize)
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
                    f"Fish returned {len(arrays)} clip(s) for {len(lines)} line(s)."
                )
            elapsed = time.perf_counter() - started
            _LOGGER.info("Fish Speech: %d line(s) in %.2fs", len(lines), elapsed)
            self._sample_rate = sample_rate
            return arrays
