"""Chatterbox speech synthesis, isolated in a child process.

Chatterbox is a pip-installed multilingual TTS model (MIT weights) with a cloned
voice and a single ``exaggeration`` style knob. It is kept out of the parent for
the same reason CosyVoice is: its torch/transformers requirements are not the
ones the Qwen checkpoint needs, and importing both in one interpreter is the kind
of conflict that only shows up as bad audio.

The child is :mod:`pipeline.chatterbox_worker`. This module builds the request,
runs it, and reads the WAVs back, so it needs no Chatterbox import of its own.
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
from .prosody import ProsodySpec, chatterbox_exaggeration, resolve_prosody

_LOGGER = logging.getLogger(__name__)

__all__ = ["ChatterboxEngine", "resolve_chatterbox_device"]

#: Fallback when the checkpoint does not expose a rate; the worker reports its own.
SAMPLE_RATE = 24_000


def resolve_chatterbox_device(device: str) -> str:
    """Chatterbox only knows ``cuda`` or ``cpu``, not an indexed device."""
    try:
        resolved = str(resolve_device(device)).lower()
    except Exception:  # noqa: BLE001 - an unusable device is reported by the model
        return "cpu"
    return "cuda" if resolved.startswith("cuda") else "cpu"


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False


class ChatterboxEngine:
    """Synthesiser backed by Chatterbox, run out of process."""

    def __init__(
        self,
        model_dir: Path | str | None = None,
        *,
        device: str = "auto",
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        temperature: float | None = None,
    ) -> None:
        #: Chatterbox fetches its weights by repo id when no local directory is
        #: given, so this is optional.
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.device = device
        self._sample_rate = SAMPLE_RATE
        #: Chatterbox's only style knob is this 0.0-1.0 intensity. ``None`` keeps
        #: the emotion-derived value; an explicit value overrides every line.
        self.exaggeration = exaggeration
        #: Adherence/expressiveness and sampling randomness. Chatterbox exposes
        #: both; ``None`` keeps the package defaults (0.5 and 0.8).
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        _LOGGER.debug(
            "Chatterbox engine: model=%s device=%s exaggeration=%s cfg_weight=%s "
            "temperature=%s",
            self.model_dir or "(package default)",
            self.device,
            exaggeration,
            cfg_weight,
            temperature,
        )

    def _line_exaggeration(self, spec: ProsodySpec) -> float:
        """Exaggeration for one line: explicit spec > engine default > emotion."""
        if spec.exaggeration is not None:
            return round(max(0.0, min(1.0, float(spec.exaggeration))), 3)
        if self.exaggeration is not None:
            return round(max(0.0, min(1.0, float(self.exaggeration))), 3)
        return chatterbox_exaggeration(spec)

    @property
    def sample_rate(self) -> int:
        """Output sample rate, learned from the first successful synthesis."""
        return self._sample_rate

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _python_executable(self) -> str:
        """The interpreter the worker runs in.

        Chatterbox pins ``torch==2.6`` and ``transformers==5.2``, while Qwen TTS
        needs ``transformers==4.57`` and the installed torch is older. Importing
        both in one environment is not an option, so the backend prefers a
        dedicated ``.venv-chatterbox`` when one exists. ``CHATTERBOX_PYTHON``
        overrides it for a deployment that keeps the environment elsewhere.
        """
        override = os.environ.get("CHATTERBOX_PYTHON")
        if override:
            return override
        candidate = self._project_root() / ".venv-chatterbox" / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
        return sys.executable

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        # The child imports ``pipeline.chatterbox_worker``; keep the project root
        # on the path without inheriting whatever the caller had.
        env["PYTHONPATH"] = str(self._project_root())
        return env

    def _ensure_available(self) -> None:
        """Whether a usable Chatterbox installation can be found.

        The package may live in the dedicated environment rather than the main
        one, so its presence on disk is enough - importing it here is neither
        necessary nor safe (it would pull in its `transformers`).
        """
        if os.environ.get("CHATTERBOX_PYTHON"):
            if Path(os.environ["CHATTERBOX_PYTHON"]).is_file():
                return
            raise DependencyError(
                f"CHATTERBOX_PYTHON points at {os.environ['CHATTERBOX_PYTHON']}, "
                "which does not exist."
            )
        if (self._project_root() / ".venv-chatterbox" / "bin" / "python").is_file():
            return
        if _module_available("chatterbox"):
            return
        raise DependencyError(
            "chatterbox is not installed. Create the isolated environment "
            "described in docs/voices.md (Chatterbox pins torch 2.6 / "
            "transformers 5.2, which conflict with Qwen), or set "
            "CHATTERBOX_PYTHON to an environment that has it. Weights: "
            "python dub.py models --download tts_chatterbox"
        )

    def _run_worker(self, request_path: Path, out_dir: Path) -> dict[str, Any]:
        """Run the child and return its ``result.json`` payload."""
        command = [
            self._python_executable(),
            "-m",
            "pipeline.chatterbox_worker",
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
            raise TTSError(f"Could not start the Chatterbox worker: {exc}") from exc
        elapsed = time.perf_counter() - started

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            if "No module named 'chatterbox'" in stderr:
                raise DependencyError(
                    "chatterbox could not be imported by the worker. Install it "
                    "with: pip install \".[chatterbox]\" (or point "
                    "CHATTERBOX_PYTHON at an environment that has it)."
                )
            tail = "\n".join(stderr.splitlines()[-12:])
            raise TTSError(
                f"Chatterbox worker failed ({completed.returncode}) after "
                f"{elapsed:.1f}s:\n{tail}"
            )

        if completed.stderr.strip():
            _LOGGER.debug(
                "Chatterbox worker [%s] output:\n%s",
                self._python_executable(),
                completed.stderr.strip(),
            )

        result_path = out_dir / "result.json"
        if not result_path.is_file():
            raise TTSError("The Chatterbox worker produced no result manifest.")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TTSError("The Chatterbox worker returned an invalid result.")
        return payload

    def synthesize(
        self,
        lines: Sequence[str],
        *,
        reference: Path | str,
        prosody: Sequence[ProsodySpec] | None = None,
        language: str | None = None,
        trace: list[dict[str, Any]] | None = None,
    ) -> list[np.ndarray]:
        """Synthesise each line in the reference voice.

        Args:
            lines: Lines to speak, one utterance each.
            reference: Clip of the voice to clone.
            prosody: One spec per line; a single value broadcasts. It is mapped
                onto Chatterbox's single ``exaggeration`` scalar.
            language: Language code (``fr``, ``en``, ...). Required by the
                multilingual checkpoint; ignored by the English-only one.

        Returns:
            One mono float32 array per line, in order.

        Raises:
            DependencyError: If Chatterbox is not installed.
            TTSError: If the worker fails or a line comes back empty.
        """
        if not lines:
            return []

        specs = resolve_prosody(lines, prosody)
        exaggerations = [self._line_exaggeration(spec) for spec in specs]
        if trace is not None:
            for line, exaggeration in zip(lines, exaggerations):
                trace.append(
                    {
                        "channel": "chatterbox",
                        "text": str(line),
                        "exaggeration": exaggeration,
                    }
                )
        self._ensure_available()

        _LOGGER.debug(
            "Chatterbox request: lines=%d language=%s reference=%s "
            "exaggerations=%s cfg_weight=%s temperature=%s interpreter=%s",
            len(lines),
            language or "default",
            reference,
            exaggerations,
            self.cfg_weight,
            self.temperature,
            self._python_executable(),
        )
        for index, line in enumerate(lines, start=1):
            _LOGGER.debug("  line %d: %r", index, line)
        started = time.perf_counter()

        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="qwen-tts-chatterbox-") as tmp:
            out_dir = Path(tmp)
            request_path = out_dir / "request.json"
            request: dict[str, Any] = {
                "model_dir": str(self.model_dir) if self.model_dir else None,
                "reference": str(reference),
                "lines": list(lines),
                "exaggerations": exaggerations,
                "language": language,
                "device": resolve_chatterbox_device(self.device),
                "verbose": _LOGGER.isEnabledFor(logging.DEBUG),
                "output_dir": str(out_dir),
            }
            if self.cfg_weight is not None:
                request["cfg_weight"] = float(self.cfg_weight)
            if self.temperature is not None:
                request["temperature"] = float(self.temperature)
            request_path.write_text(
                json.dumps(request, ensure_ascii=False),
                encoding="utf-8",
            )
            result = self._run_worker(request_path, out_dir)

            sample_rate = int(result.get("sample_rate", SAMPLE_RATE))
            arrays: list[np.ndarray] = []
            for path in result.get("files", []):
                data, rate = sf.read(str(path), dtype="float32", always_2d=False)
                if getattr(data, "ndim", 1) > 1:
                    data = data.mean(axis=-1)
                arrays.append(
                    np.ascontiguousarray(np.asarray(data, dtype=np.float32).reshape(-1))
                )
                sample_rate = int(rate)
            if len(arrays) != len(lines):
                raise TTSError(
                    f"Chatterbox returned {len(arrays)} clip(s) for {len(lines)} line(s)."
                )
            elapsed = time.perf_counter() - started
            timings = result.get("timings") or {}
            _LOGGER.info(
                "Chatterbox: %d line(s) in %.2fs "
                "(worker load %.2fs, generate %.2fs)",
                len(lines),
                elapsed,
                float(timings.get("load_seconds", 0.0)),
                float(sum(timings.get("line_seconds", []) or [])),
            )
            self._sample_rate = sample_rate
            return arrays
