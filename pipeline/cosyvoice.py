"""CosyVoice 3 speech synthesis, isolated in a child process.

The repo pins ``transformers==4.51.3`` while the Qwen TTS checkpoint this project
uses pins ``transformers==4.57.3``. Loading both in one interpreter silently
produces garbled audio - the LLM runs, but its codes decode to noise. The engine
therefore runs CosyVoice in a child whose ``PYTHONPATH`` starts with a small
transformers-4.51.3 overlay, leaving the main environment on the version Qwen
needs.

The child is :mod:`pipeline.cosyvoice_worker`. This module only builds the
request, runs it, and reads the WAVs back, so it needs no CosyVoice import of its
own and cannot poison the parent's ``transformers``.

Prosody is rendered here rather than in the worker: the parent has the neutral
:class:`pipeline.prosody.ProsodySpec` and the backend mapping, and the worker
should stay a thin adapter over the checkpoint.
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
from .prosody import (
    COSYVOICE3_SYSTEM_PROMPT,
    ProsodySpec,
    apply_inline_tokens,
    cosyvoice_instruct,
    cosyvoice_mode,
    resolve_prosody,
)

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "CosyVoiceEngine",
    "resolve_cosyvoice_device",
]

#: CosyVoice synthesises at 24 kHz; the worker reports the checkpoint's own rate.
SAMPLE_RATE = 24_000

#: Duration multipliers used to apply ``pace`` by time-stretching. On CosyVoice 3
#: a pace clause in the instruct destabilises non-English speech, so pace is
#: applied to the audio instead (>1 is slower, <1 is faster).
_PACE_RATIOS = {"slow": 1.12, "fast": 0.90}


def resolve_cosyvoice_device(device: str) -> str | None:
    """The ``CUDA_VISIBLE_DEVICES`` value that makes CosyVoice use ``device``.

    CosyVoice reads the visible devices when it imports torch, so this is passed
    to the child process in its environment. It only knows CUDA or CPU, so
    ``auto`` resolves through the project's rules first and MPS/XPU fall back to
    the CPU.
    """
    resolved = resolve_device(device)  # e.g. "cuda:0" or "cpu"
    if resolved.startswith("cuda"):
        return resolved.split(":", 1)[1] if ":" in resolved else "0"
    if resolved == "cpu":
        return ""
    return ""


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False


class CosyVoiceEngine:
    """Synthesiser backed by a CosyVoice 3 checkpoint, run out of process."""

    def __init__(
        self,
        model_dir: Path | str,
        *,
        device: str = "auto",
        fp16: bool | None = None,
        stretcher: Any | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        #: ``None`` means "follow the device": fp16 on CUDA, fp32 on CPU.
        self._fp16 = fp16
        self._sample_rate = SAMPLE_RATE
        #: Injected in tests; otherwise resolved lazily by ``_stretcher_for_pace``.
        self._stretcher = stretcher
        _LOGGER.debug(
            "CosyVoice engine: model=%s device=%s fp16=%s",
            self.model_dir,
            self.device,
            self._fp16,
        )

    @property
    def sample_rate(self) -> int:
        """Output sample rate, learned from the first successful synthesis."""
        return self._sample_rate

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _pythonpath(self) -> list[str]:
        """The child's ``PYTHONPATH``: overlay first, then CosyVoice, then root.

        ``COSYVOICE_PYTHONPATH``, when set, is prepended as a whole, which lets a
        deployment point at its own overlay or at another CosyVoice checkout.
        """
        root = self._project_root()
        candidates: list[Path] = []
        override = os.environ.get("COSYVOICE_PYTHONPATH")
        if override:
            candidates.extend(Path(part) for part in override.split(os.pathsep) if part)
        candidates.extend(
            [
                root / "third_party" / "cosyvoice-overlay",
                root / "third_party" / "CosyVoice",
                root / "third_party" / "CosyVoice" / "third_party" / "Matcha-TTS",
                root,
            ]
        )
        seen: list[str] = []
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            resolved = str(candidate.resolve())
            if resolved not in seen:
                seen.append(resolved)
        return seen

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(self._pythonpath())
        visible = resolve_cosyvoice_device(self.device)
        if visible is not None:
            env["CUDA_VISIBLE_DEVICES"] = visible
        return env

    def _fp16_enabled(self) -> bool:
        if self._fp16 is not None:
            return bool(self._fp16)
        return resolve_device(self.device).startswith("cuda")

    def _ensure_available(self) -> None:
        root = self._project_root()
        repo = root / "third_party" / "CosyVoice"
        if repo.is_dir() or _module_available("cosyvoice"):
            return
        raise DependencyError(
            "cosyvoice is not installed. Clone the CosyVoice repository under "
            "third_party/CosyVoice (or install it so 'cosyvoice' is importable), "
            "and download the checkpoint with: "
            "python dub.py models --download tts_cosyvoice3"
        )

    def _run_worker(self, request_path: Path, out_dir: Path) -> dict[str, Any]:
        """Run the child and return its ``result.json`` payload."""
        command = [
            sys.executable,
            "-m",
            "pipeline.cosyvoice_worker",
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
            raise TTSError(f"Could not start the CosyVoice worker: {exc}") from exc
        elapsed = time.perf_counter() - started

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            if (
                "No module named 'cosyvoice'" in stderr
                or "No module named 'matcha'" in stderr
            ):
                raise DependencyError(
                    "cosyvoice could not be imported by the worker. Clone the "
                    "CosyVoice repository and install its Matcha-TTS submodule "
                    "(see docs/setup.md)."
                )
            tail = "\n".join(stderr.splitlines()[-12:])
            raise TTSError(
                f"CosyVoice worker failed ({completed.returncode}) after "
                f"{elapsed:.1f}s:\n{tail}"
            )

        if completed.stderr.strip():
            _LOGGER.debug(
                "CosyVoice worker [%s] output:\n%s",
                self.model_dir,
                completed.stderr.strip(),
            )

        result_path = out_dir / "result.json"
        if not result_path.is_file():
            raise TTSError("The CosyVoice worker produced no result manifest.")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TTSError("The CosyVoice worker returned an invalid result.")
        return payload

    @property
    def _text_frontend(self) -> bool:
        """Whether CosyVoice's text frontend normalises the line.

        CosyVoice 3 reads text without a traditional frontend (its README says
        so), and the frontend's only non-Chinese path is the *English*
        normaliser - which mangles French. Keeping it off is what stops a French
        line from being normalised as English.
        """
        return False

    def _zero_shot_prompt(self, ref_text: str | None) -> str:
        """The prompt slot for ``inference_zero_shot``: system prefix, transcript."""
        return f"{COSYVOICE3_SYSTEM_PROMPT}{(ref_text or '').strip()}"

    def _cross_lingual_text(self, text: str) -> str:
        """Prefix inline-token text with the system prompt CosyVoice 3 expects."""
        return f"{COSYVOICE3_SYSTEM_PROMPT}{text}"

    def _stretcher_for_pace(self) -> Any:
        """The stretcher used to apply pace, resolved on first use."""
        if self._stretcher is None:
            from .timing import detect_stretcher

            self._stretcher = detect_stretcher("auto")
        return self._stretcher

    def _apply_pace(
        self, samples: np.ndarray, spec: ProsodySpec, sample_rate: int
    ) -> np.ndarray:
        """Apply ``slow``/``fast`` by time-stretching one line."""
        ratio = _PACE_RATIOS.get(spec.pace)
        if ratio is None:
            return samples
        stretched = self._stretcher_for_pace().stretch(
            np.asarray(samples, dtype=np.float32), int(sample_rate), ratio
        )
        out = np.ascontiguousarray(
            np.asarray(stretched, dtype=np.float32).reshape(-1)
        )
        _LOGGER.debug(
            "Pace %s: stretched by %.2fx (%d -> %d samples)",
            spec.pace,
            ratio,
            len(samples),
            len(out),
        )
        return out

    def _plan_lines(
        self,
        lines: Sequence[str],
        *,
        prosody: Sequence[ProsodySpec] | None,
        ref_text: str | None,
        language: str | None = None,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Per-line ``(text, instruct, prompt, mode)`` for the worker."""
        specs = resolve_prosody(lines, prosody)
        have_transcript = bool((ref_text or "").strip())

        texts: list[str] = []
        values: list[str] = []
        prompts: list[str] = []
        modes: list[str] = []
        for index, line in enumerate(lines):
            spec = specs[index]
            mode = cosyvoice_mode(spec)
            if mode == "zero_shot" and not have_transcript:
                # No transcript to condition on: the token path still clones
                # the speaker and keeps the language, where an empty prompt
                # would weaken the clone.
                mode = "cross_lingual"
            if mode == "instruct2":
                text = apply_inline_tokens(line, spec)
                instruct = cosyvoice_instruct(spec, language=language)
                prompt = ""
                if (
                    language
                    and str(language).strip().lower()
                    not in {"english", "chinese"}
                    and have_transcript
                ):
                    # instruct2 drops the reference speech tokens, which is
                    # what makes non-English prosody unstable. Pairing the
                    # instruction with the reference transcript keeps them,
                    # so the voice and accent stay grounded.
                    mode = "zero_shot"
                    prompt = f"{instruct}{(ref_text or '').strip()}"
                    instruct = ""
            elif mode == "cross_lingual":
                text = self._cross_lingual_text(apply_inline_tokens(line, spec))
                instruct = ""
                prompt = ""
            else:  # zero_shot
                text = line
                instruct = ""
                prompt = self._zero_shot_prompt(ref_text)
            texts.append(text)
            values.append(instruct)
            prompts.append(prompt)
            modes.append(mode)
        return texts, values, prompts, modes

    def synthesize(
        self,
        lines: Sequence[str],
        *,
        reference: Path | str,
        prosody: Sequence[ProsodySpec] | None = None,
        ref_text: str | None = None,
        language: str | None = None,
        trace: list[dict[str, Any]] | None = None,
    ) -> list[np.ndarray]:
        """Synthesise each line in the reference voice.

        How each line is rendered depends on what it asks for, because the three
        CosyVoice inference methods are not interchangeable (see
        :func:`pipeline.prosody.cosyvoice_mode`):

        * nothing asked -> ``inference_zero_shot`` with the reference transcript,
          which is what preserves the reference's native accent;
        * only inline tokens (breath/pause/laughter) -> ``inference_cross_lingual``;
        * emotion/pace/volume -> ``inference_instruct2``.

        Args:
            lines: Target-language lines to speak, one utterance each.
            reference: Path to a clip of the voice to clone. CosyVoice loads it
                itself at the rates it needs.
            prosody: One spec per line; a single value broadcasts.
            ref_text: Exact transcript of ``reference``. Used by the zero-shot
                clone path to keep the reference's accent and pronunciation.

        Returns:
            One mono float32 array per line, in order.

        Raises:
            DependencyError: If CosyVoice is not installed.
            TTSError: If the worker fails or a line comes back empty.
        """
        if not lines:
            return []

        texts, values, prompts, modes = self._plan_lines(
            lines,
            prosody=prosody,
            ref_text=ref_text,
            language=language,
        )
        if trace is not None:
            for text, instruct, prompt, mode in zip(texts, values, prompts, modes):
                trace.append(
                    {
                        "channel": "cosyvoice",
                        "text": text,
                        "instruct": instruct,
                        "prompt": prompt,
                        "mode": mode,
                    }
                )
        self._ensure_available()

        _LOGGER.debug(
            "CosyVoice request: lines=%d reference=%s language=%s modes=%s "
            "text_frontend=%s",
            len(lines),
            reference,
            language or "auto",
            modes,
            self._text_frontend,
        )
        for index, (text, instruct, prompt, mode) in enumerate(
            zip(texts, values, prompts, modes), start=1
        ):
            _LOGGER.debug(
                "  line %d [%s]: %r (instruct=%r prompt=%r)",
                index,
                mode,
                text,
                instruct,
                prompt,
            )
        started = time.perf_counter()

        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="qwen-tts-cosyvoice-") as tmp:
            out_dir = Path(tmp)
            request_path = out_dir / "request.json"
            request_path.write_text(
                json.dumps(
                    {
                        "model_dir": str(self.model_dir),
                        "reference": str(reference),
                        "ref_text": (ref_text or ""),
                        "lines": texts,
                        "instructs": values,
                        "prompts": prompts,
                        "modes": modes,
                        "text_frontend": self._text_frontend,
                        "fp16": self._fp16_enabled(),
                        "verbose": _LOGGER.isEnabledFor(logging.DEBUG),
                        "output_dir": str(out_dir),
                    },
                    ensure_ascii=False,
                ),
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
                    f"CosyVoice returned {len(arrays)} clip(s) for {len(lines)} line(s)."
                )
            # Pace is applied to the audio on CosyVoice 3 for non-English text,
            # where a pace clause in the instruct destabilises the speech.
            if (
                prosody is not None
                and language
                and str(language).strip().lower() not in {"english", "chinese"}
            ):
                specs = resolve_prosody(lines, prosody)
                arrays = [
                    self._apply_pace(array, spec, sample_rate)
                    for array, spec in zip(arrays, specs)
                ]
            elapsed = time.perf_counter() - started
            timings = result.get("timings") or {}
            _LOGGER.info(
                "CosyVoice %s: %d line(s) in %.2fs "
                "(worker load %.2fs, generate %.2fs)",
                "3",
                len(lines),
                elapsed,
                float(timings.get("load_seconds", 0.0)),
                float(sum(timings.get("line_seconds", []) or [])),
            )
            self._sample_rate = sample_rate
            return arrays
