"""Structured environment and device reports.

The CLI renders these as text; the HTTP API serialises the same facts as JSON.
Keeping the data here means both interfaces answer from one source and cannot
drift about which device, precision or model is actually available.
"""

from __future__ import annotations

import platform
import shutil
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Sequence

from .asr import asr_device, resolve_compute_type
from .config import (
    available_accelerators,
    enumerate_devices,
    pick_dtype,
    preferred_dtype,
    probe_device,
)
from .models import MVP_REQUIRED, REGISTRY, registry_status

__all__ = [
    "OPTIONAL_DEPENDENCIES",
    "SYSTEM_BINARIES",
    "device_report",
    "environment_report",
    "torch_version",
]

#: Optional dependencies and what each one unlocks.
OPTIONAL_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("demucs", "keep the original music and ambience (--no-background disables)"),
    ("funasr", "multi-speaker dubbing (--diarizer none disables)"),
    ("speechbrain", "overlapping-speech separation (--overlap separate)"),
)

#: System binaries the media layer looks for. Only ffmpeg is ever required.
SYSTEM_BINARIES: tuple[tuple[str, str], ...] = (
    ("ffmpeg", "required for video input and muxing"),
    ("ffprobe", "required for video input"),
    ("rubberband", "best-quality time stretching"),
    ("sox", "alternative time stretching"),
)


def _module_available(name: str) -> bool:
    """True when an optional dependency can be imported."""
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False


def torch_version() -> str:
    """torch's version, tagged with the CUDA or ROCm build it was compiled for."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard runtime dep
        return "not installed"

    version = getattr(torch, "version", None)
    hip = getattr(version, "hip", None)
    cuda = getattr(version, "cuda", None)
    if hip:
        return f"{torch.__version__}  (ROCm {hip})"
    if cuda:
        return f"{torch.__version__}  (CUDA {cuda})"
    return f"{torch.__version__}  (CPU build)"


def _dtype_label(device: str, requested: str) -> str:
    name = str(pick_dtype(device, requested)).replace("torch.", "")
    return name if requested == "auto" else f"{name}  (requested: {requested})"


def _device_payload(info: Any) -> dict[str, Any]:
    return {
        "label": info.label,
        "kind": info.kind,
        "index": info.index,
        "name": info.name,
        "total_memory": info.total_memory,
        "capability": list(info.capability) if info.capability else None,
        "supports_bf16": bool(info.supports_bf16),
        "supports_fp16": bool(info.supports_fp16),
        "is_accelerated": bool(info.is_accelerated),
        "requested": info.requested,
    }


def devices_report() -> dict[str, Any]:
    """Every device torch can use, plus the exact flag that selects it."""
    devices = enumerate_devices()
    return {
        "torch": torch_version(),
        "supported": [
            ("cuda", "NVIDIA, with the CUDA build of PyTorch"),
            ("rocm", "AMD, with the ROCm build of PyTorch (reported as cuda)"),
            ("xpu", "Intel Arc / integrated, with the XPU build of PyTorch"),
            ("mps", "Apple Silicon, via Metal"),
            ("cpu", "always available; slower, and the fallback for every GPU"),
        ],
        "devices": [
            {
                **_device_payload(info),
                "asr_device": asr_device(info.label),
                "tts_dtype": str(preferred_dtype(info)),
            }
            for info in devices
        ],
        "note": (
            "Vulkan is not a backend here: speech recognition runs on CTranslate2 "
            "(CUDA or CPU only) and the other stages run on PyTorch, which has no "
            "Vulkan device. AMD cards use ROCm and appear as 'cuda'."
        ),
    }


def environment_report(
    models_dir: Path | str,
    *,
    device: str = "auto",
    dtype: str = "auto",
    devices: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The facts `dub.py info` prints, as a serialisable mapping."""
    requested = list(devices) if devices else []
    device = requested[0] if requested else device
    info = probe_device(device)
    return {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "torch": torch_version(),
        "accelerators": list(available_accelerators()),
        "device": _device_payload(info),
        "resolved": info.label,
        "synthesis_devices": requested,
        "tts_dtype": _dtype_label(device, dtype),
        "asr_device": asr_device(device),
        "asr_compute_type": resolve_compute_type(asr_device(device)),
        "optional_packages": [
            {
                "name": name,
                "available": _module_available(name),
                "purpose": purpose,
            }
            for name, purpose in OPTIONAL_DEPENDENCIES
        ],
        "system_binaries": [
            {
                "name": name,
                "path": shutil.which(name),
                "purpose": purpose,
            }
            for name, purpose in SYSTEM_BINARIES
        ],
        "models_dir": str(Path(models_dir).resolve()),
        "models": [
            {
                "name": spec.name,
                "present": present,
                "required": spec.name in MVP_REQUIRED,
                "repo_id": spec.repo_id,
                "description": spec.description,
                "license": spec.license,
            }
            for spec, present in registry_status(models_dir)
        ],
        "registry": list(REGISTRY),
    }


def device_report(
    models_dir: Path | str,
    *,
    device: str = "auto",
    dtype: str = "auto",
    devices: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Alias used by the API for the environment view."""
    return environment_report(
        models_dir, device=device, dtype=dtype, devices=devices
    )
