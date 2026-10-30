"""Standalone Chatterbox worker, run in a child process.

Chatterbox ships as the ``chatterbox-tts`` package. Its multilingual entry point
is ``chatterbox.mtl_tts.ChatterboxMultilingualTTS``; older or English-only
installs expose ``chatterbox.tts.ChatterboxTTS`` instead. The worker accepts both
so a user is not blocked by whichever release they installed.

Protocol: one argument, a JSON request path. The request names the optional local
checkpoint, the reference clip, the lines, their exaggeration values, the
language code, the device and an output directory. The worker writes
``line_000.wav``, ... plus ``result.json`` with ``sample_rate`` and ``files``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

#: Fallback when the checkpoint does not expose a rate.
SAMPLE_RATE = 24_000

#: Files ``ChatterboxMultilingualTTS.from_local`` loads. A local ``models/chatterbox``
#: is used only when all of them are present; otherwise the package's own
#: ``from_pretrained`` resolves (and downloads) the weights.
MULTILINGUAL_FILES = (
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "Cangjie5_TC.json",
)


def _to_numpy(value) -> np.ndarray:
    """One utterance as a flat float32 numpy array."""
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return np.ascontiguousarray(array)


def _has_local_checkpoint(model_dir: str | None) -> bool:
    if not model_dir:
        return False
    root = Path(model_dir)
    return root.is_dir() and all((root / name).is_file() for name in MULTILINGUAL_FILES)


def _load_model(model_dir: str | None, device: str):
    """Load the multilingual model when available, else the English-only one.

    Returns ``(model, multilingual)``.
    """
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        if _has_local_checkpoint(model_dir):
            model = ChatterboxMultilingualTTS.from_local(str(model_dir), device)
        else:
            model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        return model, True
    except ImportError:
        from chatterbox.tts import ChatterboxTTS

        model = ChatterboxTTS.from_pretrained(device=device)
        return model, False


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(
            "usage: python -m pipeline.chatterbox_worker REQUEST.json",
            file=sys.stderr,
        )
        return 2

    request = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    from pipeline.logging_utils import quiet_dependencies

    quiet_dependencies()
    logger = logging.getLogger("chatterbox_worker")
    if request.get("verbose"):
        logger.setLevel(logging.DEBUG)

    import soundfile as sf

    load_started = time.perf_counter()
    model, multilingual = _load_model(request.get("model_dir"), request["device"])
    load_seconds = time.perf_counter() - load_started
    sample_rate = int(getattr(model, "sr", SAMPLE_RATE))
    logger.info(
        "Loaded %s (%s) in %.2fs (sample_rate=%d, device=%s, python=%s)",
        type(model).__name__,
        "multilingual" if multilingual else "english-only",
        load_seconds,
        sample_rate,
        request["device"],
        sys.executable,
    )

    started_all = time.perf_counter()
    line_seconds: list[float] = []
    files: list[str] = []
    for index, (line, exaggeration) in enumerate(
        zip(request["lines"], request["exaggerations"])
    ):
        kwargs: dict[str, object] = {
            "audio_prompt_path": request["reference"],
            "exaggeration": float(exaggeration),
        }
        if "cfg_weight" in request:
            kwargs["cfg_weight"] = float(request["cfg_weight"])
        if "temperature" in request:
            kwargs["temperature"] = float(request["temperature"])
        if multilingual:
            # The multilingual checkpoint requires a code; English is a safe
            # default when the caller did not supply one.
            kwargs["language_id"] = request.get("language") or "en"
        started = time.perf_counter()
        wav = model.generate(line, **kwargs)
        elapsed = time.perf_counter() - started
        line_seconds.append(elapsed)
        array = _to_numpy(wav)
        if array.size == 0:
            raise RuntimeError(f"Chatterbox returned no audio for line {index}.")
        logger.info(
            "line %d/%d: %.2fs, %d word(s), %.2fs of audio, "
            "exaggeration=%.2f, language=%s",
            index + 1,
            len(request["lines"]),
            elapsed,
            len(line.split()),
            array.size / float(sample_rate or 1),
            float(exaggeration),
            kwargs.get("language_id", "n/a"),
        )

        path = output_dir / f"line_{index:03d}.wav"
        sf.write(str(path), array, sample_rate)
        files.append(str(path))

    total_seconds = time.perf_counter() - started_all
    logger.info(
        "Generated %d line(s) in %.2fs (load %.2fs, generate %.2fs)",
        len(files),
        load_seconds + total_seconds,
        load_seconds,
        total_seconds,
    )
    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "sample_rate": sample_rate,
                "files": files,
                "model": type(model).__name__,
                "timings": {
                    "load_seconds": round(load_seconds, 3),
                    "line_seconds": [round(value, 3) for value in line_seconds],
                    "total_seconds": round(total_seconds, 3),
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
