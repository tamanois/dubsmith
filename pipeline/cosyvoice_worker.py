"""Standalone CosyVoice 2/3 worker, run in a child process.

CosyVoice and Qwen TTS pin different ``transformers`` versions (4.51.3 vs
4.57.3), and loading CosyVoice with the newer one silently produces garbled
audio. This worker is therefore launched by ``pipeline.cosyvoice`` with a
``PYTHONPATH`` whose first entry is a small transformers-4.51.3 overlay, so the
main environment keeps the version Qwen needs.

The checkpoint is loaded through ``AutoModel``, which picks CosyVoice 2 or 3 from
the config file in the model directory. CosyVoice 3 needs an instruct that ends
with ``<|endofprompt|>`` in *every* request - including a plain clone - because
its LLM asserts on that token before decoding.

Protocol: one argument, a JSON request path. The request names the checkpoint,
its generation, the reference clip, the lines, their instructions, ``fp16`` and
an output directory. The worker writes ``line_000.wav``, ... plus ``result.json``
with ``sample_rate`` and ``files``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

#: Fallback when the checkpoint does not expose ``sample_rate``.
SAMPLE_RATE = 24_000

#: The minimum instruct CosyVoice 3 accepts: it only asserts on the marker.
NEUTRAL_INSTRUCT = "You are a helpful assistant.<|endofprompt|>"


def _normalize_instruct(instruct: str) -> str:
    """CosyVoice expects the instruction to end with this delimiter."""
    text = (instruct or "").strip()
    if text and "<|endofprompt|>" not in text:
        text = f"{text}<|endofprompt|>"
    return text


def _to_numpy(value):
    """One utterance as a flat float32 numpy array."""
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32).reshape(-1))


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(
            "usage: python -m pipeline.cosyvoice_worker REQUEST.json",
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
    logger = logging.getLogger("cosyvoice_worker")
    if request.get("verbose"):
        logger.setLevel(logging.DEBUG)

    import soundfile as sf
    from cosyvoice.cli.cosyvoice import AutoModel

    load_started = time.perf_counter()
    model = AutoModel(
        model_dir=request["model_dir"], fp16=bool(request.get("fp16", False))
    )
    load_seconds = time.perf_counter() - load_started
    sample_rate = int(getattr(model, "sample_rate", SAMPLE_RATE))
    model_type = str(request.get("model_type", "cosyvoice2")).lower()
    logger.info(
        "Loaded %s from %s in %.2fs (sample_rate=%d, fp16=%s, type=%s)",
        type(model).__name__,
        request["model_dir"],
        load_seconds,
        sample_rate,
        bool(request.get("fp16", False)),
        model_type,
    )

    modes = list(request.get("modes") or [])
    prompts = list(request.get("prompts") or [])
    text_frontend = bool(request.get("text_frontend", True))
    logger.info(
        "Request: %d line(s), text_frontend=%s, modes=%s",
        len(request["lines"]),
        text_frontend,
        modes or "auto",
    )

    started_all = time.perf_counter()
    line_seconds: list[float] = []
    files: list[str] = []
    for index, line in enumerate(request["lines"]):
        instruct = (
            request["instructs"][index] if index < len(request["instructs"]) else ""
        )
        mode = modes[index] if index < len(modes) else ""
        prompt = prompts[index] if index < len(prompts) else ""
        instruct = _normalize_instruct(instruct)
        if not mode:
            # Backward compatibility with a request that predates `modes`.
            if instruct:
                mode = "instruct2"
            elif model_type == "cosyvoice3":
                mode = "instruct2"  # neutral instruct; the marker is required
            else:
                mode = "cross_lingual"
        # CosyVoice 3 asserts on <|endofprompt|> before decoding, so a neutral
        # instruct path still needs the placeholder sentence.
        if mode == "instruct2" and not instruct and model_type == "cosyvoice3":
            instruct = NEUTRAL_INSTRUCT
        started = time.perf_counter()
        if mode == "instruct2":
            outputs = model.inference_instruct2(
                line,
                instruct,
                request["reference"],
                stream=False,
                text_frontend=text_frontend,
            )
        elif mode == "zero_shot":
            outputs = model.inference_zero_shot(
                line,
                prompt,
                request["reference"],
                stream=False,
                text_frontend=text_frontend,
            )
        else:  # cross_lingual
            outputs = model.inference_cross_lingual(
                line,
                request["reference"],
                stream=False,
                text_frontend=text_frontend,
            )
        wav = None
        for item in outputs:
            wav = item["tts_speech"]
        elapsed = time.perf_counter() - started
        line_seconds.append(elapsed)
        array = _to_numpy(wav) if wav is not None else np.zeros(0, dtype=np.float32)
        if array.size == 0:
            raise RuntimeError(f"CosyVoice returned no audio for line {index}.")
        logger.info(
            "line %d/%d [%s]: %.2fs, %d word(s), %.2fs of audio, "
            "instruct=%r prompt=%r",
            index + 1,
            len(request["lines"]),
            mode,
            elapsed,
            len(line.split()),
            array.size / float(sample_rate or 1),
            instruct[:80],
            prompt[:80],
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
