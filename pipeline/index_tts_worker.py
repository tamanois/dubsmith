"""Standalone IndexTTS2 worker, run in a child process.

IndexTTS pins ``torch==2.8`` and ``transformers==4.52``, which conflict with the
Qwen checkpoint's stack, so it never runs in the parent interpreter.

Protocol matches the other workers: one argument (a JSON request path). The
worker loads the checkpoint, synthesises one WAV per line, and writes
``result.json`` with ``sample_rate`` and ``files``. Duration fitting is done by
the parent (``pipeline.index_tts``), not here.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

#: Fallback when the written clip cannot be inspected.
SAMPLE_RATE = 22_050


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(
            "usage: python -m pipeline.index_tts_worker REQUEST.json",
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
    logger = logging.getLogger("index_tts_worker")
    if request.get("verbose"):
        logger.setLevel(logging.DEBUG)

    import soundfile as sf
    import torch
    from indextts.infer_v2 import IndexTTS2

    device = str(request.get("device") or "")
    if device in {"", "auto"}:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(request["model_dir"])
    cfg_path = model_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"IndexTTS config not found: {cfg_path}")

    use_fp16 = bool(request.get("use_fp16", True)) and device.startswith("cuda")

    load_started = time.perf_counter()
    tts = IndexTTS2(
        cfg_path=str(cfg_path),
        model_dir=str(model_dir),
        use_fp16=use_fp16,
        device=device,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_qwen_emo=False,
    )
    load_seconds = time.perf_counter() - load_started
    logger.info("IndexTTS2 loaded in %.1fs on %s", load_seconds, device)

    reference = str(request["reference"])
    emotion = request.get("emotion_audio")
    emotion_alpha = float(request.get("emotion_alpha", 1.0))
    verbose = bool(request.get("verbose"))

    files: list[str] = []
    line_seconds: list[float] = []
    for index, text in enumerate(request["lines"]):
        out_path = output_dir / f"line_{index:03d}.wav"
        started = time.perf_counter()
        tts.infer(
            spk_audio_prompt=reference,
            text=text,
            output_path=str(out_path),
            emo_audio_prompt=emotion,
            emo_alpha=emotion_alpha,
            verbose=verbose,
            max_text_tokens_per_segment=120,
        )
        if not out_path.is_file():
            raise RuntimeError(f"IndexTTS2 produced no audio for line {index}.")
        files.append(str(out_path))
        line_seconds.append(round(time.perf_counter() - started, 3))
        logger.info("line %d written to %s", index, out_path)

    sample_rate = SAMPLE_RATE
    if files:
        sample_rate = int(sf.info(files[0]).samplerate)

    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "sample_rate": sample_rate,
                "files": files,
                "timings": {
                    "load_seconds": round(load_seconds, 3),
                    "line_seconds": line_seconds,
                },
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
