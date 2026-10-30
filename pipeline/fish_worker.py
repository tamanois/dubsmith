"""Standalone Fish Speech S2-Pro worker, run in a child process.

Fish pins ``torch==2.8`` and ``transformers==5.19``, which conflict with the
Qwen checkpoint's stack, so it never runs in the parent interpreter.

Protocol matches the other workers: one argument (a JSON request path). The
worker loads the checkpoint, synthesises one WAV per line, and writes
``result.json`` with ``sample_rate`` and ``files``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

#: The weights leave little headroom on a 12 GB card; reduce allocator
#: fragmentation before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

#: Fallback when the checkpoint does not expose a rate.
SAMPLE_RATE = 24_000

#: The DAC codec configuration S2-Pro / Drama 3 ship with.
DECODER_CONFIG = "modded_dac_vq"

#: S2-Pro allocates its KV cache and a max_seq_len x max_seq_len causal mask
#: from the checkpoint config (32768 -> a 1 GiB mask alone). It also reserves
#: 2048 tokens for generation, so the cap must leave prompt + 2048. 4096 keeps
#: the buffers small for line-level work; the 4.56 B weights still need ~24 GB
#: on GPU. Override with FISH_MAX_SEQ_LEN (0 disables the cap).
DEFAULT_MAX_SEQ_LEN = 4096


def _shadow_checkpoint(model_dir: Path, max_seq_len: int) -> Path:
    """A checkpoint directory whose ``config.json`` caps ``max_seq_len``.

    Everything else is symlinked, so the weights are not copied. The loader
    sizes its rotary cache, causal mask and KV cache from this value.
    """
    if max_seq_len <= 0:
        return model_dir
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    def cap(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "max_seq_len" and isinstance(value, int):
                    node[key] = min(value, max_seq_len)
                else:
                    cap(value)
        elif isinstance(node, list):
            for item in node:
                cap(item)

    cap(config)
    shadow = Path(tempfile.mkdtemp(prefix="fish-s2-ctx-"))
    for entry in model_dir.iterdir():
        if entry.name != "config.json":
            (shadow / entry.name).symlink_to(entry.resolve())
    (shadow / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return shadow


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(
            "usage: python -m pipeline.fish_worker REQUEST.json",
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
    logger = logging.getLogger("fish_worker")
    if request.get("verbose"):
        logger.setLevel(logging.DEBUG)

    import soundfile as sf
    import torch
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.models.text2semantic.fish_omni.generate import (
        resolve_drama3_devices,
        resolve_drama3_dtype,
    )
    from fish_speech.models.text2semantic.fish_omni.model import (
        is_fish_omni_checkpoint,
    )
    from fish_speech.models.text2semantic.inference import (
        launch_thread_safe_queue,
        resolve_codec_path,
    )
    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

    max_seq_len = int(os.environ.get("FISH_MAX_SEQ_LEN", DEFAULT_MAX_SEQ_LEN))
    checkpoint = _shadow_checkpoint(Path(request["model_dir"]), max_seq_len)
    if checkpoint != Path(request["model_dir"]):
        logger.info(
            "Loading S2-Pro with a %d-token context (checkpoint: 32768).",
            max_seq_len,
        )
    codec = resolve_codec_path(checkpoint, None)

    device = str(request.get("device") or "cuda")
    if device == "auto" or not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = torch.bfloat16 if device.startswith("cuda") else torch.float32

    # The released S2-Pro is a FishOmni checkpoint: it can shard across several
    # GPUs (24 GB recommended), and its family is "drama3" in the engine API.
    if is_fish_omni_checkpoint(str(checkpoint)):
        devices, decoder_device = resolve_drama3_devices(device)
        precision = resolve_drama3_dtype(precision, [*devices, decoder_device])
        model_family = "drama3"
        logger.info(
            "FishOmni checkpoint: backbone on %s, decoder on %s, %s",
            devices,
            decoder_device,
            precision,
        )
    else:
        decoder_device = device
        model_family = "s2"

    load_started = time.perf_counter()
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=str(checkpoint),
        device=device,
        precision=precision,
        compile=False,
    )
    decoder = load_decoder_model(DECODER_CONFIG, codec, decoder_device)
    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder,
        precision=precision,
        compile=False,
        model_family=model_family,
    )
    load_seconds = time.perf_counter() - load_started
    logger.info("Fish Speech loaded in %.1fs on %s", load_seconds, device)

    reference_audio = Path(request["reference"]).read_bytes()
    reference_text = str(request.get("reference_text") or "")
    if not reference_text:
        logger.warning(
            "No reference transcript; cloning quality may be lower (pass --clone-text)."
        )

    files: list[str] = []
    line_seconds: list[float] = []
    sample_rate = SAMPLE_RATE

    for index, text in enumerate(request["lines"]):
        started = time.perf_counter()
        tts_request = ServeTTSRequest(
            text=text,
            references=[
                ServeReferenceAudio(audio=reference_audio, text=reference_text)
            ],
            format="wav",
            temperature=request.get("temperature"),
            top_p=request.get("top_p"),
            max_new_tokens=request.get("max_new_tokens"),
        )
        audio = None
        for result in engine.inference(tts_request):
            if result.code == "error":
                raise RuntimeError(str(result.error))
            if result.code == "final" and result.audio is not None:
                sample_rate, audio = result.audio
        if audio is None:
            raise RuntimeError("Fish Speech produced no audio.")
        samples = np.ascontiguousarray(
            np.asarray(audio, dtype=np.float32).reshape(-1)
        )
        out_path = output_dir / f"line_{index:03d}.wav"
        sf.write(str(out_path), samples, int(sample_rate))
        files.append(str(out_path))
        line_seconds.append(round(time.perf_counter() - started, 3))
        logger.info(
            "line %d: %.2fs of audio", index, samples.size / int(sample_rate)
        )

    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "sample_rate": int(sample_rate),
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
