"""Standalone speech synthesis, shared by the CLI and the HTTP API.

``dub.py say`` and the web UI must speak text with the same backend rules, the
same prosody precedence and the same file layout. This module owns that flow; the
CLI is a thin adapter over :func:`render_speech`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import backends, prosody, tts
from .capabilities import SAY_BACKENDS, say_prosody_conflict
from .config import NAME_TO_CODE, Settings, normalize_language, pick_dtype
from .errors import ConfigError, TTSError
from .logging_utils import job_logger, timed
from .media import load_audio, write_audio
from .models import model_fingerprint, resolve_model
from .provenance import (
    LineProvenance,
    NullRecorder,
    ProvenanceRecorder,
    describe_transformations,
    timed_provenance_path,
)
from .voices import (
    find_prompt,
    load_prompt,
    load_voice,
    save_prompt,
    save_voice,
    slugify_label,
)

__all__ = ["SayResult", "SaySpec", "render_speech"]

Event = Callable[[str, dict[str, Any]], None]
Log = logging.Logger | logging.LoggerAdapter

_BACKENDS = tuple(SAY_BACKENDS)


@dataclass
class SayResult:
    """What a speech-rendering run produced."""

    name: str
    backend: str
    language: str | None
    output: Path
    lines_dir: Path
    line_count: int
    sample_rate: int
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Wrote {self.line_count} line(s) to {self.lines_dir} "
            f"(combined: {self.output})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "language": self.language,
            "output": str(self.output),
            "lines_dir": str(self.lines_dir),
            "line_count": self.line_count,
            "sample_rate": self.sample_rate,
            "seconds": round(self.seconds, 3),
            "warnings": list(self.warnings),
        }


class SaySpec:
    """Validated request for one standalone speech-rendering run.

    Kept as a small pydantic-free class so importing this module never pulls the
    pydantic model graph. The API validates the same rules through
    :func:`validate_say_spec`.
    """

    def __init__(
        self,
        *,
        text: str,
        backend: str = "qwen",
        voice: str | None = None,
        design: str | None = None,
        clone: Path | None = None,
        clone_text: str | None = None,
        speaker: str | None = None,
        language: str | None = None,
        save_as: str | None = None,
        output: Path | None = None,
        output_root: Path = Path("output"),
        device: str = "auto",
        dtype: str = "auto",
        force: bool = False,
        provenance: bool = True,
        prosody: Sequence[str] | None = None,
        emotions: Sequence[str] | None = None,
        prosody_plan: str = "off",
        style_instructs: Sequence[str] | None = None,
        style_plan: str = "off",
        style_plan_scope: str = "per-line",
        plan_chunk: int | None = None,
        chatterbox_exaggeration: float | None = None,
        chatterbox_cfg_weight: float | None = None,
        chatterbox_temperature: float | None = None,
        fish_temperature: float | None = None,
        fish_top_p: float | None = None,
        fish_quantize: str | None = None,
        fish_tag_mode: str = "keep",
    ) -> None:
        self.text = text
        self.backend = backend
        self.voice = voice
        self.design = design
        self.clone = Path(clone) if clone is not None else None
        self.clone_text = clone_text
        self.speaker = speaker
        self.language = language
        self.save_as = save_as
        self.output = Path(output) if output is not None else None
        self.output_root = Path(output_root)
        self.device = device
        self.dtype = dtype
        self.force = force
        self.provenance = provenance
        self.prosody = list(prosody) if prosody else None
        self.emotions = list(emotions) if emotions else None
        self.prosody_plan = prosody_plan
        self.style_instructs = list(style_instructs) if style_instructs else None
        self.style_plan = style_plan
        self.style_plan_scope = style_plan_scope
        self.plan_chunk = plan_chunk
        self.chatterbox_exaggeration = chatterbox_exaggeration
        self.chatterbox_cfg_weight = chatterbox_cfg_weight
        self.chatterbox_temperature = chatterbox_temperature
        self.fish_temperature = fish_temperature
        self.fish_top_p = fish_top_p
        self.fish_quantize = fish_quantize
        self.fish_tag_mode = fish_tag_mode
        validate_say_spec(self)


def validate_say_spec(spec: SaySpec) -> None:
    """Reject source/backend combinations the parser cannot express."""
    if spec.backend not in _BACKENDS:
        raise ConfigError(
            f"Unknown backend {spec.backend!r}. Use: {', '.join(_BACKENDS)}"
        )
    if not spec.text or not spec.text.strip():
        raise ConfigError("Text cannot be empty.")
    if spec.style_instructs and spec.backend != "customvoice":
        raise ConfigError(
            "--style-instruct is only valid with --backend customvoice; the other "
            "backends take --prosody/--emotion (and --design already accepts a "
            "free-form prompt)."
        )
    if spec.style_plan != "off" and spec.backend != "customvoice":
        raise ConfigError(
            "--style-plan is only valid with --backend customvoice; the other "
            "backends use --prosody-plan llm for structured prosody."
        )
    if spec.style_plan_scope != "per-line" and spec.style_plan != "llm":
        raise ConfigError("--style-plan-scope only applies with --style-plan llm.")
    if spec.fish_tag_mode != "keep" and spec.backend != "fish":
        raise ConfigError("--fish-tag-mode is only valid with --backend fish.")
    if spec.fish_quantize is not None and spec.backend != "fish":
        raise ConfigError("--fish-quantize is only valid with --backend fish.")
    if spec.fish_temperature is not None and spec.backend != "fish":
        raise ConfigError("--fish-temperature is only valid with --backend fish.")
    if spec.fish_top_p is not None and spec.backend != "fish":
        raise ConfigError("--fish-top-p is only valid with --backend fish.")
    conflict = say_prosody_conflict(
        spec.backend,
        prosody=spec.prosody,
        emotions=spec.emotions,
        prosody_plan=spec.prosody_plan,
    )
    if conflict:
        raise ConfigError(conflict)
    if spec.backend == "customvoice":
        if not spec.speaker:
            raise ConfigError(
                "--backend customvoice needs --speaker (a preset voice, e.g. "
                "'Vivian' or 'Ryan')."
            )
        if spec.voice or spec.design is not None or spec.clone is not None:
            raise ConfigError(
                "--backend customvoice uses a preset --speaker and does not take "
                "--voice/--design/--clone."
            )
        return
    if spec.speaker:
        raise ConfigError("--speaker is only valid with --backend customvoice.")
    sources = [bool(spec.voice), spec.design is not None, spec.clone is not None]
    if sum(sources) != 1:
        raise ConfigError(
            "One of --voice, --design or --clone is required (unless --backend "
            "customvoice, which needs --speaker)."
        )
    if spec.fish_tag_mode == "keep" and spec.backend == "fish":
        for index, line in enumerate(_say_lines(spec.text), start=1):
            token = prosody.fish_structural_token(line)
            if token:
                raise ConfigError(
                    f"line {index} contains {token!r}, which Fish reserves for its "
                    "own prompt; remove it."
                )


def _say_lines(text: str) -> list[str]:
    """Split text into lines, treating a literal ``\\n`` as a break."""
    normalized = str(text or "").replace("\\n", "\n")
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def _parse_emotions(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    flat = [
        part.strip() for value in values for part in value.split(",") if part.strip()
    ]
    return flat or None


def _one_per_line(
    values: Sequence[str] | None, lines: Sequence[str], flag: str
) -> list[str] | None:
    if values is None:
        return None
    items = list(values)
    if len(items) == 1:
        return items * len(lines)
    if len(items) != len(lines):
        raise ConfigError(
            f"{flag}: {len(items)} value(s) for {len(lines)} line(s); give one per "
            "line, or a single one to apply everywhere."
        )
    return items


def _say_name(spec: SaySpec) -> str:
    if spec.voice:
        return slugify_label(spec.voice)
    if spec.clone is not None:
        return slugify_label(Path(spec.clone).stem)
    if spec.speaker:
        return slugify_label(f"customvoice-{spec.speaker}")
    return slugify_label(spec.save_as or "design")


def _emit(event: Event | None, kind: str, **payload: Any) -> None:
    if event is None:
        return
    try:
        event(kind, payload)
    except Exception:  # noqa: BLE001 - an observer must not fail the run
        logging.getLogger(__name__).debug(
            "Say observer failed for %s", kind, exc_info=True
        )


def _say_plan_source(spec: SaySpec) -> str:
    if spec.backend == "customvoice":
        if spec.style_plan == "llm":
            return "llm"
        if spec.style_instructs or spec.prosody or spec.emotions:
            return "manual"
        return "off"
    if spec.fish_tag_mode == "llm":
        return "llm"
    if spec.prosody_plan == "llm":
        return "llm"
    if spec.prosody or spec.emotions:
        return "manual"
    if spec.backend == "fish":
        return "manual"
    return "off"


def _write_say_provenance(recorder: Any, name: str, output_root: Path) -> None:
    recorder.write(timed_provenance_path(output_root / "say" / name / "provenance"))


def _write_qwen_say_provenance(
    name: str,
    language: str | None,
    mode: str,
    *,
    output_root: Path,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    recorder = ProvenanceRecorder("say", "qwen", language=language, plan_source="off")
    recorder.record_output(mode=mode)
    recorder.write(
        timed_provenance_path(output_root / "say" / name / "provenance"),
        supported=False,
    )


def _resolve_prosody_specs(
    lines: Sequence[str],
    *,
    emotions: Sequence[str] | None,
    values: Sequence[str] | None,
    plan: str,
    settings: Settings,
    backend: str,
    chunk_size: int | None = None,
    recorder: Any | None = None,
) -> list[Any]:
    specs = [prosody.ProsodySpec() for _ in lines]
    if plan == "llm":
        planner = prosody.ProsodyPlanner(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
            backend=backend,
            chunk_size=chunk_size or settings.llm_plan_chunk_size,
            recorder=recorder,
        )
        specs = planner.plan(lines)
    if values:
        parsed = _one_per_line(
            [prosody.parse_prosody_spec(value) for value in values], lines, "--prosody"
        )
        assert parsed is not None
        specs = [base.merged_with(override) for base, override in zip(specs, parsed)]
    if emotions:
        chosen = _one_per_line(emotions, lines, "--emotion")
        assert chosen is not None
        specs = [
            base.merged_with(prosody.ProsodySpec(emotion=emotion))
            for base, emotion in zip(specs, chosen)
        ]
    return specs


def _neural_engine(spec: SaySpec, models_dir: Path) -> Any:
    return backends.build_expressive_engine(
        spec.backend,
        models_dir,
        spec.device,
        exaggeration=spec.chatterbox_exaggeration,
        cfg_weight=spec.chatterbox_cfg_weight,
        temperature=spec.chatterbox_temperature,
        fish_temperature=spec.fish_temperature,
        fish_top_p=spec.fish_top_p,
        fish_quantize=spec.fish_quantize or "int8",
        resolve=resolve_model,
    )


def _install_saved_prompt(
    engine: tts.TTSEngine,
    saved: Any,
    models_dir: Path,
    settings: Settings,
    log: Log,
) -> None:
    fingerprint = model_fingerprint("tts_base", models_dir)
    path = find_prompt(saved.directory, "qwen", fingerprint, ref_text=saved.ref_text)
    if path is not None:
        log.info("Loaded the stored prompt for %s", saved.label)
        engine.prepare_prompt_from_data(load_prompt(path), seconds=saved.seconds)
        return
    if saved.cards:
        log.info(
            "No usable qwen card for this model; %d existing card(s) are kept.",
            len(saved.cards),
        )
    engine.prepare_prompt_from_file(saved.wav, saved.ref_text)
    if not settings.voices_prompt_cache:
        return
    try:
        cached = save_prompt(
            saved.directory, engine.prompt_to_data(fingerprint=fingerprint)
        )
    except (OSError, TTSError) as exc:
        log.warning("Could not cache the prompt for %s: %s", saved.label, exc)
        return
    log.info("Cached the prompt for %s at %s", saved.label, cached)


def _write_single(
    samples: np.ndarray,
    sample_rate: int,
    *,
    spec: SaySpec,
    name: str,
) -> Path:
    """Write one Qwen clip where the CLI has always written it."""
    target = (
        spec.output
        if spec.output is not None
        else spec.output_root / "say" / f"{slugify_label(name)}.wav"
    )
    return write_audio(target, samples, sample_rate)


def _write_lines(
    rendered: Sequence[np.ndarray],
    sample_rate: int,
    *,
    spec: SaySpec,
    name: str,
) -> tuple[Path, Path]:
    out_dir = spec.output_root / "say" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, wav in enumerate(rendered):
        write_audio(out_dir / f"line_{index:03d}.wav", wav, sample_rate)
    combined = (
        np.concatenate(rendered) if rendered else np.zeros(0, dtype=np.float32)
    )
    combined_path = spec.output if spec.output is not None else out_dir / "speech.wav"
    write_audio(combined_path, combined, sample_rate)
    return out_dir, combined_path


def render_speech(
    spec: SaySpec,
    settings: Settings | None = None,
    *,
    models_dir: Path | None = None,
    voices_root: Path | None = None,
    event: Event | None = None,
    log: Log | None = None,
) -> SayResult:
    """Render one ``say`` request and return its artifacts."""
    settings = settings or Settings.from_env()
    models_root = Path(models_dir or settings.models_dir)
    root = Path(voices_root or settings.voices_dir)
    logger = log or job_logger("dub.say", f"say-{_say_name(spec)}")
    lines = _say_lines(spec.text)
    if not lines:
        raise ConfigError("Text contains no lines to speak.")
    name = _say_name(spec)
    emotions = _parse_emotions(spec.emotions)
    started = time.perf_counter()
    _emit(event, "stage_start", stage="say", backend=spec.backend)

    if spec.backend == "customvoice":
        result = _render_customvoice(
            spec, settings, models_root, name, lines, emotions, logger
        )
    elif spec.backend == "qwen":
        result = _render_qwen(
            spec, settings, models_root, root, name, lines, logger
        )
    else:
        result = _render_expressive(
            spec, settings, models_root, root, name, lines, emotions, logger
        )
    result.seconds = time.perf_counter() - started
    _emit(event, "stage_end", stage="say", seconds=round(result.seconds, 3))
    return result


def _render_qwen(
    spec: SaySpec,
    settings: Settings,
    models_dir: Path,
    root: Path,
    name: str,
    lines: Sequence[str],
    log: Log,
) -> SayResult:
    if spec.voice:
        saved = load_voice(root, spec.voice)
        language = normalize_language(spec.language) if spec.language else saved.language
        has_text = bool(saved.ref_text.strip())
        log.info("Qwen: cloning from saved voice %s (%s)", saved.label, language)
        engine = tts.TTSEngine(
            resolve_model("tts_base", models_dir),
            device=spec.device,
            dtype=pick_dtype(spec.device, spec.dtype),
            x_vector_only=not has_text,
        )
        with timed(log, "Prepared the clone prompt"):
            _install_saved_prompt(engine, saved, models_dir, settings, log)
        with timed(log, "Synthesised the line", lines=1, backend="qwen"):
            rendered = engine.synthesize([spec.text], language=language)
        speech = rendered[0] if rendered else None
        if speech is None:
            raise TTSError("The model returned no audio.")
        out = _write_single(
            speech, engine.sample_rate, spec=spec, name=saved.label
        )
        _write_qwen_say_provenance(
            name,
            language,
            "voice",
            output_root=spec.output_root,
            enabled=spec.provenance,
        )
        return SayResult(name, "qwen", language, out, out.parent, 1, engine.sample_rate)

    if spec.language is None:
        raise ConfigError(
            "--lang is required when the voice is described or cloned inline."
        )
    language = normalize_language(spec.language)
    if language == "Auto":
        raise ConfigError("language must be explicit, not 'auto'.")
    anchor_text = _anchor_text(language)

    if spec.design is not None:
        engine = tts.VoiceDesignEngine(
            resolve_model("tts_voice_design", models_dir),
            device=spec.device,
            dtype=pick_dtype(spec.device, spec.dtype),
        )
        speech, sample_rate = engine.generate(spec.text, spec.design, language)
        if spec.save_as:
            samples, anchor_rate = engine.generate(anchor_text, spec.design, language)
            save_voice(
                root,
                spec.save_as,
                samples,
                anchor_rate,
                language=language,
                source="design",
                ref_text=anchor_text,
                instruct=spec.design,
                force=spec.force,
                provenance={
                    "backend": "voice-design",
                    "model": str(resolve_model("tts_voice_design", models_dir)),
                    "text_source": "anchor",
                    "plan_source": "manual",
                },
            )
        out_name = spec.save_as or "design"
        out = _write_single(speech, sample_rate, spec=spec, name=out_name)
        _write_qwen_say_provenance(
            name,
            language,
            "design",
            output_root=spec.output_root,
            enabled=spec.provenance,
        )
        return SayResult(
            out_name, "qwen", language, out, out.parent, 1, sample_rate
        )

    clone_text = (spec.clone_text or "").strip()
    engine = tts.TTSEngine(
        resolve_model("tts_base", models_dir),
        device=spec.device,
        dtype=pick_dtype(spec.device, spec.dtype),
        x_vector_only=not clone_text,
    )
    if not clone_text:
        log.warning(
            "No clone transcript: cloning in speaker-only (x-vector) mode. "
            "Similarity is lower than ICL; pass the clip's transcript for a closer "
            "match."
        )
    with timed(log, "Prepared the clone prompt"):
        engine.prepare_prompt_from_file(spec.clone, clone_text)
    with timed(log, "Synthesised the line", lines=1, backend="qwen"):
        rendered = engine.synthesize([spec.text], language=language)
    speech = rendered[0] if rendered else None
    if speech is None:
        raise TTSError("The model returned no audio.")
    if spec.save_as:
        samples, sample_rate = load_audio(spec.clone)
        save_voice(
            root,
            spec.save_as,
            samples,
            sample_rate,
            language=language,
            source="clone",
            ref_text=clone_text,
            prompt=engine.prompt_to_data(
                fingerprint=model_fingerprint("tts_base", models_dir)
            ),
            force=spec.force,
            provenance={
                "backend": "voice-clone",
                "model": str(resolve_model("tts_base", models_dir)),
                "ref_audio": str(spec.clone),
                "text_source": "clone-text" if clone_text else "none",
                "fingerprint": model_fingerprint("tts_base", models_dir),
                "plan_source": "manual",
            },
        )
    out_name = spec.save_as or "clone"
    out = _write_single(
        speech, engine.sample_rate, spec=spec, name=out_name
    )
    _write_qwen_say_provenance(
        name,
        language,
        "clone",
        output_root=spec.output_root,
        enabled=spec.provenance,
    )
    return SayResult(
        out_name, "qwen", language, out, out.parent, 1, engine.sample_rate
    )


def _render_customvoice(
    spec: SaySpec,
    settings: Settings,
    models_dir: Path,
    name: str,
    lines: Sequence[str],
    emotions: list[str] | None,
    log: Log,
) -> SayResult:
    if spec.language is None:
        raise ConfigError("language is required with backend 'customvoice'.")
    language = normalize_language(spec.language)
    if language == "Auto":
        raise ConfigError("language must be explicit, not 'auto'.")
    recorder = (
        ProvenanceRecorder(
            "say", "customvoice", language=language, plan_source=_say_plan_source(spec)
        )
        if spec.provenance
        else NullRecorder()
    )
    if spec.style_instructs:
        if spec.style_plan != "off":
            log.warning("--style-instruct overrides --style-plan for this run.")
        if spec.prosody or emotions or spec.prosody_plan != "off":
            log.warning(
                "--style-instruct overrides --prosody/--emotion/--prosody-plan "
                "for this run."
            )
        instructs = _one_per_line(
            [value.strip() for value in spec.style_instructs],
            lines,
            "--style-instruct",
        )
        assert instructs is not None
        specs = None
        mode = "verbatim"
    elif spec.style_plan == "llm":
        if spec.prosody or emotions or spec.prosody_plan != "off":
            log.warning(
                "--style-plan overrides --prosody/--emotion/--prosody-plan for "
                "this run."
            )
        planner = prosody.StylePlanner(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
            chunk_size=spec.plan_chunk or settings.llm_plan_chunk_size,
            backend="customvoice",
            recorder=recorder,
        )
        if spec.style_plan_scope == "whole":
            instruction = planner.plan_whole(
                lines, speaker=spec.speaker, language=language
            )
            instructs = [instruction] * len(lines)
        else:
            instructs = planner.plan_lines(
                lines, speaker=spec.speaker, language=language
            )
        specs = None
        mode = "verbatim-llm"
    else:
        specs = _resolve_prosody_specs(
            lines,
            emotions=emotions,
            values=spec.prosody,
            plan=spec.prosody_plan,
            settings=settings,
            backend="customvoice",
            chunk_size=spec.plan_chunk,
            recorder=recorder,
        )
        instructs = [prosody.customvoice_instruct(item) for item in specs]
        mode = "rendered"
    log.info("CustomVoice instruction mode: %s", mode)

    engine = tts.CustomVoiceEngine(
        resolve_model("tts_custom_voice", models_dir),
        device=spec.device,
        dtype=pick_dtype(spec.device, spec.dtype),
    )
    speakers = engine.speakers()
    if speakers and spec.speaker.lower() not in {n.lower() for n in speakers}:
        raise ConfigError(
            f"Unknown speaker {spec.speaker!r}. Available: " + ", ".join(speakers)
        )
    rendered: list[np.ndarray] = []
    sample_rate = 0
    with timed(
        log, f"Synthesised {len(lines)} line(s)", lines=len(lines),
        backend="customvoice",
    ):
        for text, instruct in zip(lines, instructs):
            samples, sample_rate = engine.generate(
                text, spec.speaker, instruct, language
            )
            rendered.append(samples)
    out_dir, combined = _write_lines(
        rendered, sample_rate, spec=spec, name=name
    )
    for index, (text, instruct) in enumerate(zip(lines, instructs)):
        record = {
            "channel": "customvoice",
            "text": text,
            "instruct": instruct,
            "speaker": spec.speaker,
        }
        item = specs[index].model_dump() if specs is not None else None
        recorder.record_line(
            LineProvenance(
                key=index,
                text_planned=text,
                text_spoken=text,
                overrides={
                    "style_instruct": spec.style_instructs,
                    "prosody": spec.prosody,
                    "emotion": emotions,
                },
                spec=item,
                rendered=record,
                transformations=describe_transformations(
                    text, text, channel="customvoice", spec=item, instruct=instruct
                ),
            )
        )
    _write_say_provenance(recorder, name, spec.output_root)
    from .config import release_gpu_memory

    release_gpu_memory()
    return SayResult(
        name, "customvoice", language, combined, out_dir, len(rendered),
        sample_rate,
    )


def _resolve_expressive_reference(
    spec: SaySpec, root: Path, log: Log
) -> tuple[Path, str, str | None, str | None]:
    """Return ``(reference, name, language, ref_text)`` for an expressive backend."""
    if spec.voice:
        saved = load_voice(root, spec.voice)
        language = normalize_language(spec.language) if spec.language else saved.language
        if saved.cards:
            log.info(
                "%s: using voice.wav directly (stored cards are Qwen-only)",
                spec.backend,
            )
        return saved.wav, saved.label, language, saved.ref_text or None
    if spec.clone is not None:
        language = normalize_language(spec.language) if spec.language else None
        if spec.backend == "chatterbox" and language is None:
            raise ConfigError(
                "language is required with backend 'chatterbox' when the voice is "
                "cloned inline."
            )
        return (
            Path(spec.clone),
            slugify_label(Path(spec.clone).stem),
            language,
            (spec.clone_text or "").strip() or None,
        )
    raise ConfigError(f"backend {spec.backend!r} needs a saved voice or a clone clip.")


def _render_expressive(
    spec: SaySpec,
    settings: Settings,
    models_dir: Path,
    root: Path,
    name: str,
    lines: Sequence[str],
    emotions: list[str] | None,
    log: Log,
) -> SayResult:
    if spec.design is not None:
        raise ConfigError(
            f"backend {spec.backend!r} cannot be combined with an inline design."
        )
    if spec.save_as:
        raise ConfigError(f"save_as is not supported with backend {spec.backend!r}.")
    reference, name, language, ref_text = _resolve_expressive_reference(
        spec, root, log
    )
    recorder = (
        ProvenanceRecorder(
            "say", spec.backend, language=language, plan_source=_say_plan_source(spec)
        )
        if spec.provenance
        else NullRecorder()
    )
    trace: list[dict[str, Any]] = []
    specs: list[prosody.ProsodySpec] | None = None
    tagged: list[str] | None = None
    engine = _neural_engine(spec, models_dir)
    log.info(
        "Backend %s: reference=%s language=%s lines=%d",
        spec.backend,
        reference,
        language or "auto",
        len(lines),
    )
    if spec.backend == "fish" and spec.fish_tag_mode == "llm":
        planner = prosody.FishInlinePlanner(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
            chunk_size=spec.plan_chunk or settings.llm_plan_chunk_size,
            recorder=recorder,
        )
        tagged = planner.plan_lines(lines, language=language)
        spoken = tagged
    elif spec.backend == "fish":
        checked: list[str] = []
        for index, text in enumerate(lines, start=1):
            for problem in prosody.validate_fish_inline_text(text):
                log.warning("line %d: %s", index, problem)
            if not prosody.strip_fish_tags(text):
                if not checked:
                    log.warning("dropping a first line that is only tags: %r", text)
                    continue
                log.warning("line %d is only tags; merging it into the line before.", index)
                checked[-1] = f"{checked[-1]} {text}"
                continue
            checked.append(text)
        if not checked:
            raise ConfigError("The text contains only Fish tags, nothing to speak.")
        spoken = checked
    else:
        specs = _resolve_prosody_specs(
            lines,
            emotions=emotions,
            values=spec.prosody,
            plan=spec.prosody_plan,
            settings=settings,
            backend=spec.backend,
            chunk_size=spec.plan_chunk,
            recorder=recorder,
        )
        spoken = list(lines)

    with timed(
        log, f"Synthesised {len(spoken)} line(s)", lines=len(spoken),
        backend=spec.backend,
    ):
        if spec.backend == "fish":
            rendered = engine.synthesize(
                spoken,
                reference=reference,
                ref_text=ref_text,
                language=language,
                trace=trace,
            )
        elif spec.backend == "chatterbox":
            rendered = engine.synthesize(
                spoken,
                reference=reference,
                prosody=specs,
                language=NAME_TO_CODE.get(language or ""),
                trace=trace,
            )
        else:
            rendered = engine.synthesize(
                spoken,
                reference=reference,
                prosody=specs,
                ref_text=ref_text,
                language=language,
                trace=trace,
            )
    out_dir, combined = _write_lines(
        rendered, engine.sample_rate, spec=spec, name=name
    )
    for index, planned in enumerate(lines):
        record = trace[index] if index < len(trace) else None
        if record:
            text_spoken = str(record.get("text"))
        else:
            text_spoken = spoken[index] if index < len(spoken) else planned
        item = specs[index].model_dump() if specs is not None else None
        recorder.record_line(
            LineProvenance(
                key=index,
                text_planned=planned,
                text_spoken=text_spoken,
                overrides={"prosody": spec.prosody, "emotion": emotions},
                spec=item,
                rendered=record,
                transformations=describe_transformations(
                    planned,
                    text_spoken,
                    channel=str(record.get("channel")) if record else spec.backend,
                    spec=item,
                    instruct=record.get("instruct") if record else None,
                    exaggeration=record.get("exaggeration") if record else None,
                ),
            )
        )
    _write_say_provenance(recorder, name, spec.output_root)
    from .config import release_gpu_memory

    release_gpu_memory()
    return SayResult(
        name, spec.backend, language, combined, out_dir, len(rendered),
        engine.sample_rate,
    )


def _anchor_text(language: str) -> str:
    from .anchors import ANCHOR_TEXT

    try:
        return ANCHOR_TEXT[language]
    except KeyError:
        raise ConfigError(
            f"No built-in passage for {language!r}; pick a supported language."
        ) from None
