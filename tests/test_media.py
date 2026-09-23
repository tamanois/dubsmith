"""Tests for media utilities: probing, command running, loudness, audio I/O."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline import media
from pipeline.errors import DependencyError, MediaError

FFPROBE_PAYLOAD = {
    "streams": [
        {"codec_type": "video", "width": 1920, "height": 1080},
        {"codec_type": "audio", "sample_rate": "48000", "channels": 2},
    ],
    "format": {"duration": "12.5"},
}


class TestRunCommand:
    def test_returns_stdout_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            media.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        )
        assert media.run_command(["echo", "ok"]) == "ok"

    def test_raises_media_error_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            media.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom\nreason"),
        )
        with pytest.raises(MediaError, match="reason"):
            media.run_command(["ffmpeg", "-x"])

    def test_missing_executable_becomes_dependency_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*args: object, **kwargs: object) -> None:
            raise FileNotFoundError("nope")

        monkeypatch.setattr(media.subprocess, "run", boom)
        with pytest.raises(DependencyError, match="not found"):
            media.run_command(["definitely-not-real"])

    def test_timeout_becomes_media_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)

        monkeypatch.setattr(media.subprocess, "run", boom)
        with pytest.raises(MediaError, match="timed out"):
            media.run_command(["ffmpeg"], timeout=1)

    def test_shell_is_never_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def capture(*args: object, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(media.subprocess, "run", capture)
        media.run_command(["ffmpeg", "-i", "weird; rm -rf /"])
        assert captured["shell"] is False
        assert isinstance(captured, dict)


class TestRequireBinary:
    def test_found_binary_returns_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(media.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert media.require_binary("ffmpeg") == "/usr/bin/ffmpeg"

    def test_missing_binary_raises_with_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(media.shutil, "which", lambda name: None)
        with pytest.raises(DependencyError, match="Install ffmpeg"):
            media.require_binary("ffmpeg", hint="Install ffmpeg.")


class TestProbe:
    def test_parses_streams_and_duration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(media, "require_binary", lambda *a, **k: "/usr/bin/ffprobe")
        monkeypatch.setattr(media, "run_command", lambda *a, **k: json.dumps(FFPROBE_PAYLOAD))

        info = media.probe("movie.mp4")
        assert info.duration_s == 12.5
        assert info.has_audio is True
        assert info.has_video is True
        assert info.sample_rate == 48000
        assert info.channels == 2

    def test_audio_only_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {"streams": [{"codec_type": "audio", "sample_rate": "24000", "channels": 1}]}
        monkeypatch.setattr(media, "require_binary", lambda *a, **k: "/usr/bin/ffprobe")
        monkeypatch.setattr(media, "run_command", lambda *a, **k: json.dumps(payload))

        info = media.probe("clip.wav")
        assert info.has_video is False
        assert info.sample_rate == 24000
        assert info.duration_s == 0.0  # absent duration degrades to zero

    def test_duration_na_falls_back_to_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "streams": [{"codec_type": "audio", "sample_rate": "48000", "duration": "3.25"}],
            "format": {"duration": "N/A"},
        }
        monkeypatch.setattr(media, "require_binary", lambda *a, **k: "/usr/bin/ffprobe")
        monkeypatch.setattr(media, "run_command", lambda *a, **k: json.dumps(payload))
        assert media.probe("x.wav").duration_s == 3.25

    def test_malformed_json_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(media, "require_binary", lambda *a, **k: "/usr/bin/ffprobe")
        monkeypatch.setattr(media, "run_command", lambda *a, **k: "not json")
        with pytest.raises(MediaError, match="parse ffprobe"):
            media.probe("x.mp4")


class TestExtractAndMux:
    def test_extract_audio_builds_expected_arguments(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(media, "require_binary", lambda *a, **k: "/usr/bin/ffmpeg")
        monkeypatch.setattr(media, "run_command", lambda cmd, **k: captured.append(list(cmd)))

        out = media.extract_audio(tmp_path / "in.mp4", tmp_path / "out.wav", sample_rate=16000)
        command = captured[0]
        assert "-vn" in command and "-ac" in command
        assert command[command.index("-ar") + 1] == "16000"
        assert command[-1] == str(out)

    def test_mux_audio_copies_video_and_shortest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(media, "require_binary", lambda *a, **k: "/usr/bin/ffmpeg")
        monkeypatch.setattr(media, "run_command", lambda cmd, **k: captured.append(list(cmd)))

        media.mux_audio(tmp_path / "in.mp4", tmp_path / "dub.wav", tmp_path / "out.mp4")
        command = captured[0]
        assert command[command.index("-c:v") + 1] == "copy"
        assert "-shortest" in command


class TestAudioIo:
    def test_roundtrip_preserves_samples(self, tmp_path: Path) -> None:
        samples = np.linspace(-0.5, 0.5, 4800, dtype=np.float32)
        path = media.write_audio(tmp_path / "a.wav", samples, 24000)
        loaded, sample_rate = media.load_audio(path)
        assert sample_rate == 24000
        assert loaded.shape == samples.shape
        assert np.allclose(loaded, samples, atol=1e-4)

    def test_stereo_is_downmixed_to_mono(self, tmp_path: Path) -> None:
        import soundfile as sf

        stereo = np.stack([np.ones(1000, dtype=np.float32), -np.ones(1000, dtype=np.float32)], axis=1)
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, 24000)
        loaded, _ = media.load_audio(path)
        assert loaded.ndim == 1
        # 16-bit quantisation makes the cancel-out approximately, not exactly, zero.
        assert np.allclose(loaded, 0.0, atol=1e-4)

    def test_resample_changes_length(self, tmp_path: Path) -> None:
        samples = np.sin(np.linspace(0, 10, 4800)).astype(np.float32)
        path = media.write_audio(tmp_path / "a.wav", samples, 48000)
        loaded, sample_rate = media.load_audio(path, sample_rate=24000)
        assert sample_rate == 24000
        assert loaded.size == pytest.approx(2400, abs=2)

    def test_resample_noop_when_rates_match(self) -> None:
        samples = np.ones(100, dtype=np.float32)
        assert media.resample(samples, 24000, 24000) is not None
        assert np.array_equal(media.resample(samples, 24000, 24000), samples)


class TestLoudness:
    def test_silence_is_unmeasurable(self) -> None:
        assert media.measure_loudness(np.zeros(48000, dtype=np.float32), 48000) is None

    def test_empty_array_is_unmeasurable(self) -> None:
        assert media.measure_loudness(np.zeros(0, dtype=np.float32), 48000) is None

    def test_too_short_clip_is_unmeasurable(self) -> None:
        # pyloudnorm needs at least one 400 ms gating block.
        assert media.measure_loudness(np.full(100, 0.1, dtype=np.float32), 48000) is None

    def test_loud_clip_has_measurable_loudness(self) -> None:
        t = np.linspace(0, 1.0, 48000, endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        assert media.measure_loudness(tone, 48000) is not None

    def test_quiet_clip_is_boosted(self) -> None:
        t = np.linspace(0, 1.0, 48000, endpoint=False)
        tone = (0.01 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        boosted = media.normalize_loudness(tone, 48000)
        assert np.max(np.abs(boosted)) > np.max(np.abs(tone))

    def test_normalisation_never_clips(self) -> None:
        t = np.linspace(0, 1.0, 48000, endpoint=False)
        tone = (0.9 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = media.normalize_loudness(tone, 48000, target_lufs=0.0)
        assert np.max(np.abs(result)) <= 1.0

    def test_unmeasurable_input_is_returned_unchanged(self) -> None:
        quiet = np.zeros(48000, dtype=np.float32)
        assert np.array_equal(media.normalize_loudness(quiet, 48000), quiet)

    def test_save_normalized_writes_file(self, tmp_path: Path) -> None:
        t = np.linspace(0, 1.0, 48000, endpoint=False)
        tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        out = media.save_normalized(tone, 48000, tmp_path / "n.wav")
        assert out.is_file()


class TestLoudnessReporting:
    """`normalize_loudness_detailed` must describe what it actually delivered."""

    @staticmethod
    def tone(amplitude: float, seconds: float = 2.0) -> np.ndarray:
        t = np.linspace(0, seconds, int(seconds * 48000), endpoint=False)
        return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def test_reports_target_and_achieved(self) -> None:
        result = media.normalize_loudness_detailed(self.tone(0.05), 48000)
        assert result.target_lufs == media.TARGET_LUFS
        assert result.measured_lufs is not None
        assert result.gain_db > 0
        # The applied gain must actually move the result towards the target.
        assert result.achieved_lufs is not None
        assert result.achieved_lufs > result.measured_lufs

    def test_quieter_input_gets_more_gain(self) -> None:
        quiet = media.normalize_loudness_detailed(self.tone(0.02), 48000)
        loud = media.normalize_loudness_detailed(self.tone(0.2), 48000)
        assert quiet.gain_db > loud.gain_db

    def test_peak_limiting_is_flagged_when_target_is_unreachable(self) -> None:
        # Quiet body with a single near-full-scale spike: a very high crest
        # factor, so the target cannot be reached without clipping.
        spiky = self.tone(0.01)
        spiky[0] = 0.99
        result = media.normalize_loudness_detailed(spiky, 48000)
        assert result.peak_limited is True
        assert result.achieved_lufs is not None
        assert result.achieved_lufs < result.target_lufs
        assert float(np.max(np.abs(result.samples))) <= 1.0

    def test_no_limiting_is_flagged_for_easy_material(self) -> None:
        result = media.normalize_loudness_detailed(self.tone(0.9), 48000, target_lufs=-30.0)
        assert result.peak_limited is False

    def test_unmeasurable_input_is_passed_through(self) -> None:
        silent = np.zeros(48000, dtype=np.float32)
        result = media.normalize_loudness_detailed(silent, 48000)
        assert result.measured_lufs is None
        assert result.achieved_lufs is None
        assert result.gain_db == 0.0
        assert result.peak_limited is False
        assert np.array_equal(result.samples, silent)

    def test_samples_never_clip(self) -> None:
        result = media.normalize_loudness_detailed(self.tone(0.8), 48000, target_lufs=0.0)
        assert float(np.max(np.abs(result.samples))) <= 1.0

    def test_normalize_loudness_still_returns_plain_array(self) -> None:
        out = media.normalize_loudness(self.tone(0.05), 48000)
        assert isinstance(out, np.ndarray)
        assert out.ndim == 1
