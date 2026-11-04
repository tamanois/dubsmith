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

    def test_mux_audio_caps_at_video_duration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(media, "require_binary", lambda *a, **k: "/usr/bin/ffmpeg")
        monkeypatch.setattr(media, "run_command", lambda cmd, **k: captured.append(list(cmd)))
        monkeypatch.setattr(
            media,
            "probe",
            lambda *a, **k: media.MediaInfo(
                duration_s=12.5, has_audio=True, has_video=True
            ),
        )

        media.mux_audio(tmp_path / "in.mp4", tmp_path / "dub.wav", tmp_path / "out.mp4")
        command = captured[0]
        assert command[command.index("-c:v") + 1] == "copy"
        # `-shortest` would end at the SRT's last cue start and truncate the
        # video, so the mux is capped at the video duration instead.
        assert "-shortest" not in command
        assert command[command.index("-t") + 1] == "12.500"


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
    def test_unreadable_file_raises_a_media_error(self, tmp_path: Path) -> None:
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"not actually audio")

        with pytest.raises(MediaError, match="pass it with --video"):
            media.load_audio(path)
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
    """The normaliser must reach the target, not surrender to the loudest peak."""

    @staticmethod
    def tone(amplitude: float, seconds: float = 2.0) -> np.ndarray:
        t = np.linspace(0, seconds, int(seconds * 48000), endpoint=False)
        return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    @staticmethod
    def peaky(seconds: float = 2.0) -> np.ndarray:
        """Quiet speech-like body with sparse near-full-scale transients."""
        t = np.linspace(0, seconds, int(seconds * 48000), endpoint=False)
        body = (0.03 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        for position in range(0, body.size, 8000):
            body[position] = 0.999  # isolated spike every ~0.17 s
        return body

    def test_reports_target_and_achieved(self) -> None:
        result = media.normalize_loudness_detailed(self.tone(0.05), 48000)
        assert result.target_lufs == media.TARGET_LUFS
        assert result.measured_lufs is not None
        assert result.achieved_lufs is not None
        assert result.achieved_lufs > result.measured_lufs

    def test_reaches_the_target_on_peaky_material(self) -> None:
        """Regression: a handful of transients used to block all gain."""
        source = self.peaky()
        result = media.normalize_loudness_detailed(source, 48000, target_lufs=-16.0)
        assert result.achieved_lufs is not None
        # Previously this sat ~7 dB below target because the guard allowed 0 dB.
        assert result.achieved_lufs == pytest.approx(-16.0, abs=1.0)
        assert result.peak_limited is True

    def test_quieter_input_gets_more_gain(self) -> None:
        quiet = media.normalize_loudness_detailed(self.tone(0.02), 48000)
        loud = media.normalize_loudness_detailed(self.tone(0.2), 48000)
        assert quiet.gain_db > loud.gain_db

    def test_unmeasurable_input_is_passed_through(self) -> None:
        silent = np.zeros(48000, dtype=np.float32)
        result = media.normalize_loudness_detailed(silent, 48000)
        assert result.measured_lufs is None
        assert result.achieved_lufs is None
        assert result.gain_db == 0.0
        assert result.peak_limited is False
        assert np.array_equal(result.samples, silent)

    def test_samples_never_clip(self) -> None:
        result = media.normalize_loudness_detailed(self.peaky(), 48000, target_lufs=0.0)
        assert float(np.max(np.abs(result.samples))) <= media.LIMIT_CEILING + 1e-6

    def test_normalize_loudness_still_returns_plain_array(self) -> None:
        out = media.normalize_loudness(self.tone(0.05), 48000)
        assert isinstance(out, np.ndarray)
        assert out.ndim == 1


class TestLimitPeaks:
    def test_accepts_a_pure_python_sequence_of_levels(self) -> None:
        # 1 kHz-ish tone at 0.5 plus a spike; the spike must come down.
        t = np.linspace(0, 1.0, 48000, endpoint=False)
        signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        signal[100] = 1.4

        limited = media.limit_peaks(signal, 48000, ceiling=0.98)
        assert float(np.max(np.abs(limited))) <= 0.98 + 1e-6

    def test_leaves_audio_below_the_ceiling_untouched(self) -> None:
        signal = np.full(4800, 0.5, dtype=np.float32)
        assert np.array_equal(media.limit_peaks(signal, 48000, ceiling=0.98), signal)

    def test_preserves_the_body_of_the_signal(self) -> None:
        """Peaks are reduced without dragging the whole program down."""
        t = np.linspace(0, 1.0, 48000, endpoint=False)
        signal = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        signal[500] = 1.2

        limited = media.limit_peaks(signal, 48000, ceiling=0.98)
        # Away from the spike, level is essentially unchanged.
        body = limited[20000:30000]
        original = signal[20000:30000]
        assert np.allclose(body, original, atol=0.02)

    def test_handles_silence_and_empty_input(self) -> None:
        assert media.limit_peaks(np.zeros(0, dtype=np.float32), 48000).size == 0
        silent = np.zeros(1000, dtype=np.float32)
        assert np.array_equal(media.limit_peaks(silent, 48000), silent)

    def test_single_spike_is_contained(self) -> None:
        signal = np.zeros(48000, dtype=np.float32)
        signal[24000] = 5.0
        limited = media.limit_peaks(signal, 48000, ceiling=0.5)
        assert float(np.max(np.abs(limited))) <= 0.5 + 1e-6

    def test_rejects_invalid_ceiling(self) -> None:
        signal = np.ones(10, dtype=np.float32)
        assert np.array_equal(media.limit_peaks(signal, 48000, ceiling=0.0), signal)


class TestMovingAverage:
    def test_smooths_a_step(self) -> None:
        values = np.concatenate([np.zeros(100), np.ones(100)]).astype(np.float32)
        smoothed = media.moving_average(values, 11)
        assert smoothed.size == values.size
        assert 0.0 < float(smoothed[100]) < 1.0  # transition is gradual

    def test_window_larger_than_signal_is_safe(self) -> None:
        values = np.ones(10, dtype=np.float32)
        assert media.moving_average(values, 100).size == 10

    def test_single_sample_window_is_identity(self) -> None:
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert np.allclose(media.moving_average(values, 1), values)


class TestSubtitleStreams:
    def test_language_tag_normalises_common_spellings(self) -> None:
        assert media.language_tag("fra") == "fr"
        assert media.language_tag("French") == "fr"
        assert media.language_tag("ENG") == "en"
        assert media.language_tag(None) is None

    def test_metadata_code_uses_the_three_letter_form(self) -> None:
        # mov_text silently drops a track's language when handed "en".
        assert media.language_code_for_metadata("French") == "fra"
        assert media.language_code_for_metadata("English") == "eng"
        assert media.language_code_for_metadata(None) is None

    def test_probe_reads_only_subtitle_streams(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = {
            "streams": [
                {"index": 0, "codec_type": "video"},
                {"index": 1, "codec_type": "audio"},
                {"index": 2, "codec_type": "subtitle", "tags": {"language": "eng"}},
                {"index": 3, "codec_type": "subtitle"},
            ]
        }
        monkeypatch.setattr(media, "require_binary", lambda *a, **k: "/usr/bin/ffprobe")
        monkeypatch.setattr(media, "run_command", lambda *a, **k: json.dumps(payload))

        streams = media.probe_subtitle_streams("movie.mp4")
        assert [s.index for s in streams] == [2, 3]
        assert streams[0].language == "eng"
        assert streams[1].language is None

    def test_map_args_drop_only_the_target_language(self) -> None:
        existing = [
            media.SubtitleStream(2, "eng"),
            media.SubtitleStream(3, "fra"),
            media.SubtitleStream(4, None),
        ]
        args, preserved = media._build_subtitle_map_args(existing, "French")
        assert args == ["-map", "0:s?", "-map", "-0:3"]
        assert preserved == 2

    def test_add_subtitle_track_copies_and_tags_the_new_stream(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(
            media,
            "probe_subtitle_streams",
            lambda path: [media.SubtitleStream(2, "eng"), media.SubtitleStream(3, "fra")],
        )
        monkeypatch.setattr(media, "require_binary", lambda *a, **k: "/usr/bin/ffmpeg")
        monkeypatch.setattr(media, "run_command", lambda cmd, **k: captured.append(list(cmd)))

        media.add_subtitle_track(
            tmp_path / "in.mp4",
            tmp_path / "subs.srt",
            tmp_path / "out.mp4",
            language="French",
        )
        command = captured[0]
        assert command[command.index("-c:v") + 1] == "copy"
        assert command[command.index("-c:s") + 1] == "mov_text"
        assert "0:s?" in command and "1:s:0" in command
        # One preserved non-target stream precedes ours, so ours is index 1.
        assert "-c:s:1" not in command
        assert "language=fra" in command
        assert command[-1] == str(tmp_path / "out.mp4")

    def test_burn_subtitles_reencodes_with_the_subtitles_filter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(media, "probe_subtitle_streams", lambda path: [])
        monkeypatch.setattr(media, "require_binary", lambda *a, **k: "/usr/bin/ffmpeg")
        monkeypatch.setattr(media, "run_command", lambda cmd, **k: captured.append(list(cmd)))

        media.burn_subtitles(
            tmp_path / "in.mp4",
            tmp_path / "subs.srt",
            tmp_path / "out.mp4",
            language="French",
        )
        command = captured[0]
        assert command[command.index("-c:v") + 1] == "libx264"
        assert command[command.index("-c:a") + 1] == "copy"
        assert "subtitles=filename=" in command[command.index("-vf") + 1]
