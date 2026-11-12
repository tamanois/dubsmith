"""Tests for the model registry, resolution and downloading."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.errors import ConfigError, ModelNotFoundError
from pipeline.models import (
    MVP_REQUIRED,
    REGISTRY,
    download_hint,
    download_model,
    is_available,
    registry_status,
    remove_model,
    resolve_model,
)


def _populate(root: Path, name: str) -> Path:
    """Create a directory that looks like a downloaded model."""
    target = root / REGISTRY[name].dir_name
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    return target


class TestRegistry:
    def test_mvp_models_are_registered(self) -> None:
        for name in MVP_REQUIRED:
            assert name in REGISTRY

    def test_missing_model_reports_unavailable(self, tmp_path: Path) -> None:
        assert is_available("tts_base", tmp_path) is False

    def test_available_once_populated(self, tmp_path: Path) -> None:
        _populate(tmp_path, "tts_base")
        assert is_available("tts_base", tmp_path) is True

    def test_empty_directory_counts_as_missing(self, tmp_path: Path) -> None:
        (tmp_path / REGISTRY["tts_base"].dir_name).mkdir(parents=True)
        assert is_available("tts_base", tmp_path) is False

    def test_registry_status_covers_every_model(self, tmp_path: Path) -> None:
        _populate(tmp_path, "tts_base")
        status = dict((spec.name, present) for spec, present in registry_status(tmp_path))
        assert status["tts_base"] is True
        assert status["tts_custom_voice"] is False
        assert len(status) == len(REGISTRY)

    def test_unknown_model_name_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ModelNotFoundError, match="Unknown model"):
            is_available("does_not_exist", tmp_path)


class TestResolve:
    def test_returns_directory_when_present(self, tmp_path: Path) -> None:
        expected = _populate(tmp_path, "whisper_large_v3")
        assert resolve_model("whisper_large_v3", tmp_path) == expected

    def test_missing_model_error_is_actionable(self, tmp_path: Path) -> None:
        with pytest.raises(ModelNotFoundError) as excinfo:
            resolve_model("tts_base", tmp_path)
        assert "tts_base" in str(excinfo.value)
        assert excinfo.value.remedy == download_hint("tts_base")
        assert "dub.py models --download" in (excinfo.value.remedy or "")

    def test_honours_custom_models_dir(self, tmp_path: Path) -> None:
        custom = tmp_path / "elsewhere"
        _populate(custom, "tts_base")
        assert resolve_model("tts_base", custom).parent == custom


class TestDownload:
    def test_calls_snapshot_download_with_local_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_snapshot(**kwargs: object) -> str:
            captured.update(kwargs)
            return str(kwargs["local_dir"])

        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)

        target = download_model("tts_base", tmp_path)
        assert captured["repo_id"] == REGISTRY["tts_base"].repo_id
        assert Path(str(captured["local_dir"])) == target
        assert target.is_dir()

    def test_skips_existing_model_unless_forced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _populate(tmp_path, "tts_base")
        calls: list[int] = []

        def fake_snapshot(**kwargs: object) -> str:
            calls.append(1)
            return ""

        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)

        download_model("tts_base", tmp_path)
        assert calls == []  # already present -> no network call

        download_model("tts_base", tmp_path, force=True)
        assert len(calls) == 1  # force re-downloads

    def test_unknown_model_raises_before_network(self, tmp_path: Path) -> None:
        with pytest.raises(ModelNotFoundError, match="Unknown model"):
            download_model("nope", tmp_path)


class TestRemoveModel:
    def test_deletes_a_present_model_and_reports_freed_bytes(
        self, tmp_path: Path
    ) -> None:
        _populate(tmp_path, "tts_custom_voice")
        (tmp_path / REGISTRY["tts_custom_voice"].dir_name / "model.safetensors").write_bytes(
            b"x" * 16
        )

        result = remove_model("tts_custom_voice", tmp_path)

        assert result["freed_bytes"] >= 16
        assert not (tmp_path / REGISTRY["tts_custom_voice"].dir_name).exists()
        assert result["warning"] is None

    def test_warns_when_removing_a_required_model(self, tmp_path: Path) -> None:
        _populate(tmp_path, "tts_base")
        result = remove_model("tts_base", tmp_path)

        assert result["warning"] is not None
        assert "re-download" in result["warning"]
        assert is_available("tts_base", tmp_path) is False

    def test_missing_model_frees_nothing(self, tmp_path: Path) -> None:
        result = remove_model("tts_fish", tmp_path)
        assert result["freed_bytes"] == 0
        assert result["warning"] is None

    def test_package_backed_model_cannot_be_deleted(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="package-backed"):
            remove_model("demucs", tmp_path)

    def test_unknown_model_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ModelNotFoundError, match="Unknown model"):
            remove_model("nope", tmp_path)
