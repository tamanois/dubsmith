"""Device resolution, capability probing and dtype selection.

Every hardware profile is exercised through a fake ``torch`` module, so the rules
for Pascal, Volta, Ampere, CDNA, Metal, XPU and plain CPU are all proven without
owning any of that hardware - which is the entire point of the capability layer.

The behaviour these tests defend: a user with a 1080 Ti must not get bfloat16,
``--device cuda:1`` must not silently become ``cuda``, and a machine with no GPU
at all must still run.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from pipeline.asr import asr_device, resolve_compute_type
from pipeline.config import (
    DTYPE_ALIASES,
    DeviceInfo,
    available_accelerators,
    normalize_dtype,
    pick_dtype,
    preferred_dtype,
    probe_device,
    release_gpu_memory,
    resolve_device,
    supports_bf16,
    supports_fp16,
)
from pipeline.errors import ConfigError
from pipeline.tts import resolve_dtype

TorchProfile = Callable[..., Any]


class TestProbeDevice:
    def test_explicit_cpu_needs_no_accelerator(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="cpu")
        info = probe_device("cpu")

        assert info.kind == "cpu"
        assert info.label == "cpu"
        assert info.is_accelerated is False

    def test_auto_prefers_cuda(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="cuda")
        info = probe_device("auto")

        assert info.kind == "cuda"
        # Bare "cuda", so nothing downstream starts passing a device index where a
        # backend expects the current device.
        assert info.label == "cuda"
        assert info.capability == (8, 6)
        assert info.name == "NVIDIA GeForce RTX 3060"
        assert info.total_memory == 12 * 1024**3

    def test_rocm_is_reported_as_cuda_to_backends(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="rocm", capability=(9, 4))
        info = probe_device("auto")

        assert info.kind == "rocm"
        # torch has no "rocm" device: AMD cards answer to the CUDA API.
        assert info.label == "cuda"
        assert info.capability == (9, 4)

    def test_the_gpu_index_is_preserved(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="cuda")
        info = probe_device("cuda:1")

        assert info.index == 1
        assert info.label == "cuda:1"

    def test_auto_ignores_metal_and_xpu(self, torch_profile: TorchProfile) -> None:
        # Speech recognition runs on CTranslate2, which has no Metal or XPU
        # backend, so an `auto` that picked one would break a machine that works.
        torch_profile(kind="mps")
        assert probe_device("auto").kind == "cpu"

        torch_profile(kind="xpu")
        assert probe_device("auto").kind == "cpu"

    def test_an_explicit_metal_is_still_honoured(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="mps")
        info = probe_device("mps")

        assert info.kind == "mps"
        assert info.label == "mps"
        assert info.is_accelerated is True

    def test_an_unknown_device_passes_through(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="cpu")
        info = probe_device("gpu-ish")

        assert info.kind == "gpu-ish"
        assert info.label == "gpu-ish"

    def test_a_broken_probe_does_not_raise(self, torch_profile: TorchProfile) -> None:
        module = torch_profile(kind="cuda")

        def explode(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("driver exploded")

        module.cuda.get_device_capability = explode
        info = probe_device("cuda")

        assert info.kind == "cuda"
        assert info.capability is None  # unknown, rather than fatal

    def test_a_missing_torch_reports_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("pipeline.config._import_torch", lambda: None)

        assert probe_device("auto").kind == "cpu"
        assert available_accelerators() == ()


class TestAvailableAccelerators:
    def test_lists_the_present_accelerator(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="cuda")
        assert available_accelerators() == ("cuda",)

    def test_reports_metal_even_though_auto_declines_it(
        self, torch_profile: TorchProfile
    ) -> None:
        # Reported so `dub.py info` can tell a Mac user to try --device mps.
        torch_profile(kind="mps")
        assert available_accelerators() == ("mps",)

    def test_a_cpu_only_machine_reports_nothing(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="cpu")
        assert available_accelerators() == ()


class TestResolveDevice:
    @pytest.mark.parametrize("kind", ["cuda", "rocm"])
    def test_auto_picks_cuda_when_a_gpu_is_present(
        self, torch_profile: TorchProfile, kind: str
    ) -> None:
        torch_profile(kind=kind)
        assert resolve_device("auto").startswith("cuda")

    def test_auto_falls_back_to_cpu(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="cpu")
        assert resolve_device("auto") == "cpu"

    @pytest.mark.parametrize("requested", ["cpu", "cuda:1", "mps", "xpu:0"])
    def test_an_explicit_request_is_returned_unchanged(
        self, torch_profile: TorchProfile, requested: str
    ) -> None:
        torch_profile(kind="cpu")
        assert resolve_device(requested) == requested


class TestDtypeSelection:
    def test_ampere_gets_bfloat16(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="cuda", capability=(8, 6))
        assert pick_dtype("auto") == "bfloat16"

    def test_turing_gets_float16(self, torch_profile: TorchProfile) -> None:
        # sm_75 (T4, RTX 2080): bfloat16 is unsupported, float16 is fast.
        torch_profile(kind="cuda", capability=(7, 5))
        assert pick_dtype("auto") == "float16"

    def test_pascal_gets_float32(self, torch_profile: TorchProfile) -> None:
        # sm_61 (GTX 1080 Ti): no bfloat16, and emulated float16 is lossy.
        torch_profile(kind="cuda", capability=(6, 1))
        assert pick_dtype("auto") == "float32"

    def test_an_unreadable_capability_degrades_to_float32(
        self, torch_profile: TorchProfile
    ) -> None:
        torch_profile(kind="cuda", capability=None)
        assert pick_dtype("auto") == "float32"

    def test_cdna_gets_bfloat16_but_older_rocm_gets_float16(
        self, torch_profile: TorchProfile
    ) -> None:
        torch_profile(kind="rocm", capability=(9, 4))
        assert pick_dtype("auto") == "bfloat16"

        torch_profile(kind="rocm", capability=(9, 0))
        assert pick_dtype("auto") == "float16"

    def test_cpu_is_always_float32(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="cpu")
        assert pick_dtype("auto") == "float32"

    def test_xpu_gets_bfloat16_and_metal_gets_float16(
        self, torch_profile: TorchProfile
    ) -> None:
        torch_profile(kind="xpu")
        assert pick_dtype("xpu") == "bfloat16"

        torch_profile(kind="mps")
        assert pick_dtype("mps") == "float16"

    def test_auto_on_a_metal_only_machine_stays_float32(
        self, torch_profile: TorchProfile
    ) -> None:
        # `auto` deliberately declines Metal, so the dtype follows the device that
        # will actually run - the CPU.
        torch_profile(kind="mps")
        assert pick_dtype("auto") == "float32"

    def test_an_explicit_dtype_is_honoured(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="cuda", capability=(8, 6))
        assert pick_dtype("cuda", "fp16") == "float16"
        assert pick_dtype("cuda", "fp32") == "float32"

    def test_bfloat16_on_an_old_card_degrades_and_warns(
        self, torch_profile: TorchProfile, caplog: pytest.LogCaptureFixture
    ) -> None:
        torch_profile(kind="cuda", capability=(6, 1))
        with caplog.at_level("WARNING"):
            chosen = pick_dtype("cuda", "bf16")

        assert chosen == "float32"
        assert "not usable" in caplog.text

    def test_half_precision_on_cpu_degrades_and_warns(
        self, torch_profile: TorchProfile, caplog: pytest.LogCaptureFixture
    ) -> None:
        torch_profile(kind="cpu")
        with caplog.at_level("WARNING"):
            assert pick_dtype("cpu", "fp16") == "float32"
            assert pick_dtype("cpu", "bf16") == "float32"

        assert "not usable" in caplog.text

    def test_tts_resolve_dtype_delegates(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="cuda", capability=(7, 5))
        assert resolve_dtype("cuda") == "float16"
        # bf16 is not usable there, so it degrades to the best the card does.
        assert resolve_dtype("cuda", "bf16") == "float16"


class TestDtypeNames:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [("auto", "auto"), ("BFloat16", "bf16"), ("half", "fp16"), ("FULL", "fp32")],
    )
    def test_aliases_are_canonicalised(self, given: str, expected: str) -> None:
        assert normalize_dtype(given) == expected

    def test_an_unknown_dtype_is_a_config_error(self) -> None:
        with pytest.raises(ConfigError, match="Unknown dtype"):
            normalize_dtype("float8")

    def test_every_alias_maps_to_a_supported_dtype(self) -> None:
        assert set(DTYPE_ALIASES.values()) == {"auto", "bf16", "fp16", "fp32"}


class TestCapabilityFlags:
    def test_flags_follow_the_capability_rule(self) -> None:
        assert supports_bf16(DeviceInfo(kind="cuda", capability=(8, 0))) is True
        assert supports_bf16(DeviceInfo(kind="cuda", capability=(7, 5))) is False
        assert supports_fp16(DeviceInfo(kind="cuda", capability=(7, 0))) is True
        assert supports_fp16(DeviceInfo(kind="cuda", capability=(6, 1))) is False

    def test_an_unknown_capability_supports_nothing(self) -> None:
        # Guessing "probably fine" here is how a crash becomes a wrong answer.
        unknown = DeviceInfo(kind="cuda", capability=None)
        assert unknown.supports_bf16 is False
        assert unknown.supports_fp16 is False

    def test_cpu_supports_neither(self) -> None:
        info = DeviceInfo(kind="cpu")
        assert info.supports_bf16 is False
        assert info.supports_fp16 is False
        assert preferred_dtype(info) == "fp32"


class TestAsrDevice:
    def test_cuda_is_passed_through_bare(self, torch_profile: TorchProfile) -> None:
        # CTranslate2 wants "cuda" and does not take an index.
        torch_profile(kind="cuda")
        assert asr_device("auto") == "cuda"
        assert resolve_compute_type(asr_device("auto")) == "float16"

    def test_rocm_maps_to_cuda(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="rocm")
        assert asr_device("auto") == "cuda"

    @pytest.mark.parametrize("requested", ["cpu", "mps"])
    def test_unsupported_accelerators_degrade_to_cpu(
        self, torch_profile: TorchProfile, requested: str
    ) -> None:
        # Metal has no CTranslate2 backend: running ASR on the CPU is correct,
        # failing after the weights load is not.
        torch_profile(kind=requested)
        assert asr_device(requested) == "cpu"
        assert resolve_compute_type(asr_device(requested)) == "int8"

    def test_an_explicit_cpu_is_not_upgraded(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="cuda")
        assert asr_device("cpu") == "cpu"


class TestReleaseGpuMemory:
    def test_reports_true_when_cuda_released_something(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="cuda")
        assert release_gpu_memory() is True

    def test_reports_false_on_a_cpu_only_machine(self, torch_profile: TorchProfile) -> None:
        torch_profile(kind="cpu")
        assert release_gpu_memory() is False

    def test_metal_is_released_too(self, torch_profile: TorchProfile) -> None:
        # The caching allocator holds memory on Metal as well, and leaving it
        # resident is what makes the next stage fail on a small machine.
        torch_profile(kind="mps")
        assert release_gpu_memory() is True
