# SPDX-License-Identifier: Apache-2.0
"""Applicable pre-RPC regressions retained under the approved v6 redesign."""

import asyncio
import importlib
import itertools
import threading

import pytest


def api(name):
    try:
        return importlib.import_module(
            "sglang.multimodal_gen.runtime.conditioning." + name
        )
    except ModuleNotFoundError as error:
        if error.name in (
            "sglang.multimodal_gen.runtime.conditioning",
            "sglang.multimodal_gen.runtime.conditioning." + name,
        ):
            pytest.fail("Conditioning " + name + " behavior is not implemented")
        raise


def test_tenant_admission_is_atomic_and_release_idempotent():
    mod = api("resources")
    ledger = mod.AdmissionLedger({"slots": (2, 1), "bytes": (20, 12)})
    lease = ledger.reserve("a", slots=1, bytes=10)
    with pytest.raises(mod.ConditionError) as caught:
        ledger.reserve("a", slots=1, bytes=1)
    assert caught.value.code == "CONDITION_BUSY"
    assert ledger.snapshot()["global"] == {"slots": 1, "bytes": 10}
    other = ledger.reserve("b", slots=1, bytes=10)
    lease.release()
    lease.release()
    other.release()
    assert all(value == 0 for value in ledger.snapshot()["global"].values())


def test_resource_limits_reject_inconsistent_and_unbounded_configuration():
    mod = api("resources")
    with pytest.raises(ValueError):
        mod.ResourceLimits(cpu_executor_active=0)
    with pytest.raises(ValueError):
        mod.AdmissionLedger({"slots": (1, 2)})


def test_executor_full_rejects_without_submit_and_cancel_drains():
    mod = api("resources")

    async def run():
        executor = mod.BoundedExecutor(active=1, waiting=0)
        entered = threading.Event()
        finish = threading.Event()

        def work():
            entered.set()
            finish.wait(5)
            return 3

        task = asyncio.create_task(executor.run(work))
        while not entered.is_set():
            await asyncio.sleep(0.001)
        with pytest.raises(mod.ConditionError):
            await executor.run(lambda: 9)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done(), "cancelled CPU work must retain its slot until drained"
        assert executor.in_use == 1
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert executor.in_use == 0
        await executor.close()

    asyncio.run(run())


def test_default_maximum_admission_fits_tenant_inflight_budget():
    mod = api("resources")
    limits = mod.ResourceLimits()
    charge = dict(
        slots=1,
        inflight=limits.max_request_bytes
        + limits.output_reservation_bytes
        + limits.max_header_bytes
        + 4,
    )
    assert charge["inflight"] <= limits.max_inflight_bytes_per_tenant
    lease = mod.AdmissionLedger(limits.budgets()).reserve("tenant", **charge)
    lease.release()


def test_native_pipeline_excludes_dit_and_vae_before_loader():
    mod = api("native")

    class FakePipeline:
        def __init__(self, model_path, server_args, required_config_modules=None):
            self.modules = {key: object() for key in required_config_modules}
            self.create_pipeline_stages(server_args)

    pipeline = mod.build_encoder_pipeline(
        "/unused", object(), pipeline_cls=FakePipeline
    )
    assert set(pipeline.modules) == {"text_encoder", "tokenizer", "processor"}


@pytest.fixture
def native_profile_assets(tmp_path, monkeypatch):
    from dataclasses import fields

    from sglang.multimodal_gen.configs.models.encoders.qwen3vl import (
        Qwen3VLExecutionOptions,
    )

    for item in fields(Qwen3VLExecutionOptions):
        monkeypatch.setenv(f"SGLANG_H3_QWEN_{item.name.upper()}", "0")
    monkeypatch.setenv("SGLANG_H3_QWEN_FP8", "0")
    for folder in ("text_encoder", "tokenizer", "processor"):
        path = tmp_path / folder
        path.mkdir()
        (path / "config.json").write_text("{}")
        (path / "weights.safetensors").write_bytes(b"frozen-" + folder.encode())
    (tmp_path / "zing_config.json").write_text(
        '{"condition_encoder_config":{"video_sample_fps":2}}'
    )
    environment = {
        "encoder_tp": 1,
        "gpu": "fixture",
        "cc": "9.0",
        "backend": "torch_sdpa",
        "precision": "bf16",
        "software": {"fixture": "1"},
    }
    return tmp_path, environment


def test_cpu_profile_builder_hashes_all_assets_and_explicit_null_projection(
    native_profile_assets,
):
    mod = api("native")
    tmp_path, environment = native_profile_assets
    first = mod.build_profile(tmp_path, environment)
    assert first.document["projection"] is None
    assert set(first.document["model"]["files"]) == {
        "text_encoder/config.json",
        "text_encoder/weights.safetensors",
    }
    (tmp_path / "tokenizer" / "new_file.json").write_text("{}")
    second = mod.build_profile(tmp_path, environment)
    assert first.profile_id != second.profile_id


@pytest.mark.parametrize(
    "option",
    ["HF_RMSNORM", "FUSED_MROPE", "FUSED_GATE_UP", "TEXT_FA4", "VISION_FA4", "FP8"],
)
@pytest.mark.parametrize("value", [None, "1"])
def test_native_rejects_unprofiled_switches_before_build_and_saved_profile_load(
    native_profile_assets, monkeypatch, option, value
):
    import torch

    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

    mod = api("native")
    root, environment = native_profile_assets
    saved_profile = mod.build_profile(root, environment)
    if value is None:
        monkeypatch.delenv(f"SGLANG_H3_QWEN_{option}")
    else:
        monkeypatch.setenv(f"SGLANG_H3_QWEN_{option}", value)

    def unexpected_cuda_probe():
        pytest.fail("Unprofiled execution must be rejected before CUDA initialization")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_probe)
    with pytest.raises(ConditionError, match="execution options differ") as error:
        mod.build_profile(root, saved_profile.document["numerical_environment"])
    assert error.value.code == "CONDITION_PROFILE_MISMATCH"
    with pytest.raises(ConditionError, match="execution options differ"):
        mod.NativeConditionEncoder(root, saved_profile)

    # Restoring the explicit baseline retains the same asset/profile identity.
    monkeypatch.setenv(f"SGLANG_H3_QWEN_{option}", "0")
    assert mod.build_profile(root, environment).profile_id == saved_profile.profile_id


def test_remote_all_off_requires_sdpa_profile(native_profile_assets):
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

    root, environment = native_profile_assets
    with pytest.raises(ConditionError, match="attention backend"):
        api("native").build_profile(root, dict(environment, backend="fa"))


def test_native_all_execution_combinations_have_distinct_profiles(
    native_profile_assets, monkeypatch
):
    root, environment = native_profile_assets
    mod = api("native")
    names = (
        "hf_rmsnorm",
        "fused_mrope",
        "fused_gate_up",
        "text_fa4",
        "vision_fa4",
        "fp8",
    )
    identities = set()
    for values in itertools.product((False, True), repeat=6):
        options = dict(zip(names, values))
        for name, enabled in options.items():
            monkeypatch.setenv("SGLANG_H3_QWEN_" + name.upper(), str(int(enabled)))
        profile = mod.build_profile(
            root, dict(environment, backend=mod._attention_backend(options))
        )
        assert profile.document["numerical_environment"]["execution_options"] == options
        identities.add(profile.profile_id)
    assert len(identities) == 64


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"fp8": True},
        dict.fromkeys(
            (
                "hf_rmsnorm",
                "fused_mrope",
                "fused_gate_up",
                "text_fa4",
                "vision_fa4",
                "fp8",
            ),
            0,
        ),
    ],
)
def test_native_rejects_incomplete_or_non_boolean_execution_options(
    native_profile_assets, options
):
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

    root, environment = native_profile_assets
    with pytest.raises(ConditionError):
        api("native").build_profile(root, dict(environment, execution_options=options))


@pytest.mark.parametrize(
    "text,vision", [(False, False), (True, False), (False, True), (True, True)]
)
def test_native_profile_records_options_and_mixed_attention(
    native_profile_assets, monkeypatch, text, vision
):
    root, environment = native_profile_assets
    mod = api("native")
    baseline = mod.build_profile(root, environment)
    monkeypatch.setenv("SGLANG_H3_QWEN_TEXT_FA4", str(int(text)))
    monkeypatch.setenv("SGLANG_H3_QWEN_VISION_FA4", str(int(vision)))
    backend = (
        "fa4"
        if text and vision
        else "fa4+torch_sdpa"
        if text or vision
        else "torch_sdpa"
    )
    profile = mod.build_profile(root, dict(environment, backend=backend))
    options = profile.document["numerical_environment"]["execution_options"]
    assert options == dict(
        hf_rmsnorm=False,
        fused_mrope=False,
        fused_gate_up=False,
        text_fa4=text,
        vision_fa4=vision,
        fp8=False,
    )
    assert (profile.profile_id != baseline.profile_id) == (text or vision)
    receipt = dict(profile.document["numerical_environment"], folding_mode=None)
    mod.validate_runtime_receipt(profile, receipt)
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

    with pytest.raises(ConditionError):
        mod.validate_runtime_receipt(
            profile, dict(receipt, execution_options=dict(options, fp8=True))
        )


@pytest.mark.parametrize(
    "option",
    ["HF_RMSNORM", "FUSED_MROPE", "FUSED_GATE_UP", "TEXT_FA4", "VISION_FA4", "FP8"],
)
def test_native_new_profile_accepts_each_optimization(
    native_profile_assets, monkeypatch, option
):
    root, environment = native_profile_assets
    mod = api("native")
    baseline = mod.build_profile(root, environment)
    monkeypatch.setenv("SGLANG_H3_QWEN_" + option, "1")
    if option in {"TEXT_FA4", "VISION_FA4"}:
        environment = dict(environment, backend="fa4+torch_sdpa")
    profile = mod.build_profile(root, environment)
    assert profile.profile_id != baseline.profile_id
    assert (
        profile.document["numerical_environment"]["execution_options"][option.lower()]
        is True
    )


def test_native_actual_receipt_rejects_fold_backend_precision_and_gpu_mismatch(
    tmp_path,
):
    mod = api("native")
    from sglang.multimodal_gen.runtime.conditioning.contract import (
        ConditionError,
        DeploymentProfile,
    )

    document = dict(
        schema_version=1,
        model={"sha256": "1" * 64},
        projection=None,
        tokenizer={"sha256": "2" * 64},
        processor={"sha256": "3" * 64},
        runtime={"kind": "zing"},
        implementation_version="v1",
        preprocess_version="rgb-v1",
        numerical_environment={
            "encoder_tp": 1,
            "gpu": "fixture",
            "cc": "9.0",
            "backend": "torch_sdpa",
            "precision": "bf16",
            "software": {"torch": "fixture"},
        },
    )
    profile = DeploymentProfile(document)
    receipt = dict(document["numerical_environment"], folding_mode=None)
    mod.validate_runtime_receipt(profile, receipt)
    for key, value in [
        ("encoder_tp", 2),
        ("folding_mode", "fold"),
        ("backend", "fa"),
        ("precision", "fp16"),
        ("gpu", "different"),
        ("cc", "8.0"),
        ("software", {"torch": "wrong"}),
    ]:
        with pytest.raises(ConditionError):
            mod.validate_runtime_receipt(profile, dict(receipt, **{key: value}))
