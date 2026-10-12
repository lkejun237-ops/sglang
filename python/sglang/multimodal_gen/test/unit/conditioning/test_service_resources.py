# SPDX-License-Identifier: Apache-2.0
"""Applicable pre-RPC regressions retained under the approved v6 redesign."""

import asyncio
import importlib
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


def test_cpu_profile_builder_hashes_all_assets_and_explicit_null_projection(tmp_path):
    mod = api("native")
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
    first = mod.build_profile(tmp_path, environment)
    assert first.document["projection"] is None
    assert set(first.document["model"]["files"]) == {
        "text_encoder/config.json",
        "text_encoder/weights.safetensors",
    }
    (tmp_path / "tokenizer" / "new_file.json").write_text("{}")
    second = mod.build_profile(tmp_path, environment)
    assert first.profile_id != second.profile_id


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
