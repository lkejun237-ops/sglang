# SPDX-License-Identifier: Apache-2.0
"""Reject cloud-injected transport mismatches before model startup."""

import json
import os

import pytest

from sglang.multimodal_gen.runtime.realtime.role_pool.reporting import (
    validate_worker_configuration,
)


@pytest.fixture
def managed_config(tmp_path, monkeypatch):
    path = tmp_path / "worker.json"
    monkeypatch.setenv("SGLANG_ROLE_POOL_WORKER_CONFIG", str(path))
    for name in ("NCCL_MIN_NCHANNELS", "NCCL_MIN_CTAS"):
        # Register the original absence too: startup may set this variable.
        monkeypatch.setenv(name, "2")
        monkeypatch.delenv(name)

    def write(role):
        path.write_text(
            json.dumps(
                dict(
                    role=role,
                    service_count=1,
                    hardware_ids=["gpu-test"],
                    profile_limits={"480": {}},
                )
            )
        )

    return write


@pytest.mark.parametrize("role", ["encoder", "dit", "decoder"])
@pytest.mark.parametrize("inherited", [None, "2", "4"])
def test_managed_channel_policy_before_startup(
    managed_config, monkeypatch, role, inherited
):
    managed_config(role)
    if inherited is not None:
        monkeypatch.setenv("NCCL_MIN_NCHANNELS", inherited)
    if inherited == "4":
        with pytest.raises(ValueError, match="NCCL_MIN_NCHANNELS=2"):
            validate_worker_configuration(role, 1)
        assert os.environ["NCCL_MIN_NCHANNELS"] == "4"
    else:
        validate_worker_configuration(role, 1)
        assert os.environ["NCCL_MIN_NCHANNELS"] == "2"


def test_modern_min_ctas_alias_cannot_override_transport_policy(
    managed_config, monkeypatch
):
    managed_config("dit")
    monkeypatch.setenv("NCCL_MIN_NCHANNELS", "2")
    monkeypatch.setenv("NCCL_MIN_CTAS", "4")
    with pytest.raises(ValueError, match="NCCL_MIN_CTAS=2"):
        validate_worker_configuration("dit", 1)


def test_unmanaged_launcher_environment_is_preserved(monkeypatch):
    monkeypatch.delenv("SGLANG_ROLE_POOL_WORKER_CONFIG", raising=False)
    monkeypatch.setenv("NCCL_MIN_NCHANNELS", "4")
    validate_worker_configuration("decoder", 1, static_connection=True)
    assert os.environ["NCCL_MIN_NCHANNELS"] == "4"
