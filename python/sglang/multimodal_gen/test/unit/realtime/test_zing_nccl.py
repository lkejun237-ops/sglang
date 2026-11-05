# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from sglang.multimodal_gen.runtime.realtime.zing_nccl import configure_zing_nccl


@pytest.mark.parametrize("inherited", [None, "2", "32"])
def test_worker_policy_clears_channel_limits_but_preserves_network(
    monkeypatch, caplog, inherited
):
    removed = (
        "NCCL_MAX_NCHANNELS",
        "NCCL_MAX_NRINGS",
        "NCCL_MIN_CTAS",
        "NCCL_MAX_CTAS",
    )
    for key in ("NCCL_MIN_NCHANNELS", *removed):
        if inherited is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, inherited)
    preserved = {
        "NCCL_MIN_NRINGS": "8",  # Ignored when MIN_NCHANNELS is present.
        "NCCL_NET": "Socket",
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_IB_DISABLE": "1",
        "NCCL_DEBUG": "INFO",
    }
    for key, value in preserved.items():
        monkeypatch.setenv(key, value)

    configure_zing_nccl()
    assert os.environ["NCCL_MIN_NCHANNELS"] == "2"
    assert all(key not in os.environ for key in removed)
    assert {key: os.environ[key] for key in preserved} == preserved
    assert "Zing NCCL startup policy" in caplog.text
    assert "NCCL_MIN_NCHANNELS=2" in caplog.text

    once = dict(os.environ)
    configure_zing_nccl()
    assert dict(os.environ) == once
