# SPDX-License-Identifier: Apache-2.0
import pytest

from sglang.multimodal_gen.runtime.realtime import block0_rdma as rdma


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("MC_FORCE_TCP", "MC_USE_TENT", "MC_USE_TEV1", "MC_MS_AUTO_DISC"):
        monkeypatch.delenv(name, raising=False)


def test_default_chooses_active_container_hcas(tmp_path, monkeypatch):
    devices = tmp_path / "dev"
    devices.mkdir()
    for idx, (visible, state) in enumerate(
        (
            (True, "4: ACTIVE"),
            (False, "4: ACTIVE"),
            (True, "2: INIT"),
            (True, "4: ACTIVE"),
        )
    ):
        hca = f"mlx5_{idx}"
        node = f"uverbs{idx}"
        if visible:
            (devices / node).touch()
        uc = tmp_path / "infiniband_verbs" / node
        uc.mkdir(parents=True)
        (uc / "ibdev").write_text(hca)
        port = tmp_path / "infiniband" / hca / "ports/1"
        port.mkdir(parents=True)
        (port / "state").write_text(state)
    active = rdma._active_hcas(devices, tmp_path)
    assert active == "mlx5_0,mlx5_3"
    monkeypatch.setattr(rdma, "_active_hcas", lambda: active)
    assert rdma.block0_rdma_devices({}) == active


def test_missing_hca_fails_instead_of_enabling_tcp(monkeypatch):
    monkeypatch.setattr(rdma, "_active_hcas", lambda: "")
    with pytest.raises(ValueError, match="No active RDMA HCA"):
        rdma.block0_rdma_devices({"protocol": "rdma"})


def test_explicit_hcas():
    assert (
        rdma.block0_rdma_devices({"rdma_devices": "mlx5_2, mlx5_3"}) == "mlx5_2,mlx5_3"
    )


@pytest.mark.parametrize("protocol", ["tcp", "auto", "", None])
def test_legacy_or_unknown_protocol_is_rejected(protocol):
    with pytest.raises(ValueError, match="protocol"):
        rdma.block0_rdma_devices({"protocol": protocol})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MC_FORCE_TCP", "1"),
        ("MC_FORCE_TCP", "0"),
        ("MC_FORCE_TCP", ""),
        ("MC_MS_AUTO_DISC", "1"),
        ("MC_USE_TENT", "1"),
        ("MC_USE_TEV1", "1"),
    ],
)
def test_environment_cannot_override_rdma(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        rdma.block0_rdma_devices({"rdma_devices": "mlx5_0"})
