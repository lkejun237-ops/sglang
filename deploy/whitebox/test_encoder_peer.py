"""CPU tests for cross-pod access control, independent of CUDA imports."""

import importlib.util
from pathlib import Path

import pytest

PATH = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/zing_h3/encoder_peer.py"
)
SPEC = importlib.util.spec_from_file_location("encoder_peer_cpu", PATH)
peer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(peer)


def test_loopback_is_backward_compatible():
    endpoint = peer.EncoderPeer()
    assert endpoint.fields() == {}
    assert endpoint.accepts(dict(op="open", bridge_port=31002))
    assert not endpoint.accepts(
        dict(op="open", bridge_host="10.0.0.2", bridge_port=31002)
    )


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "8.8.8.8", "169.254.169.254", "::1", "localhost"]
)
def test_no_public_wildcard_or_metadata_address(host):
    with pytest.raises(ValueError):
        peer.EncoderPeer(host=host, bridge_host=host)


def test_private_requires_secret_and_rejects_cross_node_and_wrong_peer(
    tmp_path, monkeypatch
):
    original = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda p: b"node-a" if p.name == "boot_id" else original(p)
    )
    token = tmp_path / "peer-token"
    token.write_text("a" * 64)
    token.chmod(0o600)
    args = dict(
        host="10.128.1.44", bridge_host="10.128.1.45", auth_token_file=str(token)
    )
    endpoint = peer.EncoderPeer(**args)
    request = dict(
        op="open", bridge_host="10.128.1.44", bridge_port=31002, **endpoint.fields()
    )
    assert endpoint.accepts(request)
    for change in (
        {"peer_token": "wrong"},
        {"peer_token": "错误"},
        {"peer_node": "other-node"},
        {"bridge_host": "10.0.0.1"},
        {"bridge_port": True},
        {"bridge_port": 80},
    ):
        assert not endpoint.accepts({**request, **change})
    assert not endpoint.accepts(dict(op="abort", session="owner"))
    assert not endpoint.accepts(None)
    token.chmod(0o644)
    with pytest.raises(ValueError, match="other users"):
        peer.EncoderPeer(**args)
    with pytest.raises(ValueError, match="token file"):
        peer.EncoderPeer(host=args["host"], bridge_host=args["bridge_host"])


def test_private_and_loopback_cannot_mix():
    with pytest.raises(ValueError, match="both"):
        peer.EncoderPeer(host="10.128.1.44")


def test_scheduler_binding_across_distinct_boot_ids(tmp_path, monkeypatch):
    import json

    token = tmp_path / "token"
    token.write_text("b" * 64)
    token.chmod(0o600)
    path = tmp_path / "binding.json"
    binding = dict(
        version=1,
        verified_by="pai-control-plane",
        node_name="approved-node",
        pods=[
            dict(
                pod_name="dit-pod",
                pod_uid="dit-uid",
                node_name="approved-node",
                host="10.128.1.44",
            ),
            dict(
                pod_name="encoder-pod",
                pod_uid="encoder-uid",
                node_name="approved-node",
                host="10.128.1.45",
            ),
        ],
    )
    path.write_text(json.dumps(binding))
    path.chmod(0o600)
    args = dict(auth_token_file=str(token), node_binding_file=str(path))
    monkeypatch.setenv("HOSTNAME", "dit-pod")
    dit = peer.EncoderPeer(host="10.128.1.45", bridge_host="10.128.1.44", **args)
    monkeypatch.setenv("HOSTNAME", "encoder-pod")
    encoder = peer.EncoderPeer(host="10.128.1.44", bridge_host="10.128.1.45", **args)
    assert dit.node == encoder.node
    request = dict(
        op="open", bridge_host="10.128.1.44", bridge_port=32001, **dit.fields()
    )
    assert encoder.accepts(request)
    assert not encoder.accepts({**request, "peer_token": "c" * 64})
    assert not encoder.accepts({**request, "peer_node": "d" * 64})
    monkeypatch.setenv("HOSTNAME", "unapproved-pod")
    with pytest.raises(ValueError, match="this pod"):
        peer.EncoderPeer(host="10.128.1.44", bridge_host="10.128.1.45", **args)
    monkeypatch.setenv("HOSTNAME", "encoder-pod")
    binding["pods"][1]["pod_uid"] = "replacement-uid"
    path.write_text(json.dumps(binding))
    replaced = peer.EncoderPeer(host="10.128.1.44", bridge_host="10.128.1.45", **args)
    assert not replaced.accepts(request)
    binding["pods"][1]["node_name"] = "other-node"
    path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="pod binding"):
        peer.EncoderPeer(host="10.128.1.44", bridge_host="10.128.1.45", **args)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private file"):
        peer.EncoderPeer(host="10.128.1.44", bridge_host="10.128.1.45", **args)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_uid",
        "wrong_authority",
        "other_node",
        "wrong_address",
        "missing_pod",
        "extra_peer",
    ],
)
def test_scheduler_binding_rejects_invalid_assignment(tmp_path, monkeypatch, change):
    import json

    token = tmp_path / "token"
    token.write_text("a" * 64)
    token.chmod(0o600)
    value = dict(
        version=1,
        verified_by="pai-control-plane",
        node_name="node-a",
        pods=[
            dict(
                pod_name="dit", pod_uid="uid-a", node_name="node-a", host="10.128.1.44"
            ),
            dict(
                pod_name="encoder",
                pod_uid="uid-b",
                node_name="node-a",
                host="10.128.1.45",
            ),
        ],
    )
    if change == "duplicate_uid":
        value["pods"][1]["pod_uid"] = "uid-a"
    elif change == "wrong_authority":
        value["verified_by"] = "caller"
    elif change == "other_node":
        value["pods"][1]["node_name"] = "node-b"
    elif change == "wrong_address":
        value["pods"][0]["host"] = "10.128.1.46"
    elif change == "missing_pod":
        value["pods"][0]["pod_name"] = "other"
    elif change == "extra_peer":
        value["pods"].append(value["pods"][0])
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps(value))
    binding.chmod(0o600)
    monkeypatch.setenv("HOSTNAME", "dit")
    with pytest.raises(ValueError):
        peer.EncoderPeer(
            host="10.128.1.45",
            bridge_host="10.128.1.44",
            auth_token_file=str(token),
            node_binding_file=str(binding),
        )
