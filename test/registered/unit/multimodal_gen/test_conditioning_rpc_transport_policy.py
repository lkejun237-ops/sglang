# SPDX-License-Identifier: Apache-2.0
"""Plaintext HTTP is loopback-only; bind and TLS checks run before the model loads."""

if __name__ == "__main__":
    import runpy
    import sys
    from pathlib import Path

    test_file = Path(__file__).resolve()
    runner = (
        test_file.parents[4] / "benchmark/zing_h3_causal/run_conditioning_cpu_tests.py"
    )
    arguments = ["-x" if value == "-f" else value for value in sys.argv[1:]]
    sys.argv = [str(runner), str(test_file), *arguments]
    runpy.run_path(str(runner), run_name="__main__")

from types import SimpleNamespace

import pytest
from sglang.multimodal_gen.runtime.conditioning import cli
from sglang.multimodal_gen.runtime.conditioning.client import (
    ConditioningClient,
    validate_service_url,
)
from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8093",
        "http://localhost:8093",
        "http://[::1]:8093",
        "http://127.0.0.2:8093",
        "https://127.0.0.1:8093",
        "https://qwen.internal:8093",
        "https://10.0.0.5:8093/",
    ],
)
def test_allowed_service_urls(url):
    assert validate_service_url(url) == url.rstrip("/")


@pytest.mark.parametrize(
    "url",
    [
        "http://10.0.0.5:8093",
        "http://qwen.internal:8093",
        "http://localhost.evil.example",
        "http://127.0.0.1.evil.example",
        "ftp://127.0.0.1",
        "https://",
        "http://user:pw@127.0.0.1:8093",
        "",
        None,
    ],
)
def test_rejected_service_urls(url):
    with pytest.raises(ConditionError) as raised:
        validate_service_url(url)
    assert raised.value.code == "CONDITION_INVALID"


def test_client_constructor_rejects_plaintext_remote_host():
    with pytest.raises(ConditionError):
        ConditioningClient("http://qwen.internal:8093", "secret", "a" * 64)


def test_serve_rejects_external_plaintext_bind_before_loading_the_model(
    monkeypatch, tmp_path
):
    events = []
    profile = SimpleNamespace(profile_id="a" * 64)
    limits = SimpleNamespace()
    config = {"native": {"model_path": "m", "gpu_id": 0}, "tokens": {}}
    monkeypatch.setattr(
        cli, "load_configuration", lambda args: (config, profile, limits)
    )

    class Encoder:
        def __init__(self, *args, **kwargs):
            events.append("load")

    from sglang.multimodal_gen.runtime.conditioning import native

    monkeypatch.setattr(native, "NativeConditionEncoder", Encoder)
    monkeypatch.setattr(cli, "warm_encoder", lambda encoder: events.append("warm"))
    with pytest.raises(ValueError, match="TLS"):
        cli.main(["serve", "--config", "c", "--profile", "p", "--host", "0.0.0.0"])
    assert events == []


def test_validate_bind_accepts_loopback_and_tls_and_verified_proxy():
    cli.validate_bind("127.0.0.1", {})
    cli.validate_bind("::1", {})
    cli.validate_bind("0.0.0.0", {"tls": {"certificate": "c"}})
    cli.validate_bind("0.0.0.0", {"internal_auth_proxy_verified": True})
    with pytest.raises(ValueError):
        cli.validate_bind("0.0.0.0", {"internal_auth_proxy_verified": "yes"})


@pytest.mark.parametrize(
    "host,tls,warned", [("127.0.0.1", None, False), ("0.0.0.0", {"x": 1}, True)]
)
def test_serve_warns_when_leaving_the_validated_loopback_scope(
    monkeypatch, caplog, host, tls, warned
):
    import uvicorn
    from sglang.multimodal_gen.runtime.conditioning import native, service

    config = {
        "native": {"model_path": "m", "gpu_id": 0},
        "tokens": {},
        "tls": tls and {"certificate": "c", "private_key": "k", "ca": "a"},
    }
    monkeypatch.setattr(
        cli,
        "load_configuration",
        lambda args: (
            config,
            SimpleNamespace(),
            SimpleNamespace(
                max_condition_rows=1, max_active_requests=1, max_header_bytes=1
            ),
        ),
    )
    monkeypatch.setattr(native, "NativeConditionEncoder", lambda *a, **k: None)
    monkeypatch.setattr(service, "ConditioningService", lambda **k: None)
    monkeypatch.setattr(cli, "warm_encoder", lambda encoder: None)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    with caplog.at_level("WARNING", logger=cli.logger.name):
        cli.main(["serve", "--config", "c", "--profile", "p", "--host", host])
    assert any("experimental" in r.message for r in caplog.records) is warned
