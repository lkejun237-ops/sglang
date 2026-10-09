# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from continuous_source_client import websocket_options
from encoder_overlap_profile_server import configure_monitoring
from run_encoder_overlap_matrix import endpoint_phase_groups, server_command


def test_profile_uses_separate_warmed_server_without_changing_five_samples():
    assert endpoint_phase_groups() == [
        (False, ["warmup", *[f"measured-{i}" for i in range(5)]]),
        (True, ["profile-warmup", "profile"]),
    ]
    assert endpoint_phase_groups(capture=True) == [(False, ["capture"])]
    assert endpoint_phase_groups(soak=True) == [(False, ["warmup", "measured-0"])]


def test_only_profile_server_changes_monitoring_and_keeps_model_arguments():
    regular = server_command("model", "config", 4)
    profile = server_command("model", "config", 4, profile_monitoring=True)
    assert regular[1:3] == ["-m", "sglang.multimodal_gen.runtime.launch_server"]
    assert "--realtime-session-max-lifetime-s" not in regular
    assert profile[1].endswith("encoder_overlap_profile_server.py")
    for flag in (
        "--realtime-session-max-lifetime-s",
        "--realtime-session-idle-timeout-s",
    ):
        assert profile[profile.index(flag) + 1] == "3600"
    assert (
        regular[regular.index("--model-path") :]
        == profile[profile.index("--model-path") :]
    )


def test_uvicorn_override_retains_protocol_and_does_not_mutate_package():
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return "started"

    package = SimpleNamespace(run=run)
    launch = SimpleNamespace(uvicorn=package)
    configure_monitoring(launch)
    assert (
        launch.uvicorn.run("app", ws="websockets-sansio", ws_max_queue=1) == "started"
    )
    assert calls == [
        (("app",), dict(ws="websockets-sansio", ws_max_queue=1, ws_ping_timeout=3600))
    ]
    assert package.run is run


def test_client_profile_ping_budget_is_finite_and_regular_default_unchanged():
    ordinary = websocket_options(profile=False, timeout=3600)
    profile = websocket_options(profile=True, timeout=3600)
    assert ordinary.pop("ping_timeout") == 20
    assert profile.pop("ping_timeout") == 3600
    assert ordinary == profile


def test_client_profile_options_preserve_negotiated_transport_budget():
    for profile in (False, True):
        options = websocket_options(
            profile=profile, timeout=3600, message_budget=48 << 20
        )
        assert options["max_size"] == 48 << 20
        assert options["max_queue"] == 1
        assert options["compression"] is None
    assert websocket_options(profile=False, timeout=600)["max_size"] == 64 << 20
