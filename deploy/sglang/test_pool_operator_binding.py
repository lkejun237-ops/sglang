import json
import subprocess

import pytest
from pool_operator_binding import launch_args


@pytest.fixture
def binding():
    return {
        "pool_id": "fixture",
        "parameter_name": "fixture-operator",
        "parameter_id": "p-fixture",
        "version": 2,
    }


def responses(
    monkeypatch, binding, *, phase="issued", oos_version=2, local_version="2"
):
    def run(args, **kwargs):
        if args[0] == "aliyun":
            assert args[args.index("--WithDecryption") + 1] == "false"
            assert args[args.index("--ParameterVersion") + 1] == "2"
            return json.dumps(
                {
                    "Parameter": {
                        "Type": "Secret",
                        "Name": binding["parameter_name"],
                        "Id": "p-fixture",
                        "ParameterVersion": oos_version,
                        "Tags": {
                            "environment": "test-shanghai-copy",
                            "role-pool": "fixture",
                            "phase": phase,
                        },
                    }
                }
            )
        assert args[:4] == [
            "podman",
            "secret",
            "inspect",
            "sglang-operator-p-fixture-v2",
        ]
        assert "--showsecret" not in args
        return f"object2 sglang-operator-p-fixture-v{local_version}"

    monkeypatch.setattr(subprocess, "check_output", run)


def test_launch_argument_pins_local_object(monkeypatch, binding):
    responses(monkeypatch, binding)
    args = launch_args(binding, "test")
    assert args == ["--secret", "object2,type=env,target=ROLE_POOL_OPERATOR_TOKEN"]


@pytest.mark.parametrize(
    "changes", [{"phase": "staged"}, {"oos_version": 3}, {"local_version": "1"}]
)
def test_binding_rejects_unreleased_or_stale_version(monkeypatch, binding, changes):
    responses(monkeypatch, binding, **changes)
    with pytest.raises(ValueError):
        launch_args(binding, "test")


def test_room_credentials_cannot_use_an_operator_pin(monkeypatch, binding):
    from pool_operator_binding import check_reference

    def metadata(*args, **kwargs):
        return json.dumps(
            {
                "Parameter": {
                    "Type": "Secret",
                    "Name": binding["parameter_name"],
                    "Id": binding["parameter_id"],
                    "ParameterVersion": binding["version"],
                    "Tags": {
                        "environment": "test-shanghai-copy",
                        "role-pool": "fixture",
                        "purpose": "operator-rotation",
                    },
                }
            }
        )

    monkeypatch.setattr(subprocess, "check_output", metadata)
    with pytest.raises(ValueError, match="purpose"):
        check_reference(binding, "test", "pool_room")
    assert check_reference(binding, "test", "operator") == ("p-fixture", 2)
