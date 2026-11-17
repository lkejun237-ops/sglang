"""The test listener must resolve to a running, correctly scoped container."""

import json
import re

import pytest
from prepare_test_runtime_collector import CONTROL_RUNTIME


def namespace(obj):
    class Connection:
        def request(self, method, path):
            assert method == "GET"
            assert path == "/containers/" + "a" * 64 + "/json"

        def getresponse(self):
            return self

        status = 200

        def read(self, limit):
            return json.dumps(obj).encode()

        def close(self):
            pass

    ns = dict(
        json=json,
        re=re,
        CopyReadError=ValueError,
        CONTROL_PORTS={"room": 8092, "gateway": 18080},
        _UnixHTTP=Connection,
        _listener=lambda port: 100,
        _process=lambda pid: {"container_id": "a" * 64, "pid": pid},
    )
    exec(CONTROL_RUNTIME, ns)
    return ns


def container():
    return {
        "Id": "a" * 64,
        "Name": "/whitebox-sh-copy-room-any-new-release",
        "State": {"Running": True, "Pid": 90, "StartedAt": "now"},
        "Config": {"Env": ["WHITEBOX_ROLE=room"]},
        "HostConfig": {"NetworkMode": "host"},
    }


def test_follows_listener_across_release_names():
    obj = container()
    ns = namespace(obj)
    assert ns["_control_identity"]("room")[1]["container_name"] == obj["Name"][1:]
    obj["Name"] = "/whitebox-sh-copy-room-next-release"
    assert ns["_control_identity"]("room")[1]["container_name"] == obj["Name"][1:]


@pytest.mark.parametrize(
    "change",
    [
        {"Name": "/whitebox-production-room-new"},
        {"Config": {"Env": ["WHITEBOX_ROLE=gateway"]}},
        {"State": {"Running": False, "Pid": 90}},
        {"HostConfig": {"NetworkMode": "bridge"}},
        {"Id": "b" * 64},
    ],
)
def test_rejects_wrong_scope_role_or_runtime(change):
    obj = container()
    obj.update(change)
    with pytest.raises(ValueError, match="control_container_identity"):
        namespace(obj)["_control_identity"]("room")


def test_rejects_replaced_init_process():
    ns = namespace(container())
    ns["_process"] = lambda pid: {"container_id": ("a" if pid == 100 else "b") * 64}
    with pytest.raises(ValueError, match="listener_container_mismatch"):
        ns["_control_identity"]("room")
