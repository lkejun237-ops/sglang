import json

import pytest
from control_bindings import REFS_ENV, marker, parse_manifest, verify


def manifest():
    purposes = (
        "room_internal",
        "pool_room",
        "operator",
        "registration",
        "signing",
        "rtc",
        "room_admin",
    )
    return {
        "environment": "test-shanghai-copy",
        "region": "cn-shanghai",
        "pool_id": "fixture",
        "audiences": {
            "room_internal": "http://localhost:8001/internal/worlds/runtime",
            "pool_room": "http://localhost:8002/v1/role-pool/capacity",
        },
        "refs": {
            purpose: {
                "parameter_name": purpose,
                "parameter_id": "p-" + str(i),
                "version": 2,
            }
            for i, purpose in enumerate(purposes)
        },
    }


def inject(env, m, assignments):
    for variable, purpose in assignments.items():
        env[variable] = "unit-test-value-" + purpose
        env[variable + "_OOS_REF"] = marker(m["refs"][purpose])


@pytest.mark.parametrize("role", ["room", "gateway"])
def test_serving_components_check_the_actual_pool_route_and_version(role, tmp_path):
    m = manifest()
    route = tmp_path / "role-pool.json"
    route.write_text(
        json.dumps(
            {
                "room_token_env": "DEPLOYED_ROOM_CLIENT",
                "coordinator_url": "http://localhost:8002",
            }
        )
    )
    env = {
        REFS_ENV: json.dumps(m),
        "WHITEBOX_ROLE_POOL_CONFIG": str(route),
        "WHITEBOX_NATIVE_RTC_ENABLED": "1",
    }
    inject(
        env,
        m,
        {
            "WHITEBOX_INTERNAL_TOKEN": "room_internal",
            "WHITEBOX_RTC_AUTH_SECRET": "rtc",
            "DEPLOYED_ROOM_CLIENT": "pool_room",
        },
    )
    assert verify(role, environ=env) == verify(role, environ=env, preparing=True)
    del env["WHITEBOX_INTERNAL_TOKEN_OOS_REF"]
    env["DEPLOYED_ROOM_CLIENT_OOS_REF"] = "stale"
    with pytest.raises(ValueError, match="pool_room, room_internal"):
        verify(role, environ=env)
    inject(
        env,
        m,
        {
            "WHITEBOX_INTERNAL_TOKEN": "room_internal",
            "DEPLOYED_ROOM_CLIENT": "pool_room",
        },
    )
    route.write_text(
        json.dumps(
            {
                "room_token_env": "DEPLOYED_ROOM_CLIENT",
                "coordinator_url": "http://other.invalid",
            }
        )
    )
    with pytest.raises(ValueError, match="audience"):
        verify(role, environ=env)


def test_coordinators_publishers_and_workers_consume_only_their_scopes(tmp_path):
    m = manifest()
    base = {REFS_ENV: json.dumps(m)}
    pool = {
        "pool_id": "fixture",
        "room_token_env": "ROOM_CLIENT",
        "operator_token_env": "OP",
        "role_token_env": "REG",
        "signing_secret_env": "SIGN",
    }
    path = tmp_path / "c.json"
    path.write_text(json.dumps(pool))
    model = dict(base, SGLANG_ROLE_POOL_COORDINATOR_CONFIG=str(path))
    inject(
        model,
        m,
        {
            "OP": "operator",
            "REG": "registration",
            "SIGN": "signing",
            "ROOM_CLIENT": "pool_room",
        },
    )
    digest = verify("coordinator", environ=model)
    directory = dict(base, WHITEBOX_DYNAMIC_ROOMS="1")
    inject(
        directory,
        m,
        {
            "WHITEBOX_ROOM_ADMIN_TOKEN": "room_admin",
            "WHITEBOX_INTERNAL_TOKEN": "room_internal",
        },
    )
    assert verify("coordinator", environ=directory) == digest
    worker = dict(base)
    inject(worker, m, {"REG": "registration", "SIGN": "signing"})
    assert verify("worker", pool, environ=worker) == digest
    with pytest.raises(ValueError, match="different pool"):
        verify("worker", dict(pool, pool_id="other"), environ=worker)
    publisher = dict(base)
    inject(
        publisher,
        m,
        {"ROLE_POOL_OPERATOR_TOKEN": "operator", "ROLE_POOL_SIGNING_KEY": "signing"},
    )
    assert verify("model_publisher", environ=publisher) == digest
    with pytest.raises(ValueError, match="route_publisher"):
        verify("route_publisher", environ=publisher)
    route = dict(base)
    inject(
        route,
        m,
        {
            "WHITEBOX_INTERNAL_TOKEN": "room_internal",
            "ADAPTIVE_ROLE_POOL_ROOM_TOKEN": "pool_room",
        },
    )
    for name, purpose in (
        ("WHITEBOX_INTERNAL_TOKEN", "room_internal"),
        ("ADAPTIVE_ROLE_POOL_ROOM_TOKEN", "pool_room"),
    ):
        route[name + "_AUDIENCE"] = m["audiences"][purpose]
    assert verify("route_publisher", environ=route) == digest
    route["ADAPTIVE_ROLE_POOL_ROOM_TOKEN_AUDIENCE"] = "http://other.invalid"
    with pytest.raises(ValueError, match="pool_room"):
        verify("route_publisher", environ=route)


def test_manifest_disallows_aliases_literal_credentials_and_non_status_audiences():
    m = manifest()
    m["refs"]["pool_room"] = m["refs"]["operator"]
    with pytest.raises(ValueError, match="reused"):
        parse_manifest(json.dumps(m))
    m = manifest()
    m["refs"]["pool_room"]["value"] = "should-not-be-in-a-manifest"
    with pytest.raises(ValueError):
        parse_manifest(json.dumps(m))
    m = manifest()
    m["audiences"]["pool_room"] = "http://localhost:8002/v1/role-pool/drain"
    with pytest.raises(ValueError, match="audience"):
        parse_manifest(json.dumps(m))
