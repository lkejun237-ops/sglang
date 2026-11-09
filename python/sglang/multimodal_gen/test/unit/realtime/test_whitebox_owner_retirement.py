# SPDX-License-Identifier: Apache-2.0
"""Original control-record retirement: no GPU, cloud, or business retry."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from test_whitebox_room_control import CALLER, SECRET, request_body, ticket

from sglang.multimodal_gen.runtime.realtime.whitebox.room_control import (
    RoomControlCapabilities,
    install_gateway_room_control,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    MemoryRoomStore,
    RoomDirectory,
)


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


async def original():
    store = MemoryRoomStore()
    directory = RoomDirectory(store, {"a": "http://room"}, clock=lambda: 1793258394)
    body = request_body()
    route = await directory.control_route(CALLER, body, create=True)
    await directory.touch(route["game_id"], "a", "2" * 32, end=True)
    key = next(key for key in store.values if key.startswith("control:"))
    route = json.loads(store.values[route["game_id"]])
    return directory, body, key, route


def stored_retirement(body, key, route):
    """Fixture for a previously accepted maintenance proof, not a wire input."""
    return {
        "schema_version": 1,
        "control_key": key,
        "invocation_sha256": canonical_digest(
            {k: body[k] for k in ("createContext", "videoSpec", "cliFps")}
        ),
        "game_id": route["game_id"],
        "owner_id": route["owner_id"],
        "owner_epoch": route["owner_epoch"],
        "container_id": "c" * 64,
        "evidence_sha256": "e" * 64,
        "stopped_at": "2026-10-29T09:06:09.285092029Z",
    }


def test_query_original_retired_create_after_runtime_restart_never_dispatches():
    async def run():
        directory, body, key, route = await original()
        value = json.loads(directory.store.values[key])
        value["retirement"] = stored_retirement(body, key, route)
        directory.store.values[key] = json.dumps(value)
        calls = []

        async def runtime(request):
            calls.append(request)
            return httpx.Response(200, json={"status": "unknown"})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(runtime)
        ) as upstream:
            app = FastAPI()
            install_gateway_room_control(
                app,
                SimpleNamespace(
                    directory=directory,
                    owners={"a": "http://room"},
                    token="internal",
                    client=upstream,
                ),
                verifier=RoomControlCapabilities(SECRET, {"wtp"}, now=lambda: 100),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://gateway"
            ) as client:
                for index, action in enumerate(
                    ["getOperation", "createRoom", "getOperation"], 1
                ):
                    body["action"] = action
                    raw = json.dumps(body).encode()
                    response = await client.post(
                        "/v1/whitebox/room-control",
                        content=raw,
                        headers={
                            "Authorization": "Bearer "
                            + ticket(raw, jti=str(index) * 32)
                        },
                    )
                    assert response.status_code == 200
                    assert response.json() == {
                        "operationId": "op-original",
                        "parametersSha256": "a" * 64,
                        "status": "failed",
                        "cleanupComplete": True,
                        "errorCode": "ROOM_OWNER_RETIRED",
                    }
        assert not calls

    asyncio.run(run())


def test_ended_route_without_proof_remains_unknown_not_retired():
    async def run():
        directory, body, key, route = await original()
        response = await directory.control_route(CALLER, request_body("getOperation"))
        assert response == route and "retirement" not in json.loads(
            directory.store.values[key]
        )

    asyncio.run(run())


@pytest.mark.parametrize(
    "field,value",
    [
        ("control_key", "control:wrong"),
        ("game_id", "f" * 32),
        ("invocation_sha256", "f" * 64),
        ("owner_epoch", None),
        ("evidence_sha256", ""),
    ],
)
def test_malformed_or_changed_retirement_record_cannot_be_consumed(field, value):
    async def run():
        directory, body, key, route = await original()
        record = json.loads(directory.store.values[key])
        proof = stored_retirement(body, key, route)
        proof[field] = value
        record["retirement"] = proof
        directory.store.values[key] = json.dumps(record)
        with pytest.raises(ValueError):
            await directory.control_route(CALLER, request_body("getOperation"))

    asyncio.run(run())


def observations(body, key, route):
    from sglang.multimodal_gen.runtime.realtime.whitebox.owner_retirement import (
        STARTUP_SOURCE_PATHS,
    )

    sources = {
        path: ("reviewed source " + path).encode() for path in STARTUP_SOURCE_PATHS
    }
    container = {
        "Id": "c" * 64,
        "Image": "sha256:" + "d" * 64,
        "State": {
            "Status": "exited",
            "Running": False,
            "Pid": 0,
            "StartedAt": "2026-10-29T05:42:54Z",
            "FinishedAt": "2026-10-29T09:06:09Z",
        },
        "HostConfig": {"RestartPolicy": {"Name": "no"}},
    }
    audit = {
        "schema_version": 1,
        "kind": "native-create-owner-retirement",
        "caller": CALLER,
        "operation_id": "op-original",
        "parameters_sha256": "a" * 64,
        "game_id": route["game_id"],
        "owner_id": "a",
        "owner_epoch": "2" * 32,
        "container_id": container["Id"],
        "image_id": container["Image"],
        "started_at": container["State"]["StartedAt"],
        "stopped_at": container["State"]["FinishedAt"],
        "epoch_observation": {
            "container_id": container["Id"],
            "owner_id": "a",
            "owner_epoch": "2" * 32,
            "at": "2026-10-29T07:25:00Z",
            "evidence_sha256": "b" * 64,
        },
        "stop_receipt": {
            "container_id": container["Id"],
            "action": "stop",
            "at": "2026-10-29T09:06:01Z",
            "finished_at": container["State"]["FinishedAt"],
            "evidence_sha256": "f" * 64,
        },
        "startup_boundary": {
            "classification": "sealed-loader-before-driver",
            "error_code": "CREATOR_MULTIPLAYER_EVIDENCE_INVALID",
            "review_evidence_sha256": "9" * 64,
            "source_paths": {k: "/app/" + k for k in sources},
            "source_sha256": {
                k: hashlib.sha256(v).hexdigest() for k, v in sources.items()
            },
        },
    }
    return audit, container, sources


def verified(directory, body, key, route, *, mutate=None):
    from sglang.multimodal_gen.runtime.realtime.whitebox.owner_retirement import (
        verify_retired_create,
    )

    audit, container, sources = observations(body, key, route)
    audit.update(
        invocation_sha256=canonical_digest(
            {k: body[k] for k in ("createContext", "videoSpec", "cliFps")}
        ),
        control_value_sha256=hashlib.sha256(
            directory.store.values[key].encode()
        ).hexdigest(),
        route_value_sha256=hashlib.sha256(
            directory.store.values[route["game_id"]].encode()
        ).hexdigest(),
    )
    if mutate:
        mutate(audit, container, sources)
    raw = json.dumps(audit).encode()
    return verify_retired_create(
        CALLER,
        body,
        directory.store.values[key],
        directory.store.values[route["game_id"]],
        audit_bytes=raw,
        approved_audit_sha256=hashlib.sha256(raw).hexdigest(),
        container=container,
        startup_log=b"2026-10-29T07:20:00Z CLI startup failed trace_id=op-original code=CREATOR_MULTIPLAYER_EVIDENCE_INVALID\n",
        source_bytes=sources,
    )


def test_maintenance_requires_full_original_scope_and_positive_stopped_owner():
    async def run():
        directory, body, key, route = await original()
        proof = verified(directory, body, key, route)
        result = await directory.confirm_retired_create(CALLER, body, proof)
        assert result["status"] == "failed" and result["cleanupComplete"] is True
        assert await directory.confirm_retired_create(CALLER, body, proof) == result
        restored = RoomDirectory(directory.store, {"a": "http://room"})
        del directory.store.values[
            route["game_id"]
        ]  # Original route retention expires after proof acceptance.
        assert await restored.control_route(CALLER, request_body("getOperation")) == {
            "retired_create": result
        }

    asyncio.run(run())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a, c, s: c["State"].update(Running=True, Pid=42),
        lambda a, c, s: c["HostConfig"]["RestartPolicy"].update(Name="always"),
        lambda a, c, s: c.update(Id="f" * 64),
        lambda a, c, s: c.update(Image="sha256:" + "f" * 64),
        lambda a, c, s: a.update(owner_epoch="3" * 32),
        lambda a, c, s: a["epoch_observation"].update(owner_epoch="3" * 32),
        lambda a, c, s: a["stop_receipt"].update(action="observed-absent"),
        lambda a, c, s: a["startup_boundary"].update(classification="unknown"),
        lambda a, c, s: s.update({next(iter(s)): b"unreviewed source"}),
        lambda a, c, s: a.update(parameters_sha256="f" * 64),
        lambda a, c, s: a.update(caller={**CALLER, "subject": "other"}),
    ],
)
def test_incomplete_or_changed_maintenance_evidence_cannot_mint_proof(mutate):
    async def run():
        directory, body, key, route = await original()
        with pytest.raises(ValueError):
            verified(directory, body, key, route, mutate=mutate)
        assert "retirement" not in json.loads(directory.store.values[key])

    asyncio.run(run())


def test_control_cas_and_route_binding_are_rechecked_before_retirement():
    async def run():
        for changed in ["control", "route"]:
            directory, body, key, route = await original()
            proof = verified(directory, body, key, route)
            target = key if changed == "control" else route["game_id"]
            value = json.loads(directory.store.values[target])
            value["changed"] = True
            directory.store.values[target] = json.dumps(value)
            with pytest.raises(ValueError):
                await directory.confirm_retired_create(CALLER, body, proof)
            assert "retirement" not in json.loads(directory.store.values[key])
        directory, body, key, route = await original()
        with pytest.raises(ValueError):
            await directory.confirm_retired_create(
                CALLER, body, {"cleanupComplete": True}
            )

    asyncio.run(run())


def test_same_maintenance_proof_concurrently_commits_only_once():
    async def run():
        directory, body, key, route = await original()
        proof = verified(directory, body, key, route)
        original_cas = directory.store.cas
        writes = []

        async def cas(*args, **kwargs):
            await asyncio.sleep(0)
            result = await original_cas(*args, **kwargs)
            if result:
                writes.append(args)
            return result

        directory.store.cas = cas
        results = await asyncio.gather(
            *(directory.confirm_retired_create(CALLER, body, proof) for _ in range(2))
        )
        assert results[0] == results[1] and len(writes) == 1 and writes[0][-1] is None

    asyncio.run(run())


def test_failed_retirement_cas_does_not_overwrite_a_changed_original_record():
    async def run():
        directory, body, key, route = await original()
        proof = verified(directory, body, key, route)

        async def cas(*args, **kwargs):
            value = json.loads(directory.store.values[key])
            value["invocation"]["cliFps"] = 1
            directory.store.values[key] = json.dumps(value)
            return False

        directory.store.cas = cas
        with pytest.raises(ValueError):
            await directory.confirm_retired_create(CALLER, body, proof)
        assert "retirement" not in json.loads(directory.store.values[key])
        del directory.store.values[key]
        with pytest.raises(ValueError):
            await directory.confirm_retired_create(CALLER, body, proof)
        assert key not in directory.store.values

    asyncio.run(run())


def test_existing_retirement_cannot_match_another_caller_or_frozen_invocation():
    async def run():
        directory, body, key, route = await original()
        proof = verified(directory, body, key, route)
        await directory.confirm_retired_create(CALLER, body, proof)
        assert (
            await directory.control_route(
                {**CALLER, "subject": "another"}, request_body("getOperation")
            )
            is None
        )
        changed = request_body("getOperation")
        changed["createContext"]["frozenPublication"]["world"]["publicationId"] = (
            "other"
        )
        with pytest.raises(ValueError):
            await directory.control_route(CALLER, changed)

    asyncio.run(run())


def test_retirement_uses_original_redis_adapter_cas_without_a_new_ttl():
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import RedisRoomStore

    async def run():
        directory, body, key, route = await original()
        proof = verified(directory, body, key, route)
        original_store = directory.store
        prefix = "original:rooms:"
        calls = []

        class Client:
            async def get(self, key):
                return original_store.values.get(key.removeprefix(prefix))

            async def eval(
                self,
                script,
                numkeys,
                redis_key,
                route_key,
                old,
                new,
                retention,
                expected_route,
            ):
                calls.append(
                    (script, numkeys, redis_key, route_key, old, new, retention)
                )
                assert numkeys == 2 and redis_key == prefix + key
                assert route_key == prefix + route["game_id"]
                return int(
                    await original_store.cas(
                        key,
                        old,
                        new,
                        None,
                        expected_route=(route["game_id"], expected_route),
                    )
                )

        client = Client()
        directory.store = RedisRoomStore(lambda: client, prefix)
        result = await directory.confirm_retired_create(CALLER, body, proof)
        assert result["cleanupComplete"] is True
        assert (
            len(calls) == 1
            and calls[0][0] == RedisRoomStore.SCRIPT
            and calls[0][-1] == 0
        )
        assert await directory.control_route(CALLER, request_body("getOperation")) == {
            "retired_create": result
        }

    asyncio.run(run())


def test_original_single_control_and_retirement_share_nonexpiring_redis_cas():
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import RedisRoomStore

    async def run():
        calls = []

        class Client:
            async def eval(self, script, count, *values):
                calls.append((count, values))
                return 1

        store = RedisRoomStore(Client, "original:")
        # Existing single-player control uses zero; retirement supplies no TTL.
        # Both must keep their original tombstone rather than fail or expire it.
        for retention in (0, None):
            assert await store.cas("single-control:original", "old", "new", retention)
        assert calls == [(1, ("original:single-control:original", "old", "new", 0))] * 2
        for invalid in (-1, True, 1.5):
            with pytest.raises(ValueError, match="Invalid room record retention"):
                await store.cas("single-control:original", "old", "new", invalid)
        assert len(calls) == 2

    asyncio.run(run())


def maintenance_payload(directory, body, key, route):
    import base64

    audit, container, sources = observations(body, key, route)
    references = {
        "epoch": b"original epoch audit",
        "stop": b"completed stop audit",
        "source_review": b"independent review of exact pre-driver source boundary",
    }
    audit["epoch_observation"]["evidence_sha256"] = hashlib.sha256(
        references["epoch"]
    ).hexdigest()
    audit["stop_receipt"]["evidence_sha256"] = hashlib.sha256(
        references["stop"]
    ).hexdigest()
    audit["startup_boundary"]["review_evidence_sha256"] = hashlib.sha256(
        references["source_review"]
    ).hexdigest()
    audit.update(
        invocation_sha256=canonical_digest(
            {k: body[k] for k in ("createContext", "videoSpec", "cliFps")}
        ),
        control_value_sha256=hashlib.sha256(
            directory.store.values[key].encode()
        ).hexdigest(),
        route_value_sha256=hashlib.sha256(
            directory.store.values[route["game_id"]].encode()
        ).hexdigest(),
    )
    raw = json.dumps(audit).encode()

    def encode(data):
        return base64.b64encode(data).decode()

    return {
        "audit": encode(raw),
        "container": container,
        "startup_log": encode(
            b"2026-10-29T07:20:00Z CLI startup failed trace_id=op-original code=CREATOR_MULTIPLAYER_EVIDENCE_INVALID\n"
        ),
        "sources": {k: encode(v) for k, v in sources.items()},
        "references": {k: encode(v) for k, v in references.items()},
    }, hashlib.sha256(raw).hexdigest()


def test_maintenance_entry_defaults_to_read_only_and_explicit_apply_is_idempotent():
    from sglang.multimodal_gen.runtime.realtime.whitebox.owner_retirement_maintenance import (
        reconcile_retirement_payload,
    )

    async def run():
        directory, body, key, route = await original()
        payload, approval = maintenance_payload(directory, body, key, route)
        before = dict(directory.store.values)
        dry = await reconcile_retirement_payload(
            directory, payload, approval, apply=False
        )
        assert (
            dry["validated"] is True
            and dry["applied"] is False
            and directory.store.values == before
        )
        actual = await reconcile_retirement_payload(
            directory, payload, approval, apply=True
        )
        assert actual["applied"] is True
        repeated = await reconcile_retirement_payload(
            directory, payload, approval, apply=True
        )
        assert (
            repeated["already_applied"] is True
            and repeated["operation"] == actual["operation"]
        )

    asyncio.run(run())


@pytest.mark.parametrize(
    "change", ["approval", "reference", "source", "missing-original"]
)
def test_maintenance_entry_missing_evidence_or_record_cannot_write(change):
    from sglang.multimodal_gen.runtime.realtime.whitebox.owner_retirement_maintenance import (
        reconcile_retirement_payload,
    )

    async def run():
        directory, body, key, route = await original()
        payload, approval = maintenance_payload(directory, body, key, route)
        if change == "approval":
            approval = "0" * 64
        elif change == "reference":
            payload["references"]["source_review"] = ""
        elif change == "source":
            payload["sources"].pop(next(iter(payload["sources"])))
        else:
            del directory.store.values[key]
        before = dict(directory.store.values)
        with pytest.raises(ValueError):
            await reconcile_retirement_payload(directory, payload, approval, apply=True)
        assert directory.store.values == before

    asyncio.run(run())


def load_retirement_tool():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[6]
        / "deploy/whitebox/confirm_retired_create.py"
    )
    spec = importlib.util.spec_from_file_location("retirement_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_host_tool_reads_real_docker_observations_and_never_starts_the_old_owner(
    tmp_path,
):
    import base64
    import io
    import tarfile

    async def run():
        directory, body, key, route = await original()
        payload, approval = maintenance_payload(directory, body, key, route)
        audit = json.loads(base64.b64decode(payload["audit"]))
        audit_path = tmp_path / "audit.json"
        audit_path.write_bytes(base64.b64decode(payload["audit"]))
        references = {}
        for name, data in payload["references"].items():
            references[name] = tmp_path / name
            references[name].write_bytes(base64.b64decode(data))
        calls = []

        def docker(args, limit):
            calls.append(args)
            if args[1] == "inspect":
                return json.dumps([payload["container"]]).encode()
            if args[1] == "logs":
                return base64.b64decode(payload["startup_log"])
            assert args[1] == "cp"
            path = args[2].split(":", 1)[1]
            logical = next(
                k
                for k, v in audit["startup_boundary"]["source_paths"].items()
                if v == path
            )
            data = base64.b64decode(payload["sources"][logical])
            out = io.BytesIO()
            with tarfile.open(fileobj=out, mode="w") as archive:
                entry = tarfile.TarInfo("source.ts")
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
            return out.getvalue()

        tool = load_retirement_tool()
        observed = tool.collect_observations(
            audit_path, approval, references, run=docker
        )
        assert observed == payload
        assert {call[1] for call in calls} == {"inspect", "cp", "logs"}
        assert len([call for call in calls if call[1] == "cp"]) == 12
        with pytest.raises(ValueError):
            tool.collect_observations(audit_path, "0" * 64, references, run=docker)

    asyncio.run(run())


def test_route_change_between_observation_and_control_cas_rejects_retirement():
    async def run():
        directory, body, key, route = await original()
        proof = verified(directory, body, key, route)
        original_cas = directory.store.cas

        async def race(*args, **kwargs):
            changed = json.loads(directory.store.values[route["game_id"]])
            changed["owner_epoch"] = "3" * 32
            directory.store.values[route["game_id"]] = json.dumps(changed)
            return await original_cas(*args, **kwargs)

        directory.store.cas = race
        with pytest.raises(ValueError):
            await directory.confirm_retired_create(CALLER, body, proof)
        assert "retirement" not in json.loads(directory.store.values[key])

    asyncio.run(run())


def test_retirement_traverses_original_internal_directory_http_but_no_new_business_proof_field():
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        HTTPRoomDirectory,
        install_room_routes,
    )

    async def run():
        directory, body, key, route = await original()
        proof = verified(directory, body, key, route)
        await directory.confirm_retired_create(CALLER, body, proof)
        coordinator = FastAPI()
        install_room_routes(coordinator, directory, "internal")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(coordinator), base_url="http://coordinator"
        ) as internal:
            remote = HTTPRoomDirectory(internal, "http://coordinator", "internal")

            def no_runtime(request):
                pytest.fail("Retired create must not dispatch a new runtime operation")

            async with httpx.AsyncClient(
                transport=httpx.MockTransport(no_runtime)
            ) as runtime:
                gateway = FastAPI()
                install_gateway_room_control(
                    gateway,
                    SimpleNamespace(
                        directory=remote,
                        owners={"a": "http://room"},
                        token="internal",
                        client=runtime,
                    ),
                    verifier=RoomControlCapabilities(SECRET, {"wtp"}, now=lambda: 100),
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(gateway), base_url="http://gateway"
                ) as client:
                    query = request_body("getOperation")
                    raw = json.dumps(query).encode()
                    response = await client.post(
                        "/v1/whitebox/room-control",
                        content=raw,
                        headers={"Authorization": "Bearer " + ticket(raw)},
                    )
                    assert response.json()["errorCode"] == "ROOM_OWNER_RETIRED"
                    query["retirement"] = {"cleanupComplete": True}
                    raw = json.dumps(query).encode()
                    response = await client.post(
                        "/v1/whitebox/room-control",
                        content=raw,
                        headers={
                            "Authorization": "Bearer " + ticket(raw, jti="2" * 32)
                        },
                    )
                    assert response.status_code == 400
                    assert (
                        await internal.post(
                            "/v1/whitebox/rooms/confirm-retired-create", json={}
                        )
                    ).status_code == 404

    asyncio.run(run())
