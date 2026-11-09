# SPDX-License-Identifier: Apache-2.0
"""CPU Room CLI boundary: real subprocesses/sockets, no model or cloud tasks."""

import asyncio
import importlib
import json
import sys
from pathlib import Path

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec


@pytest.fixture
def driver_module():
    return importlib.import_module(
        "sglang.multimodal_gen.runtime.realtime.whitebox.worldkit_room"
    )


FAKE_CLI = r"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
args=sys.argv[1:]
def arg(flag):return args[args.index(flag)+1]
root=Path(os.environ['TEST_ROOM_ROOT'])
log=root/'commands.jsonl'
with log.open('a') as file:file.write(json.dumps(args)+'\n')
assert args[0]=='room'
op=args[1]
room={'protocolVersion':1,'roomId':'sdk-room','roomEpoch':0,'runtimeHash':'a'*64,'contentHash':'c'*64}
if op=='create':
 auth=Path(arg('--authorization-file'));assert auth.stat().st_mode&0o777==0o600
 rows=json.loads(auth.read_text());assert len({r['actorId'] for r in rows})==len(rows)
 (root/'auth.json').write_text(json.dumps(rows))
 print(json.dumps({'kind':'worldkit-room-ready','schemaVersion':1,'roomId':'sdk-room','room':room,'artifact':{'worldBuildHash':'b'*64,'runtimeHash':'a'*64,'sourceHash':'c'*64,'archiveSha256':'d'*64},'capacity':2,'playConfiguration':{'playMode':'multiplayer','multiplayer':{'maxPlayers':2}}}),flush=True)
 while not (root/'closed').exists():time.sleep(.01)
elif op=='join':
 credential=Path(arg('--credential-file'));assert credential.stat().st_mode&0o777==0o600
 secret=json.loads(credential.read_text())['credential']
 row=next(r for r in json.loads((root/'auth.json').read_text()) if r['credential']==secret)
 request=arg('--request-id');target=root/(request+'.json')
 if not target.exists():
  session='session-'+row['actorId'];binding={'schemaVersion':1,'room':room,'playerSessionId':'member-'+row['subject'],'connectionEpoch':0,'viewportId':'view-'+row['subject'],'viewportBindingRevision':0,'mode':'local','render':{'sessionId':session,'epoch':0}}
  record={'kind':'worldkit-room-member','schemaVersion':1,'roomId':'sdk-room','binding':binding,'session':{'sessionId':session,'epoch':0,'baseUrl':os.environ['TEST_ROOM_URL'],'accessToken':hashlib.sha256(secret.encode()).hexdigest()[:48],'accessMode':'capability'}}
  target.write_text(json.dumps(record));target.chmod(0o600)
 record=json.loads(target.read_text());result={'kind':'worldkit-room-member','schemaVersion':1,'roomId':'sdk-room','memberFile':str(target),'sessionId':record['session']['sessionId'],'epoch':0,'binding':record['binding'],'localResources':{'uiRoot':str(root/'ui')}}
 if os.environ.get('TEST_DROP_JOIN')=='1' and not (root/(request+'.dropped')).exists():
  (root/(request+'.dropped')).touch();sys.exit(1)
 print(json.dumps(result))
elif op=='inspect':
 if '--room' in args and os.environ.get('TEST_LEAVE_PROGRESS'):
  retired=json.loads((root/'leaving.json').read_text())
  count=root/'cleanup-observations';seen=int(count.read_text()) if count.exists() else 0;count.write_text(str(seen+1))
  row={'playerSessionId':retired['binding']['playerSessionId'],'connectionEpoch':retired['binding']['connectionEpoch'],'phase':'left' if seen else 'cleaning','cleanupBlocked':False}
  mode=os.environ['TEST_LEAVE_PROGRESS']
  if mode=='blocked':row['cleanupBlocked']=True
  if mode=='pending':row['phase']='cleaning'
  if mode=='wrong-epoch':row['connectionEpoch']+=1
  if mode=='missing':sys.exit(1)
  observed_room={**room,'roomEpoch':1} if mode=='wrong-room' else room
  print(json.dumps({'room':observed_room,'sessions':None if mode=='malformed' else [row]}))
 else:print(json.dumps({'phase':'active'}))
elif op=='resume':
 target=Path(arg('--member-file'));record=json.loads(target.read_text());record['session']['sessionId']+='-resumed';record['binding']['render']['sessionId']=record['session']['sessionId'];record['binding']['connectionEpoch']+=1;record['binding']['viewportBindingRevision']+=1;target.write_text(json.dumps(record));target.chmod(0o600)
 if os.environ.get('TEST_DROP_RESUME')=='1':sys.exit(1)
 print(json.dumps({'kind':'worldkit-room-member','schemaVersion':1,'roomId':'sdk-room','memberFile':str(target),'sessionId':record['session']['sessionId'],'epoch':0,'binding':record['binding'],'localResources':{'uiRoot':str(root/'ui')}}))
elif op=='leave':
 if os.environ.get('TEST_LEAVE_PROGRESS'):(root/'leaving.json').write_text(Path(arg('--member-file')).read_text())
 print(json.dumps({'phase':'cleaning' if os.environ.get('TEST_BLOCK_LEAVE')=='1' or os.environ.get('TEST_LEAVE_PROGRESS') else 'left'}))
elif op=='close':
 (root/'closed').touch();print(json.dumps({'kind':'worldkit-room-control','schemaVersion':1,'roomId':'sdk-room','status':'closing'}))
else:raise ValueError('Unsupported room operation')
"""


class Renderer:
    def __init__(self, **kwargs):
        self.metadata = {"mode": "test"}
        self.events = []

    async def consume_ui(self, value):
        self.events.append(value)

    async def asset(self, path):
        return b"own-ui"

    async def close(self):
        pass


def configure(tmp_path, monkeypatch, module):
    entry = tmp_path / "worldkit.py"
    entry.write_text(FAKE_CLI)
    monkeypatch.setenv("WHITEBOX_NODE", sys.executable)
    monkeypatch.setenv("WHITEBOX_CLI_ENTRYPOINT", str(entry))
    monkeypatch.setenv("WHITEBOX_MAX_PLAYERS", "3")
    monkeypatch.setenv("TEST_ROOM_ROOT", str(tmp_path))

    # Only the public SDK readSession/UI-package boundary is mocked. CLI process
    # ownership, private files and authenticated WS input remain real.
    async def descriptor(member, *_, purpose="viewer", public_origin=None):
        record = json.loads(Path(member["memberFile"]).read_text())
        session = record["session"]
        base = session["baseUrl"]
        sid = session["sessionId"]
        return {
            **session,
            "kind": "worldkit-stream-session",
            "schemaVersion": 1,
            "protocolVersion": 1,
            "status": "ready",
            "worldId": "world",
            "worldBuildHash": "b" * 64,
            "uiBundleHash": "e" * 64,
            "playerBinding": record["binding"],
            "width": 32,
            "height": 32,
            "fps": 24,
            "media": {"kind": "pixels", "pixelFormat": "rgb24", "colorSpace": "srgb"},
            "ui": {"catalog": {"actions": {}}, "document": {}},
            "streams": {
                "ui": {
                    "url": base.replace("http:", "ws:") + f"/v1/sessions/{sid}/control"
                },
                "pixels": {
                    "url": base.replace("http:", "ws:") + f"/v1/sessions/{sid}/pixels"
                },
            },
        }

    async def prepared(*_):
        return {
            "archive": str(tmp_path / "delivery.tar.gz"),
            "operation": {
                "operationId": "op-original",
                "requestId": "request-original",
                "kind": "createRoom",
                "parametersSha256": "e" * 64,
            },
            "receipt": {},
            "identity": {
                "worldBuildHash": "b" * 64,
                "runtimeHash": "a" * 64,
                "sourceHash": "c" * 64,
                "archiveSha256": "d" * 64,
            },
        }

    monkeypatch.setattr(module, "prepare_delivery", prepared)
    monkeypatch.setattr(module, "read_member_session", descriptor)
    monkeypatch.setattr(
        module,
        "load_local_ui_package",
        lambda _: {"manifest": {}, "root": str(tmp_path)},
    )
    monkeypatch.setattr(module, "load_plugin", lambda _: Renderer)


def test_room_cli_two_private_views_use_binding_ack_and_leave_independently(
    tmp_path, monkeypatch, driver_module
):
    from urllib.parse import urlsplit

    from websockets.asyncio.server import serve

    async def run():
        observed = []

        async def socket(ws):
            assert "token=" not in ws.request.path
            assert "worldkit.cap." in ws.request.headers["Sec-WebSocket-Protocol"]
            path = urlsplit(ws.request.path).path
            if path.endswith("/control"):
                await ws.send(
                    json.dumps(
                        {"type": "session.ready", "epoch": 0, "role": "controller"}
                    )
                )
                await ws.send(
                    json.dumps(
                        {"type": "input.binding", "epoch": 0, "bindingSequence": 7}
                    )
                )
                async for wire in ws:
                    observed.append((path, json.loads(wire)))
            else:
                await ws.wait_closed()

        async with serve(
            socket, "127.0.0.1", 0, subprotocols=["worldkit.stream.v1"]
        ) as server:
            monkeypatch.setenv(
                "TEST_ROOM_URL",
                "http://127.0.0.1:" + str(server.sockets[0].getsockname()[1]),
            )
            configure(tmp_path, monkeypatch, driver_module)
            room = await driver_module.worldkit_room(
                world_id="source/world", spec=VideoSpec(32, 32), trace_id="test"
            )
            try:
                assert (
                    room.max_players == 2
                )  # sealed SDK capacity, not Host slot-pool size 3
                alice = await room.join(player_id="a" * 32, spec=VideoSpec(32, 32))
                bob = await room.join(player_id="b" * 32, spec=VideoSpec(32, 32))
                assert alice.descriptor["sessionId"] != bob.descriptor["sessionId"]
                assert alice.descriptor["accessToken"] != bob.descriptor["accessToken"]
                await bob.send_input(0, {"held_keys": ["KeyW"]})
                await asyncio.sleep(0.02)
                messages = [
                    value
                    for path, value in observed
                    if bob.descriptor["sessionId"] in path
                ]
                assert messages[0] == {
                    "type": "input.binding.applied",
                    "epoch": 0,
                    "bindingSequence": 7,
                }
                assert (
                    messages[1]["type"] == "input.state"
                    and messages[1]["bindingSequence"] == 7
                )
                assert messages[1]["input"]["heldKeys"] == ["KeyW"]
                await alice.close()
                assert not bob.closed
                assert bob.metadata["max_players"] == 2
                assert "accessToken" not in json.dumps(bob.metadata)
            finally:
                await room.close()
        commands = [
            json.loads(line)
            for line in (tmp_path / "commands.jsonl").read_text().splitlines()
        ]
        assert sum(row[:2] == ["room", "create"] for row in commands) == 1
        assert sum(row[:2] == ["room", "join"] for row in commands) == 2
        assert sum(row[:2] == ["room", "leave"] for row in commands) == 1
        assert sum(row[:2] == ["room", "close"] for row in commands) == 1

    asyncio.run(run())


def test_unknown_join_reuses_its_private_request_and_never_allocates_another_actor(
    tmp_path, monkeypatch, driver_module
):
    async def run():
        configure(tmp_path, monkeypatch, driver_module)
        monkeypatch.setenv("TEST_ROOM_URL", "http://127.0.0.1:1234")
        monkeypatch.setenv("TEST_DROP_JOIN", "1")

        async def start(view):
            view.renderer = Renderer()

        monkeypatch.setattr(driver_module.NativeCLIView, "start", start)
        room = await driver_module.worldkit_room(
            world_id="source/world", spec=VideoSpec(32, 32), trace_id="test"
        )
        try:
            with pytest.raises(ValueError, match="UNKNOWN"):
                await room.join(player_id="a" * 32, spec=VideoSpec(32, 32))
            view = await room.join(player_id="a" * 32, spec=VideoSpec(32, 32))
            assert view.descriptor["playerBinding"]["room"]["roomId"] == "sdk-room"
            joins = [
                json.loads(line)
                for line in (tmp_path / "commands.jsonl").read_text().splitlines()
                if json.loads(line)[:2] == ["room", "join"]
            ]
            assert joins[0] == joins[1]
            assert len(list(tmp_path.glob("join-*.json"))) == 1
        finally:
            await room.close()

    asyncio.run(run())


def test_unknown_resume_reads_the_replaced_private_member_without_resending_resume(
    tmp_path, monkeypatch, driver_module
):
    async def run():
        configure(tmp_path, monkeypatch, driver_module)
        monkeypatch.setenv("TEST_ROOM_URL", "http://127.0.0.1:1234")
        monkeypatch.setenv("TEST_DROP_RESUME", "1")

        async def start(view):
            view.renderer = Renderer()

        monkeypatch.setattr(driver_module.NativeCLIView, "start", start)
        room = await driver_module.worldkit_room(
            world_id="source/world", spec=VideoSpec(32, 32), trace_id="test"
        )
        try:
            original = await room.join(player_id="a" * 32, spec=VideoSpec(32, 32))
            with pytest.raises(ValueError, match="UNKNOWN"):
                await room.resume("a" * 32)
            recovered = await room.resume("a" * 32)
            assert (
                recovered.descriptor["sessionId"]
                == original.descriptor["sessionId"] + "-resumed"
            )
            assert (
                recovered.descriptor["playerBinding"]["playerSessionId"]
                == original.descriptor["playerBinding"]["playerSessionId"]
            )
            commands = [
                json.loads(line)
                for line in (tmp_path / "commands.jsonl").read_text().splitlines()
            ]
            assert sum(row[:2] == ["room", "resume"] for row in commands) == 1
            assert any(row[:2] == ["room", "inspect"] for row in commands)
        finally:
            await room.close()

    asyncio.run(run())


def test_deployment_capacity_cannot_silently_reduce_sealed_game(
    tmp_path, monkeypatch, driver_module
):
    async def run():
        configure(tmp_path, monkeypatch, driver_module)
        monkeypatch.setenv("WHITEBOX_MAX_PLAYERS", "1")
        with pytest.raises(ValueError, match="ROOM_HOST_CAPACITY_INSUFFICIENT"):
            await driver_module.worldkit_room(
                world_id="source/world", spec=VideoSpec(32, 32), trace_id="test"
            )

    asyncio.run(run())


@pytest.mark.parametrize("cancelled", [False, True])
def test_controlled_factory_emits_cleanup_receipt_after_original_cli_is_reaped(
    tmp_path, monkeypatch, driver_module, cancelled
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_startup import (
        FactoryCleanupReceipt,
        RoomStartupFailure,
        startup_cleanup_receipt,
    )

    async def run():
        configure(tmp_path, monkeypatch, driver_module)
        prepared = driver_module.prepare_delivery

        async def controlled_prepare(*args, **_):
            return await prepared(*args)

        monkeypatch.setattr(driver_module, "prepare_delivery", controlled_prepare)
        monkeypatch.setenv("WHITEBOX_MAX_PLAYERS", "3" if cancelled else "1")
        instances = []
        cls = driver_module.WorldkitRoom
        started = asyncio.Event()
        original_start = cls.start

        async def hold_started(instance):
            await original_start(instance)
            started.set()
            await asyncio.Event().wait()

        if cancelled:
            monkeypatch.setattr(cls, "start", hold_started)

        def owned_room(**kwargs):
            instance = cls(**kwargs)
            instances.append(instance)
            return instance

        monkeypatch.setattr(driver_module, "WorldkitRoom", owned_room)
        task = asyncio.create_task(
            driver_module.worldkit_room(
                world_id="source/world",
                spec=VideoSpec(32, 32),
                trace_id="op-original",
                create_context={
                    "frozenPublication": {
                        "playConfiguration": {
                            "playMode": "multiplayer",
                            "multiplayer": {"maxPlayers": 2},
                        }
                    }
                },
                verified_caller={"serviceId": "wtp", "subject": "manager"},
            )
        )
        if cancelled:
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
        with pytest.raises(
            asyncio.CancelledError if cancelled else RoomStartupFailure
        ) as caught:
            await task
        assert startup_cleanup_receipt(caught.value) == FactoryCleanupReceipt(
            "op-original"
        )
        if cancelled:
            assert task.cancelled()
        else:
            assert caught.value.code == "CLI_STARTUP_FAILED"
            assert "ROOM_HOST_CAPACITY_INSUFFICIENT" in str(caught.value.__cause__)
        assert instances[0].process.returncode is not None
        assert not instances[0].directory.exists()
        assert (tmp_path / "closed").exists()

    asyncio.run(run())


@pytest.mark.parametrize(
    "progress",
    [
        "normal",
        "blocked",
        "wrong-epoch",
        "wrong-room",
        "missing",
        "pending",
        "malformed",
    ],
)
def test_member_cleanup_observes_original_session_before_releasing_capacity(
    tmp_path, monkeypatch, driver_module, progress
):
    async def run():
        configure(tmp_path, monkeypatch, driver_module)
        monkeypatch.setenv("TEST_ROOM_URL", "http://127.0.0.1:1234")

        async def start(view):
            view.renderer = Renderer()

        monkeypatch.setattr(driver_module.NativeCLIView, "start", start)
        room = await driver_module.worldkit_room(
            world_id="source/world", spec=VideoSpec(32, 32), trace_id="test"
        )
        try:
            alice = await room.join(player_id="a" * 32, spec=room.spec)
            bob = await room.join(player_id="b" * 32, spec=room.spec)
            room.environment["TEST_LEAVE_PROGRESS"] = progress
            if progress == "pending":
                room.timeout = 0.25
            if progress == "normal":
                await alice.close()
                assert "a" * 32 not in room.members
                assert int((tmp_path / "cleanup-observations").read_text()) == 2
                del room.environment["TEST_LEAVE_PROGRESS"]
                await room.join(player_id="c" * 32, spec=room.spec)
            else:
                with pytest.raises(ValueError, match="ROOM_LEAVE_UNCONFIRMED"):
                    await alice.close()
                assert "a" * 32 in room.members
                with pytest.raises(ValueError, match="ROOM_PLAYER_CAPACITY_EXCEEDED"):
                    await room.join(player_id="c" * 32, spec=room.spec)
            assert not bob.closed
            commands = [
                json.loads(line)
                for line in (tmp_path / "commands.jsonl").read_text().splitlines()
            ]
            assert sum(row[:2] == ["room", "leave"] for row in commands) == 1
        finally:
            await room.close()

    asyncio.run(run())


def test_blocked_member_cleanup_keeps_capacity_and_does_not_stop_peer(
    tmp_path, monkeypatch, driver_module
):
    async def run():
        configure(tmp_path, monkeypatch, driver_module)
        monkeypatch.setenv("TEST_ROOM_URL", "http://127.0.0.1:1234")

        async def start(view):
            view.renderer = Renderer()

        monkeypatch.setattr(driver_module.NativeCLIView, "start", start)
        room = await driver_module.worldkit_room(
            world_id="source/world", spec=VideoSpec(32, 32), trace_id="test"
        )
        try:
            alice = await room.join(player_id="a" * 32, spec=room.spec)
            bob = await room.join(player_id="b" * 32, spec=room.spec)
            room.environment["TEST_BLOCK_LEAVE"] = "1"
            with pytest.raises(ValueError, match="ROOM_LEAVE_UNCONFIRMED"):
                await alice.close()
            with pytest.raises(ValueError, match="ROOM_PLAYER_CAPACITY_EXCEEDED"):
                await room.join(player_id="c" * 32, spec=room.spec)
            assert not bob.closed
            del room.environment["TEST_BLOCK_LEAVE"]
            await room.leave("a" * 32)
            replacement = await room.join(player_id="c" * 32, spec=room.spec)
            assert replacement.descriptor["sessionId"] != bob.descriptor["sessionId"]
            assert not bob.closed
        finally:
            await room.close()

    asyncio.run(run())


def test_slow_cancelled_join_does_not_block_peer_and_reconciles_before_release(
    tmp_path, monkeypatch, driver_module
):
    async def run():
        configure(tmp_path, monkeypatch, driver_module)
        monkeypatch.setenv("TEST_ROOM_URL", "http://127.0.0.1:1234")

        async def start(view):
            view.renderer = Renderer()

        monkeypatch.setattr(driver_module.NativeCLIView, "start", start)
        room = await driver_module.worldkit_room(
            world_id="source/world", spec=VideoSpec(32, 32), trace_id="test"
        )
        entered, release = asyncio.Event(), asyncio.Event()
        original_call = room.call
        first = True

        async def delayed(operation, *args):
            nonlocal first
            if operation == "join" and first:
                first = False
                result = await original_call(operation, *args)
                entered.set()
                await release.wait()
                return result
            return await original_call(operation, *args)

        room.call = delayed
        try:
            joining = asyncio.create_task(room.join(player_id="a" * 32, spec=room.spec))
            await asyncio.wait_for(entered.wait(), 2)
            bob = await asyncio.wait_for(
                room.join(player_id="b" * 32, spec=room.spec), 2
            )
            joining.cancel()
            with pytest.raises(asyncio.CancelledError):
                await joining
            assert len(room.members) == 2
            await room.leave("a" * 32)
            assert len(room.members) == 1 and not bob.closed
            commands = [
                json.loads(line)
                for line in (tmp_path / "commands.jsonl").read_text().splitlines()
            ]
            joins = [row for row in commands if row[:2] == ["room", "join"]]
            assert (
                joins[0] == joins[2]
            )  # reconcile the same request, credential and actor
        finally:
            release.set()
            await room.close()

    asyncio.run(run())


def test_resume_replaces_only_media_and_stale_view_cannot_leave_successor(
    tmp_path, monkeypatch, driver_module
):
    async def run():
        configure(tmp_path, monkeypatch, driver_module)
        monkeypatch.setenv("TEST_ROOM_URL", "http://127.0.0.1:1234")

        async def start(view):
            view.renderer = Renderer()

        monkeypatch.setattr(driver_module.NativeCLIView, "start", start)
        room = await driver_module.worldkit_room(
            world_id="source/world", spec=VideoSpec(32, 32), trace_id="test"
        )
        try:
            original = await room.join(player_id="a" * 32, spec=room.spec)
            bob = await room.join(player_id="b" * 32, spec=room.spec)
            resumed = await room.resume("a" * 32)
            await original.close()
            assert room.views["a" * 32] is resumed
            assert not resumed.closed and not bob.closed
            assert (
                resumed.metadata["player_binding"]["viewportId"]
                == original.metadata["player_binding"]["viewportId"]
            )
            assert resumed.metadata["player_binding"]["connectionEpoch"] == 1
        finally:
            await room.close()

    asyncio.run(run())


def test_ready_must_match_the_downloaded_receipt(tmp_path, monkeypatch, driver_module):
    async def run():
        configure(tmp_path, monkeypatch, driver_module)
        original = driver_module.prepare_delivery

        async def wrong_receipt(*args):
            value = await original(*args)
            value["identity"]["archiveSha256"] = "f" * 64
            return value

        monkeypatch.setattr(driver_module, "prepare_delivery", wrong_receipt)
        with pytest.raises(ValueError, match="ROOM_READY_IDENTITY_INVALID"):
            await driver_module.worldkit_room(
                world_id="source/world", spec=VideoSpec(32, 32), trace_id="test"
            )
        assert (tmp_path / "closed").exists()

    asyncio.run(run())


def test_real_room_runtime_consumes_private_driver_views_and_same_game_route(
    tmp_path, monkeypatch, driver_module
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import EngineFrame
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        MemoryRoomStore,
        RoomDirectory,
        new_id,
    )

    async def run():
        configure(tmp_path, monkeypatch, driver_module)
        monkeypatch.setenv("TEST_ROOM_URL", "http://127.0.0.1:1234")

        async def start(view):
            view.renderer = Renderer()

        async def frames(view):
            for index in range(3):
                yield EngineFrame(
                    index,
                    view.spec.pts(index),
                    bytes(view.spec.frame_bytes),
                    {"member": view.descriptor["playerBinding"]["playerSessionId"]},
                )

        monkeypatch.setattr(driver_module.NativeCLIView, "start", start)
        monkeypatch.setattr(driver_module.NativeCLIView, "frames", frames)
        directory = RoomDirectory(MemoryRoomStore(), {"pod-a": "http://a"})
        game = new_id()
        await directory.resolve(game, "source/world", create=True)
        runtime = RoomRuntime(
            directory=directory,
            owner_id="pod-a",
            driver_factory=driver_module.worldkit_room,
            spool=tmp_path / "spool",
            spec=VideoSpec(32, 32),
            max_players=3,
        )

        def request():
            return dict(
                game_id=game,
                world_id="source/world",
                player_id=new_id(),
                stream_id=new_id(),
                session_id=new_id(),
                user_id="test",
                trace_id="test",
                spec={"width": 32, "height": 32, "fps": 24},
            )

        try:
            alice, bob = await runtime.join(request()), await runtime.join(request())
            assert alice["game_id"] == bob["game_id"] == game
            assert alice["owner_epoch"] == bob["owner_epoch"] == runtime.epoch
            assert (
                alice["engine"]["player_binding"]["playerSessionId"]
                != bob["engine"]["player_binding"]["playerSessionId"]
            )
            a_frame, b_frame = (
                await runtime.next_frame(alice["stream_id"]),
                await runtime.next_frame(bob["stream_id"]),
            )
            assert a_frame["states"] != b_frame["states"]
            assert a_frame["provenance"]["stream_id"] == alice["stream_id"]
            await runtime.leave(alice["stream_id"])
            assert (await runtime.next_frame(bob["stream_id"]))["index"] == 1
            commands = [
                json.loads(line)
                for line in (tmp_path / "commands.jsonl").read_text().splitlines()
            ]
            assert sum(row[:2] == ["room", "create"] for row in commands) == 1
        finally:
            await runtime.close()

    asyncio.run(run())


def test_room_runtime_cancelled_join_retires_unknown_member_without_stream_entry(
    tmp_path, monkeypatch, driver_module
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        MemoryRoomStore,
        RoomDirectory,
        new_id,
    )

    async def run():
        configure(tmp_path, monkeypatch, driver_module)
        monkeypatch.setenv("TEST_ROOM_URL", "http://127.0.0.1:1234")

        async def start(view):
            view.renderer = Renderer()

        monkeypatch.setattr(driver_module.NativeCLIView, "start", start)
        directory = RoomDirectory(MemoryRoomStore(), {"pod-a": "http://a"})
        game = new_id()
        await directory.resolve(game, "source/world", create=True)
        runtime = RoomRuntime(
            directory=directory,
            owner_id="pod-a",
            driver_factory=driver_module.worldkit_room,
            spool=tmp_path / "spool",
            spec=VideoSpec(32, 32),
        )

        def request():
            return dict(
                game_id=game,
                world_id="source/world",
                player_id=new_id(),
                stream_id=new_id(),
                session_id=new_id(),
                user_id="test",
                trace_id="test",
                spec={"width": 32, "height": 32, "fps": 24},
            )

        try:
            bob = await runtime.join(request())
            driver = runtime.rooms[game]["driver"]
            original = driver.call
            entered = asyncio.Event()

            async def delayed(operation, *args):
                result = await original(operation, *args)
                if operation == "join" and not entered.is_set():
                    entered.set()
                    await asyncio.Event().wait()
                return result

            driver.call = delayed
            cancelled = request()
            joining = asyncio.create_task(runtime.join(cancelled))
            await asyncio.wait_for(entered.wait(), 2)
            joining.cancel()
            with pytest.raises(asyncio.CancelledError):
                await joining
            assert cancelled["stream_id"] not in runtime.players
            assert cancelled["player_id"] not in driver.members
            assert len(driver.members) == 1
            assert bob["stream_id"] in runtime.players and not driver.closed
            assert not driver.views[bob["player_id"]].closed
        finally:
            await runtime.close()

    asyncio.run(run())


def test_rejected_duplicate_join_cannot_rollback_existing_view(
    tmp_path, monkeypatch, driver_module
):
    async def run():
        configure(tmp_path, monkeypatch, driver_module)
        monkeypatch.setenv("TEST_ROOM_URL", "http://127.0.0.1:1234")

        async def start(view):
            view.renderer = Renderer()

        monkeypatch.setattr(driver_module.NativeCLIView, "start", start)
        room = await driver_module.worldkit_room(
            world_id="source/world", spec=VideoSpec(32, 32), trace_id="test"
        )
        try:
            view = await room.join(player_id="a" * 32, spec=room.spec)
            with pytest.raises(ValueError, match="ROOM_PLAYER_ALREADY_CONNECTED"):
                await room.join(player_id="a" * 32, spec=room.spec)
            await room.cancel_join("a" * 32)
            assert room.views["a" * 32] is view and not view.closed
        finally:
            await room.close()

    asyncio.run(run())


def test_direct_driver_leave_closes_local_view_before_releasing_slot(
    tmp_path, monkeypatch, driver_module
):
    async def run():
        configure(tmp_path, monkeypatch, driver_module)
        monkeypatch.setenv("TEST_ROOM_URL", "http://127.0.0.1:1234")
        closed = []

        class CountedRenderer(Renderer):
            async def close(self):
                closed.append(self)

        async def start(view):
            view.renderer = CountedRenderer()

        monkeypatch.setattr(driver_module.NativeCLIView, "start", start)
        room = await driver_module.worldkit_room(
            world_id="source/world", spec=VideoSpec(32, 32), trace_id="test"
        )
        try:
            view = await room.join(player_id="a" * 32, spec=room.spec)
            await room.leave("a" * 32)
            assert view.closed and closed == [view.renderer]
            assert not room.views and not room.members
        finally:
            await room.close()
        assert closed == [view.renderer]

    asyncio.run(run())


def test_model_member_join_uses_spectator_transport_and_public_keeps_viewer(
    tmp_path, monkeypatch, driver_module
):
    from urllib.parse import parse_qs, urlsplit

    from websockets.asyncio.server import serve

    async def run():
        observed = []

        async def socket(ws):
            observed.append(
                (
                    parse_qs(urlsplit(ws.request.path).query),
                    ws.request.headers["Sec-WebSocket-Protocol"],
                )
            )
            if ws.request.path.split("?")[0].endswith("/control"):
                await ws.send(
                    json.dumps(dict(type="session.ready", epoch=0, role="spectator"))
                )
            await ws.wait_closed()

        async with serve(
            socket, "127.0.0.1", 0, subprotocols=["worldkit.stream.v1"]
        ) as server:
            monkeypatch.setenv(
                "TEST_ROOM_URL",
                f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}",
            )
            configure(tmp_path, monkeypatch, driver_module)
            original = driver_module.read_member_session

            async def descriptor(
                member, entry, *, purpose="viewer", public_origin=None
            ):
                result = await original(member, entry)
                result["modelContext"] = dict(
                    protocol="worldkit-model-context-v1",
                    required=True,
                    configHash="c" * 64,
                    gameBuildHash="b" * 64,
                    compilerVersion=1,
                    visible=False,
                )
                if purpose == "model-source":
                    result["accessToken"] = "f" * 48
                return result

            monkeypatch.setattr(driver_module, "read_member_session", descriptor)
            room = await driver_module.worldkit_room(
                world_id="source/world", spec=VideoSpec(32, 32), trace_id="model-member"
            )
            try:
                view = await asyncio.wait_for(
                    room.join(player_id="a" * 32, spec=room.spec), 2
                )
                assert view.external_input and not view.controller_ready
                assert len(observed) == 2
                assert all(query["role"] == ["spectator"] for query, _ in observed)
                assert all(
                    "worldkit.cap." + "f" * 48 in protocols for _, protocols in observed
                )
                public = await room.public_session("a" * 32, "https://public.example")
                assert public["accessToken"] != view.descriptor["accessToken"]
            finally:
                await room.close()

    asyncio.run(run())
