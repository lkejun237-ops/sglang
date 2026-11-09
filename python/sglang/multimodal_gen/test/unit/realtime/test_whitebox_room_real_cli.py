# SPDX-License-Identifier: Apache-2.0
"""Opt-in actual sealed SDK Room/Chromium/native frames; no cloud, model or UI compositor."""

import asyncio
import json
import logging
import os
import re
import signal
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec


def test_local_fixture_diagnostics_redact_urls_and_bound_bytes():
    # Evaluate only the diagnostic functions; SDK imports and browser launch are
    # deliberately excluded from this portable probe.
    fixture = Path(__file__).with_name("worldkit_room_delivery_fixture.mjs").read_text()
    source = fixture.split("const diagnosticFile =", 1)[1].split(
        "if (diagnosticFile) {", 1
    )[0]
    script = "const source = " + json.dumps("const diagnosticFile =" + source) + ";\n"
    script += r"""
import assert from 'node:assert/strict';
const writes = [];
const {safeUrl, diagnostic} = new Function('appendFileSync', 'process',
  source + '; return {safeUrl, diagnostic};')(
    (_file, line) => writes.push(line),
    {env:{WHITEBOX_TEST_BROWSER_DIAGNOSTICS:'/unused-probe'}});
for (const url of ['data:text/html,<html>PRIVATE_MARKER</html>',
    'file:///PRIVATE_MARKER', 'javascript:PRIVATE_MARKER', 'about:PRIVATE_MARKER']) {
  assert.equal(safeUrl(url), '<unsupported-url>');
}
assert.equal(safeUrl('https://USER:PRIVATE_MARKER@example.test/path?q=PRIVATE_MARKER#PRIVATE_MARKER'),
  'https://example.test/path');
assert.equal(safeUrl('http://example.test/path'), 'http://example.test/path');
assert.equal(safeUrl('not a url PRIVATE_MARKER'), '<invalid-url>');
assert.ok(Buffer.byteLength(safeUrl('http://example.test/' + 'x'.repeat(1_000_000))) <= 512);
assert.ok(Buffer.byteLength(safeUrl('https://example.test/' + '界'.repeat(10_000))) <= 512);
diagnostic('oversized', {message:'界'.repeat(4096)});
assert.equal(writes.length, 0);
for (let i=0; i<1000; i++) diagnostic('probe', {message:'界'.repeat(900)});
assert.ok(writes.length > 0 && writes.length < 256);
assert.ok(writes.every(line => Buffer.byteLength(line) <= 4096));
assert.ok(writes.reduce((total,line) => total + Buffer.byteLength(line), 0) <= 65536);
assert.ok(writes.every(line => JSON.parse(line).event === 'probe'));
console.log('diagnostic probe: URL redaction, UTF-8 entry and total bounds PASS');
"""
    result = subprocess.run(
        ["node", "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_local_fixture_rpc_diagnostics_preserve_evaluate_and_filter_private_values():
    fixture = Path(__file__).with_name("worldkit_room_delivery_fixture.mjs").read_text()
    source = fixture.split("const diagnosticFile =", 1)[1].split(
        "if (diagnosticFile) {", 1
    )[0]
    script = "const source = " + json.dumps("const diagnosticFile =" + source) + ";\n"
    script += r"""
import assert from 'node:assert/strict';
const writes = [], calls = [];
const {observeAuthorityRpc} = new Function('appendFileSync', 'process',
  source + '; return {observeAuthorityRpc};')(
    (_file, line) => writes.push(line),
    {env:{WHITEBOX_TEST_BROWSER_DIAGNOSTICS:'/unused-probe'}});
const privateValue = {value:'PRIVATE_MARKER'};
let result = Promise.resolve(privateValue), failure;
const page = {evaluate(...args) {
  assert.equal(this, page); calls.push(args);
  if (failure) throw failure;
  return result;
}};
observeAuthorityRpc(page);
for (const method of ['describeContent','acquire','progress','close']) {
  const fn=()=>{}, arg={method,args:[privateValue],name:'PRIVATE_MARKER'};
  assert.equal(page.evaluate(fn,arg),result);
  assert.equal(await result,privateValue);
  assert.deepEqual(calls.at(-1),[fn,arg]);
}
const knownCount = writes.length;
for (const method of ['resource','PRIVATE_MARKER','acquirePRIVATE_MARKER',null]) {
  assert.equal(page.evaluate(()=>{}, {method,args:[privateValue]}),result);
  await result;
}
assert.equal(writes.length,knownCount);
const error=Error('PRIVATE_MARKER');
result=Promise.reject(error);
assert.equal(page.evaluate(()=>{},{method:'progress'}),result);
await assert.rejects(result,e=>e===error);
failure=error;
assert.throws(()=>page.evaluate(()=>{},{method:'acquire'}),e=>e===error);
const rows=writes.map(line=>JSON.parse(line));
assert.equal(rows.length,12);
assert.deepEqual(rows.map(row=>row.stage),[
 'start','finish','start','finish','start','finish','start','finish','start','error','start','error']);
assert.ok(!writes.join('').includes('PRIVATE_MARKER'));
for(const row of rows) {
 assert.deepEqual(Object.keys(row).sort(),['schemaVersion','event','method','stage','durationMilliseconds','callId'].sort());
 assert.ok(Number.isSafeInteger(row.callId) && row.callId > 0);
 assert.ok(Number.isFinite(row.durationMilliseconds) && row.durationMilliseconds >= 0);
}
failure=undefined;
result=Promise.resolve(privateValue);
let methodReads=0;
const dynamicArgument={get method() {
  return ++methodReads===1 ? 'progress' : 'PRIVATE_MARKER';
}};
const callsBefore=calls.length, writesBefore=writes.length;
assert.equal(page.evaluate(()=>{},dynamicArgument),result);
assert.equal(await result,privateValue);
assert.equal(calls.length,callsBefore+1);
assert.equal(calls.at(-1)[1],dynamicArgument);
assert.equal(writes.length,writesBefore+2);
const allowedMethods=new Set(['describeContent','acquire','progress','close']);
assert.ok(writes.every(line=>allowedMethods.has(JSON.parse(line).method)));
assert.ok(!writes.join('').includes('PRIVATE_MARKER'));
assert.equal(methodReads,1);
console.log('RPC probe: exact allowlist, single getter read, original promise/result/error and redaction PASS');
"""
    result = subprocess.run(
        ["node", "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


async def tap_cli_production_failures(lines):
    """Observe existing SDK failures while the original consumer owns stderr."""
    stages = {
        "room.authority",
        "room.execution",
        "room.start",
        "player.media",
        "member.lease",
        "member.media-cleanup",
        "member.join",
        "member.rollback",
        "member.resume",
        "member.leave",
    }
    emitted = 0
    async for line in lines:
        try:
            if emitted < 32 and len(line) <= 4096:
                row = json.loads(line)
                if isinstance(row, dict):
                    stage = row.get("stage")
                    error = row.get("error")
                    code = error.get("code") if isinstance(error, dict) else None
                    if (
                        type(row.get("schemaVersion")) is int
                        and row["schemaVersion"] == 1
                        and row.get("event") == "worldkit.production.blocked"
                        and isinstance(stage, str)
                        and stage in stages
                        and isinstance(code, str)
                        and re.fullmatch(r"[A-Z][A-Z0-9_]{1,95}", code)
                    ):
                        emitted += 1
                        logging.getLogger(__name__).warning(
                            "SDK production failure: %s",
                            json.dumps(
                                dict(event=row["event"], stage=stage, code=code)
                            ),
                        )
        except Exception:
            # Parsing/logging failures must not alter the original drain or exit.
            pass
        yield line


def test_cli_production_diagnostics_filter_fields_and_preserve_drain(
    caplog, monkeypatch
):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_diagnostics

    valid = dict(
        schemaVersion=1,
        event="worldkit.production.blocked",
        stage="room.authority",
        error=dict(code="AUTHORITY_RPC_TIMEOUT", message="PRIVATE_MARKER"),
        context="PRIVATE_MARKER",
    )
    invalid = [
        b"PRIVATE_MARKER\n",
        b"[]\n",
        b"{\n",
        json.dumps(dict(valid, event="PRIVATE_MARKER")).encode(),
        json.dumps(dict(valid, stage="PRIVATE_MARKER")).encode(),
        json.dumps(dict(valid, error=dict(code="Bearer PRIVATE_MARKER"))).encode(),
        json.dumps(dict(valid, error=dict(code="A" * 97))).encode(),
        json.dumps(dict(valid, schemaVersion=2)).encode(),
    ]
    allowed = json.dumps(valid).encode()
    terminal = json.dumps(
        dict(kind="worldkit-cli-error", error=dict(code="ROOM_EXECUTION_FAILED"))
    ).encode()
    execution = json.dumps(dict(valid, stage="room.execution")).encode()
    lines = invalid + [execution] + [allowed] * 40 + [terminal]

    async def run():
        forwarded = []

        async def source():
            for line in lines:
                yield line

        async def observe():
            async for line in tap_cli_production_failures(source()):
                forwarded.append(line)
                yield line

        result = await cli_diagnostics(SimpleNamespace(stderr=observe()), "probe")
        assert (
            result.code == "CLI_STARTUP_FAILED"
        )  # Existing public-code normalization.
        assert len(forwarded) == len(lines)
        assert all(actual is expected for actual, expected in zip(forwarded, lines))

    asyncio.run(run())
    rows = [
        json.loads(record.args[0])
        for record in caplog.records
        if record.msg == "SDK production failure: %s"
    ]
    assert rows == [
        dict(
            event="worldkit.production.blocked",
            stage=stage,
            code="AUTHORITY_RPC_TIMEOUT",
        )
        for stage in ["room.execution"] + ["room.authority"] * 31
    ]
    assert "PRIVATE_MARKER" not in caplog.text

    def failed_sink(*args, **kwargs):
        raise RuntimeError("PRIVATE_MARKER")

    monkeypatch.setattr(logging.getLogger(__name__), "warning", failed_sink)
    asyncio.run(run())  # Observer failure cannot change the original drain/result.


def read_sealed_delivery_record():
    """Opt in to an externally owned sealed input; never create an implicit cache."""
    name = os.environ.get("WHITEBOX_TEST_SEALED_DELIVERY_RECORD")
    if name is None:
        return None
    try:
        if not Path(name).is_absolute():
            raise ValueError
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid()
                or info.st_size > 1 << 20
            ):
                raise ValueError
            payload = stream.read((1 << 20) + 1)
        if len(payload) > 1 << 20:
            raise ValueError
        value = json.loads(payload)
        if (
            not isinstance(value, dict)
            or any(
                not isinstance(value.get(key), dict) or not value[key]
                for key in ("receipt", "identity", "reference")
            )
            or not isinstance(value.get("archive"), str)
            or not Path(value["archive"]).is_absolute()
            or not Path(value["archive"]).is_file()
        ):
            raise ValueError
        for key in ("archiveSha256", "worldBuildHash", "runtimeHash", "sourceHash"):
            identity = value["identity"].get(key)
            if (
                not isinstance(identity, str)
                or not re.fullmatch(r"[a-f0-9]{64}", identity)
                or value["receipt"].get(key) != identity
            ):
                raise ValueError
        return value
    except (OSError, ValueError, TypeError):
        raise ValueError("SEALED_DELIVERY_RECORD_INVALID") from None


def test_sealed_delivery_record_selects_external_input_and_preserves_ownership(
    tmp_path, monkeypatch
):
    from sglang.multimodal_gen.runtime.realtime.whitebox import worldkit_room as module

    archive = tmp_path / "creator-delivery.tar.gz"
    archive.write_bytes(b"sealed bytes retained for the real CLI verifier")
    identity = dict(
        archiveSha256="a" * 64,
        worldBuildHash="b" * 64,
        runtimeHash="c" * 64,
        sourceHash="d" * 64,
    )
    prepared = dict(
        archive=str(archive),
        identity=identity,
        receipt=dict(
            identity,
            archiveByteLength=archive.stat().st_size,
            deliveryManifestSha256="e" * 64,
        ),
        reference=dict(
            platformUrl="http://127.0.0.1",
            sourceId="local-fixture",
            worldId="room-fixture",
            attempt=1,
        ),
    )
    record = tmp_path / "sealed-record.json"
    record.write_text(json.dumps(prepared))
    record.chmod(0o600)
    monkeypatch.delenv("WHITEBOX_TEST_SEALED_DELIVERY_RECORD", raising=False)
    assert read_sealed_delivery_record() is None
    monkeypatch.setenv("WHITEBOX_TEST_SEALED_DELIVERY_RECORD", str(record))
    assert read_sealed_delivery_record() == prepared
    monkeypatch.setenv("WHITEBOX_TEST_CLI_SDK", str(tmp_path))
    failure = RuntimeError("CONTROLLED_ROOM_STOP")
    consumed = []

    async def consume_delivery(**kwargs):
        consumed.append(await module.prepare_delivery())
        raise failure

    async def forbidden_process(*args, **kwargs):
        raise AssertionError("External delivery must not launch a Creator fixture")

    monkeypatch.setattr(module, "worldkit_room", consume_delivery)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_process)
    with pytest.raises(RuntimeError) as caught:
        test_real_sealed_room_cli_private_native_views_and_cleanup(
            tmp_path, monkeypatch, True, True
        )
    assert caught.value is failure
    assert consumed == [prepared]
    assert archive.read_bytes() == b"sealed bytes retained for the real CLI verifier"
    assert json.loads(record.read_text()) == prepared
    assert not any(
        "cleanup failed" in note.lower() for note in getattr(failure, "__notes__", [])
    )


def test_sealed_delivery_record_rejects_nonprivate_or_incomplete_input(
    tmp_path, monkeypatch
):
    record = tmp_path / "sealed-record.json"
    record.write_text("{}")
    record.chmod(0o644)
    monkeypatch.setenv("WHITEBOX_TEST_SEALED_DELIVERY_RECORD", str(record))
    with pytest.raises(ValueError, match="SEALED_DELIVERY_RECORD_INVALID"):
        read_sealed_delivery_record()
    record.chmod(0o600)
    with pytest.raises(ValueError, match="SEALED_DELIVERY_RECORD_INVALID"):
        read_sealed_delivery_record()
    link = tmp_path / "linked-record.json"
    link.symlink_to(record)
    monkeypatch.setenv("WHITEBOX_TEST_SEALED_DELIVERY_RECORD", str(link))
    with pytest.raises(ValueError, match="SEALED_DELIVERY_RECORD_INVALID"):
        read_sealed_delivery_record()


class InertUiRenderer:
    """Observe UI protocol only. This test makes no composed-output claim."""

    def __init__(self, **kwargs):
        self.events = []
        self.closed = False

    async def consume_ui(self, value):
        self.events.append(value)

    async def close(self):
        self.closed = True


async def stop_local_fixture(fixture, *, grace_seconds=20, terminate_seconds=10):
    """Bound EOF/TERM cleanup even when the fixture or its group is wedged."""
    if fixture.returncode is not None:
        return
    fixture.stdin.close()
    try:
        await asyncio.wait_for(fixture.wait(), grace_seconds)
        return
    except asyncio.TimeoutError:
        pass
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(fixture.pid, signum)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(fixture.wait(), terminate_seconds)
            # A terminated leader does not prove all of its children exited.
            try:
                os.killpg(fixture.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return
        except asyncio.TimeoutError:
            if signum == signal.SIGKILL:
                raise


def test_local_fixture_cleanup_kills_term_resistant_process():
    async def run():
        fixture = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "print('ready',flush=True); time.sleep(60)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            assert await asyncio.wait_for(fixture.stdout.readline(), 2) == b"ready\n"
            await stop_local_fixture(
                fixture, grace_seconds=0.02, terminate_seconds=0.05
            )
            assert fixture.returncode == -signal.SIGKILL
            await stop_local_fixture(fixture)  # Already-reaped owners remain harmless.
        finally:
            if fixture.returncode is None:
                try:
                    os.killpg(fixture.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await asyncio.wait_for(fixture.wait(), 2)

    asyncio.run(run())


@pytest.mark.skipif(
    not os.environ.get("WHITEBOX_TEST_CLI_SDK"),
    reason="Set WHITEBOX_TEST_CLI_SDK to a SDK checkout with Chromium",
)
@pytest.mark.parametrize(
    "external_input,model_context",
    [(False, False), (True, False), (True, True)],
    ids=["cli-controller", "public-browser-controller", "trusted-model-members"],
)
def test_real_sealed_room_cli_private_native_views_and_cleanup(
    tmp_path, monkeypatch, external_input, model_context
):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox import worldkit_room as module

    original_diagnostics = module.cli_diagnostics

    async def diagnostics(process, *args, **kwargs):
        return await original_diagnostics(
            SimpleNamespace(stderr=tap_cli_production_failures(process.stderr)),
            *args,
            **kwargs,
        )

    monkeypatch.setattr(module, "cli_diagnostics", diagnostics)

    async def run():
        sdk = Path(os.environ["WHITEBOX_TEST_CLI_SDK"])
        entry = sdk / "packages/cli/bin/worldkit.mjs"
        monkeypatch.setenv("WHITEBOX_CLI_ENTRYPOINT", str(entry))
        monkeypatch.setenv("WHITEBOX_MAX_PLAYERS", "2")
        monkeypatch.setenv("WHITEBOX_TEST_MODEL_CONTEXT", "1" if model_context else "0")
        monkeypatch.setenv("WHITEBOX_MODEL_STARTUP_PREPARE", "1")
        monkeypatch.setenv("WHITEBOX_CLI_UI_FACTORY", __name__ + ":InertUiRenderer")
        monkeypatch.setenv("WHITEBOX_CLI_STARTUP_TIMEOUT_SECONDS", "120")
        diagnostic_file = tmp_path / "sdk-browser-diagnostics.jsonl"
        diagnostic_file.touch(mode=0o600)
        prepared = read_sealed_delivery_record()
        fixture = errors = None
        if prepared is None:
            fixture = await asyncio.create_subprocess_exec(
                "node",
                "--import",
                "tsx",
                str(Path(__file__).with_name("worldkit_room_delivery_fixture.mjs")),
                cwd=sdk,
                env={
                    **os.environ,
                    "WHITEBOX_TEST_BROWSER_DIAGNOSTICS": str(diagnostic_file),
                },
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                limit=4 << 20,
            )
            errors = asyncio.create_task(fixture.stderr.read())
        room = None
        streams = []
        owned_pids = set()

        async def descendants(parent):
            if sys.platform == "linux":
                # Slim runtime images omit procps. Observe the kernel's process
                # table directly so cleanup verification needs no system package.
                rows = []
                for entry in Path("/proc").iterdir():
                    if not entry.name.isdecimal():
                        continue
                    try:
                        fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                    except (FileNotFoundError, ProcessLookupError, PermissionError):
                        continue
                    rows.append((int(entry.name), int(fields[1])))
            else:
                process = await asyncio.create_subprocess_exec(
                    "ps", "-axo", "pid=,ppid=", stdout=asyncio.subprocess.PIPE
                )
                output, _ = await process.communicate()
                rows = [tuple(map(int, line.split())) for line in output.splitlines()]
            selected = {parent}
            while True:
                updated = selected | {pid for pid, ppid in rows if ppid in selected}
                if updated == selected:
                    return selected - {parent}
                selected = updated

        def alive(pid):
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False

        async def assert_exited(pids):
            async with asyncio.timeout(15):
                while any(alive(pid) for pid in pids):
                    await asyncio.sleep(0.05)

        fixture_directory = None
        primary_error = None
        try:
            if fixture is not None:
                line = await asyncio.wait_for(fixture.stdout.readline(), 240)
                if not line:
                    await fixture.wait()
                    pytest.fail(
                        "Local SDK fixture failed: " + (await errors).decode()[-6000:]
                    )
                prepared = json.loads(line)
                fixture_directory = Path(prepared.pop("directory"))
            # Captured pytest output preserves exact sealed input identity on
            # failure without retaining archives or exposing credentials.
            print(
                "Local Room delivery identity: "
                + json.dumps(prepared["identity"], sort_keys=True)
            )

            async def local_delivery(*_, **_kwargs):
                return prepared

            # Only cloud retrieval is substituted. The CLI verifies/extracts the
            # actual SDK-sealed receipt/archive; all Room commands remain real.
            monkeypatch.setattr(module, "prepare_delivery", local_delivery)
            room = await module.worldkit_room(
                world_id="local-fixture/room-fixture",
                # The model path shares the existing native pixel dimensions;
                # MemberEngine does not resize. Its logical H3 clock stays 24 FPS
                # while NativeCLIView preserves the independent capture timestamps.
                spec=(
                    VideoSpec(832, 480)
                    if os.environ.get("WHITEBOX_TEST_GPU_COORDINATOR_URL")
                    else VideoSpec(160, 96)
                    if model_context
                    else VideoSpec(160, 90)
                ),
                cli_fps=12,
                trace_id="local-real-room",
                **(
                    {
                        "create_context": {
                            "frozenPublication": {
                                "playConfiguration": {
                                    "playMode": "multiplayer",
                                    "multiplayer": {"maxPlayers": 2},
                                }
                            }
                        },
                        "verified_caller": {
                            "serviceId": "fixture",
                            "subject": "fixture",
                        },
                    }
                    if external_input
                    else {}
                ),
            )
            pid, room_directory = room.process.pid, room.directory
            owned_pids.update(await descendants(pid))
            alice = await room.join(player_id="a" * 32, spec=room.spec)
            bob = await room.join(player_id="b" * 32, spec=room.spec)
            owned_pids.update(await descendants(pid))
            assert alice.descriptor["sessionId"] != bob.descriptor["sessionId"]
            assert alice.descriptor["accessToken"] != bob.descriptor["accessToken"]
            assert (
                alice.descriptor["playerBinding"]["room"]
                == bob.descriptor["playerBinding"]["room"]
            )
            actors = [
                room.slots[room.members[player]["slot"]]["actorId"]
                for player in ("a" * 32, "b" * 32)
            ]
            members = [
                room.members[player]["member"] for player in ("a" * 32, "b" * 32)
            ]
            for member, actor in zip(members, actors):
                inspected = await room.call(
                    "inspect", "--member-file", member["memberFile"]
                )
                assert inspected["ui"] == {"player": actor, "secret": actor}
            if model_context:
                from test_whitebox_player_transport import verify_real_public_players

                await verify_actual_member_model_sources(room, (alice, bob), tmp_path)
                await verify_real_public_players(room, alice, bob)
                await room.close()
                assert room.process.returncode == 0 and not room_directory.exists()
                await assert_exited(owned_pids)
                return
            streams = [view.frames() for view in (alice, bob)]
            frames = await asyncio.gather(
                *(asyncio.wait_for(anext(stream), 20) for stream in streams)
            )
            for frame, view in zip(frames, (alice, bob)):
                assert len(frame.rgb) == room.spec.frame_bytes
                assert (
                    frame.provenance["cli_frame_header"]["sessionId"]
                    == view.descriptor["sessionId"]
                )
            if external_input:
                from test_whitebox_player_transport import verify_real_public_players

                await verify_real_public_players(room, alice, bob)
                await room.close()
                assert room.process.returncode == 0 and not room_directory.exists()
                assert bob.renderer.closed
                await assert_exited(owned_pids)
                return
            await alice.send_input(0, {"held_keys": ["KeyW"]})
            await bob.send_input(0, {"held_keys": ["KeyS"]})

            async def receipt(view, sequence=0):
                async with asyncio.timeout(15):
                    while not any(
                        item.get("type") == "input.receipt"
                        and item.get("clientSequence") == sequence
                        for item in view.receipts
                    ):
                        await asyncio.sleep(0.02)
                return next(
                    item
                    for item in view.receipts
                    if item.get("type") == "input.receipt"
                    and item.get("clientSequence") == sequence
                )

            for view, result in zip(
                (alice, bob), await asyncio.gather(receipt(alice), receipt(bob))
            ):
                assert result["epoch"] == view.descriptor["epoch"]
                assert (
                    isinstance(view.binding_sequence, int) and view.binding_sequence > 0
                )
                assert result["sdkInputSequence"] >= 1
                assert result["simulationTick"] >= 0
                assert result["clientSequence"] == 0
            await alice.close()
            assert alice.renderer.closed and not bob.renderer.closed
            await bob.send_input(1, {"held_keys": ["KeyD"]})
            assert (await receipt(bob, 1))["sdkInputSequence"] >= 1
            # Drain the bounded old credit window and require fresh post-leave
            # captures from the remaining player's live producer.
            for _ in range(3):
                after = await asyncio.wait_for(anext(streams[1]), 20)
            assert (
                after.provenance["cli_frame_header"]["outputFrameId"]
                > frames[1].provenance["cli_frame_header"]["outputFrameId"] + 2
            )
            assert (
                after.provenance["cli_frame_header"]["sessionId"]
                == bob.descriptor["sessionId"]
            )
            remaining = await room.call(
                "inspect", "--member-file", members[1]["memberFile"]
            )
            assert remaining["ui"]["secret"] == actors[1]
            assert room.process.returncode is None
            await room.close()
            assert room.process.returncode == 0
            assert not room_directory.exists()
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            assert bob.renderer.closed
            await assert_exited(owned_pids)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            cleanup_errors = []
            try:
                results = await asyncio.gather(
                    *(stream.aclose() for stream in streams), return_exceptions=True
                )
                cleanup_errors.extend(
                    error for error in results if isinstance(error, BaseException)
                )
                if room:
                    try:
                        owned_pids.update(await descendants(room.process.pid))
                    except BaseException as error:
                        cleanup_errors.append(error)
                    try:
                        await room.close()
                    except BaseException as error:
                        cleanup_errors.append(error)
            finally:
                # Room.close may cache a failed cleanup. Fixture EOF/termination
                # still runs, and cleanup diagnostics cannot replace the test's
                # original admission/assertion failure.
                if fixture is not None:
                    try:
                        await stop_local_fixture(fixture)
                    except BaseException as error:
                        cleanup_errors.append(error)
                    try:
                        await asyncio.wait_for(errors, 10)
                    except BaseException as error:
                        cleanup_errors.append(error)
                if room:
                    try:
                        await assert_exited(owned_pids)
                    except BaseException as error:
                        cleanup_errors.append(error)
            if primary_error is not None and diagnostic_file.exists():
                rows = []
                for line in diagnostic_file.read_text().splitlines():
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        # A terminated fixture may leave its final diagnostic incomplete.
                        continue
                counts = {
                    event: sum(row["event"] == event for row in rows)
                    for event in {row["event"] for row in rows}
                }
                summary = f"Browser diagnostics retained at {diagnostic_file}: {json.dumps(counts, sort_keys=True)}"
                logging.getLogger(__name__).error(summary)
                primary_error.add_note(summary)
            if cleanup_errors:
                diagnostic = "Local Room fixture cleanup failed: " + "; ".join(
                    f"{type(error).__name__}: {error}" for error in cleanup_errors
                )
                if primary_error is not None:
                    logging.getLogger(__name__).error(diagnostic)
                    add_note = getattr(primary_error, "add_note", None)
                    if add_note:
                        add_note(diagnostic)
                else:
                    raise RuntimeError(diagnostic) from cleanup_errors[0]
        if fixture is not None:
            assert fixture.returncode == 0
            assert fixture_directory is not None and not fixture_directory.exists()

    asyncio.run(run())


@pytest.mark.parametrize("owner", ["fixture", "driver"])
def test_owned_group_cleanup_reaps_resistant_child_after_leader_exits(owner):
    from sglang.multimodal_gen.runtime.realtime.whitebox.worldkit_room import (
        stop_process,
    )

    async def run():
        child_code = (
            "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "print('ready',flush=True); time.sleep(60)"
        )
        parent_code = (
            "import subprocess,sys,time; "
            f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}], "
            "stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL); "
            "assert child.stdout.readline()==b'ready\\n'; "
            "print(child.pid,flush=True); time.sleep(60)"
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            parent_code,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            child = int(await asyncio.wait_for(process.stdout.readline(), 2))
            if owner == "fixture":
                await stop_local_fixture(
                    process, grace_seconds=0.02, terminate_seconds=0.05
                )
            else:
                await stop_process(process)
            assert process.returncode == -signal.SIGTERM
            async with asyncio.timeout(2):
                while True:
                    probe = await asyncio.create_subprocess_exec(
                        "ps",
                        "-p",
                        str(child),
                        "-o",
                        "stat=",
                        stdout=asyncio.subprocess.PIPE,
                    )
                    state, _ = await probe.communicate()
                    if not state.strip() or state.strip().startswith(b"Z"):
                        break
                    await asyncio.sleep(0.02)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await asyncio.wait_for(process.wait(), 2)

    asyncio.run(run())


async def verify_actual_member_model_sources(room, views, tmp_path):
    """Real SDK/CLI/context/journal; only the model and local runtime registry are CPU fixtures."""
    import hashlib
    import io
    from contextlib import asynccontextmanager
    from dataclasses import asdict
    from types import SimpleNamespace
    from uuid import uuid4

    import httpx
    import msgspec
    from PIL import Image
    from test_whitebox_interactive import BrowserUI
    from test_whitebox_render_lifecycle import StrictLeases
    from test_whitebox_startup import request

    from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
        AsyncArchiveManager,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.player_media import (
        MemberEngine,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderJournal,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig

    gpu_url = os.environ.get("WHITEBOX_TEST_GPU_COORDINATOR_URL")
    manual_inference = os.environ.get("WHITEBOX_TEST_MANUAL_INFERENCE") == "1"
    if manual_inference and gpu_url:
        raise RuntimeError("The manual inference gate is CPU-only")
    coordinator = None
    run_id = uuid4().hex
    if gpu_url:
        from websockets.asyncio.client import connect

        from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
            HTTPCoordinatorClient,
        )
        from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
            ModelLeases,
        )

        coordinator = HTTPCoordinatorClient(gpu_url)

    class ObservedLeases:
        """Observe original GPU leases/wires without replacing admission or identity."""

        def __init__(self):
            self.active, self.opens, self.frames = 0, [], 0
            self.output_hash = hashlib.sha256()

        @property
        def resource_state(self):
            return self.delegate.resource_state

        def bind(self, user_id, segment):
            self.delegate = ModelLeases(
                coordinator,
                connect,
                user_id=user_id,
                trace_id=segment["segment_id"],
                model_revision=os.environ["WHITEBOX_TEST_GPU_MODEL_REVISION"],
                vae_fingerprint="local",
                max_bytes=64 << 20,
                strict_release=True,
                session_id=segment["model_session_id"],
                generation_id=segment["generation_id"],
            )
            return self

        @asynccontextmanager
        async def open(self):
            async with self.delegate.open() as (wire, assignment):
                pool = self

                class ObservedWire:
                    def __init__(self):
                        self.prompt_updates = []

                    async def send(self, data):
                        message = msgspec.msgpack.decode(data)
                        if message.get("type") == "init":
                            self.init = message
                        elif message.get("kind") == "prompt.update":
                            self.prompt_updates.append(message["payload"])
                        await wire.send(data)

                    async def recv(self):
                        data = await wire.recv()
                        try:
                            message = msgspec.msgpack.decode(data)
                        except (msgspec.DecodeError, UnicodeDecodeError):
                            message = None
                        if isinstance(message, dict) and message.get("type") in {
                            "frame_batch",
                            "frame_batch_header",
                        }:
                            pool.frames += message["num_frames"]
                            pool.output_hash.update(data)
                        return data

                    async def close(self):
                        await wire.close()

                observer = ObservedWire()
                self.opens.append(observer)
                self.active += 1
                self.assignment = assignment
                try:
                    yield observer, assignment
                finally:
                    self.active -= 1

    class ObservedCPULeases(StrictLeases):
        @asynccontextmanager
        async def open(self):
            async with super().open() as (wire, assignment):
                original = wire.send
                wire.prompt_updates = []

                async def send(data):
                    message = msgspec.msgpack.decode(data)
                    if message.get("kind") == "prompt.update":
                        wire.prompt_updates.append(message["payload"])
                    await original(data)

                wire.send = send
                yield wire, assignment

    owners, tasks, iterators, journals, pools, engines = [], [], [], [], [], []
    instruction_results = []
    archives = AsyncArchiveManager(recorder_slots=6)

    async def wait(predicate):
        async with asyncio.timeout(90 if gpu_url else 35):
            while not predicate():
                for task in tasks:
                    if task.done():
                        task.result()
                await asyncio.sleep(0.02)

    try:
        # Sequential subscription fixes the fixture's private first-receiver identity.
        for index, view in enumerate(views[:1] if gpu_url else views):
            descriptor = view.descriptor
            assert descriptor["accessToken"] != view.player_descriptor["accessToken"]
            async with httpx.AsyncClient(trust_env=False) as client:
                url = (
                    descriptor["baseUrl"]
                    + "/v1/sessions/"
                    + descriptor["sessionId"]
                    + "/model-context"
                )
                model = await client.get(
                    url,
                    headers={"Authorization": "Bearer " + descriptor["accessToken"]},
                )
                ordinary = await client.get(
                    url,
                    headers={
                        "Authorization": "Bearer "
                        + view.player_descriptor["accessToken"]
                    },
                )
                assert model.status_code == 200 and ordinary.status_code in {401, 403}
                resources = model.json()["resources"]
                assert [ref["resource"] for ref in resources] == ["landmark"]
                denied = await client.get(
                    resources[0]["url"],
                    headers={
                        "Authorization": "Bearer "
                        + view.player_descriptor["accessToken"]
                    },
                )
                assert denied.status_code in {401, 403}
            iterator = view.frames()
            iterators.append(iterator)
            entry = dict(engine=view)

            async def next_frame(stream_id, *, consumer=None, iterator=iterator):
                return asdict(await anext(iterator))

            async def event(stream_id, kind, payload):
                assert kind == "inference.init"

            runtime = SimpleNamespace(
                player=lambda *_, entry=entry, **__: entry,
                next_frame=next_frame,
                event=event,
            )
            source = SimpleNamespace(
                runtime=runtime, entry=entry, check=lambda entry=entry: entry
            )
            engine = MemberEngine(
                source,
                dict(
                    stream_id=f"member-{index}",
                    game_id=room.ready["roomId"],
                    player_id=descriptor["playerBinding"]["playerSessionId"],
                    session_id=f"source-{index}",
                    user_id=f"harness-{run_id}-{index}",
                    owner_id="test-owner",
                    owner_epoch="room",
                    joined_room_time_us=0,
                    game_object_prefix="test-room",
                    object_prefix=f"test-room/member-{index}",
                    engine={},
                    ui=dict(mode="browser", files={}),
                ),
            )
            entry["media_reader"] = engine
            engines.append(engine)
            leases = ObservedLeases() if gpu_url else ObservedCPULeases(room.spec)
            pools.append(leases)
            journal = RenderJournal(
                MemoryRoomStore(),
                user_id=f"harness-{run_id}-{index}",
                source_session_id=f"source-{index}",
                source_epoch=f"epoch-{index}",
                owner_epoch="room",
            )
            await journal.create()
            journals.append(journal)

            async def factory(*, engine=engine, **_):
                return engine

            owner = SourceSession(
                SimpleNamespace(
                    runtime=SimpleNamespace(players={}),
                    make_segment_leases=(
                        leases.bind if gpu_url else lambda *_, leases=leases: leases
                    ),
                ),
                journal,
                source_epoch=f"epoch-{index}",
                user_id=f"harness-{run_id}-{index}",
                session_id=f"source-{index}",
                trace_id=f"actual-member-{index}",
                config=WhiteboxConfig(
                    engine_factory=factory,
                    ui_factory=lambda **_: BrowserUI(),
                    spool=tmp_path / str(index),
                    spec=room.spec,
                    room_owned=True,
                    startup_seconds=90 if gpu_url else 35,
                    num_inference_steps=2 if gpu_url else 4,
                    drain_seconds=3,
                ),
                archives=archives,
            )
            owners.append(owner)

            replies = []
            instruction_results.append(replies)

            async def consume(owner=owner, replies=replies):
                while True:
                    message = msgspec.msgpack.decode(await owner.recv())
                    if message.get("type") in {
                        "model.context.intervention.result",
                        "operation.rejected",
                    }:
                        replies.append(message)
                    if message.get("type") == "error":
                        raise AssertionError(
                            "Actual member SourceSession failed: "
                            + str(message.get("code"))
                        )
                    if message.get("type") in {"whitebox.frame", "frame_batch"}:
                        owner.media_delivered()

            await owner.send(
                msgspec.msgpack.encode(
                    dict(
                        type="init",
                        whitebox=dict(world_id="world"),
                        prompt="legacy preview",
                        startup=request(False),
                        **(
                            dict(
                                source_operation="create.manual",
                                inference_control="manual",
                            )
                            if manual_inference
                            else {}
                        ),
                    )
                )
            )
            tasks.append(asyncio.create_task(consume()))
            if manual_inference:
                await wait(lambda: owner.ready.is_set() and owner.preview_count >= 2)
                assert not leases.opens and leases.active == 0
                assert journal.snapshot["inference"] == dict(
                    enabled=False, revision=0, state="disabled"
                )
                assert owner.context_channel.suspended
                await owner.command(
                    dict(
                        type="command",
                        op="model.inference.configure",
                        request_id=f"actual-enable-{index}",
                        source_session_id=owner.session_id,
                        source_epoch=owner.source_epoch,
                        enabled=True,
                        expected_inference_revision=0,
                    )
                )
            await wait(
                lambda: (
                    journal.snapshot.get("segment", {}).get("applied_version", 0) >= 1
                    if journal.snapshot.get("segment")
                    else False
                )
            )
            assert len(leases.opens) == 1
            initial = leases.opens[0].init["prompt"]
            assert "Two player courtyard." in initial
            if index == 0:
                assert (
                    "Private sunshine." in initial
                    and "Private golden landmark." in initial
                )
            else:
                assert "Shared rain." in initial and "Private" not in initial
            references = leases.opens[0].init["references"]
            assert len(references) == 1
            reference = references[0]
            assert reference["id"] == "<Picture 1>" and reference["format"] == "png"
            assert "Landmark reference <Picture 1>." in initial
            assert (
                hashlib.sha256(reference["payload"]).hexdigest()
                == resources[0]["sha256"]
            )
            with Image.open(io.BytesIO(reference["payload"])) as image:
                assert image.format == "PNG" and image.size == (128, 128)
                image.verify()
            with Image.open(io.BytesIO(reference["payload"])) as image:
                assert image.getpixel((0, 0)) == (255, 0, 0)
            assert owner.world_driven and owner.startup is None
            assert owner.source_conditions is not None
        # Confirmation is authored only after the SDK receives the actual later
        # applied receipt. Its next compiled condition proves the round trip.
        await wait(
            lambda: (
                journals[0].snapshot["segment"]["applied_version"] >= 2
                and owners[0].source_conditions.candidate is not None
                and "SDK confirmed later model application."
                in owners[0].source_conditions.candidate["condition"]["prompt"]
            )
        )
        operations = list(journals[0].snapshot["operations"].values())
        applied = [op for op in operations if op.get("state") == "applied"]
        assert len(applied) >= 2
        assert len({op["request_id"] for op in applied}) == len(applied)
        if len(journals) == 2:
            assert (
                journals[0].snapshot["segment"]["segment_id"]
                != journals[1].snapshot["segment"]["segment_id"]
            )
        if gpu_url:
            assert pools[0].frames >= 17
            observer = pools[0].opens[0]
            applied_by_version = {op["prompt_version"]: op for op in applied}
            updates = [
                update
                for update in observer.prompt_updates
                if update["version"] in applied_by_version
                and "A later opening after initial model application." in update["text"]
            ]
            assert len(updates) == 1
            update = updates[0]
            version = update["version"]
            # Inspect the real model wire, not a separately recompiled expectation.
            # A later required event must preserve both independent persistent slots.
            for text in (
                "Two player courtyard.",
                "Private sunshine.",
                "Private golden landmark.",
                "Landmark reference <Picture 1>.",
                "A later opening after initial model application.",
            ):
                assert text in update["text"]
            assert "Shared rain." not in update["text"]

            def reference_evidence(references):
                return [
                    dict(
                        id=reference["id"],
                        format=reference["format"],
                        sha256=hashlib.sha256(reference["payload"]).hexdigest(),
                    )
                    for reference in references
                ]

            initial_references = observer.init["references"]
            updated_references = update.get("references", initial_references)
            assert reference_evidence(updated_references) == reference_evidence(
                initial_references
            )
            assert 1 in applied_by_version and version in applied_by_version
            condition_inputs = [
                dict(
                    prompt_version=number,
                    request_id=applied_by_version[number]["request_id"],
                    prompt=text,
                    prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    references=reference_evidence(references),
                )
                for number, text, references in (
                    (1, observer.init["prompt"], initial_references),
                    (version, update["text"], updated_references),
                )
            ]
            evidence = dict(
                condition_inputs=condition_inputs,
                persistent_state_preserved=True,
                reference_order_preserved=True,
                frames=pools[0].frames,
                model_output_wire_sha256=pools[0].output_hash.hexdigest(),
                worker_id=pools[0].assignment.denoiser.worker_id,
                model_revision=pools[0].assignment.denoiser.model_revision,
                model_session_id=pools[0].assignment.session_id,
                generation_id=pools[0].assignment.generation_id,
                applied_requests=[op["request_id"] for op in applied],
                applied_version=journals[0].snapshot["segment"]["applied_version"],
                sdk_confirmed=True,
            )
            (tmp_path / "actual-gpu-evidence.json").write_text(
                json.dumps(evidence, indent=2)
            )
            print("Actual GPU evidence: " + json.dumps(evidence, sort_keys=True))

        # The browser instruction path crosses the actual authenticated CLI
        # connection and Host compiler. Admission alone is not model application.
        async def instruction(index, request_id, *, revision=None, prompt=None):
            owner, journal = owners[index], journals[index]
            ids = journal.snapshot["model_context_segment_ids"]
            command = dict(
                type="command",
                op="model.context.intervention" + (".get" if revision is None else ""),
                request_id=request_id,
                source_session_id=owner.session_id,
                source_epoch=owner.source_epoch,
                segment_id=ids["segment_id"],
                segment_epoch=ids["segment_epoch"],
            )
            if revision is not None:
                command.update(expectedRevision=revision, prompt=prompt)
            offset = len(instruction_results[index])
            await owner.command(command)
            await wait(lambda: len(instruction_results[index]) > offset)
            result = instruction_results[index][offset]
            assert result["request_id"] == request_id
            assert result["type"] == "model.context.intervention.result", result
            return result

        async def instruction_applied(text, after_version):
            journal = journals[0]

            def applied_update():
                for update in pools[0].opens[0].prompt_updates:
                    if update["version"] <= after_version:
                        continue
                    matches = ("Use cinematic lighting." in update["text"]) == (
                        text is not None
                    )
                    if matches and any(
                        op.get("state") == "applied"
                        and op.get("prompt_version") == update["version"]
                        for op in journal.snapshot["operations"].values()
                    ):
                        return update
                return None

            # SourceConditions releases its transient candidate on application.
            # The original journal plus actual model wire proves the result.
            await wait(lambda: applied_update() is not None)
            actual = applied_update()["text"]
            for retained in (
                "Two player courtyard.",
                "Private sunshine.",
                "Private golden landmark.",
                "Landmark reference <Picture 1>.",
                "SDK confirmed later model application.",
            ):
                assert retained in actual
            return actual

        current = await instruction(0, "manual-current")
        assert (
            current["status"] == "current"
            and current["revision"] == 0
            and current["prompt"] is None
        )
        manual_floor = journals[0].snapshot["segment"]["applied_version"]
        accepted = await instruction(
            0, "manual-write", revision=0, prompt="Use cinematic lighting."
        )
        assert accepted["status"] == "accepted" and accepted["revision"] == 1
        manual_prompt = await instruction_applied(
            "Use cinematic lighting.", manual_floor
        )
        duplicate = await instruction(
            0, "manual-write", revision=0, prompt="Use cinematic lighting."
        )
        assert duplicate["status"] == "duplicate" and duplicate["revision"] == 1
        reused = await instruction(
            0, "manual-write", revision=0, prompt="Different content."
        )
        assert (
            reused["status"] == "conflict"
            and reused["code"] == "MODEL_CONTEXT_REQUEST_ID_CONFLICT"
        )
        stale = await instruction(
            0, "manual-stale", revision=0, prompt="Stale content."
        )
        assert stale["status"] == "conflict" and stale["revision"] == 1
        if len(owners) == 2:
            other = await instruction(1, "other-current")
            assert (
                other["status"] == "current"
                and other["revision"] == 0
                and other["prompt"] is None
            )
            assert all(
                "Use cinematic lighting." not in update["text"]
                for update in pools[1].opens[0].prompt_updates
            )
        original_consumer = dict(owners[0].context_channel.consumer)
        await views[0].recover_model_context_transport(views[0].transport_generation)
        recovered = await instruction(0, "manual-recover")
        assert recovered["status"] == "current" and recovered["revision"] == 1
        assert recovered["prompt"] == "Use cinematic lighting."
        assert owners[0].context_channel.consumer == original_consumer
        clear_floor = journals[0].snapshot["segment"]["applied_version"]
        cleared = await instruction(0, "manual-clear", revision=1, prompt=None)
        assert (
            cleared["status"] == "accepted"
            and cleared["revision"] == 2
            and cleared["prompt"] is None
        )
        cleared_prompt = await instruction_applied(None, clear_floor)
        if gpu_url:
            actual_updates = pools[0].opens[0].prompt_updates
            assert any(update["text"] == manual_prompt for update in actual_updates)
            assert any(update["text"] == cleared_prompt for update in actual_updates)
        print(
            "Actual manual instruction: accepted/applied; duplicate/conflict; independent consumer; reconnect/read; clear retained game state"
        )
        assert all(pool.active == 1 and len(pool.opens) == 1 for pool in pools)
        if manual_inference:
            owner, journal = owners[0], journals[0]
            old_segment = journal.snapshot["segment"]["segment_id"]
            previous_frame = owner.latest.version
            await owner.command(
                dict(
                    type="command",
                    op="model.inference.configure",
                    request_id="actual-disable",
                    source_session_id=owner.session_id,
                    source_epoch=owner.source_epoch,
                    enabled=False,
                    expected_inference_revision=1,
                )
            )
            await wait(lambda: journal.snapshot["inference"]["state"] == "disabled")
            assert pools[0].active == 0
            assert all(pool.active == 1 for pool in pools[1:])
            await wait(lambda: owner.latest.version > previous_frame)
            await owner.command(
                dict(
                    type="command",
                    op="model.inference.configure",
                    request_id="actual-reenable",
                    source_session_id=owner.session_id,
                    source_epoch=owner.source_epoch,
                    enabled=True,
                    expected_inference_revision=2,
                )
            )
            await wait(lambda: journal.snapshot["inference"]["state"] == "enabled")
            assert pools[0].active == 1 and len(pools[0].opens) == 2
            assert journal.snapshot["segment"]["segment_id"] != old_segment
            assert "Two player courtyard." in pools[0].opens[-1].init["prompt"]
            assert (
                pools[0].opens[-1].init["references"][0]["payload"]
                == pools[0].opens[0].init["references"][0]["payload"]
            )
            print(
                "Actual manual inference: initial zero leases; enable; release with live world/peer; fresh-segment re-enable"
            )
        print(
            (
                "Actual SDK/CLI/SourceSession GPU integration: "
                if gpu_url
                else "Actual two-member SDK/CLI/SourceSession CPU integration: "
            )
            + "sealed PNG hash/decode/auth; private prompts; initial applied; "
            "later required applied and SDK confirmation; "
            "independent segments; one lease each"
        )
    finally:
        # Keep bounded, credential-free ownership evidence when an actual chain
        # stalls; never replace the original journal with a retry condition.
        (tmp_path / "actual-context-diagnostics.json").write_text(
            json.dumps(
                [
                    dict(
                        source_session_id=owner.session_id,
                        candidate=(
                            (owner.source_conditions.candidate or {}).get("condition")
                            if owner.source_conditions
                            else None
                        ),
                        operations=list(journal.snapshot["operations"].values()),
                        instruction_results=replies,
                        channel=dict(
                            sequence=owner.context_channel.sequence,
                            suspended=owner.context_channel.suspended,
                            bound=owner.context_channel.bound.is_set(),
                            pending_controls=owner.context_channel.queue.qsize(),
                            samples=len(owner.context_channel.samples),
                            error=str(owner.context_channel.error),
                        ),
                    )
                    for owner, journal, replies in zip(
                        owners, journals, instruction_results
                    )
                ],
                indent=2,
            )
        )
        for owner in owners:
            await owner.close()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for iterator in iterators:
            await iterator.aclose()
        await archives.close()
        if coordinator:
            try:
                assert all(
                    pool.active == 0 and pool.resource_state == "released"
                    for pool in pools
                    if pool.opens
                )
                # Verified worker release is authoritative; capacity also includes
                # the last heartbeat and can lag until its next five-second report.
                async with asyncio.timeout(10):
                    while True:
                        capacity = await coordinator.capacity()
                        if capacity["roles"]["denoiser"]["active_sessions"] == 0:
                            break
                        await asyncio.sleep(0.2)
                print("Actual GPU cleanup: original leases released; active_sessions=0")
            finally:
                await coordinator.close()
    assert all(pool.active == 0 for pool in pools)
    assert all(engine.closed for engine in engines)
