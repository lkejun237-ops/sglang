# SPDX-License-Identifier: Apache-2.0
"""Dedicated decoder service contracts: dispatch references, identity and frame parity.

The DiT never receives pixels, so these cases pin the boundary between it and
the decoder host: one shared-memory reference per dispatch carrying the current
dispatch identity, and frames identical to the local stream decoder.
"""

import ast
import json
import logging
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zmq
from test_zing_lynn_stream import ARCHITECTURE, CONFIG, _latent, _provenance

from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.model import (
    StreamingDecoder,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder import (
    DECODERS,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker import (
    _DecodeService,
    _output_identity,
    _OutputPublisher,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.remote_decoder import (
    RemoteLynnDecoder,
    create_decoder,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_decoder import (
    SourceLynnDecoder,
)

ADAPTER = DECODERS["lynn_streaming"]


@pytest.fixture(scope="module")
def model():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        decoder = StreamingDecoder(CONFIG, ARCHITECTURE)
        with torch.no_grad():
            for parameter in decoder.parameters():
                if not parameter.any():
                    parameter.normal_(std=0.5)
    decoder.to_compute("cpu", torch.bfloat16).eval().requires_grad_(False)
    yield decoder
    torch.set_num_threads(previous)


class _FakeGroup:
    rank_in_group = 0
    ranks = [0]
    world_size = 1
    cpu_group = object()
    device_group = object()


class _FakeDecoderModule(torch.nn.Module):
    def __init__(self, protocol):
        super().__init__()
        self.protocol = protocol
        self.spatial_ratio = 2
        self.anchor = torch.nn.Parameter(torch.zeros(1))


class _FakeSocket:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def send_json(self, payload):
        self.sent.append(payload)

    def recv_json(self):
        return self.replies.pop(0)

    def close(self):
        pass


def _remote_decoder(monkeypatch, protocol):
    codecs = SimpleNamespace(
        modules={"lynn_decoder": _FakeDecoderModule(protocol)},
        decoder=SimpleNamespace(module_key="lynn_decoder"),
    )
    from sglang.multimodal_gen.runtime.distributed import parallel_state

    monkeypatch.setattr(parallel_state, "get_world_group", lambda: _FakeGroup())
    monkeypatch.setattr(RemoteLynnDecoder, "_connect", lambda self: None)
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
        "zing_h3.remote_decoder.send_latent",
        lambda *args, **kwargs: None,
    )
    decoder = RemoteLynnDecoder(
        codecs,
        capacity_latents=40,
        provenance_for=_provenance,
        config={"port": 31002, "session_id": "session-1"},
    )
    decoder.bind_dispatch(
        request_id="request-1", chunk_index=3, generation_id="generation-1"
    )
    return decoder


class _RecordingPublisher:
    def __init__(self):
        self.timeout = 10
        self.published = []
        self.failed = []

    def publish(self, reference, frames):
        self.published.append((reference, frames))

    def fail(self, reference, error):
        self.failed.append((reference, error))


@pytest.fixture
def decode_group(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda values, value, group=None: values.__setitem__(0, value),
    )
    return object()


def test_service_decodes_the_same_frames_as_the_local_stream(model, decode_group):
    """Remote output must preserve both frame boundaries and rectangular geometry."""
    protocol = model.protocol
    latent = _latent(5, seed=3, height=3, width=5)
    frames = protocol.frame_count(5)
    publisher = _RecordingPublisher()
    service = _DecodeService(
        model,
        capacity_latents=40,
        group=None,
        publisher=publisher,
        decode_group=decode_group,
        device=torch.device("cpu"),
    )
    local = SourceLynnDecoder(
        SimpleNamespace(modules={ADAPTER.module_key: model}, decoder=ADAPTER),
        capacity_latents=40,
        provenance_for=_provenance,
    )
    try:
        service.append(
            latent,
            latent_start=0,
            valid_frame_end=frames,
            source_dependency_end=frames,
        )
        service.submit(frames, {"dispatch": 0})
        local.append(
            latent,
            latent_start=0,
            valid_frame_end=frames,
            source_dependency_end=frames,
        )
        expected = local.pop()
        deadline = time.monotonic() + 30
        while not publisher.published and not publisher.failed:
            assert time.monotonic() < deadline, "decoder service did not publish"
            time.sleep(0.01)
        assert not publisher.failed
        reference, values = publisher.published[0]

        assert reference == {"dispatch": 0}
        assert values.shape[2] == expected.end - expected.start
        assert (service.height, service.width) == tuple(values.shape[-2:])
        assert torch.equal(values, expected.values)
    finally:
        service.drain(10)
        service.dispose()
        local.dispose()


def test_next_transfer_waits_for_the_submitted_decode(model, decode_group):
    """D2 must finish tile work before the command thread starts another transfer."""
    entered, release, returned = (threading.Event() for _ in range(3))
    publisher = _RecordingPublisher()
    original = publisher.publish

    def publish(reference, frames):
        entered.set()
        assert release.wait(10)
        original(reference, frames)

    publisher.publish = publish
    service = _DecodeService(
        model,
        capacity_latents=40,
        group=None,
        publisher=publisher,
        decode_group=decode_group,
        device=torch.device("cpu"),
    )
    waiter = None
    try:
        latent = _latent(1, seed=3, height=3, width=5)
        frames = model.protocol.frame_count(1)
        service.append(
            latent,
            latent_start=0,
            valid_frame_end=frames,
            source_dependency_end=frames,
        )
        service.submit(frames, {"dispatch": 0})
        assert entered.wait(10)

        def transfer():
            service.wait_for_transfer()
            returned.set()

        waiter = threading.Thread(target=transfer)
        waiter.start()
        assert not returned.wait(0.05)
        release.set()
        assert returned.wait(10)
        assert not publisher.failed
    finally:
        release.set()
        if waiter is not None:
            waiter.join(timeout=10)
        service.drain(10)
        service.dispose()


def test_one_dispatch_reserves_one_reference_with_its_identity(monkeypatch):
    """The API rejects a reference whose dispatch identity is stale or missing."""
    protocol = SimpleNamespace(frame_count=lambda count: 4 * count)
    decoder = _remote_decoder(monkeypatch, protocol)
    # _connect is stubbed out, so set the channel open would have returned.
    decoder.channel = 7
    decoder.socket = _FakeSocket(
        [
            {"ok": True},
            {"ok": True},
            {
                "ok": True,
                "reference": {"num_bytes": 80},
                "metadata": {"format": "rgb24", "width": 4, "height": 4},
            },
        ]
    )
    latent = torch.zeros(1, 24, 5, 2, 2)
    decoder.append(latent, latent_start=0, source_dependency_end=20, valid_frame_end=20)
    decoder.append(latent, latent_start=5, source_dependency_end=40, valid_frame_end=40)

    decoded = decoder.pop()
    sent = decoder.socket.sent
    outputs = [item for item in sent if item["op"] == "output"]
    assert len(outputs) == 1
    assert outputs[0] == {
        "op": "output",
        "channel": 7,
        "frames": 40,
        "request_id": "request-1",
        "chunk_index": 3,
        "generation_id": "generation-1",
    }
    assert decoded.start == 0 and decoded.end == 40 and decoded.frames == 40
    assert decoded.reference == {"num_bytes": 80}
    assert decoded.transport_metadata["width"] == 4
    assert decoder.pop() is None


def test_remote_append_rejects_discontinuous_latents(monkeypatch):
    protocol = SimpleNamespace(frame_count=lambda count: 4 * count)
    decoder = _remote_decoder(monkeypatch, protocol)
    decoder.socket = _FakeSocket([{"ok": True}])
    latent = torch.zeros(1, 24, 5, 2, 2)
    with pytest.raises(ValueError, match="contiguous"):
        decoder.append(
            latent, latent_start=1, source_dependency_end=20, valid_frame_end=20
        )


def test_remote_decoder_requires_lynn_for_remote_service():
    codecs = SimpleNamespace(
        modules={"video_vae": torch.nn.Linear(1, 1)},
        decoder=SimpleNamespace(module_key="video_vae"),
    )
    with pytest.raises(ValueError, match="Lynn"):
        create_decoder(
            codecs,
            policy={"remote": {"port": 31002}},
            capacity_latents=8,
            provenance_for=_provenance,
        )


def test_leader_failure_is_raised_by_the_next_pop_on_every_rank(monkeypatch):
    """A leader-only raise in append would strand the peer DiT rank in SP."""
    protocol = SimpleNamespace(frame_count=lambda count: 4 * count)
    decoder = _remote_decoder(monkeypatch, protocol)
    broadcasts = []
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda values, **kwargs: broadcasts.append(list(values)),
    )
    decoder.socket = _FakeSocket([])
    latent = torch.zeros(1, 24, 1, 2, 2)
    decoder.append(latent, latent_start=0, source_dependency_end=4, valid_frame_end=4)
    assert decoder.failure is not None

    with pytest.raises(RuntimeError, match="Remote decoder failed"):
        decoder.pop()
    # Only the failure flag crosses ranks; the reference never leaves the leader.
    assert broadcasts == [[True]]
    decoder.dispose()
    assert decoder.closed
    with pytest.raises(ValueError, match="closed"):
        decoder.append(
            latent, latent_start=1, source_dependency_end=8, valid_frame_end=8
        )


def test_failed_finish_is_raised_by_the_final_pop_on_every_rank(monkeypatch):
    """A rejected finish must not let an EOS dispatch without new latents report
    success while the decoder host has already asked to be restarted."""
    protocol = SimpleNamespace(frame_count=lambda count: 4 * count)
    decoder = _remote_decoder(monkeypatch, protocol)
    broadcasts = []
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda values, **kwargs: broadcasts.append(list(values)),
    )
    decoder.socket = _FakeSocket(
        [
            {"ok": True},
            {"ok": True, "reference": {"num_bytes": 48}, "metadata": {"width": 2}},
            {"ok": False, "code": "DECODER_RESTART_REQUIRED"},
        ]
    )
    latent = torch.zeros(1, 24, 1, 2, 2)
    decoder.append(latent, latent_start=0, source_dependency_end=4, valid_frame_end=4)
    assert decoder.pop() is not None

    decoder.finish(total_frames=4)
    assert decoder.failure is not None
    # No new latent arrives in the EOS dispatch, yet the failure must surface.
    assert decoder.ready
    with pytest.raises(RuntimeError, match="Remote decoder failed"):
        decoder.pop()
    assert broadcasts[-1] == [True]


def test_publisher_publishes_the_api_rgb24_contract(monkeypatch, tmp_path):
    """The API consumer reads these fields and this byte layout verbatim."""
    monkeypatch.setenv("SGLANG_REALTIME_VAE_SHM_DIR", str(tmp_path))
    publisher = _OutputPublisher(timeout_seconds=10)
    reference = publisher.reserve(
        frames=2,
        width=2,
        height=2,
        identity=dict(
            session_id="session-1",
            generation_id="generation-1",
            request_id="request-1",
            chunk_index=3,
        ),
    )
    assert reference["kind"] == "raw_rgb24_frames"
    assert reference["version"] == 1
    assert reference["batch_num_frames"] == [2]
    assert reference["num_bytes"] == 2 * 2 * 2 * 3
    assert reference["chunk_index"] == 3

    # B/C/T/H/W FP32 -> B/T/H/W/C uint8, the layout the API slices into frames.
    frames = torch.linspace(0.0, 1.0, 1 * 3 * 2 * 2 * 2).reshape(1, 3, 2, 2, 2)
    publisher.publish(reference, frames)
    expected = np.transpose(
        (frames.numpy() * 255).astype(np.uint8), (0, 2, 3, 4, 1)
    ).tobytes()
    assert open(reference["path"], "rb").read() == expected

    open(reference["ack_path"], "wb").close()
    publisher.close()
    assert not any(tmp_path.iterdir())


class _LoopExit(BaseException):
    """Ends the worker loop once the scripted commands are consumed."""


def _loop_service(*, fail_on=None):
    class Service:
        def __init__(self, *args, **kwargs):
            self.width = self.height = 8
            self.valid_frame_end = self.submitted_frames = self.buffered_latents = 0

        def check(self):
            pass

        def append(
            self, latent, *, latent_start, valid_frame_end, source_dependency_end
        ):
            if fail_on == "append":
                raise RuntimeError("injected append failure")
            self.valid_frame_end = valid_frame_end

        def submit(self, frames, reference):
            self.submitted_frames += frames

        def finish(self, total_frames):
            pass

        def drain(self, timeout):
            pass

        def dispose(self):
            pass

    return Service


def _run_decoder_loop(commands, *, fail_on=None, live=None):
    """Run the worker control loop with fake devices and optional real ZMQ."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        encoder_control,
    )

    path = (
        Path(__file__).resolve().parents[2]
        / "runtime/pipelines_core/stages/model_specific_stages/zing_h3/decoder_worker.py"
    )
    tree = ast.parse(path.read_text())
    main = next(
        x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "main"
    )
    script = iter(commands)
    channel_from_open = {}

    def take():
        try:
            command = next(script)
        except StopIteration:
            raise _LoopExit
        # Commands after open carry the channel open returned, as the DiT client does.
        if command.get("channel") == "$open":
            command = dict(command, channel=channel_from_open["channel"])
        return command

    replies = []

    def send(reply):
        if isinstance(reply, dict) and "channel" in reply:
            channel_from_open["channel"] = reply["channel"]
        replies.append(reply)

    socket = SimpleNamespace(
        setsockopt=lambda *a: None,
        bind=lambda *a: None,
        poll=lambda *_: True,
        recv_multipart=lambda: [json.dumps(take()).encode()],
        send_json=send,
        close=lambda: None,
    )
    context = SimpleNamespace(socket=lambda *a: socket, term=lambda: None)
    args = SimpleNamespace(
        model=Path("model"),
        port=31003,
        decoder_config="{}",
        timeout_seconds=120,
        startup_timeout_seconds=600,
    )

    def all_gather_object(values, value, group=None):
        values[:] = [value]

    dist = SimpleNamespace(
        init_process_group=lambda *a, **kw: None,
        get_rank=lambda: 0,
        get_world_size=lambda *a, **kw: 1,
        new_group=lambda **kw: object(),
        broadcast_object_list=lambda *a, **kw: None,
        all_gather_object=all_gather_object,
        TCPStore=lambda *a, **kw: object(),
    )
    stub_cuda = SimpleNamespace(
        set_device=lambda n: None,
        current_device=lambda: 0,
        memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
        max_memory_allocated=lambda: 0,
    )
    namespace = dict(
        __doc__="test",
        argparse=SimpleNamespace(
            ArgumentParser=lambda **kw: SimpleNamespace(
                add_argument=lambda *a, **k: None, parse_args=lambda: args
            )
        ),
        Path=Path,
        json=json,
        logging=logging,
        time=SimpleNamespace(time_ns=lambda: 0),
        timedelta=timedelta,
        os=SimpleNamespace(environ={"LOCAL_RANK": "0"}),
        torch=SimpleNamespace(
            device=torch.device,
            float32=torch.float32,
            empty=torch.empty,
            cuda=stub_cuda,
        ),
        zmq=SimpleNamespace(Context=lambda: context, REP=1, LINGER=2, SNDTIMEO=3),
        dist=dist,
        initialize_runtime_imports=lambda: (
            SimpleNamespace(from_kwargs=lambda **kw: None),
            lambda server_args: None,
        ),
        load_decoder=lambda config: None,
        bridge_group=lambda *a, **kw: object(),
        receive_latent=lambda group, shape, dtype, device: torch.zeros(
            shape, dtype=dtype
        ),
        _DecodeService=_loop_service(fail_on=fail_on),
        _OutputPublisher=lambda **kw: SimpleNamespace(
            reserve=lambda **kw: dict(frames=kw["frames"], **kw["identity"]),
            close=lambda: None,
        ),
        _output_identity=_output_identity,
        EncoderCommandControl=encoder_control.EncoderCommandControl,
        EncoderControlError=encoder_control.EncoderControlError,
    )
    if live is not None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        stop = live["stop"]

        class LiveSocket:
            def setsockopt(self, *args):
                socket.setsockopt(*args)

            def bind(self, _endpoint):
                live["port"].append(socket.bind_to_random_port("tcp://127.0.0.1"))
                live["ready"].set()

            def poll(self, timeout):
                if stop.is_set():
                    raise _LoopExit
                return socket.poll(timeout)

            def recv_multipart(self):
                return socket.recv_multipart()

            def send_json(self, reply):
                socket.send_json(reply)

            def close(self):
                socket.close(linger=0)

        namespace["zmq"] = SimpleNamespace(
            Context=lambda: SimpleNamespace(
                socket=lambda *_: LiveSocket(), term=context.term
            ),
            REP=zmq.REP,
            LINGER=zmq.LINGER,
            SNDTIMEO=zmq.SNDTIMEO,
        )
        reserve = namespace["_OutputPublisher"]().reserve

        def delayed_reserve(**kwargs):
            live["output_entered"].set()
            assert live["release_output"].wait(10)
            return reserve(**kwargs)

        namespace["_OutputPublisher"] = lambda **kw: SimpleNamespace(
            reserve=delayed_reserve, close=lambda: None
        )
    exec(
        compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"), namespace
    )
    try:
        namespace["main"]()
    except _LoopExit:
        return replies, None
    except encoder_control.EncoderControlError as error:
        return replies, error


def test_timed_out_output_releases_owner_before_next_session(monkeypatch):
    """A late REP reply must not leave the worker owned after its REQ times out."""
    live = {
        name: threading.Event()
        for name in ("ready", "stop", "output_entered", "release_output")
    }
    live["port"] = []
    errors = []

    def worker():
        try:
            _run_decoder_loop([], live=live)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    decoder = None
    context = zmq.Context()

    def request(payload):
        with context.socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, 2000)
            socket.setsockopt(zmq.RCVTIMEO, 2000)
            socket.connect(f"tcp://127.0.0.1:{live['port'][0]}")
            socket.send_json(payload)
            return socket.recv_json()

    try:
        assert live["ready"].wait(10)
        protocol = SimpleNamespace(frame_count=lambda count: 4 * count)
        decoder = _remote_decoder(monkeypatch, protocol)
        decoder.port = live["port"][0]
        decoder.timeout = 1
        decoder.context = zmq.Context()
        decoder.socket = decoder.context.socket(zmq.REQ)
        decoder.socket.setsockopt(zmq.LINGER, 0)
        decoder.socket.setsockopt(zmq.SNDTIMEO, 1000)
        decoder.socket.setsockopt(zmq.RCVTIMEO, 1000)
        decoder.socket.connect(f"tcp://127.0.0.1:{decoder.port}")
        # Mirror _connect: the channel open returned is what later commands carry.
        decoder.channel = decoder._request(
            "open", bridge_port=1001, capacity_latents=40
        )["channel"]
        assert decoder._request(
            "append",
            shape=[1, 24, 1, 2, 2],
            dtype="float32",
            latent_start=0,
            valid_frame_end=4,
            source_dependency_end=4,
        )["ok"]
        with pytest.raises(RuntimeError, match="Remote decoder failed"):
            decoder._request(
                "output",
                frames=4,
                request_id="request-1",
                generation_id="generation-1",
                chunk_index=0,
            )
        assert live["output_entered"].is_set()
        live["release_output"].set()
        assert request(dict(op="ping"))["busy"] is True
        decoder.dispose()
        # The same live worker, not a restart, must accept another owner.
        assert request(
            dict(op="open", session="next-owner", bridge_port=1001, capacity_latents=40)
        )["ok"]
        assert request(dict(op="abort", session="session-1"))["released"] is False
        assert request(dict(op="ping"))["busy"] is True
        assert request(dict(op="abort", session="next-owner"))["released"] is True
    finally:
        live["release_output"].set()
        if decoder is not None:
            decoder.dispose()
        live["stop"].set()
        thread.join(timeout=10)
        context.term()
    assert not thread.is_alive()
    assert not errors


@pytest.mark.parametrize("session_field", [{}, {"session": "forged-owner"}])
def test_decoder_loop_acknowledges_every_payload_free_op(session_field):
    """Append must reply ok; a null reply reached the DiT and failed the whole session."""
    replies, error = _run_decoder_loop(
        [
            dict(op="open", session="owner", bridge_port=1001, capacity_latents=40),
            dict(
                op="append",
                channel="$open",
                shape=[1, 24, 1, 2, 2],
                dtype="float32",
                latent_start=0,
                valid_frame_end=4,
                source_dependency_end=4,
            ),
            dict(
                op="output",
                channel="$open",
                frames=4,
                request_id="request-1",
                chunk_index=0,
                generation_id="generation-1",
                **session_field,
            ),
            dict(op="finish", channel="$open", total_frames=4),
            dict(op="close", channel="$open"),
            dict(op="ping"),
        ]
    )
    assert error is None
    assert len(replies) == 6
    assert all(reply is not None and reply.get("ok") is True for reply in replies)
    assert replies[2]["reference"] == {
        "frames": 4,
        "session_id": "owner",
        "request_id": "request-1",
        "generation_id": "generation-1",
        "chunk_index": 0,
    }
    assert replies[-1]["busy"] is False


def test_a_command_with_a_stale_channel_is_rejected_and_ends_the_service():
    """A late command from a replaced owner is refused, and it ends the service.

    Refusing without ending would let a stale client keep a damaged stream alive;
    the same failure rule as any other operation applies.
    """
    replies, error = _run_decoder_loop(
        [
            dict(op="open", session="owner", bridge_port=1001, capacity_latents=40),
            dict(op="finish", channel="stale-channel", total_frames=4),
        ]
    )
    assert error is not None and "participating rank" in str(error)
    # open acks with its channel; the refused command then gets the restart reply.
    assert [reply.get("ok") for reply in replies] == [True, False]
    assert replies[-1]["code"] == "DECODER_RESTART_REQUIRED"


def test_decoder_loop_failed_append_ends_service_without_ok_reply():
    """A failed payload-free op must not default to ok; the DiT would accept a damaged stream."""
    replies, error = _run_decoder_loop(
        [
            dict(op="open", session="owner", bridge_port=1001, capacity_latents=40),
            dict(
                op="append",
                channel="$open",
                shape=[1, 24, 1, 2, 2],
                dtype="float32",
                latent_start=0,
                valid_frame_end=4,
                source_dependency_end=4,
            ),
        ],
        fail_on="append",
    )
    assert error is not None and "participating rank" in str(error)
    # Open acks, the failed append sends nothing ok, and the restart reply ends the session.
    assert [reply.get("ok") for reply in replies] == [True, False]
    assert replies[-1]["code"] == "DECODER_RESTART_REQUIRED"
