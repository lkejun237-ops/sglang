# SPDX-License-Identifier: Apache-2.0
"""Real Gloo/ZMQ service-loop regressions; CUDA and VAE work are boundaries.

The default Gloo group stands in for the production NCCL group. Its short
watchdog makes the production idle bug reproducible without a GPU. No command
transport or rank-agreement collective is mocked.
"""

import importlib.util
import json
import multiprocessing as mp
import queue
import socket as socket_module
import sys
import time
import unittest
import uuid
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import zmq

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=120, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4]
_WORKER_DIR = _ROOT / (
    "python/sglang/multimodal_gen/runtime/pipelines_core/stages/"
    "model_specific_stages/zing_h3"
)
_DEVICE_TIMEOUT = 1.2
_IDLE_PERIOD = 2.8
_EXECUTION_TIMEOUT = 6
_BOOTSTRAP_TIMEOUT = 90


def _free_port():
    with socket_module.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(
    rank, world, rendezvous, port, events, gate, bridge_gate, bootstrap_release, mode
):
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)

    def event(kind, **fields):
        events.put(dict(kind=kind, rank=rank, **fields))

    class DistBoundary:
        def __getattr__(self, name):
            return getattr(dist, name)

        def init_process_group(self, backend, **kwargs):
            # Preserve production's distinct default device and CPU groups.
            dist.init_process_group(
                "gloo",
                init_method=rendezvous,
                rank=rank,
                world_size=world,
                timeout=timedelta(seconds=_DEVICE_TIMEOUT),
            )

        def new_group(self, *args, **kwargs):
            # Production's 30-second idle watchdog is shortened. An explicitly
            # configured operation budget remains long enough for bridge I/O.
            timeout = kwargs.get("timeout", timedelta(seconds=30))
            if timeout.total_seconds() > _EXECUTION_TIMEOUT:
                kwargs["timeout"] = timedelta(seconds=_DEVICE_TIMEOUT)
            return dist.new_group(*args, **kwargs)

        def broadcast_object_list(self, values, *args, **kwargs):
            if kwargs.get("group") is None:
                event("device_control")
            return dist.broadcast_object_list(values, *args, **kwargs)

        def broadcast(self, tensor, *args, **kwargs):
            event("input_collective")
            return dist.broadcast(tensor, *args, **kwargs)

        def TCPStore(self, *args, **kwargs):
            # The independent DiT bridge is the external tensor-I/O boundary.
            return None

    class CPUEvent:
        """Replace only CUDA completion observation, without GPU timing claims."""

        def __init__(self, *args, **kwargs):
            pass

        def record(self, stream=None):
            return self

        def wait(self, stream=None):
            pass

        def synchronize(self):
            pass

        def query(self):
            return True

        def elapsed_time(self, other):
            return 0.0

    class CPUStream:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait_event(self, event):
            pass

        def wait_stream(self, stream):
            pass

        def record_event(self, event=None):
            return (event or CPUEvent()).record(self)

        def synchronize(self):
            pass

        def query(self):
            return True

    current_stream = CPUStream()

    class TorchBoundary:
        cuda = SimpleNamespace(
            Event=CPUEvent,
            Stream=CPUStream,
            current_stream=lambda *a, **kw: current_stream,
            default_stream=lambda *a, **kw: current_stream,
            stream=lambda stream: stream,
            current_device=lambda: rank,
            is_available=lambda: False,
            synchronize=lambda *a, **kw: None,
            reset_peak_memory_stats=lambda *a, **kw: None,
            set_device=lambda *_: None,
            memory_allocated=lambda: 0,
            memory_reserved=lambda: 0,
            max_memory_allocated=lambda: 0,
            max_memory_reserved=lambda: 0,
        )

        def __getattr__(self, name):
            return getattr(torch, name)

        def empty(self, *args, **kwargs):
            if kwargs.get("device") == "cuda":
                if rank == world - 1 and mode == "delayed_prepare":
                    event("prepare_wait")
                    if not gate.wait(_EXECUTION_TIMEOUT):
                        raise RuntimeError("CPU fixture preparation gate expired")
                if rank == world - 1 and mode == "failed_prepare":
                    event("prepare_failure")
                    raise RuntimeError("private allocation failure sentinel")
                kwargs["device"] = "cpu"
                result = torch.empty(*args, **kwargs)
                event("prepared")
                return result
            return torch.empty(*args, **kwargs)

    class Session:
        # Replace the GPU model/cache boundary, retaining service admission,
        # command transport, input distribution and output ordering verbatim.
        ready = False
        buffered_frames = 0
        emitted_latents = 0

        def __init__(self, codecs, *, capacity_frames, cache):
            self.cache = cache
            self.closed = False

        def append(self, pixels, *, frame_start):
            self.buffered_frames += pixels.shape[1]
            self.ready = True
            event("append")

        def finish(self):
            event("finish")

        def pop(self):
            event("encode")
            self.emitted_latents += 1
            self.ready = False
            return SimpleNamespace(
                values=torch.ones((1, 48, 1, 1, 1)),
                start=self.emitted_latents - 1,
                end=self.emitted_latents,
                source_dependency_end=self.buffered_frames,
            )

        def dispose(self):
            self.closed = True
            event("disposed")

        def state_usage(self):
            return SimpleNamespace(retained_bytes=0, temporal_positions=0)

    class Bridge:
        def send(self, *args):
            event("bridge_started")

            def wait():
                if not bridge_gate.wait(_EXECUTION_TIMEOUT):
                    raise RuntimeError("CPU fixture bridge gate expired")
                event("bridge_done")

            return SimpleNamespace(wait=wait)

    # Load the production module under an isolated package name, avoiding the
    # unrelated pipeline registry's GPU imports. Its relative imports are real.
    package_name = "_h3_encoder_worker_cpu_fixture"
    package = ModuleType(package_name)
    package.__path__ = [str(_WORKER_DIR)]
    module_name = f"{package_name}.encoder_worker"
    parallel_name = (
        "sglang.multimodal_gen.runtime.models.vaes."
        "minimax_h3_video_vae.parallel_encoder"
    )
    parallel = ModuleType(parallel_name)
    parallel.create_encoder_cache = lambda *a, **kw: SimpleNamespace(last_execution={})
    try:
        with patch.dict(sys.modules, {package_name: package, parallel_name: parallel}):
            spec = importlib.util.spec_from_file_location(
                module_name, _WORKER_DIR / "encoder_worker.py"
            )
            worker = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = worker
            spec.loader.exec_module(worker)
            worker.dist = DistBoundary()
            worker.torch = TorchBoundary()
            worker.H3ProgressiveEncoderSession = Session
            worker.initialize_runtime_imports = lambda: (
                SimpleNamespace(from_kwargs=lambda **kw: None),
                lambda *_: None,
            )
            worker.load_encoder = lambda *_: SimpleNamespace(
                modules={"video_vae": object()},
                warmup_encoder=lambda: None,
                runtime={
                    "vae_config": {
                        "encode_mode": "performance",
                        "encode_dtype": "bfloat16",
                    }
                },
            )
            worker.bridge_group = lambda *a, **kw: Bridge()

            def capture_print(line, **kwargs):
                data = json.loads(line)
                if data.get("encoder_service_ready"):
                    event("ready")
                elif "encoder_step" in data:
                    event("operation_done", op=data["encoder_step"])

            worker.print = capture_print
            with (
                patch.dict("os.environ", {"LOCAL_RANK": str(rank)}),
                patch.object(
                    sys,
                    "argv",
                    [
                        "encoder_worker",
                        "--model",
                        "unused",
                        "--port",
                        str(port),
                        "--timeout-seconds",
                        str(_EXECUTION_TIMEOUT),
                    ],
                ),
            ):
                if rank == world - 1 and mode == "delayed_bootstrap":
                    time.sleep(2 * _DEVICE_TIMEOUT)
                event("bootstrapped")
                if not bootstrap_release.wait(_BOOTSTRAP_TIMEOUT):
                    raise RuntimeError("CPU fixture bootstrap gate expired")
                worker.main()
    except BaseException as exc:
        event("error", error_type=type(exc).__name__, detail=str(exc))
    finally:
        event("exited")


class _Service:
    def __init__(self, path, world=3, mode="normal"):
        context = mp.get_context("spawn")
        self.events = context.Queue()
        self.gate = context.Event()
        self.bridge_gate = context.Event()
        self.bootstrap_release = context.Event()
        self.mode = mode
        if mode != "slow_bridge":
            self.bridge_gate.set()
        self.rows = []
        self.port = _free_port()
        self.context = zmq.Context()
        self.client = self.context.socket(zmq.REQ)
        self.client.setsockopt(zmq.LINGER, 0)
        self.client.setsockopt(zmq.RCVTIMEO, 1200)
        self.client.setsockopt(zmq.SNDTIMEO, 1200)
        self.client.connect(f"tcp://127.0.0.1:{self.port}")
        self.processes = [
            context.Process(
                target=_worker,
                args=(
                    rank,
                    world,
                    "file://" + str(path),
                    self.port,
                    self.events,
                    self.gate,
                    self.bridge_gate,
                    self.bootstrap_release,
                    mode,
                ),
            )
            for rank in range(world)
        ]
        for process in self.processes:
            process.start()
        try:
            # Imports and fixture setup have their own budget; only release the
            # ranks into the real short-watchdog rendezvous once all are ready.
            self.wait_for(
                lambda rows: (
                    sum(row["kind"] == "bootstrapped" for row in rows) == world
                ),
                _BOOTSTRAP_TIMEOUT,
            )
            self.bootstrap_release.set()
            self.wait_for(
                lambda rows: sum(row["kind"] == "ready" for row in rows) == world,
                30,
            )
            self.check_healthy()
        except BaseException:
            self.close()
            raise

    def drain(self):
        while True:
            try:
                self.rows.append(self.events.get_nowait())
            except queue.Empty:
                return self.rows

    def wait_for(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.drain()
            if predicate(self.rows):
                return
            try:
                self.rows.append(self.events.get(timeout=0.05))
            except queue.Empty:
                pass
        raise AssertionError(
            f"Encoder events did not reach expected state: {self.rows}"
        )

    def check_healthy(self):
        errors = [row for row in self.drain() if row["kind"] == "error"]
        assert not errors, f"Healthy encoder control loop failed: {errors}"
        assert all(process.is_alive() for process in self.processes), self.rows

    def request(self, command, payload=None):
        parts = [json.dumps(command).encode()]
        if payload is not None:
            parts.append(payload)
        self.client.send_multipart(parts)
        try:
            return self.client.recv_json()
        except zmq.Again:
            raise AssertionError(
                f"Encoder did not answer {command['op']}: {self.drain()}"
            ) from None

    def open(self, owner="owner"):
        reply = self.request(
            dict(op="open", session=owner, bridge_port=1001, capacity_frames=51)
        )
        assert reply["ok"], reply

    def append_command(self):
        return dict(
            op="append",
            session="owner",
            shape=[3, 1, 2, 2],
            capacity_bytes=12,
            frame_start=0,
        )

    def close(self):
        if getattr(self, "closed", False):
            return
        self.closed = True
        self.gate.set()
        self.bridge_gate.set()
        self.bootstrap_release.set()
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
        self.client.close()
        self.context.term()
        self.events.close()
        self.events.join_thread()


class TestH3EncoderCommandSync(CustomTestCase):
    def setUp(self):
        import tempfile

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def service(self, world=3, mode="normal"):
        service = _Service(
            Path(self.directory.name) / f"rendezvous-{uuid.uuid4().hex}", world, mode
        )
        self.addCleanup(service.close)
        return service

    def _assert_idle_outlasts_device_watchdog_and_next_session_works(self, world):
        service = self.service(world=world)
        try:
            time.sleep(_IDLE_PERIOD)
            service.check_healthy()
            self.assertFalse(
                any(row["kind"] == "device_control" for row in service.drain()),
                "Idle command transport must not leave a device-group collective pending",
            )
            self.assertTrue(service.request({"op": "ping"})["ok"])
            service.open("first")
            self.assertTrue(service.request({"op": "finish", "session": "first"})["ok"])
            self.assertTrue(service.request({"op": "close", "session": "first"})["ok"])
            # Returning from close must restore healthy indefinite business idle,
            # rather than merely permit an immediately following session.
            time.sleep(_IDLE_PERIOD)
            service.check_healthy()
            self.assertFalse(service.request({"op": "ping"})["busy"])
            service.open("next")
            self.assertTrue(service.request({"op": "finish", "session": "next"})["ok"])
            self.assertTrue(service.request({"op": "close", "session": "next"})["ok"])
            self.assertFalse(service.request({"op": "ping"})["busy"])
        finally:
            service.close()

    def test_idle_and_next_session_with_two_ranks(self):
        self._assert_idle_outlasts_device_watchdog_and_next_session_works(2)

    def test_idle_and_next_session_with_three_ranks(self):
        self._assert_idle_outlasts_device_watchdog_and_next_session_works(3)

    def test_idle_and_next_session_with_four_ranks(self):
        self._assert_idle_outlasts_device_watchdog_and_next_session_works(4)

    def test_startup_skew_outlasts_device_watchdog(self):
        service = self.service(mode="delayed_bootstrap")
        self.assertTrue(service.request({"op": "ping"})["ok"])
        service.check_healthy()

    def test_every_rank_prepares_before_any_input_collective(self):
        service = self.service(mode="delayed_prepare")
        service.open()
        service.client.send_multipart(
            [json.dumps(service.append_command()).encode(), bytes(12)]
        )
        service.wait_for(
            lambda rows: any(row["kind"] == "prepare_wait" for row in rows)
        )
        # Other ranks have had a chance to run; the final participant is still
        # at the allocation boundary, so no device input work may have started.
        time.sleep(0.15)
        self.assertFalse(
            any(
                row["kind"] in ("input_collective", "encode") for row in service.drain()
            ),
            service.rows,
        )
        service.gate.set()
        self.assertTrue(service.client.recv_json()["ok"])
        self.assertTrue(service.request({"op": "pop", "session": "owner"})["ok"])

    def test_local_prepare_failure_prevents_all_input_collectives(self):
        service = self.service(mode="failed_prepare")
        service.open()
        service.client.send_multipart(
            [json.dumps(service.append_command()).encode(), bytes(12)]
        )
        service.wait_for(
            lambda rows: any(row["kind"] == "prepare_failure" for row in rows)
        )
        time.sleep(0.2)
        self.assertFalse(
            any(
                row["kind"] in ("input_collective", "encode") for row in service.drain()
            ),
            "A failed local preparation must stop every rank before GPU input work",
        )
        service.wait_for(
            lambda rows: sum(row["kind"] == "exited" for row in rows) == 3,
            _EXECUTION_TIMEOUT + 2,
        )

    def test_pop_metadata_precedes_slow_bridge_and_all_ranks_finish(self):
        service = self.service(mode="slow_bridge")
        service.open()
        self.assertTrue(service.request(service.append_command(), bytes(12))["ok"])
        reply = service.request({"op": "pop", "session": "owner"})
        self.assertEqual(reply["shape"], [1, 48, 1, 1, 1])
        self.assertEqual(reply["dtype"], "float32")
        service.wait_for(
            lambda rows: any(row["kind"] == "bridge_started" for row in rows)
        )
        time.sleep(_IDLE_PERIOD)
        service.check_healthy()
        self.assertFalse(
            any(
                row["kind"] == "operation_done" and row.get("op") == "pop"
                for row in service.drain()
            ),
            "All ranks must finish leader output before returning to idle",
        )
        service.bridge_gate.set()
        service.wait_for(
            lambda rows: (
                sum(
                    row["kind"] == "operation_done" and row.get("op") == "pop"
                    for row in rows
                )
                == 3
            )
        )
        self.assertTrue(service.request({"op": "ping"})["ok"])

    def test_lost_idle_rank_terminates_every_survivor_within_budget(self):
        service = self.service()
        service.processes[-1].terminate()
        service.processes[-1].join(5)
        deadline = time.monotonic() + _EXECUTION_TIMEOUT + 2
        while time.monotonic() < deadline and any(
            process.is_alive() for process in service.processes[:-1]
        ):
            time.sleep(0.05)
        self.assertFalse(
            any(process.is_alive() for process in service.processes[:-1]),
            f"Lost rank stranded survivors in command wait: {service.drain()}",
        )

    def test_control_protocol_accepts_single_rank(self):
        service = self.service(world=1)
        self.assertTrue(service.request({"op": "ping"})["ok"])
        service.open()
        self.assertTrue(service.request(service.append_command(), bytes(12))["ok"])
        self.assertTrue(service.request({"op": "pop", "session": "owner"})["ok"])
        self.assertTrue(service.request({"op": "close", "session": "owner"})["ok"])
        self.assertFalse(service.request({"op": "ping"})["busy"])


if __name__ == "__main__":
    unittest.main()
