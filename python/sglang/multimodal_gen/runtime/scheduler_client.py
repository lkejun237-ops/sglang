import itertools

# import json
# import os
import pickle
import time
import zlib

# from pathlib import Path
from typing import Any, Optional

import zmq
import zmq.asyncio

from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
    ListLorasReq,
    MergeLoraWeightsReq,
    ReleaseRealtimeSessionReq,
    ReplaceQueuedRealtimeReq,
    SetLoraReq,
    ShutdownReq,
    UnmergeLoraWeightsReq,
    ZingCacheReq,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    GetWeightsChecksumReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromTensorCheckerReqInput,
    UpdateWeightFromTensorReqInput,
)
from sglang.multimodal_gen.runtime.ipc_array import materialize_file_refs
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.server_args import (
    MAX_SCHEDULER_RPC_TIMEOUT_S,
    ServerArgs,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.request_logger import (
    DiffusionRequestLogger,
)

logger = init_logger(__name__)

# Control ops mutate replica state (weights, LoRA, memory, shutdown), so with
# DP they must reach every replica rather than one.
_CONTROL_REQ_TYPES = (
    SetLoraReq,
    MergeLoraWeightsReq,
    UnmergeLoraWeightsReq,
    ListLorasReq,
    ShutdownReq,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromTensorReqInput,
    UpdateWeightFromTensorCheckerReqInput,
    GetWeightsChecksumReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)


def _configure_recv_timeout(socket: Any, timeout_ms: int | None) -> None:
    if timeout_ms is None:
        return
    max_timeout_ms = MAX_SCHEDULER_RPC_TIMEOUT_S * 1000
    if (
        not isinstance(timeout_ms, int)
        or isinstance(timeout_ms, bool)
        or not 0 < timeout_ms <= max_timeout_ms
    ):
        raise ValueError(
            f"timeout_ms must be None or an integer between 1 and {max_timeout_ms}"
        )
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)


def _resolve_timeout_ms(server_args: ServerArgs, timeout_ms: int | None) -> int | None:
    if timeout_ms is not None:
        return timeout_ms
    timeout_s = server_args.scheduler_rpc_timeout
    return None if timeout_s is None else timeout_s * 1000


async def run_zeromq_broker(server_args: ServerArgs):
    """
    This function runs as a background task in the FastAPI process.
    It listens for TCP requests from offline clients (e.g., DiffGenerator).
    """
    ctx = zmq.asyncio.Context()
    socket = ctx.socket(zmq.REP)
    broker_endpoint = f"tcp://127.0.0.1:{server_args.broker_port}"
    socket.bind(broker_endpoint)
    logger.info(f"ZMQ Broker is listening for offline jobs on {broker_endpoint}")

    try:
        while True:
            try:
                # 1. Receive a request from an offline client
                payload = await socket.recv()
                request_batch = pickle.loads(payload)
                logger.debug("Broker received an offline job from a client.")

                # 2. Forward the request to the main Scheduler via the shared client
                response_batch = await async_scheduler_client.forward(request_batch)

                # 3. Send the Scheduler's reply back to the offline client
                await socket.send(pickle.dumps(response_batch))

            except Exception as e:
                logger.error(f"Error in ZMQ Broker: {e}", exc_info=True)
                # A reply must be sent to prevent the client from hanging
                try:
                    await socket.send(
                        pickle.dumps({"status": "error", "message": str(e)})
                    )
                except Exception:
                    pass
    finally:
        socket.close(linger=0)
        ctx.destroy(linger=0)


def _session_key(batch: Any) -> str | None:
    """Realtime sessions hold GPU state on one replica, so every request of a
    session must land on the same one."""
    reqs = batch if isinstance(batch, list) else [batch]
    for req in reqs:
        if isinstance(req, Req) and req.realtime_session_id:
            return req.realtime_session_id
        if isinstance(req, (ReleaseRealtimeSessionReq, ReplaceQueuedRealtimeReq)):
            return req.session_id
    return None


def _select_replica(batch: Any, dp_size: int, counter: "itertools.count") -> int:
    if isinstance(batch, ZingCacheReq) and dp_size != 1:
        raise ValueError("Zing L1 cache management requires DP=1")
    if dp_size <= 1:
        return 0
    session = _session_key(batch)
    if session is not None:
        return zlib.crc32(session.encode()) % dp_size
    return next(counter) % dp_size


def _merge_fanout_results(results: list[Any]) -> Any:
    """One reply for a control op sent to every replica: the first error wins,
    because "succeeded" must mean succeeded everywhere."""
    for result in results:
        if isinstance(result, OutputBatch) and result.error:
            return result
    return results[0]


# Legacy shared-HTTP multi-DiT routing is disabled; retain its source for reference.
# Release removed session ownership before acknowledgement, so a failed release
# could retry on the wrong DiT. Post-send timeouts could replay a stateful request
# on another DiT and leave duplicate sessions. Re-enabling requires confirmed
# owner cleanup and safe request retry semantics. Use explicit worker groups.
# _DYNAMIC_DENOISER_PORTS: list[int] | None = None
#
#
# def set_dynamic_denoiser_ports(ports: list[int] | tuple[int, ...] | None) -> None:
# """Dynamically override the pooled DiT scheduler ports for single-entrypoint routing."""
# global _DYNAMIC_DENOISER_PORTS
# if ports is None:
# _DYNAMIC_DENOISER_PORTS = None
# return
# validated = [int(p) for p in ports if type(p) is int and 1024 <= p <= 65535]
# if not validated or len(validated) != len(ports):
# raise ValueError("Invalid dynamic denoiser scheduler ports")
# _DYNAMIC_DENOISER_PORTS = validated
#
#
# def _resolve_dynamic_denoiser_ports(server_args: Any = None) -> list[int] | None:
# if _DYNAMIC_DENOISER_PORTS:
# return list(_DYNAMIC_DENOISER_PORTS)
# reg_path = (
# getattr(server_args, "pool_registry_path", None)
# if server_args is not None
# else None
# ) or os.environ.get("SGLANG_DISAGG_POOL_REGISTRY")
# if reg_path:
# try:
# p = Path(reg_path)
# if p.exists():
# data = json.loads(p.read_text(encoding="utf-8"))
# ports = data.get("denoiser_ports")
# if isinstance(ports, list) and ports:
# return [int(x) for x in ports]
# except Exception:
# pass
# return None
#
#
# def _resolve_scheduler_endpoints(server_args: Any) -> tuple[list[str], bool]:
# host = getattr(server_args, "host", None)
# if host is None or host == "localhost":
# host = "127.0.0.1"
# dyn_ports = _resolve_dynamic_denoiser_ports(server_args)
# if dyn_ports:
# return [f"tcp://{host}:{p}" for p in dyn_ports], True
# dit_ports = getattr(server_args, "denoiser_scheduler_ports", None)
# if isinstance(dit_ports, (list, tuple)) and dit_ports:
# return [f"tcp://{host}:{int(p)}" for p in dit_ports], True
# endpoints = getattr(server_args, "scheduler_endpoints", None)
# if endpoints:
# is_pooled = (
# bool(getattr(server_args, "scheduler_ports", None))
# and (getattr(server_args, "dp_size", 1) or 1) == 1
# )
# return list(endpoints), is_pooled
# port = getattr(server_args, "scheduler_port", 31000)
# return [f"tcp://{host}:{port}"], False
#
#
# def _is_release_session_req(batch: Any) -> bool:
# reqs = batch if isinstance(batch, list) else [batch]
# return any(isinstance(r, ReleaseRealtimeSessionReq) for r in reqs)
#
#
# def _select_endpoint_candidates(
# batch: Any,
# endpoints: list[str],
# is_pooled: bool,
# counter: "itertools.count",
# session_endpoints: dict[str, str],
# ) -> list[str]:
# if len(endpoints) <= 1:
# session = _session_key(batch)
# if session is not None:
# if _is_release_session_req(batch):
# session_endpoints.pop(session, None)
# elif endpoints:
# session_endpoints[session] = endpoints[0]
# return list(endpoints)
#
# session = _session_key(batch)
# if session is not None:
# if _is_release_session_req(batch):
# pinned = session_endpoints.pop(session, None)
# if pinned in endpoints:
# return [pinned]
# elif session in session_endpoints and session_endpoints[session] in endpoints:
# return [session_endpoints[session]]
# if is_pooled:
# primary_idx = next(counter) % len(endpoints)
# else:
# primary_idx = _select_replica(batch, len(endpoints), counter)
# chosen = endpoints[primary_idx]
# if not _is_release_session_req(batch):
# session_endpoints[session] = chosen
# return [chosen] + [ep for i, ep in enumerate(endpoints) if i != primary_idx]
#
# primary_idx = _select_replica(batch, len(endpoints), counter)
# return [endpoints[primary_idx]] + [
# ep for i, ep in enumerate(endpoints) if i != primary_idx
# ]
#
#
class SchedulerClient:
    """
    A synchronous, singleton client for communicating with the Scheduler service.
    Designed for use in DiffGenerator, where synchronous usage is preferred
    """

    def __init__(self):
        self.context = None
        self.server_args = None
        self.request_logger: Optional[DiffusionRequestLogger] = None
        self._replica_counter = itertools.count()
        # self._session_endpoints: dict[str, str] = {}

    def initialize(self, server_args: ServerArgs):
        if self.context is not None and not self.context.closed:
            logger.warning("SchedulerClient is already initialized. Re-initializing.")
            self.close()

        self.server_args = server_args
        self.request_logger = DiffusionRequestLogger.from_server_args(server_args)
        self.context = zmq.Context()
        # self._session_endpoints.clear()

    def forward(self, batch: Any, timeout_ms: int | None = None) -> Any:
        """Sends a batch or request to the scheduler and waits for the response."""
        return self._forward_routed(batch, timeout_ms)

    def _forward_one(self, endpoint: str, batch: Any, timeout_ms: int | None) -> Any:
        socket = self.context.socket(zmq.REQ)
        try:
            socket.setsockopt(zmq.LINGER, 0)
            effective_timeout = _resolve_timeout_ms(self.server_args, timeout_ms)
            _configure_recv_timeout(socket, effective_timeout)
            socket.connect(endpoint)
            socket.send_pyobj(batch)
            output_batch = socket.recv_pyobj()
            _materialize_output_batch_file_refs(output_batch)
            return output_batch
        except zmq.error.Again:
            logger.error("Timeout waiting for response from %s.", endpoint)
            raise TimeoutError("Scheduler did not respond in time.")
        finally:
            socket.close()

    def _forward_routed(self, batch: Any, timeout_ms: int | None) -> Any:
        self.request_logger.log_received_request(batch)
        # endpoints, is_pooled = _resolve_scheduler_endpoints(self.server_args)
        endpoints = self.server_args.scheduler_endpoints
        if isinstance(batch, _CONTROL_REQ_TYPES):
            results = [self._forward_one(ep, batch, timeout_ms) for ep in endpoints]
            output_batch = _merge_fanout_results(results)
        else:
            # Disabled legacy candidate retry; see the ownership/timeout notes above.
            # candidates = _select_endpoint_candidates(
            # batch,
            # endpoints,
            # is_pooled,
            # self._replica_counter,
            # # self._session_endpoints,
            # )
            # last_err: Exception | None = None
            # output_batch = None
            # session = _session_key(batch)
            # for idx, ep in enumerate(candidates):
            # try:
            # output_batch = self._forward_one(ep, batch, timeout_ms)
            # if (
            # idx > 0
            # and session is not None
            # and not _is_release_session_req(batch)
            # ):
            # # self._session_endpoints[session] = ep
            # last_err = None
            # break
            # except (TimeoutError, zmq.error.ZMQError) as exc:
            # last_err = exc
            # if not is_pooled or idx == len(candidates) - 1:
            # raise
            # if last_err is not None:
            # raise last_err
            replica = _select_replica(batch, len(endpoints), self._replica_counter)
            output_batch = self._forward_one(endpoints[replica], batch, timeout_ms)
        self.request_logger.log_finished_request(batch, output_batch)
        return output_batch

    def ping(self) -> bool:
        """
        Checks if the scheduler server is alive using a temporary socket.
        """
        if self.context is None or self.context.closed:
            logger.error("Cannot ping: client is not initialized.")
            return False

        # endpoints, _ = _resolve_scheduler_endpoints(self.server_args)
        # for endpoint in endpoints:
        for endpoint in self.server_args.scheduler_endpoints:
            ping_socket = self.context.socket(zmq.REQ)
            ping_socket.setsockopt(zmq.LINGER, 0)
            ping_socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2-second timeout for pings
            try:
                ping_socket.connect(endpoint)
                ping_socket.send_pyobj({"method": "ping"})
                ping_socket.recv_pyobj()
            except zmq.error.Again:
                return False
            finally:
                ping_socket.close()
        return True

    def close(self):
        """Terminates the context."""
        if self.context:
            self.context.term()
            self.context = None
        # self._session_endpoints.clear()


class AsyncSchedulerClient:
    """
    An asynchronous, singleton client for communicating with the Scheduler service.
    Designed for use in asynchronous environments like FastAPI entrypoints.

    To support high concurrency, it creates a new REQ socket for each request
    rather than sharing a single one (which would cause ZMQ state errors).
    """

    def __init__(self):
        self.context = None
        self.server_args = None
        self.request_logger: Optional[DiffusionRequestLogger] = None
        self._replica_counter = itertools.count()
        # self._session_endpoints: dict[str, str] = {}

    def initialize(self, server_args: ServerArgs):
        if self.context is not None and not self.context.closed:
            logger.warning(
                "AsyncSchedulerClient is already initialized. Re-initializing."
            )
            self.close()

        self.server_args = server_args
        self.request_logger = DiffusionRequestLogger.from_server_args(server_args)
        self.context = zmq.asyncio.Context()
        # self._session_endpoints.clear()
        logger.debug("AsyncSchedulerClient initialized with zmq.asyncio.Context")

    async def forward(self, batch: Any, timeout_ms: int | None = None) -> Any:
        """Sends a batch or request to the scheduler and waits for the response."""
        self.request_logger.log_received_request(batch)
        if self.context is None:
            raise RuntimeError(
                "AsyncSchedulerClient is not initialized. Call initialize() first."
            )

        # endpoints, is_pooled = _resolve_scheduler_endpoints(self.server_args)
        endpoints = self.server_args.scheduler_endpoints
        if isinstance(batch, _CONTROL_REQ_TYPES):
            # replica state (weights, LoRA, memory) must change everywhere
            results = [
                await self._forward_one(ep, batch, timeout_ms) for ep in endpoints
            ]
            output_batch = _merge_fanout_results(results)
        else:
            # Disabled legacy candidate retry; see the ownership/timeout notes above.
            # candidates = _select_endpoint_candidates(
            # batch,
            # endpoints,
            # is_pooled,
            # self._replica_counter,
            # # self._session_endpoints,
            # )
            # last_err: Exception | None = None
            # output_batch = None
            # session = _session_key(batch)
            # for idx, ep in enumerate(candidates):
            # try:
            # output_batch = await self._forward_one(ep, batch, timeout_ms)
            # if (
            # idx > 0
            # and session is not None
            # and not _is_release_session_req(batch)
            # ):
            # # self._session_endpoints[session] = ep
            # last_err = None
            # break
            # except (TimeoutError, zmq.error.ZMQError) as exc:
            # last_err = exc
            # if not is_pooled or idx == len(candidates) - 1:
            # raise
            # if last_err is not None:
            # raise last_err
            replica = _select_replica(batch, len(endpoints), self._replica_counter)
            output_batch = await self._forward_one(
                endpoints[replica], batch, timeout_ms
            )
        self.request_logger.log_finished_request(batch, output_batch)
        return output_batch

    async def _forward_one(
        self, endpoint: str, batch: Any, timeout_ms: int | None
    ) -> Any:
        # a temporary REQ socket per request keeps concurrent requests from
        # interleaving on one socket's strict send/recv alternation
        socket = self.context.socket(zmq.REQ)
        try:
            socket.setsockopt(zmq.LINGER, 0)
            effective_timeout = _resolve_timeout_ms(self.server_args, timeout_ms)
            _configure_recv_timeout(socket, effective_timeout)
            socket.connect(endpoint)
            await socket.send(pickle.dumps(batch))
            payload = await socket.recv()
            output_batch = pickle.loads(payload)
            _materialize_output_batch_file_refs(output_batch)
            return output_batch
        except zmq.error.Again:
            logger.error("Timeout waiting for response from %s.", endpoint)
            raise TimeoutError("Scheduler did not respond in time.")
        finally:
            socket.close()

    async def ping(self) -> bool:
        """
        Checks if the scheduler server is alive using a temporary socket.
        """
        if self.context is None or self.context.closed:
            logger.error("Cannot ping: client is not initialized.")
            return False

        # endpoints, _ = _resolve_scheduler_endpoints(self.server_args)
        # for endpoint in endpoints:
        for endpoint in self.server_args.scheduler_endpoints:
            ping_socket = self.context.socket(zmq.REQ)
            ping_socket.setsockopt(zmq.LINGER, 0)
            ping_socket.setsockopt(zmq.RCVTIMEO, 2000)
            try:
                ping_socket.connect(endpoint)
                await ping_socket.send(pickle.dumps({"method": "ping"}))
                await ping_socket.recv()
            except zmq.error.Again:
                return False
            finally:
                ping_socket.close()
        return True

    def close(self):
        """Closes the socket and terminates the context."""
        if self.context:
            self.context.term()
            self.context = None
        # self._session_endpoints.clear()


# Singleton instances for easy access
async_scheduler_client = AsyncSchedulerClient()
sync_scheduler_client = SchedulerClient()


def _materialize_output_batch_file_refs(output_batch: Any) -> None:
    if not isinstance(output_batch, OutputBatch):
        return

    start_time = time.perf_counter()
    output_batch.output = materialize_file_refs(output_batch.output)
    if output_batch.metrics is not None:
        output_batch.metrics.record_stage(
            "SchedulerClient.materialize_file_refs",
            time.perf_counter() - start_time,
        )
