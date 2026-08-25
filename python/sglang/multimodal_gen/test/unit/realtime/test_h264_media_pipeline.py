# SPDX-License-Identifier: Apache-2.0

import asyncio
import shutil

import pytest

import sglang.multimodal_gen.runtime.realtime.h264_media_pipeline as pipeline_module

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    LatentChunkHeader,
    decode_message,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_worker import (
    EncodedFrameBatch,
)
from sglang.multimodal_gen.runtime.realtime.h264_media_pipeline import (
    H264MediaPipeline,
    H264PipelineConfig,
)


class _FakeStdin:
    def __init__(self):
        self.frames = []

    def write(self, value):
        self.frames.append(value)

    async def drain(self):
        await asyncio.sleep(0)

    def close(self):
        pass


class _FakeStdout:
    def __init__(self):
        self.calls = 0

    async def read(self, _size):
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(0)
            return b"fmp4-payload"
        return b""


class _FakeStderr:
    async def readline(self):
        await asyncio.Future()


class _FakeProcess:
    def __init__(self):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout()
        self.stderr = _FakeStderr()

    async def wait(self):
        return 0

    def kill(self):
        pass


class _FlushOnCloseStdin(_FakeStdin):
    def __init__(self, closed):
        super().__init__()
        self.closed = closed

    def close(self):
        self.closed.set()


class _FlushOnCloseStdout:
    def __init__(self, closed):
        self.closed = closed
        self.flushed = False

    async def read(self, _size):
        await self.closed.wait()
        if self.flushed:
            return b""
        self.flushed = True
        return b"tail-fmp4-payload"


class _FlushOnCloseProcess(_FakeProcess):
    def __init__(self):
        self.closed = asyncio.Event()
        self.stdin = _FlushOnCloseStdin(self.closed)
        self.stdout = _FlushOnCloseStdout(self.closed)
        self.stderr = _FakeStderr()

    async def wait(self):
        await self.closed.wait()
        return 0


class _NeverFlushStdout:
    async def read(self, _size):
        await asyncio.Future()


class _NeverFlushProcess(_FakeProcess):
    def __init__(self):
        self.killed = asyncio.Event()
        self.stdin = _FakeStdin()
        self.stdout = _NeverFlushStdout()
        self.stderr = _FakeStderr()

    async def wait(self):
        await self.killed.wait()
        return -9

    def kill(self):
        self.killed.set()


def _header(*, is_final_chunk: bool = False) -> LatentChunkHeader:
    return LatentChunkHeader(
        session_id="session",
        generation_id="generation",
        request_id="request-0",
        chunk_index=0,
        dtype="float16",
        shape=(1, 1, 1, 1, 1),
        byte_length=2,
        checksum="checksum",
        is_final_chunk=is_final_chunk,
    )


def _batch(*frames: bytes) -> EncodedFrameBatch:
    return EncodedFrameBatch(
        payloads=frames,
        content_type="application/x-raw-rgb",
        width=2,
        height=2,
        frame_batch_index=0,
        is_final=True,
        encode_ms=0.0,
        source_width=2,
        source_height=2,
        preview_width=2,
        preview_height=2,
    )


def _top_level_mp4_box_types(payload: bytes) -> list[bytes]:
    """解析完整 fMP4 的顶层 box；不接受截断数据，防止测试把半包算成功。"""

    types = []
    offset = 0
    while offset < len(payload):
        if len(payload) - offset < 8:
            raise AssertionError("truncated MP4 box header")
        size = int.from_bytes(payload[offset : offset + 4], "big")
        box_type = payload[offset + 4 : offset + 8]
        header_size = 8
        if size == 1:
            if len(payload) - offset < 16:
                raise AssertionError("truncated extended MP4 box header")
            size = int.from_bytes(payload[offset + 8 : offset + 16], "big")
            header_size = 16
        elif size == 0:
            size = len(payload) - offset
        if size < header_size or offset + size > len(payload):
            raise AssertionError("truncated MP4 box payload")
        types.append(box_type)
        offset += size
    return types


def test_h264_pipeline_bounds_rgb_before_encoding_and_emits_fmp4(monkeypatch):
    async def run():
        process = _FakeProcess()

        async def create_subprocess(*_args, **_kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        wires = []

        async def send(wire):
            wires.append(wire)

        pipeline = H264MediaPipeline(
            session_id="session",
            generation_id="generation",
            send=send,
            config=H264PipelineConfig(
                enabled=True,
                max_queued_frames=1,
                live_edge_frames=0,
            ),
        )
        first = bytes([1, 2, 3]) * 4
        latest = bytes([4, 5, 6]) * 4
        pipeline.enqueue(_header(), _batch(first, latest))
        pipeline.enqueue_completion(_header(), num_frames=2)

        for _ in range(100):
            messages = [decode_message(wire) for wire in wires]
            if any(message["type"] == "media_payload" for message in messages):
                break
            await asyncio.sleep(0.001)

        messages = [decode_message(wire) for wire in wires]
        assert pipeline.dropped_frames == 1
        assert process.stdin.frames == [latest]
        assert [message["type"] for message in messages] == [
            "media_init",
            "media_batch",
            "media_encode_timing",
            "media_chunk_complete",
            "media_payload",
        ]
        media = messages[-1]
        assert media["codec"] == "h264"
        assert media["container"] == "fmp4"
        assert media["payload"] == b"fmp4-payload"
        assert all(
            message.get("content_type") != "application/x-raw-rgb"
            for message in messages
        )
        await pipeline.close()

    asyncio.run(run())


def test_h264_pipeline_finalizes_final_chunk_before_session_close(monkeypatch):
    """最终块本身必须触发完整收尾，不能依赖上游 WebSocket 拆除。"""

    async def run():
        process = _FlushOnCloseProcess()
        ffmpeg_commands = []

        async def create_subprocess(*args, **_kwargs):
            ffmpeg_commands.append(args)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        wires = []

        async def send(wire):
            wires.append(wire)

        pipeline = H264MediaPipeline(
            session_id="session",
            generation_id="generation",
            send=send,
            config=H264PipelineConfig(
                enabled=True,
                # 队列放得下整块，避免掉帧策略干扰本用例
                max_queued_frames=8,
                live_edge_frames=8,
            ),
        )
        frame = bytes([1, 2, 3]) * 4
        final_header = _header(is_final_chunk=True)
        pipeline.enqueue(final_header, _batch(frame, frame))
        pipeline.enqueue_completion(final_header, num_frames=2)

        # 不调用 close()。final completion 入队后应主动排空帧、关闭 ffmpeg，
        # 等 stdout EOF 后把最终 marker 发给网关。
        for _ in range(1000):
            if any(
                decode_message(wire)["type"] == "media_chunk_complete" for wire in wires
            ):
                break
            await asyncio.sleep(0.001)

        messages = [decode_message(wire) for wire in wires]
        types = [message["type"] for message in messages]
        assert types.count("media_batch") == 2, f"尾帧被丢了: {types}"
        completions = [m for m in messages if m["type"] == "media_chunk_complete"]
        assert len(completions) == 1, f"最后一块的完成标记被丢了: {types}"
        assert completions[0]["chunk_index"] == 0
        assert completions[0]["media_transport"] == "h264"
        assert process.stdin.frames == [frame, frame]
        # final completion 是媒体 EOF 屏障，必须排在 ffmpeg 冲出的尾部 payload 后。
        assert types.index("media_payload") < types.index("media_chunk_complete")
        # rawvideo 管线不应启用 demuxer 的 nobuffer；真实 ffmpeg 会因此吞尾帧。
        assert len(ffmpeg_commands) == 1
        assert "nobuffer" not in ffmpeg_commands[0]
        # 会话 finally 再次 close 只等待同一个已完成任务。
        await pipeline.close()

    asyncio.run(run())


def test_h264_pipeline_does_not_claim_completion_when_ffmpeg_drain_times_out(
    monkeypatch,
):
    """编码器没到 EOF 时不能发送 final marker，否则严格验收会被伪造为成功。"""

    async def run():
        process = _NeverFlushProcess()

        async def create_subprocess(*_args, **_kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        monkeypatch.setattr(pipeline_module, "CLOSE_DRAIN_TIMEOUT_S", 0.01)
        wires = []

        async def send(wire):
            wires.append(wire)

        pipeline = H264MediaPipeline(
            session_id="session",
            generation_id="generation",
            send=send,
            config=H264PipelineConfig(
                enabled=True,
                max_queued_frames=8,
                live_edge_frames=8,
            ),
        )
        frame = bytes([1, 2, 3]) * 4
        final_header = _header(is_final_chunk=True)
        pipeline.enqueue(final_header, _batch(frame))
        pipeline.enqueue_completion(final_header, num_frames=1)

        with pytest.raises(RuntimeError, match="H.264 encoder drain timed out"):
            await pipeline.close()

        message_types = [decode_message(wire)["type"] for wire in wires]
        assert "media_chunk_complete" not in message_types

    asyncio.run(run())


def test_h264_pipeline_does_not_claim_completion_when_queue_drain_times_out(
    monkeypatch,
):
    """FIFO 哨兵没到达时说明仍有帧未喂入编码器，final marker 必须缺席。"""

    async def run():
        process = _FakeProcess()

        async def create_subprocess(*_args, **_kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
        monkeypatch.setattr(pipeline_module, "CLOSE_DRAIN_TIMEOUT_S", 0.01)
        wires = []
        media_batch_blocked = asyncio.Event()

        async def send(wire):
            message = decode_message(wire)
            wires.append(wire)
            if message["type"] == "media_batch":
                media_batch_blocked.set()
                await asyncio.Future()

        pipeline = H264MediaPipeline(
            session_id="session",
            generation_id="generation",
            send=send,
            config=H264PipelineConfig(
                enabled=True,
                max_queued_frames=8,
                live_edge_frames=8,
            ),
        )
        frame = bytes([1, 2, 3]) * 4
        final_header = _header(is_final_chunk=True)
        pipeline.enqueue(final_header, _batch(frame))
        pipeline.enqueue_completion(final_header, num_frames=1)
        await asyncio.wait_for(media_batch_blocked.wait(), timeout=1)

        with pytest.raises(RuntimeError, match="pipeline drain timed out"):
            await pipeline.close()

        message_types = [decode_message(wire)["type"] for wire in wires]
        assert "media_chunk_complete" not in message_types

    asyncio.run(run())


@pytest.mark.parametrize("frame_count", [1, 5, 30])
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is unavailable")
def test_h264_pipeline_real_ffmpeg_preserves_every_input_frame(frame_count):
    """真实 ffmpeg 必须为每个 stdin 帧生成一个 frag_every_frame 分片。"""

    async def run():
        wires = []

        async def send(wire):
            wires.append(wire)

        pipeline = H264MediaPipeline(
            session_id="session",
            generation_id="generation",
            send=send,
            config=H264PipelineConfig(
                enabled=True,
                ffmpeg_bin=shutil.which("ffmpeg") or "ffmpeg",
                threads=1,
                preset="ultrafast",
                bitrate_kbps=250,
                vbv_buffer_ms=40,
                gop_seconds=1,
                max_queued_frames=frame_count,
                max_frame_age_ms=0,
                live_edge_frames=frame_count,
            ),
        )
        frame = bytes([31, 127, 223]) * 4
        final_header = _header(is_final_chunk=True)
        pipeline.enqueue(final_header, _batch(*([frame] * frame_count)))
        pipeline.enqueue_completion(final_header, num_frames=frame_count)
        await pipeline.close()

        messages = [decode_message(wire) for wire in wires]
        assert (
            sum(message["type"] == "media_batch" for message in messages) == frame_count
        )
        fmp4 = b"".join(
            message["payload"]
            for message in messages
            if message["type"] == "media_payload"
        )
        box_types = _top_level_mp4_box_types(fmp4)
        assert box_types.count(b"moof") == frame_count
        assert box_types.count(b"mdat") == frame_count
        assert messages[-1]["type"] == "media_chunk_complete"

    asyncio.run(run())
