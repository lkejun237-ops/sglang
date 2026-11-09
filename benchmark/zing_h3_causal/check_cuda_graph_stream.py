# SPDX-License-Identifier: Apache-2.0
"""Check Zing capture isolation after the process-wide stream pool wraps."""

import argparse
import json
import threading
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        CudaGraphBackend,
    )
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_prefix_graph import (
        PrefixGraphBackend,
    )
    from sglang.multimodal_gen.runtime.utils.zing_capture_stream import (
        capture_stream,
        conditioning_nccl_stream,
        decoder_stream,
        dit_io_stream,
        remote_encoder_stream,
    )

    result = dict(
        passed=False,
        torch=torch.__version__,
        private_streams={},
        captures=[],
    )
    try:
        x = torch.ones(16, device="cuda")
        streams = [torch.cuda.Stream() for _ in range(256)]
        pool = {stream.cuda_stream for stream in streams}
        private_stream_objs = {
            "capture": capture_stream(x.device),
            "decoder": decoder_stream(x.device),
            "remote_encoder": remote_encoder_stream(x.device),
            "dit_io_peer0": dit_io_stream(x.device, key="127.0.0.1:29500:0"),
            "dit_io_peer1": dit_io_stream(x.device, key="127.0.0.1:29501:0"),
            "conditioning_nccl": conditioning_nccl_stream(x.device),
        }
        private_handles = {
            name: int(s.cuda_stream) for name, s in private_stream_objs.items()
        }
        result["private_streams"] = private_handles
        if len(set(private_handles.values())) != len(private_handles):
            raise RuntimeError(
                f"Private CUDA streams alias each other: {private_handles}"
            )
        overlap = set(private_handles.values()) & pool
        if overlap:
            raise RuntimeError(
                f"Private CUDA streams overlap with the 32-stream pool: {overlap}"
            )

        for backend, increment in [(CudaGraphBackend(), 1), (PrefixGraphBackend(), 2)]:
            streams = [torch.cuda.Stream() for _ in range(256)]
            pool = {stream.cuda_stream for stream in streams}
            entered, returned = threading.Event(), threading.Event()
            row = dict(backend=type(backend).__name__, pool_unique_streams=len(pool))

            def synchronize():
                try:
                    if not entered.wait(15):
                        raise RuntimeError("Capture did not reach the sync probe")
                    streams[0].synchronize()
                    for priv_s in private_stream_objs.values():
                        if priv_s.cuda_stream != capture_stream(x.device).cuda_stream:
                            priv_s.synchronize()
                    row["background_sync_ok"] = True
                except BaseException:
                    row["background_error"] = traceback.format_exc()
                finally:
                    returned.set()

            thread = threading.Thread(target=synchronize, daemon=True)
            thread.start()

            def forward():
                stream = torch.cuda.current_stream().cuda_stream
                row["capture_stream"] = stream
                if stream in pool:
                    raise RuntimeError("Zing capture aliases a pooled CUDA stream")
                if backend.capturing:
                    entered.set()
                    if not returned.wait(15):
                        raise RuntimeError("Background stream synchronization hung")
                    if not row.get("background_sync_ok"):
                        raise RuntimeError("Background stream synchronization failed")
                return x + increment

            before = torch.cuda.current_stream()
            graph, output = backend.capture(forward)
            thread.join(timeout=20)
            if thread.is_alive():
                raise RuntimeError("Background stream probe did not exit")
            backend.replay(graph)
            torch.testing.assert_close(output, x + increment, rtol=0, atol=0)
            if torch.cuda.current_stream() != before:
                raise RuntimeError("Capture did not restore the caller stream")
            row["output_exact"] = True
            result["captures"].append(row)
        result["passed"] = True
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
