# SPDX-License-Identifier: Apache-2.0
"""Untimed real-GPU transport, bounded-credit, cancellation and EOS witness.

Run on one consumer GPU against a dedicated encoder worker on separate GPUs.
This is a component diagnostic; it does not replace the six-reference WS test.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


def wait_full(session, timeout=120):
    deadline = time.monotonic() + timeout
    while session.queue_snapshot()["pending_results"] < 2:
        session._open()
        if time.monotonic() >= deadline:
            raise TimeoutError("Encoder did not fill the two result credits")
        time.sleep(0.05)
    first = session.queue_snapshot()
    time.sleep(0.25)
    second = session.queue_snapshot()
    if first["pending_results"] != second["pending_results"] or session.emitted_latents:
        raise RuntimeError("Encoder credit bound was not maintained")
    return dict(before=first, after=second, held_seconds=0.25)


def initialize_runtime_imports():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
        initialize_runtime_imports as initialize,
    )

    return initialize()


def compare_moments(actual, expected):
    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or not actual.numel()
    ):
        raise RuntimeError("Transport moments shape or dtype differs from reference")
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return dict(
        passed=finite and torch.equal(actual, expected),
        finite=finite,
        max_abs=(
            (actual.float() - expected.float()).abs().max().item() if finite else None
        ),
        atol=0,
        rtol=0,
    )


def dispose_checked(session, *, dispose=None):
    """Local queue release is insufficient if the worker did not ACK close."""
    (dispose or session.dispose)()
    if session.failure is not None:
        raise RuntimeError(
            "Encoder producer or remote close failed"
        ) from session.failure
    if (
        session.thread is not None and session.thread.is_alive()
    ) or session.state_usage().retained_bytes:
        raise RuntimeError("Encoder did not release session state")


def reset_checked(session):
    # reset() clears failure in __init__; validate the old producer before that
    # happens, while still exercising the real reset implementation.
    original = session.dispose
    session.dispose = lambda: dispose_checked(session, dispose=original)
    try:
        session.reset()
    finally:
        del session.dispose


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--source-rgb", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--exploratory",
        action="store_true",
        help="Record numerical failures while continuing lifecycle diagnostics; never merge acceptance",
    )
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()
    if args.width < 1 or args.height < 1:
        p.error("width and height must be positive")
    ServerArgs, set_global_server_args = initialize_runtime_imports()
    args.output.mkdir(parents=True, exist_ok=False)
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
        load_encoder,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
        H3ProgressiveEncoderSession,
        expected_progressive_slices,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.remote_encoder import (
        RemoteEncoderSession,
    )

    torch.cuda.set_device(0)
    set_global_server_args(
        ServerArgs.from_kwargs(
            model_path=str(args.model),
            num_gpus=1,
            video_decoder_backend="native",
            attention_backend="torch_sdpa",
            warmup_mode="off",
        )
    )
    with tempfile.TemporaryDirectory() as td:
        init_distributed_environment(
            distributed_init_method="file://" + td + "/rdzv",
            backend="nccl",
            timeout=180,
        )
        codecs = load_encoder(args.model)
        frame_bytes = args.width * args.height * 3
        pixels = torch.from_numpy(
            np.fromfile(args.source_rgb, dtype=np.uint8, count=34 * frame_bytes)
            .reshape(34, args.height, args.width, 3)
            .copy()
        ).permute(3, 0, 1, 2)
        rows = []
        numerical_checks = []
        for frames in (34, 19):
            config = dict(
                port=args.port,
                mode="overlap",
                capacity_bytes=frames * frame_bytes,
                timeout_seconds=120,
            )
            session = RemoteEncoderSession(codecs, capacity_frames=51, config=config)
            reference = H3ProgressiveEncoderSession(codecs, capacity_frames=51)
            try:
                session.append(pixels[:, :frames], frame_start=0)
                reference.append(pixels[:, :frames], frame_start=0)
                witness = wait_full(session)
                try:
                    session.append(pixels[:, :1], frame_start=frames)
                except ValueError as exc:
                    if "capacity" not in str(exc):
                        raise
                else:
                    raise RuntimeError("Encoder byte capacity was not enforced")
                session.finish()
                reference.finish()
                slices = []
                while reference.ready:
                    expected, actual = reference.pop(), session.pop()
                    check = compare_moments(actual.values, expected.values)
                    numerical_checks.append(
                        dict(frames=frames, start=actual.start, end=actual.end, **check)
                    )
                    (args.output / "numerical-checks.json").write_text(
                        json.dumps(numerical_checks, indent=2)
                    )
                    if not check["passed"] and not args.exploratory:
                        raise AssertionError(
                            "Transport moments failed exact numerical acceptance"
                        )
                    slices.append(
                        (actual.start, actual.end, actual.source_dependency_end)
                    )
                if session.ready or slices != expected_progressive_slices(frames):
                    raise RuntimeError("Transport EOS intervals differ from reference")
                rows.append(
                    dict(
                        frames=frames,
                        credit_witness=witness,
                        byte_capacity_rejected=True,
                        exact_moments=all(
                            x["passed"]
                            for x in numerical_checks
                            if x["frames"] == frames
                        ),
                        slices=slices,
                    )
                )
            finally:
                reference.dispose()
                dispose_checked(session)
        session = RemoteEncoderSession(
            codecs,
            capacity_frames=51,
            config=dict(port=args.port, mode="overlap", timeout_seconds=120),
        )
        try:
            session.append(pixels, frame_start=0)
            cancel_witness = wait_full(session)
            reset_checked(session)
            session.append(pixels[:, :1], frame_start=0)
            session.finish()
            item = session.pop()
            if (item.start, item.end, item.source_dependency_end) != (
                0,
                1,
                1,
            ) or session.ready:
                raise RuntimeError("Encoder reset retained previous session state")
        finally:
            dispose_checked(session)
        (args.output / "acceptance.json").write_text(
            json.dumps(
                dict(
                    passed=not args.exploratory
                    and all(x["passed"] for x in numerical_checks),
                    exploratory=args.exploratory,
                    merge_eligible=False,
                    performance_validated=False,
                    correctness_passed=all(x["passed"] for x in numerical_checks),
                    lifecycle_passed=True,
                    numerical_checks=numerical_checks,
                    rows=rows,
                    cancellation_at_credit_limit=cancel_witness,
                    reset_and_short_eos=True,
                ),
                indent=2,
            )
        )
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
