# SPDX-License-Identifier: Apache-2.0
"""Real-GPU session disposal probes; no claim about WebSocket transport cleanup.

Install instead of window_acceptance in an isolated benchmark process. The
normal finite-window generation still runs and returns its unchanged output.
"""

import gc
import json
import os
import weakref
from pathlib import Path

import torch


def install(cls):
    original = cls.sample

    @torch.no_grad()
    def sample(self, inputs, visual, audio=None):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
            SourceReadySession,
        )
        from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
            PixelRange,
            Provenance,
            SourceSpan,
        )

        result = original(self, inputs, visual, audio)
        metrics = self.last_run_metrics
        assert visual.is_cuda and inputs.source_visual is not None
        assert self.window_policy.window_chunks == 21
        snapshot = (
            torch.get_rng_state().clone(),
            torch.cuda.get_rng_state(visual.device).clone(),
        )
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        reports = {}
        chunks = [self.processor.chunk_plan.chunk(i) for i in range(2)]
        for mode in (
            "pending_cancel",
            "committed_cancel",
            "terminal_exception",
            "eos_success",
        ):
            gc.collect()
            torch.cuda.synchronize(visual.device)
            before = torch.cuda.memory_allocated(visual.device)
            session = SourceReadySession(
                self,
                inputs,
                seed=42,
                max_pending_chunks=1,
                max_pending_bytes=256 * 2**20,
                initial_rng_state=snapshot,
            )
            refs = [weakref.ref(t) for t in session.prompt[0].values()]
            cursor = 0

            def append(chunk):
                nonlocal cursor
                indices = (
                    []
                    if audio is None
                    else self.processor._audio_indices_for_video_span(
                        audio.shape[2], chunk.start, chunk.end, visual.shape[2]
                    )
                )
                session.append(
                    inputs.source_visual[:, :, chunk.start : chunk.end],
                    visual[:, :, chunk.start : chunk.end],
                    audio=None if audio is None else audio[:, :, indices],
                    audio_start=cursor,
                    final=mode != "committed_cancel" or chunk.chunk_id == 1,
                    provenance=Provenance(
                        PixelRange(chunk.start, chunk.end),
                        (
                            SourceSpan(
                                1, chunk.start, chunk.start, chunk.end - chunk.start
                            ),
                        ),
                        inputs.pixel_frame_count,
                    ),
                )
                cursor += len(indices)
                refs.extend(
                    weakref.ref(t)
                    for t in (
                        session.pending[0].source,
                        session.pending[0].noise,
                        session.pending[0].audio,
                    )
                    if t is not None
                )

            try:
                append(chunks[0])
                if mode == "eos_success":
                    session.step()
                    assert session.complete and session.closed
                elif mode == "committed_cancel":
                    session.step()
                    assert len(session.cache.committed) == 1
                    refs.extend(
                        weakref.ref(t)
                        for c in session.cache.committed
                        for layer in c.layers
                        for t in layer
                    )
                    append(chunks[1])
                elif mode == "terminal_exception":
                    forward = self._forward

                    def fail(packed, cache, update):
                        output = forward(packed, cache, update)
                        if update == "final":
                            raise RuntimeError(
                                "Injected terminal window acceptance error"
                            )
                        return output

                    self._forward = fail
                    try:
                        session.step()
                    except RuntimeError as error:
                        assert str(error) == "Injected terminal window acceptance error"
                    else:
                        raise AssertionError("Expected terminal test error")
                    finally:
                        self._forward = forward
                    assert session.committed_chunk_id == 0 and not session.pending
                    assert session.closed and not session.complete
                    assert all(v == 0 for v in session.memory_snapshot().values())
                held = session.memory_snapshot()
            finally:
                session.dispose()
                session.dispose()  # Cleanup is idempotent.
            gc.collect()
            torch.cuda.synchronize(visual.device)
            after = torch.cuda.memory_allocated(visual.device)
            assert all(ref() is None for ref in refs)
            assert all(v == 0 for v in session.memory_snapshot().values())
            assert session.rng.cpu_rng is session.rng.cuda_rng is None
            assert session.complete == (mode == "eos_success")
            assert after <= before + 2**20, (mode, before, after)
            reports[mode] = dict(
                passed=True,
                held=held,
                released=session.memory_snapshot(),
                gpu_allocated_before=before,
                gpu_allocated_after=after,
                all_owned_tensor_refs_released=True,
            )
        root = Path(os.environ["ZING_WINDOW_RESULT"])
        (root / f"window-lifecycle-rank{rank}.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    rank=rank,
                    modes=reports,
                    scope="GPU SourceReadySession dispose; WebSocket path not exercised",
                ),
                indent=2,
            )
        )
        self.last_run_metrics = metrics
        return result

    cls.sample = sample
