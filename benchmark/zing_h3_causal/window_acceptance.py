# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-checkpoint sampler instrumentation for temporal-window acceptance.

Install with install(ZingH3Sampler) in the isolated benchmark process. The finite
and ready paths must use the same configured policy. Codec pixel provenance is
explicitly a fixture; this does not validate the #114 network/codec bridge.
"""

import gc
import hashlib
import json
import os
import weakref
from collections import deque
from copy import copy
from pathlib import Path

import torch


def _rss():
    # Current resident pages, not the historical high-water mark.
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf(
        "SC_PAGE_SIZE"
    )


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

        assert (
            self.window_policy.window_chunks == 21
            and self.window_policy.sink_chunks == 3
        )
        assert inputs.source_visual is not None
        assert self.processor.chunk_plan.count(visual.shape[2]) >= 30
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        root = Path(os.environ["ZING_WINDOW_RESULT"])
        root.mkdir(parents=True, exist_ok=True)
        device = visual.device
        snapshot = (
            torch.get_rng_state().clone(),
            torch.cuda.get_rng_state(device).clone(),
        )
        expected_v, expected_a = original(self, inputs, visual, audio)
        baseline_metrics = copy(self.last_run_metrics)
        expected_v = expected_v.cpu()
        expected_a = None if expected_a is None else expected_a.cpu()
        post = (torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone())
        max_length = max(self.processor.chunk_plan.lengths)
        # One ready entry, with enough bytes for its exact modality geometry.
        byte_budget = (
            max_length
            * (visual[:, :, :1].numel() + inputs.source_visual[:, :, :1].numel())
            * 4
        )
        byte_budget += 0 if audio is None else audio.numel() * audio.element_size()
        session = SourceReadySession(
            self,
            inputs,
            seed=42,
            max_pending_chunks=1,
            max_pending_bytes=byte_budget,
            initial_rng_state=snapshot,
        )
        cursor = 0
        tail = deque(maxlen=10)
        count = 0
        output_digest = hashlib.sha256()
        before_allocated = torch.cuda.memory_allocated(device)
        log = (root / f"window-memory-rank{rank}.jsonl").open("w")
        try:
            for chunk in self.processor.chunk_plan.chunks(visual.shape[2]):
                indices = (
                    []
                    if audio is None
                    else self.processor._audio_indices_for_video_span(
                        audio.shape[2], chunk.start, chunk.end, visual.shape[2]
                    )
                )
                provenance = Provenance(
                    PixelRange(chunk.start, chunk.end),
                    (SourceSpan(1, chunk.start, chunk.start, chunk.end - chunk.start),),
                    inputs.pixel_frame_count,
                )
                session.append(
                    inputs.source_visual[:, :, chunk.start : chunk.end],
                    visual[:, :, chunk.start : chunk.end],
                    audio=None if audio is None else audio[:, :, indices],
                    audio_start=cursor,
                    provenance=provenance,
                    final=chunk.end == visual.shape[2],
                )
                cursor += len(indices)
                result = session.step()
                actual = result.visual.cpu()
                assert torch.equal(actual, expected_v[:, :, chunk.start : chunk.end])
                output_digest.update(actual.contiguous().numpy().tobytes())
                if audio is not None:
                    actual_audio = result.audio.cpu()
                    assert torch.equal(actual_audio, expected_a[:, :, indices])
                    output_digest.update(actual_audio.contiguous().numpy().tobytes())
                    del actual_audio
                i = chunk.chunk_id
                ids = [c.temporal_chunk_id for c in session.cache.committed]
                expected_ids = [*range(min(i + 1, 3)), *range(max(3, i - 17), i + 1)]
                if not session.complete:
                    assert ids == expected_ids
                else:
                    assert not ids
                del result, actual
                torch.cuda.synchronize(device)
                gc.collect()
                measurement = dict(
                    chunk=i,
                    retained_ids=ids,
                    **session.memory_snapshot(),
                    gpu_allocated=torch.cuda.memory_allocated(device),
                    gpu_reserved=torch.cuda.memory_reserved(device),
                    host_rss=_rss(),
                )
                log.write(json.dumps(measurement) + "\n")
                log.flush()
                if i >= 23 and not session.complete:
                    tail.append(measurement)
                count += 1
                print("ZING_WINDOW_CHUNK", rank, i, ids, flush=True)
            assert count >= 30 and session.closed
            assert all(value == 0 for value in session.memory_snapshot().values())
            plateau = {
                key: max(x[key] for x in tail) - min(x[key] for x in tail)
                for key in (
                    "gpu_allocated",
                    "gpu_reserved",
                    "host_rss",
                    "kv_storage_bytes",
                )
            }
            # Variable [1,2,2] chunks make a bounded periodic working set.
            # Report allocated span; allocator reservation is observable, not a leak oracle.
            assert len(tail) >= 5
            assert plateau["host_rss"] < 128 * 2**20, plateau
            # Independently replay through the first eviction, then fail after
            # final KV commit. The successful path above remains uninterrupted.
            failed = SourceReadySession(
                self,
                inputs,
                seed=42,
                max_pending_chunks=1,
                max_pending_bytes=byte_budget,
                initial_rng_state=snapshot,
            )
            failure_refs = []
            cursor = 0
            forward = self._forward
            try:
                for chunk in self.processor.chunk_plan.chunks(visual.shape[2]):
                    indices = (
                        []
                        if audio is None
                        else self.processor._audio_indices_for_video_span(
                            audio.shape[2], chunk.start, chunk.end, visual.shape[2]
                        )
                    )
                    failed.append(
                        inputs.source_visual[:, :, chunk.start : chunk.end],
                        visual[:, :, chunk.start : chunk.end],
                        audio=None if audio is None else audio[:, :, indices],
                        audio_start=cursor,
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
                    if chunk.chunk_id < 21:
                        failed.step()
                        continue
                    failure_refs = [
                        weakref.ref(t)
                        for c in failed.cache.committed
                        for layer in c.layers
                        for t in layer
                    ]
                    caller_rng = (
                        torch.get_rng_state().clone(),
                        torch.cuda.get_rng_state(device).clone(),
                    )

                    def fail_after_commit(packed, cache, update):
                        value = forward(packed, cache, update)
                        if update == "final":
                            assert [c.temporal_chunk_id for c in cache.committed] == [
                                0,
                                1,
                                2,
                                *range(4, 22),
                            ]
                            raise RuntimeError("Injected window acceptance failure")
                        return value

                    self._forward = fail_after_commit
                    try:
                        failed.step()
                    except RuntimeError as exc:
                        assert str(exc) == "Injected window acceptance failure"
                    else:
                        raise AssertionError("Expected injected failure")
                    assert failed.closed and not failed.complete
                    assert all(v == 0 for v in failed.memory_snapshot().values())
                    assert failed.rng.cpu_rng is failed.rng.cuda_rng is None
                    assert torch.equal(caller_rng[0], torch.get_rng_state())
                    assert torch.equal(caller_rng[1], torch.cuda.get_rng_state(device))
                    try:
                        failed.step()
                    except ValueError as exc:
                        assert "closed" in str(exc)
                    else:
                        raise AssertionError("Failed session accepted retry")
                    break
            finally:
                self._forward = forward
                failed.dispose()
            gc.collect()
            torch.cuda.synchronize(device)
            assert failure_refs and all(ref() is None for ref in failure_refs)
            report = dict(
                passed=True,
                chunks=count,
                chunk_lengths=list(self.processor.chunk_plan.lengths),
                window_chunks=21,
                sink_chunks=3,
                rope=self.rope_position_mode,
                rope_max_frame_gap=self.rope_max_frame_gap,
                video_exact=True,
                audio_exact=audio is not None,
                failure_closes_after_eviction=True,
                output_sha256=output_digest.hexdigest(),
                plateau_span_bytes=plateau,
                before_session_gpu_allocated=before_allocated,
                after_eos_gpu_allocated=torch.cuda.memory_allocated(device),
                released=session.memory_snapshot(),
                provenance="synthetic fixture; codec/network bridge not validated",
                full_history_quality="not assessed",
            )
            (root / f"window-acceptance-rank{rank}.json").write_text(
                json.dumps(report, indent=2)
            )
            return expected_v.to(device), None if expected_a is None else expected_a.to(
                device
            )
        finally:
            log.close()
            session.dispose()
            torch.set_rng_state(post[0])
            torch.cuda.set_rng_state(post[1], device)
            self.last_run_metrics = baseline_metrics

    cls.sample = sample
