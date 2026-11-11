# SPDX-License-Identifier: Apache-2.0
"""Capture-clock admission and selection; no timer, playout queue or new frames."""

from __future__ import annotations

import asyncio
import math
from collections import Counter, OrderedDict
from itertools import islice

from .contracts import capture_pts


class InputChangeTracker:
    """Priority hint for the existing CLI held-state envelope, never a filter.

    Only an exact, valid held-key/button renewal without events is coalesced.
    Unknown envelopes and every event remain changes. All original messages,
    sequences and receipts still reach the SDK and the recording unchanged.
    """

    def __init__(self):
        self.revision = 0
        self.held = None
        self.receipt_revisions = OrderedDict()

    def observe(self, payload, sequence=None):
        keys, buttons = payload.get("held_keys", []), payload.get("held_buttons", [])
        known = (
            bool(payload)
            and not set(payload) - {"held_keys", "held_buttons", "events"}
            and isinstance(keys, list)
            and len(keys) <= 64
            and all(isinstance(k, str) and len(k) <= 32 for k in keys)
            and isinstance(buttons, list)
            and len(buttons) <= 64
            and all(type(b) is int and b in (0, 1, 2) for b in buttons)
            and isinstance(payload.get("events", []), list)
        )
        held = (
            (tuple(sorted(set(keys))), tuple(sorted(set(buttons)))) if known else None
        )
        if not known or payload.get("events") or held != self.held:
            self.revision += 1
        self.held = held
        if sequence is not None:
            self.receipt_revisions[sequence] = self.revision
            while len(self.receipt_revisions) > 1024:
                self.receipt_revisions.popitem(last=False)
        return self.revision

    def acknowledged(self, receipts):
        # A new-state receipt remains a freshness hint if dispatch preceded
        # actual SDK consumption. It is not an effective-input claim.
        return max(
            (
                self.receipt_revisions.get(r.get("clientSequence"), 0)
                for r in receipts
                if isinstance(r, dict)
                and r.get("type") == "input.receipt"
                and type(r.get("clientSequence")) is int
            ),
            default=0,
        )


class QueueFeedback:
    """Small P correction for reported *unprocessed source*, never total in-flight.

    No integral or derivative: a stale/missing native observation cannot slow
    admission, and starvation immediately removes the correction.
    """

    def __init__(self):
        self.updates = 0
        self.max_oldest_age = 0.0
        self.freeze()

    def freeze(self):
        self.scale = 1.0
        self.integral = 0.0  # Kept in archived diagnostics; no integrator is run.
        self.last = None
        self.error = 0.0
        self.target = None

    def suspend(self):
        self.freeze()

    def observe(
        self,
        now,
        *,
        buffered,
        reserve,
        capacity,
        oldest_age,
        period,
        enabled,
        queued_source_frames=None,
    ):
        if oldest_age is not None and math.isfinite(oldest_age):
            self.max_oldest_age = max(self.max_oldest_age, oldest_age)
        if (
            not enabled
            or period is None
            or queued_source_frames is None
            or capacity <= reserve
            or not math.isfinite(now)
        ):
            self.freeze()
            return
        self.target = max(1, reserve)
        if queued_source_frames <= self.target:
            self.scale = 1.0
            self.error = 0.0
            self.last = now
            return
        if self.last is not None and 0 <= now - self.last < 0.5:
            return
        self.last = now
        self.error = min(1.0, (queued_source_frames - self.target) / self.target)
        desired = 1.0 + 0.05 * self.error
        self.scale += max(-0.02, min(0.01, desired - self.scale))
        self.updates += 1

    def snapshot(self):
        return dict(
            kind="bounded_p_native_source",
            period_scale=self.scale,
            integral=0.0,
            error=self.error,
            target_frames=self.target,
            updates=self.updates,
            max_oldest_inflight_ms=self.max_oldest_age * 1000,
        )


class AdaptiveSourceSampler:
    """A model-segment-local selector over the source owner's bounded history.

    Learn model-completion cadence only across intervals that had sufficient model
    input. A slow/empty source must not teach the model a lower capacity. Capture
    time and the worker's indexed protocol clock remain separate identities.
    """

    max_candidates = 8
    max_age_seconds = 0.1
    observation_seconds = 0.5
    reset_seconds = 2.0

    def __init__(self):
        self.period_us = None
        self.next_pts = None
        self.last_version = None
        self.last_pts = None
        self.last_input = None
        self.control_sequence = None
        self.anchor = None
        self.last_completed = None
        self.fed = False
        self.counts = Counter()
        self.max_extra_age_us = 0
        self.control_refill_pts = None
        self.admission_target = None
        self.last_deferred_version = None
        self.admission_counts = Counter()
        self.feedback = QueueFeedback()
        self.native = None
        self.native_required = 0
        self.native_accepted = self.native_queued = self.native_sequence = 0
        self.native_dispatched = 0
        self.native_waiting = False
        self.native_wait_ms = 0.0
        self.native_seen = None
        self.measuring = False
        self.slow_windows = 0
        self.capture_sample = None
        self.capture_period_us = None

    def configure_native(self, ready, *, session_id, generation_id, capacity):
        enabled = ready.get("source_demand_protocol") == "source-demand-v1"
        flow = ready.get("source_flow")
        if enabled and (
            any(not isinstance(v, str) or not v for v in (session_id, generation_id))
            or type(capacity) is not int
            or capacity < 1
            or not isinstance(flow, dict)
            or flow.get("protocol") != "clip17-source-flow-v1"
            or flow.get("input_groups") != [1, 8, 8]
        ):
            raise ValueError("Invalid source demand identity")
        self.native = dict(
            enabled=enabled,
            session_id=session_id,
            generation_id=generation_id,
            capacity=capacity,
            input_quantum=max(flow["input_groups"]) if enabled else None,
        )
        self.native_required = self.native_accepted = self.native_queued = 0
        self.native_dispatched = 0
        self.native_sequence = 0
        self.native_waiting = False
        self.native_wait_ms = 0.0
        self.native_seen = None
        self.reset()

    def model_demand(self, message, *, sent, now):
        if self.native is None or not self.native["enabled"]:
            raise ValueError("Source demand was not negotiated")
        fields = [
            message.get(k)
            for k in (
                "sequence",
                "accepted_source_frame_end",
                "required_source_frame_end",
                "queued_source_frames",
                "dispatched_source_frame_end",
            )
        ]
        seq, accepted, required, queued, dispatched = fields
        waited = message.get("input_wait_ms")
        waiting = message.get("waiting_for_source")
        if (
            message.get("protocol") != "source-demand-v1"
            or any(
                message.get(k) != self.native[k]
                for k in ("session_id", "generation_id")
            )
            or any(type(v) is not int for v in fields)
            or seq <= self.native_sequence
            or not self.native_accepted <= accepted <= sent
            or not self.native_dispatched <= dispatched <= accepted
            or not 0 <= required <= sent + self.native["capacity"]
            or not 0 <= queued <= min(accepted, self.native["capacity"])
            or queued != accepted - dispatched
            or type(waiting) is not bool
            or (waiting and accepted >= required)
            or type(waited) not in (int, float)
            or not math.isfinite(waited)
            or waited < self.native_wait_ms
        ):
            raise ValueError("Invalid native source demand")
        if waiting or waited > self.native_wait_ms:
            self.fed = False
            self.feedback.freeze()
            self.counts["native_source_waits"] += 1
        self.native_sequence, self.native_accepted = seq, accepted
        self.native_dispatched = dispatched
        self.native_required, self.native_queued = required, queued
        self.native_waiting, self.native_wait_ms, self.native_seen = (
            waiting,
            waited,
            now,
        )

    def native_healthy(self, now):
        return (
            self.native is not None
            and self.native["enabled"]
            and self.native_seen is not None
            and not self.native_waiting
            and 0
            <= now - self.native_seen
            <= max(2.0, 2 * self.native["capacity"] * (self.period_us or 0) / 1_000_000)
        )

    def needs_input(self, sent):
        return self.native is not None and self.native_required > sent

    def observe_queue(self, now, *, sent, **kwargs):
        healthy = self.native_healthy(now)
        if healthy:
            # The source queue target is one native encoder group, independent
            # of the AR first-output minimum or total pipeline in-flight count.
            kwargs = {**kwargs, "reserve": self.native["input_quantum"]}
        self.feedback.observe(
            now,
            **kwargs,
            queued_source_frames=(
                self.native_queued + max(0, sent - self.native_accepted)
                if healthy
                else None
            ),
        )
        if self.native is not None and not healthy:
            self.fed = False

    def reset(self):
        self.period_us = self.next_pts = self.last_pts = self.last_version = None
        self.last_input = self.control_sequence = self.anchor = None
        self.last_completed = None
        self.fed = False
        self.control_refill_pts = self.admission_target = None
        self.last_deferred_version = None
        self.counts["resets"] += 1
        self.feedback.freeze()
        self.measuring = self.native is not None and self.native["enabled"]
        self.slow_windows = 0
        self.capture_sample = self.capture_period_us = None

    def control_changed(self, identity, sequence, semantic_control=False):
        if semantic_control and self.last_input is not None:
            # Receipt sequence/tick churn is not a new input. Effective input,
            # display/context changes and non-keyboard action receipts remain
            # independent freshness fences, without interpreting game payloads.
            def actions(value):
                return [
                    r
                    for r in (value[2] or [])
                    if isinstance(r, dict) and r.get("type") != "input.receipt"
                ]

            return (
                sequence != self.control_sequence
                or identity[0] != self.last_input[0]
                or identity[4:] != self.last_input[4:]
                or actions(identity) != actions(self.last_input)
            )
        return (
            self.last_input is None
            or sequence != self.control_sequence
            or identity != self.last_input
        )

    def admit(
        self,
        latest,
        *,
        after,
        buffered,
        reserve,
        startup=False,
        urgent=False,
        control_sequence=None,
        semantic_control=False,
    ):
        """Wait for capture/progress notifications when ordinary input is early.

        A new control may advance one sampling opportunity. Further controls
        coalesce until that opportunity's capture time, so mouse motion or held
        key renewals cannot turn the model input back into an unbounded burst.
        The separate SDK input/control path never waits on this decision.
        """
        self.admission_target = None
        version, (frame, _) = latest
        pts = capture_pts(frame)
        if version <= after:
            return False
        if self.capture_sample is None or version > self.capture_sample[0]:
            if self.capture_sample is not None and pts > self.capture_sample[1]:
                measured = (pts - self.capture_sample[1]) / (
                    version - self.capture_sample[0]
                )
                self.capture_period_us = (
                    measured
                    if self.capture_period_us is None
                    else 0.8 * self.capture_period_us + 0.2 * measured
                )
            self.capture_sample = (version, pts)
        if self.native is not None and not self.native["enabled"]:
            return True  # Old peers keep bounded latest, not unobserved feedback.
        if self.measuring:
            return True  # Bounded latest is an independent capacity probe.
        identity = self.input_identity(frame)
        reset = self.last_version is not None and (
            after != self.last_version
            or pts <= self.last_pts
            or pts - self.last_pts > self.reset_seconds * 1_000_000
            or identity[5:] != self.last_input[5:]
        )
        if reset or startup or urgent or buffered < reserve or self.period_us is None:
            self.control_refill_pts = None
            return True
        target = self.next_pts
        if target is None:
            return True
        if pts >= target:
            self.control_refill_pts = None
            if pts - target <= self.max_age_seconds * 1_000_000:
                self.admission_target = target
            return True
        if (
            self.control_changed(identity, control_sequence, semantic_control)
            and self.feedback.scale <= 1.01
            and (self.control_refill_pts is None or pts >= self.control_refill_pts)
        ):
            self.control_refill_pts = target
            # Preserve the long-term admission budget. Repeated held-state
            # renewals no longer consume priority opportunities, and an aging
            # queue disables borrowing without delaying SDK input dispatch.
            self.admission_target = target
            self.admission_counts["control_advances"] += 1
            return True
        if version != self.last_deferred_version:
            self.admission_counts["early_captures"] += 1
            self.last_deferred_version = version
        # A completion window shaped by our own admission gate is not an
        # independent measurement of a slower model. It may still prove a
        # faster sustainable rate, but must not ratchet capacity downward.
        self.fed = False
        return False

    def supply(self, buffered, reserve, *, available=False):
        if self.native is not None:
            return  # Native readiness, not total GPU in-flight, owns this fence.
        if buffered < reserve and not available:
            self.fed = False
            self.feedback.suspend()

    def completed(self, end, now, *, buffered, reserve, learning_allowed=True):
        """Observe a batch with its input still counted in `buffered`.

        A normal completion/refill dip is not starvation. Source waits below the
        input floor invalidate the window separately through `supply`.
        """
        if self.last_completed is not None and end <= self.last_completed:
            return
        self.last_completed = end
        if self.native is not None and not self.native["enabled"]:
            return
        if not learning_allowed:
            self.anchor = None
            self.fed = False
            self.feedback.freeze()
            return
        self.supply(buffered, reserve)
        if self.anchor is not None:
            previous, started = self.anchor
            if end <= previous:
                return  # Duplicate receipts do not invent a cadence sample.
            elapsed = now - started
            if not 0 < elapsed <= max(self.reset_seconds, 4.0 if self.native else 0):
                self.anchor = None
            elif elapsed >= (
                2.0
                if self.measuring
                else 1.0
                if self.native
                else self.observation_seconds
            ):
                measured = elapsed * 1_000_000 / (end - previous)
                healthy = self.fed and (self.native is None or self.native_healthy(now))
                source_headroom = (
                    self.capture_period_us is not None
                    and self.capture_period_us < 0.9 * measured
                )
                if self.measuring and source_headroom:
                    # Measure the attainable full pipeline with no soft
                    # admission gate. Native credit/batch bubbles are part of
                    # that capacity; a genuinely source-limited window is not.
                    self.period_us = measured
                    self.measuring = False
                    self.slow_windows = 0
                    self.counts["capacity_probes"] += 1
                    self.counts["rate_observations"] += 1
                elif not self.measuring and (
                    healthy
                    or (self.period_us is not None and measured < self.period_us)
                ):
                    self.period_us = (
                        measured
                        if self.period_us is None
                        else 0.8 * self.period_us + 0.2 * measured
                    )
                    self.counts["rate_observations"] += 1
                    self.slow_windows = 0
                else:
                    self.counts["capacity_samples_rejected"] += 1
                    if (
                        self.native is not None
                        and not self.measuring
                        and self.period_us is not None
                        and source_headroom
                        and measured > 1.1 * self.period_us
                    ):
                        self.slow_windows += 1
                        if self.slow_windows >= 2:
                            self.measuring = True
                            self.feedback.freeze()
                            self.counts["capacity_rechecks"] += 1
                self.anchor = None
        if self.anchor is None:
            self.anchor = (end, now)
            self.fed = (
                self.native_healthy(now)
                if self.native is not None
                else buffered >= reserve
            )
        else:
            self.supply(buffered, reserve)

    @staticmethod
    def input_identity(frame):
        # Do not interpret application-owned input schemas, or promote observed
        # receipt evidence into an effective-input claim.
        observed = frame.observed_input or {}
        provenance = frame.provenance or {}
        header = provenance.get("cli_frame_header", {})
        source = header.get("source", {})
        return (
            frame.effective_input,
            observed.get("latest"),
            observed.get("receipts"),
            observed.get("events"),
            observed.get("display"),
            provenance.get("source_session_id"),
            provenance.get("source_epoch"),
            header.get("sessionId"),
            header.get("epoch"),
            source.get("presentationId"),
            source.get("sdkEpoch"),
        )

    def choose(
        self,
        latest,
        history,
        *,
        after,
        now,
        buffered,
        reserve,
        startup=False,
        urgent=False,
        control_sequence=None,
        semantic_control=False,
    ):
        """Return a real existing capture immediately, including under supply loss.

        `latest` has already passed pinned-start/context/stop fences. The history
        is an ephemeral snapshot of at most eight entries from SourceFrames;
        this object never retains RGB or allocates another frame pool.
        """
        version, (frame, captured) = latest
        pts = capture_pts(frame)
        identity = self.input_identity(frame)
        observed = frame.observed_input or {}
        reset = self.last_version is not None and (
            after != self.last_version
            or pts <= self.last_pts
            or pts - self.last_pts > self.reset_seconds * 1_000_000
            or identity[5:] != self.last_input[5:]
        )
        if reset:
            self.reset()
        changed_input = self.control_changed(
            identity, control_sequence, semantic_control
        )
        # Unverified input still requires the freshest pixels. A persistent
        # receipt does not, however, represent a new control on every capture.
        unverified_input = frame.effective_input is None and bool(
            observed.get("latest") or observed.get("receipts") or observed.get("events")
        )
        if startup:
            reason = "startup"
        elif self.measuring:
            reason = "capacity_probe"
        elif urgent or changed_input:
            reason = "control"
        elif buffered < reserve:
            reason = "low_water"
        elif self.period_us is None:
            reason = "unmeasured"
        else:
            reason = "selection"
        chosen = latest
        candidates = [latest]
        if reason in {"low_water", "selection"} and not unverified_input:
            for v, candidate, timestamp, _ in islice(history, self.max_candidates):
                p = capture_pts(candidate)
                if (
                    after < v < version
                    and (self.last_pts is None or p > self.last_pts)
                    and 0 <= now - timestamp <= self.max_age_seconds
                    and 0 <= pts - p <= self.max_age_seconds * 1_000_000
                    and self.input_identity(candidate) == identity
                ):
                    candidates.append((v, (candidate, timestamp)))
        if reason == "low_water":
            # Do not throw away usable, already captured input while the model
            # needs feeding. Repeated calls drain the short fresh tail in order.
            chosen = min(candidates, key=lambda x: x[0])
        target = self.next_pts
        if target is None and self.last_pts is not None and self.period_us is not None:
            target = self.last_pts + self.period_us
        if reason == "selection" and target is not None:
            if pts - target > self.max_age_seconds * 1_000_000:
                reason = "phase_reset"
            else:
                # Prefer fresh on ties. There is never a wait for a future frame.
                chosen = min(
                    candidates,
                    key=lambda x: (abs(capture_pts(x[1][0]) - target), -x[0]),
                )
                reason = (
                    "fresh_unverified"
                    if unverified_input
                    else "uniform"
                    if chosen[0] != version
                    else "latest"
                )
        selected_pts = capture_pts(chosen[1][0])
        self.max_extra_age_us = max(self.max_extra_age_us, pts - selected_pts)
        self.counts[reason] += 1
        self.last_version, self.last_pts = chosen[0], selected_pts
        self.last_input, self.control_sequence = identity, control_sequence
        phase = self.admission_target
        if phase is None or startup or urgent or reason in {"low_water", "phase_reset"}:
            phase = (
                target
                if reason in {"uniform", "latest", "fresh_unverified"}
                else selected_pts
            )
        self.next_pts = (
            phase + self.period_us * self.feedback.scale
            if self.period_us is not None
            else None
        )
        self.admission_target = None
        return chosen

    def snapshot(self):
        return dict(
            mode="adaptive",
            effective_mode=(
                "latest_unverified_supply"
                if self.native is not None and not self.native["enabled"]
                else "adaptive_probe"
                if self.measuring
                else "adaptive"
            ),
            native_source_demand=dict(
                negotiated=self.native is not None and self.native["enabled"],
                accepted_frame_end=self.native_accepted,
                dispatched_frame_end=self.native_dispatched,
                required_frame_end=self.native_required,
                queued_source_frames=self.native_queued,
                waiting_for_source=self.native_waiting,
                input_wait_ms=self.native_wait_ms,
                sequence=self.native_sequence,
            ),
            selection_counts={
                k: v
                for k, v in self.counts.items()
                if k not in {"resets", "rate_observations"}
            },
            resets=self.counts["resets"],
            rate_observations=self.counts["rate_observations"],
            estimated_model_completion_period_us=self.period_us,
            max_extra_capture_age_us=self.max_extra_age_us,
            max_candidates=self.max_candidates,
            max_candidate_age_ms=self.max_age_seconds * 1000,
            admission_counts=dict(self.admission_counts),
            queue_feedback=self.feedback.snapshot(),
        )


async def wait_for_capture_or_progress(source, after, condition, progress, observed):
    """A deferred capture can become necessary when model demand changes."""

    async def changed():
        async with condition:
            await condition.wait_for(lambda: progress() != observed)

    capture = asyncio.create_task(source.wait_for_capture(after))
    demand = asyncio.create_task(changed())
    try:
        await asyncio.wait((capture, demand), return_when=asyncio.FIRST_COMPLETED)
        # This notification cannot consume a pinned/context-bound capture,
        # including when completion and cancellation happen in the same turn.
    finally:
        capture.cancel()
        demand.cancel()
        await asyncio.gather(capture, demand, return_exceptions=True)
