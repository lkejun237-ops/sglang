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
    """Small PI correction around measured completion cadence, not a hard gate.

    Time-average a half-second window to avoid chasing batch completion jumps.
    Low-water, startup, condition and supply fences freeze/reset the integrator.
    All limits are dimensionless; hard native frame/byte credits stay separate.
    """

    def __init__(self):
        self.updates = 0
        self.max_oldest_age = 0.0
        self.freeze()

    def freeze(self):
        self.scale = 1.0
        self.integral = 0.0
        self.last = None
        self.last_seen = None
        self.elapsed = self.queue_area = self.age_area = 0.0
        self.error = 0.0
        self.target = None

    def suspend(self):
        # Do not integrate a low-water/source wait, but retain the correction
        # across ordinary batch refill dips. Long gaps invalidate it below.
        self.last = None

    def observe(self, now, *, buffered, reserve, capacity, oldest_age, period, enabled):
        if (
            not enabled
            or period is None
            or oldest_age is None
            or not all(math.isfinite(v) for v in (now, oldest_age, period))
            or period <= 0
            or oldest_age < 0
            or capacity <= reserve
        ):
            self.freeze()
            return
        if self.last_seen is not None and now - self.last_seen > 2.0:
            self.freeze()
        self.last_seen = now
        if buffered < reserve:
            self.suspend()
            return
        self.max_oldest_age = max(self.max_oldest_age, oldest_age)
        self.target = min(capacity, reserve + max(2, (reserve - 1) / 2))
        old, self.last = self.last, (now, buffered, oldest_age)
        if old is None:
            return
        dt = now - old[0]
        if not 0 < dt <= 0.25:
            self.freeze()
            return
        self.elapsed += dt
        self.queue_area += old[1] * dt
        self.age_area += min(old[2] + dt / 2, 10 * period * capacity) * dt
        if self.elapsed < 0.5:
            return
        queue = self.queue_area / self.elapsed
        age = self.age_area / self.elapsed
        queue_error = (queue - self.target) / max(1, reserve)
        age_error = max(0.0, age / (self.target * period) - 1.0)
        error = max(-1.0, min(1.0, queue_error + 0.5 * age_error))
        self.error = 0.0 if abs(error) < 0.05 else error
        candidate = max(
            -0.06, min(0.06, self.integral + 0.04 * self.error * self.elapsed)
        )
        raw = 1.0 + 0.12 * self.error + candidate
        # Conditional integration: do not accumulate error into saturation.
        if (
            1.0 <= raw <= 1.15
            or (raw > 1.15 and self.error < 0)
            or (raw < 1.0 and self.error > 0)
        ):
            self.integral = candidate
        desired = max(1.0, min(1.15, 1.0 + 0.12 * self.error + self.integral))
        self.scale += max(-0.02, min(0.02, desired - self.scale))
        self.updates += 1
        self.elapsed = self.queue_area = self.age_area = 0.0

    def snapshot(self):
        return dict(
            kind="bounded_pi",
            period_scale=self.scale,
            integral=self.integral,
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

    def reset(self):
        self.period_us = self.next_pts = self.last_pts = self.last_version = None
        self.last_input = self.control_sequence = self.anchor = None
        self.last_completed = None
        self.fed = False
        self.control_refill_pts = self.admission_target = None
        self.last_deferred_version = None
        self.counts["resets"] += 1
        self.feedback.freeze()

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
        return False

    def supply(self, buffered, reserve, *, available=False):
        if buffered < reserve and not available:
            self.fed = False
            self.feedback.suspend()

    def completed(self, end, now, *, buffered, reserve):
        """Observe a batch with its input still counted in `buffered`.

        A normal completion/refill dip is not starvation. Source waits below the
        input floor invalidate the window separately through `supply`.
        """
        if self.last_completed is not None and end <= self.last_completed:
            return
        self.last_completed = end
        self.supply(buffered, reserve)
        if self.anchor is not None:
            previous, started = self.anchor
            if end <= previous:
                return  # Duplicate receipts do not invent a cadence sample.
            elapsed = now - started
            if not self.fed or not 0 < elapsed <= self.reset_seconds:
                self.anchor = None
            elif elapsed >= self.observation_seconds:
                measured = elapsed * 1_000_000 / (end - previous)
                self.period_us = (
                    measured
                    if self.period_us is None
                    else 0.8 * self.period_us + 0.2 * measured
                )
                self.counts["rate_observations"] += 1
                self.anchor = None
        if self.anchor is None:
            self.anchor = (end, now)
            self.fed = buffered >= reserve
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
