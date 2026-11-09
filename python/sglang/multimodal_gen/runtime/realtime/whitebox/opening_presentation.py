# SPDX-License-Identifier: Apache-2.0
"""Attachment-local evidence for one model opening presentation."""

from collections import OrderedDict
from time import monotonic
from uuid import uuid4

from .native_pixels import read_frame_identity, safe_int
from .render_contract import RenderError

PROTOCOL = "worldkit-model-context-first-presented-v1"
CONFIRM = "model.context.presentation.confirm"
MAX_RECENT_OFFERS = 128
OFFER_LIFETIME_SECONDS = 3


def applied_context_operation(state):
    segment = state.get("segment")
    if (
        state.get("source_state") != "ready"
        or not state.get("inference", {}).get("enabled", True)
        or not segment
        or segment["state"] not in {"initializing", "streaming"}
    ):
        return None
    version = segment.get("refresh_condition_version", segment["applied_version"])
    request_id = segment.get("model_context_applied_request_id")
    if request_id is None and version == 1:
        request_id = segment["request_id"]
    operation = state["operations"].get(request_id)
    if (
        not operation
        or operation["op"] not in {"render.start", "prompt.replace"}
        or operation.get("prompt_version") != version
        or not operation.get("model_context_lifecycle")
        or operation.get("model_context", {}).get("segmentId")
        != segment.get("context_segment_id")
        or operation.get("model_context_execution", {}).get(
            "state", operation.get("state")
        )
        != "applied"
    ):
        return None
    return operation


def retained_first_presented(state):
    return state.get("operations", {}).get(
        state.get("model_context_first_presented_request_id")
    )


def opening_operation(state):
    if state.get("display_mode") != "model":
        return None
    operation = applied_context_operation(state)
    if (
        not operation
        or operation["prompt_version"] != state["segment"]["accepted_version"]
    ):
        return None
    prior = retained_first_presented(state)
    if (
        prior
        and prior["model_context_lifecycle"] == operation["model_context_lifecycle"]
    ):
        if prior["first_presented"] == "acknowledged" or (
            prior["first_presented"] == "pending"
            and prior["model_context"]["segmentId"]
            == operation["model_context"]["segmentId"]
        ):
            return None
    # After a negative Host ACK only a fresh applied condition can retry.
    return (
        None if operation.get("first_presented") not in (None, "expired") else operation
    )


class OpeningPresentationOffers:
    """An offer comes from an exact model RTP mapping, never browser input."""

    def __init__(self, *, clock=monotonic):
        self.clock = clock
        self.recent = OrderedDict()

    def current(self, session, state):
        channel = getattr(session, "context_channel", None)
        if (
            channel is None
            or not getattr(channel, "first_presented_supported", False)
            or channel.suspended
            or session.binding_ended
            or session.close_requested
            or getattr(session, "closing", False)
        ):
            return None
        operation = opening_operation(state)
        if not operation:
            return None
        segment = state["segment"]
        if (
            segment["segment_epoch"] != session.current_model_stream
            or channel.segment != operation["model_context"]["segmentId"]
            or getattr(channel, "consumer", None)
            != operation["model_context"].get("audience")
        ):
            return None
        return operation

    def offer(self, session, frame):
        state = session.journal.snapshot or {}
        operation = self.current(session, state)
        if operation is None or frame.get("stream") != "model":
            return None
        segment = state["segment"]
        rendering = frame.get("rendering") or {}
        if (
            frame.get("media_stream_id") != segment["segment_epoch"]
            or not safe_int(frame.get("output_frame_start"))
            or not safe_int(frame.get("pts_us"))
            or rendering.get("segment_epoch") != segment["segment_epoch"]
            or rendering.get("segment_id") != segment["segment_id"]
            or rendering.get("prompt_version") != operation["prompt_version"]
        ):
            return None
        try:
            header = read_frame_identity(
                (frame.get("source") or {}).get("cli_frame_header")
            )
        except ValueError:
            return None
        expected = operation["model_context"]["source"]
        source = header["source"]
        if (
            header["sessionId"] != expected["sessionId"]
            or header["epoch"] != expected["epoch"]
            or any(source[k] != expected[k] for k in ("presentationId", "sdkEpoch"))
            or any(
                source[k] < v
                for k, v in operation.get("model_context_effective", {}).items()
                if k != "policy"
            )
        ):
            return None
        offer_id = uuid4().hex
        self.recent[offer_id] = (
            self.clock() + OFFER_LIFETIME_SECONDS,
            operation["request_id"],
            segment["segment_epoch"],
            state["display_revision"],
            header["uiRevision"],
        )
        if len(self.recent) > MAX_RECENT_OFFERS:
            self.recent.popitem(last=False)
        return offer_id

    def resolve(self, session, offer_id, ui_revision, *, state=None):
        state = state if state is not None else session.journal.snapshot or {}
        operation = self.current(session, state)
        offer = self.recent.get(offer_id)
        if (
            operation is None
            or offer is None
            or offer[0] <= self.clock()
            or offer[1:]
            != (
                operation["request_id"],
                state["segment"]["segment_epoch"],
                state["display_revision"],
                ui_revision,
            )
        ):
            raise RenderError("MODEL_PRESENTATION_STALE", retryable=True)
        return operation
