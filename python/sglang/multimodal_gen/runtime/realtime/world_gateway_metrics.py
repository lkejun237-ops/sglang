# SPDX-License-Identifier: Apache-2.0

"""Low-cardinality transport counters for authorized world sessions."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from prometheus_client import Counter

from sglang.multimodal_gen.runtime.realtime.critical_path_metrics import (
    lane_label,
    model_label,
    service_label,
)

_VALID_ROLES = frozenset({"primary", "comparison"})
_VALID_PHASES = frozenset({"before_first_frame", "playing", "finishing"})
_VALID_DISCONNECT_KINDS = frozenset(
    {"close_frame", "connection_lost", "protocol_close"}
)
_VALID_CLOSE_CLASSES = frozenset(
    {
        "normal",
        "going_away",
        "abnormal",
        "protocol",
        "service",
        "application",
        "other",
    }
)
_VALID_DETACH_CLASSES = frozenset(
    {
        "send_timeout",
        "send_error",
        "queue_message_limit",
        "queue_byte_limit",
        "finish_deadline",
        "other",
    }
)
_PROFILE_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

WORLD_GATEWAY_FIRST_MEDIA_TOTAL = Counter(
    "world_model_gateway_first_media_total",
    "Authorized world sessions that delivered their first real media.",
    ("service", "model", "lane", "profile", "role"),
)
WORLD_GATEWAY_BROWSER_INPUT_DISCONNECTS_TOTAL = Counter(
    "world_model_gateway_browser_input_disconnects_total",
    "Authorized world browser input disconnect observations.",
    ("service", "model", "lane", "profile", "role", "phase", "kind", "close_class"),
)
WORLD_GATEWAY_BROWSER_DELIVERY_DETACHES_TOTAL = Counter(
    "world_model_gateway_browser_delivery_detaches_total",
    "Authorized world browser output delivery detach observations.",
    ("service", "model", "lane", "profile", "role", "phase", "detach_class"),
)


def _trusted_profile_label(value: Any) -> str:
    # Profile ids are validated when the deployment configuration is loaded.
    # Keep a defensive bound here so a future internal caller cannot create an
    # unbounded label value by mistake.
    label = str(value or "").strip()
    if _PROFILE_LABEL_RE.fullmatch(label) is None:
        return "unknown"
    return label


def _base_labels(
    *,
    service: str,
    model: str,
    lane: str | None,
    profile: str,
    role: str,
) -> tuple[str, str, str, str, str] | None:
    if role not in _VALID_ROLES:
        return None
    return (
        service_label(service),
        model_label(model),
        lane_label(lane or "default"),
        _trusted_profile_label(profile),
        role,
    )


@lru_cache(maxsize=256)
def _first_media_child(
    service: str,
    model: str,
    lane: str,
    profile: str,
    role: str,
):
    return WORLD_GATEWAY_FIRST_MEDIA_TOTAL.labels(
        service=service,
        model=model,
        lane=lane,
        profile=profile,
        role=role,
    )


@lru_cache(maxsize=2048)
def _input_disconnect_child(
    service: str,
    model: str,
    lane: str,
    profile: str,
    role: str,
    phase: str,
    kind: str,
    close_class: str,
):
    return WORLD_GATEWAY_BROWSER_INPUT_DISCONNECTS_TOTAL.labels(
        service=service,
        model=model,
        lane=lane,
        profile=profile,
        role=role,
        phase=phase,
        kind=kind,
        close_class=close_class,
    )


@lru_cache(maxsize=1024)
def _delivery_detach_child(
    service: str,
    model: str,
    lane: str,
    profile: str,
    role: str,
    phase: str,
    detach_class: str,
):
    return WORLD_GATEWAY_BROWSER_DELIVERY_DETACHES_TOTAL.labels(
        service=service,
        model=model,
        lane=lane,
        profile=profile,
        role=role,
        phase=phase,
        detach_class=detach_class,
    )


def observe_world_gateway_first_media(
    *,
    service: str = "gateway",
    model: str,
    lane: str | None = None,
    profile: str,
    role: str,
) -> bool:
    labels = _base_labels(
        service=service,
        model=model,
        lane=lane,
        profile=profile,
        role=role,
    )
    if labels is None:
        return False
    try:
        _first_media_child(*labels).inc()
    except Exception:  # noqa: BLE001 - metrics must never break session traffic.
        return False
    return True


def observe_world_gateway_browser_input_disconnect(
    *,
    service: str = "gateway",
    model: str,
    lane: str | None = None,
    profile: str,
    role: str,
    phase: str,
    kind: str,
    close_class: str,
) -> bool:
    labels = _base_labels(
        service=service,
        model=model,
        lane=lane,
        profile=profile,
        role=role,
    )
    if (
        labels is None
        or phase not in _VALID_PHASES
        or kind not in _VALID_DISCONNECT_KINDS
        or close_class not in _VALID_CLOSE_CLASSES
    ):
        return False
    try:
        _input_disconnect_child(*labels, phase, kind, close_class).inc()
    except Exception:  # noqa: BLE001 - metrics must never break session traffic.
        return False
    return True


def observe_world_gateway_browser_delivery_detach(
    *,
    service: str = "gateway",
    model: str,
    lane: str | None = None,
    profile: str,
    role: str,
    phase: str,
    detach_class: str,
) -> bool:
    labels = _base_labels(
        service=service,
        model=model,
        lane=lane,
        profile=profile,
        role=role,
    )
    if (
        labels is None
        or phase not in _VALID_PHASES
        or detach_class not in _VALID_DETACH_CLASSES
    ):
        return False
    try:
        _delivery_detach_child(*labels, phase, detach_class).inc()
    except Exception:  # noqa: BLE001 - metrics must never break session traffic.
        return False
    return True
