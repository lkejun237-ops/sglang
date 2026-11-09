# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.realtime.world_gateway_metrics import (
    WORLD_GATEWAY_BROWSER_DELIVERY_DETACHES_TOTAL,
    WORLD_GATEWAY_BROWSER_INPUT_DISCONNECTS_TOTAL,
    WORLD_GATEWAY_FIRST_MEDIA_TOTAL,
    observe_world_gateway_browser_delivery_detach,
    observe_world_gateway_browser_input_disconnect,
    observe_world_gateway_first_media,
)


def _clear_metric_label_overrides(monkeypatch):
    for name in (
        "WORLD_MODEL_METRIC_SERVICE",
        "SERVICE_NAME",
        "WORLD_MODEL_METRIC_MODEL",
        "WORLD_MODEL_METRIC_LANE",
        "REALTIME_LANE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_world_gateway_transport_counters_use_only_bounded_labels(monkeypatch):
    _clear_metric_label_overrides(monkeypatch)
    common = {
        "service": "gateway",
        "model": "minwm",
        "lane": "test",
        "profile": "world-test-k1",
        "role": "primary",
    }
    first_media = WORLD_GATEWAY_FIRST_MEDIA_TOTAL.labels(**common)
    input_disconnect = WORLD_GATEWAY_BROWSER_INPUT_DISCONNECTS_TOTAL.labels(
        **common,
        phase="playing",
        kind="connection_lost",
        close_class="abnormal",
    )
    delivery_detach = WORLD_GATEWAY_BROWSER_DELIVERY_DETACHES_TOTAL.labels(
        **common,
        phase="playing",
        detach_class="send_timeout",
    )
    before = (
        first_media._value.get(),
        input_disconnect._value.get(),
        delivery_detach._value.get(),
    )

    assert observe_world_gateway_first_media(**common)
    assert observe_world_gateway_browser_input_disconnect(
        **common,
        phase="playing",
        kind="connection_lost",
        close_class="abnormal",
    )
    assert observe_world_gateway_browser_delivery_detach(
        **common,
        phase="playing",
        detach_class="send_timeout",
    )

    assert first_media._value.get() == before[0] + 1
    assert input_disconnect._value.get() == before[1] + 1
    assert delivery_detach._value.get() == before[2] + 1


def test_world_gateway_transport_counters_reject_unbounded_dimensions(monkeypatch):
    _clear_metric_label_overrides(monkeypatch)
    common = {
        "model": "minwm",
        "profile": "world-test-k1",
        "role": "primary",
    }

    assert not observe_world_gateway_browser_input_disconnect(
        **common,
        phase="run-123",
        kind="connection_lost",
        close_class="abnormal",
    )
    assert not observe_world_gateway_browser_input_disconnect(
        **common,
        phase="playing",
        kind="raw-close-reason",
        close_class="abnormal",
    )
    assert not observe_world_gateway_browser_delivery_detach(
        **common,
        phase="playing",
        detach_class="browser_send_CustomError",
    )
    assert not observe_world_gateway_first_media(
        model="minwm",
        profile="world-test-k1",
        role="run-123",
    )
