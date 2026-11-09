# SPDX-License-Identifier: Apache-2.0
import hashlib
import io

import pytest
from PIL import Image

from sglang.multimodal_gen.runtime.realtime.source_prepare import (
    MAX_SNAPSHOT_PIXELS,
    V2_PROTOCOL,
    condition_fingerprint,
    snapshot_rgb,
    validate_preview_rgb,
)


def frame(mode="RGB", color="red", *, size=(32, 32)):
    image = Image.new(mode, size, color)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return dict(
        format="png",
        width=size[0],
        height=size[1],
        payload=out.getvalue(),
        encoded_sha256=hashlib.sha256(out.getvalue()).hexdigest(),
        rgb_sha256=hashlib.sha256(image.convert("RGB").tobytes()).hexdigest(),
    )


def test_snapshot_preserves_source_bytes_and_rejects_tampering():
    value = frame()
    rgb = snapshot_rgb(value, 32, 32)
    assert rgb == bytes([255, 0, 0]) * 32 * 32
    with pytest.raises(ValueError, match="integrity"):
        snapshot_rgb({**value, "payload": value["payload"] + b"changed"}, 32, 32)
    with pytest.raises(ValueError):
        snapshot_rgb(value, 64, 32)
    with pytest.raises(ValueError, match="opaque"):
        snapshot_rgb(frame("RGBA", (1, 2, 3, 254)), 32, 32)


def test_fingerprint_includes_order_seed_prompt_and_actual_rgb():
    kwargs = dict(
        rgb=b"abc",
        prompt="test",
        references=[
            dict(id="a", format="png", payload=b"1"),
            dict(id="b", format="png", payload=b"2"),
        ],
        seed=1,
        source_stream=dict(width=32, height=32, fps=24, format="rgb24"),
    )
    original = condition_fingerprint(**kwargs)
    for patch in [
        dict(rgb=b"abd"),
        dict(prompt="next"),
        dict(seed=2),
        dict(references=list(reversed(kwargs["references"]))),
    ]:
        assert condition_fingerprint(**{**kwargs, **patch}) != original
    assert condition_fingerprint(**kwargs) == original


def test_worker_cache_preserves_prepared_state_across_attach_and_real_block_zero():
    from types import SimpleNamespace as NS

    from sglang.multimodal_gen.runtime.realtime.session import (
        BaseRealtimeState,
        RealtimeSessionCache,
    )

    class State(BaseRealtimeState):
        disposed = False

        def dispose(self):
            self.disposed = True

    clock = [0]
    cache = RealtimeSessionCache(
        max_sessions=1, stale_after_s=10, clock=lambda: clock[0]
    )

    def req(op=None, generation="g", preparation="p"):
        return NS(
            realtime_session_id="s",
            realtime_generation_id=generation,
            block_idx=0,
            session=None,
            extra=dict(
                realtime_preparation_id=preparation,
                zing_source_command=(
                    dict(startup=dict(op=op)) if op else dict(appends=[])
                ),
            ),
        )

    first = req("prepare")
    cache.attach(first)
    state = first.session.get_or_create_state(State)
    for next_request in (req("attach"), req(), req("prepare")):
        cache.attach(next_request)
        assert next_request.session is first.session and not state.disposed
    with pytest.raises(ValueError, match="identity changed"):
        cache.attach(req("prepare", generation="other"))
    plain = req()
    plain.extra = {}
    with pytest.raises(ValueError, match="requires its preparation identity"):
        cache.attach(plain)
    assert not state.disposed
    clock[0] = 11
    with pytest.raises(ValueError, match="Missing realtime preparation"):
        cache.attach(req("attach"))
    assert state.disposed
    fresh = req("prepare", generation="next")
    cache.attach(fresh)
    assert fresh.session is not first.session
    assert cache.release("s") and cache._preparations == {}


def test_v2_native_condition_preview_preserves_pixels_without_v1_dimension_relaxation():
    from sglang.multimodal_gen.runtime.realtime.source_prepare import V2_PROTOCOL

    value = frame()
    assert (
        snapshot_rgb(value, 64, 64, protocol=V2_PROTOCOL)
        == bytes([255, 0, 0]) * 32 * 32
    )
    with pytest.raises(ValueError):
        snapshot_rgb(value, 64, 64)
    for dimensions in ((True, 32), (0, 32), (4097, 4096)):
        with pytest.raises(ValueError):
            snapshot_rgb(
                {**value, "width": dimensions[0], "height": dimensions[1]},
                64,
                64,
                protocol=V2_PROTOCOL,
            )
    with pytest.raises(ValueError):
        snapshot_rgb({**value, "width": 16, "height": 64}, 64, 64, protocol=V2_PROTOCOL)


def test_v2_fingerprint_matches_original_owner_and_keeps_v1_domain():
    from sglang.multimodal_gen.runtime.realtime.source_prepare import V2_PROTOCOL

    values = dict(
        rgb=bytes([1, 2, 3]) * 32 * 32,
        prompt="snow",
        references=[dict(id="r", format="png", payload=b"fixture-reference")],
        seed=17,
        source_stream=dict(format="rgb24", width=64, height=64, fps=24),
        profile_sha256="a" * 64,
    )
    assert (
        condition_fingerprint(**values)
        == "df093c7d29b00c2e3428382a9f3ab2a4bde995c3b30331d61061e8e59e2335ec"
    )
    actual = condition_fingerprint(
        **values, protocol=V2_PROTOCOL, preview_width=32, preview_height=32
    )
    assert actual == "4442500517f6f7bf7f2f2e10ae94c13f85784c8f3095475bb5457cbda8dc35b8"
    assert (
        condition_fingerprint(
            **values, protocol=V2_PROTOCOL, preview_width=16, preview_height=64
        )
        != actual
    )
    with pytest.raises(ValueError):
        condition_fingerprint(
            **values, protocol=V2_PROTOCOL, preview_width=16, preview_height=16
        )
    with pytest.raises(ValueError):
        condition_fingerprint(**values, protocol="future-unknown")


def test_worker_request_schema_preserves_explicit_startup_version():
    from pydantic import ValidationError

    from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
        RealtimeVideoGenerationsRequest,
    )
    from sglang.multimodal_gen.runtime.realtime.source_prepare import (
        PROTOCOL,
        V2_PROTOCOL,
    )

    for protocol in (PROTOCOL, V2_PROTOCOL):
        value = RealtimeVideoGenerationsRequest(
            type="init", prompt="snow", source_prepare_protocol=protocol
        )
        assert value.source_prepare_protocol == protocol
    with pytest.raises(ValidationError):
        RealtimeVideoGenerationsRequest(
            type="init", prompt="snow", source_prepare_protocol="unknown-v3"
        )


def test_snapshot_uses_declared_native_dimensions_and_bounds_allocation():
    value = frame(size=(960, 540))
    assert (
        snapshot_rgb(value, 32, 32, protocol=V2_PROTOCOL)
        == bytes([255, 0, 0]) * 960 * 540
    )
    for width, height in [
        (0, 540),
        (True, 540),
        (960, -1),
        (MAX_SNAPSHOT_PIXELS + 1, 1),
    ]:
        with pytest.raises(ValueError, match="dimensions"):
            snapshot_rgb(
                {**value, "width": width, "height": height},
                32,
                32,
                protocol=V2_PROTOCOL,
            )
    with pytest.raises(ValueError, match="pixels"):
        validate_preview_rgb(b"abc", 960, 540)


def test_png_dimensions_are_checked_before_image_decoder(monkeypatch):
    value = frame()
    raw = bytearray(value["payload"])
    raw[16:20] = (MAX_SNAPSHOT_PIXELS + 1).to_bytes(4, "big")
    raw = bytes(raw)
    monkeypatch.setattr(Image, "open", lambda *_: pytest.fail("Image decoder reached"))
    with pytest.raises(ValueError, match="dimensions"):
        snapshot_rgb(
            {
                **value,
                "payload": raw,
                "encoded_sha256": hashlib.sha256(raw).hexdigest(),
            },
            32,
            32,
            protocol=V2_PROTOCOL,
        )
