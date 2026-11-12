# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for Zing media routing; model inference is mocked."""

import contextlib
import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.configs.zing_media import ZingResizeLimits
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    codecs as codecs_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    media_geometry,
    remote_encoder,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder import (
    DECODERS,
)


@pytest.fixture
def codecs():
    instance = object.__new__(codecs_module.ZingH3Codecs)
    instance.runtime = {}
    instance.decoder = DECODERS["official_h3"]
    instance.spatial_ratio = 16
    instance.patch_size = (1, 2, 2)
    instance.temporal_ratio = 4
    instance.vae_frames_per_chunk = 17
    instance.vae_latents_per_chunk = 5
    instance.target_fps = 24.0
    instance.max_short_edge = 32
    instance.canvas_multiple = 32
    instance.max_area_config = dict(
        reference_image=1024, reference_video=1024, source=1024
    )
    instance.max_source_target_aspect_ratio_factor = 2.0
    instance.audio_sample_rate = 32000
    return instance


def make_request(kind, *, references=0, source=True):
    count = 1 if kind == "image" else 125
    media = [
        {"type": "image", "id": f"ref-{i}", "uri": f"reference-{i}.png"}
        for i in range(references)
    ]
    target = {
        "role": "target",
        "content": [
            {"type": kind, "output": {"frames": count, "height": 32, "width": 32}}
        ],
        "conditions": [
            {
                "content_id": item["id"],
                "role": "reference",
                "latent_modalities": ["visual"],
            }
            for item in media
        ],
    }
    if source:
        target["source"] = {
            "type": "image" if kind == "image" else "video",
            "uri": "source",
        }
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Edit the scene"}, *media],
            },
            target,
        ]
    }


@pytest.fixture
def cached_codecs(codecs, monkeypatch):
    from sglang.multimodal_gen.runtime.conditioning.cache import ConditioningCache

    codecs.reference_cache = ConditioningCache(1 << 20)
    codecs.remote_read_fs = None
    codecs.max_short_edge = 64
    codecs.runtime = {
        "vae_config": {"posterior_seed": 42, "encode_mode": "deterministic"}
    }
    codecs.va = SimpleNamespace(latents_mean=[0.0], latents_std=[1.0])
    codecs.modules = {"video_vae": torch.nn.Linear(1, 1)}
    codecs.reference_encodes = []

    def encode(frames, *, metric_operation="source"):
        codecs.reference_encodes.append(frames.clone())
        return frames.float().mean().reshape(1, 1, 1, 1, 1)

    monkeypatch.setattr(codecs, "encode_visual", encode)
    monkeypatch.setattr(codecs, "encode_condition", lambda *args: {})
    return codecs


def reference_bytes(color="red", media_id="ref-0"):
    from io import BytesIO

    from PIL import Image

    output = BytesIO()
    Image.new("RGB", (96, 64), color).save(output, format="PNG")
    return dict(type="image", id=media_id, format="png", payload=output.getvalue())


def test_reference_cache_reuses_across_sessions_prompts_and_updates(
    cached_codecs, monkeypatch
):
    from contextlib import nullcontext

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        media_input,
    )

    codec = cached_codecs
    reads = []
    read_image = media_input._read_image
    monkeypatch.setattr(
        media_input,
        "_read_image",
        lambda *a, **kw: reads.append(1) or read_image(*a, **kw),
    )
    request = make_request("video", references=1, source=False)
    request["messages"][0]["content"][1] = reference_bytes()

    def use(_):
        return nullcontext()

    first = codec.prepare(request, use, retain_condition_media=True)
    expected = first.visual["ref-0"].clone()
    first.visual["ref-0"].fill_(999)
    first.condition_media[0]["frames"].zero_()
    request["messages"][0]["content"][0]["text"] = "Changed prompt"
    second = codec.prepare(request, use, retain_condition_media=True)
    torch.testing.assert_close(second.visual["ref-0"], expected, rtol=0, atol=0)
    updated = codec.prepare_reference_update(
        second,
        [reference_bytes(media_id="renamed")],
        "another prompt",
        use,
        conditioning=lambda *a: {},
    )
    assert list(updated.visual) == ["renamed"]
    assert reads == [1]
    assert len(codec.reference_encodes) == 1
    assert codec.reference_cache.snapshot(tenant="preprocess")["hit"] == 2
    assert codec.reference_cache.snapshot(tenant="vae")["hit"] == 2
    # Source/video encoding never enters the image-reference cache.
    codec.encode_reference(
        dict(type="video", frames=second.condition_media[0]["frames"])
    )
    assert len(codec.reference_encodes) == 2


def test_reference_preprocess_key_uses_file_contents_and_geometry(
    cached_codecs, tmp_path
):
    codec = cached_codecs
    path = tmp_path / "reference.png"
    path.write_bytes(reference_bytes()["payload"])
    item = dict(type="image", id="ref-0", uri=str(path))

    def decode():
        return codec._decode_reference_image(
            item, max_area_config=codec.max_area_config
        )

    red = decode()["frames"]
    assert torch.equal(decode()["frames"], red)
    path.write_bytes(reference_bytes("blue")["payload"])
    assert not torch.equal(decode()["frames"], red)
    codec.max_area_config["reference_image"] = 4096
    assert decode()["frames"].shape[-2:] != red.shape[-2:]
    assert codec.reference_cache.snapshot(tenant="preprocess")["miss"] == 3
    invalid = reference_bytes()
    invalid["format"] = "jpeg"
    with pytest.raises(ValueError, match="format"):
        codec._decode_reference_image(invalid, max_area_config=codec.max_area_config)


@pytest.mark.parametrize("change", ["pixel", "seed", "dtype", "mode", "mean", "model"])
def test_reference_latent_key_invalidates_effective_encoder_inputs(
    cached_codecs, change
):
    codec = cached_codecs
    item = dict(type="image", frames=torch.ones(3, 1, 32, 32, dtype=torch.uint8))
    codec.encode_reference(item)
    if change == "pixel":
        item["frames"][0, 0, 0, 0] = 2
    elif change == "mean":
        codec.va.latents_mean = [1.0]
    elif change == "model":
        codec.modules["video_vae"] = torch.nn.Linear(1, 1)
    else:
        key, value = {
            "seed": ("posterior_seed", 43),
            "dtype": ("encode_dtype", "float32"),
            "mode": ("encode_mode", "performance"),
        }[change]
        codec.runtime["vae_config"][key] = value
    codec.encode_reference(item)
    assert len(codec.reference_encodes) == 2


def test_reference_partial_sp_hit_reencodes_on_every_rank(cached_codecs, monkeypatch):
    from sglang.multimodal_gen.runtime.conditioning import worker

    codec = cached_codecs
    item = dict(type="image", frames=torch.ones(3, 1, 32, 32, dtype=torch.uint8))
    codec.encode_reference(item)
    monkeypatch.setattr(
        worker, "coordinate_validation", lambda receipt: [receipt, (receipt[0], False)]
    )
    codec.encode_reference(item)
    assert len(codec.reference_encodes) == 2
    monkeypatch.setattr(
        worker,
        "coordinate_validation",
        lambda receipt: [receipt, ("different-input", True)],
    )
    with pytest.raises(ValueError, match="SP ranks"):
        codec.encode_reference(item)
    assert len(codec.reference_encodes) == 2


@pytest.mark.parametrize("capacity", [0, 1])
def test_reference_disabled_or_oversize_preserves_uncached_results(
    cached_codecs, capacity
):
    from sglang.multimodal_gen.runtime.conditioning.cache import ConditioningCache

    codec = cached_codecs
    codec.reference_cache = ConditioningCache(capacity)
    item = reference_bytes()
    decoded = codec._decode_media(item, need_audio=False)
    target = make_request("video", references=1, source=False)["messages"][1]
    expected, *_ = codec._normalize_inference_media(
        [decoded],
        None,
        target["conditions"],
        target["content"][0],
        max_area_config=codec.max_area_config,
    )
    for _ in range(2):
        actual = codec._decode_reference_image(
            item, max_area_config=codec.max_area_config
        )
        actual, *_ = codec._normalize_inference_media(
            [actual],
            None,
            target["conditions"],
            target["content"][0],
            max_area_config=codec.max_area_config,
        )
        assert torch.equal(actual[0]["frames"], expected[0]["frames"])
        codec.encode_reference(actual[0])
    assert len(codec.reference_encodes) == 2
    assert codec.reference_cache.current_bytes == 0


def test_reference_resize_cache_matches_uncached_pixels_across_limit_changes(
    cached_codecs,
):
    from contextlib import nullcontext
    from io import BytesIO

    import numpy as np
    from PIL import Image

    from sglang.multimodal_gen.runtime.conditioning.cache import ConditioningCache

    codec = cached_codecs
    codec.max_short_edge = 128
    codec.max_area_config["reference_image"] = 128 * 96
    pixels = np.random.default_rng(42).integers(0, 256, (97, 131, 3), dtype=np.uint8)
    output = BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    request = make_request("video", references=1, source=False)
    request["messages"][0]["content"][1] = dict(
        type="image", id="ref-0", format="png", payload=output.getvalue()
    )
    results = []
    for capacity in (0, 1 << 20):
        codec.reference_cache = ConditioningCache(capacity)
        frames = []
        for limits in (None, ZingResizeLimits(32, 2048), None):
            codec.reference_resize = limits
            prepared = codec.prepare(
                request, lambda _: nullcontext(), retain_condition_media=True
            )
            frames.append(prepared.condition_media[0]["frames"])
        results.append(frames)
    for uncached, cached in zip(*results):
        assert torch.equal(uncached, cached)
    assert results[1][0].shape != results[1][1].shape
    assert codec.reference_cache.snapshot(tenant="preprocess")["hit"] == 1


@pytest.mark.parametrize("format_name", ["PNG", "JPEG", "WEBP"])
def test_init_and_reference_update_decode_file_bytes_in_memory(
    codecs, monkeypatch, format_name
):
    from contextlib import nullcontext
    from io import BytesIO

    from PIL import Image

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        media_input,
    )

    def item(color):
        output = BytesIO()
        Image.new("RGB", (32, 32), color).save(output, format=format_name)
        return dict(
            type="image",
            id="ref-0",
            format=format_name.lower(),
            payload=output.getvalue(),
        )

    monkeypatch.setattr(
        media_input,
        "_local_media",
        lambda *a, **k: pytest.fail("Inline images must not access a URI"),
    )
    monkeypatch.setattr(
        codecs,
        "encode_visual",
        lambda frames, *, metric_operation="source": (
            frames.float().mean().reshape(1, 1, 1, 1, 1)
        ),
    )

    def condition(prompt, media):
        return dict(context=torch.ones(1, 4), row_role=torch.zeros(1, dtype=torch.long))

    monkeypatch.setattr(codecs, "encode_condition", condition)
    request = make_request("video", references=1, source=False)
    request["messages"][0]["content"][1] = item("red")
    active = codecs.prepare(
        request, lambda _: nullcontext(), retain_condition_media=True
    )
    candidate = codecs.prepare_reference_update(
        active, [item("blue")], "new", lambda _: nullcontext(), conditioning=condition
    )
    before, after = (
        active.condition_media[0]["frames"],
        candidate.condition_media[0]["frames"],
    )
    assert before.dtype == after.dtype == torch.uint8
    assert before.shape == after.shape == (3, 1, 32, 32)
    assert before[0].float().mean() > before[2].float().mean()
    assert after[2].float().mean() > after[0].float().mean()
    assert (
        "payload" not in candidate.condition_media[0]
        and "uri" not in candidate.condition_media[0]
    )
    assert candidate.conditions == active.conditions


@pytest.mark.parametrize("reference_ids", [[], ["ref-0"], ["new", "ref-0"]])
def test_reference_snapshot_recomputes_all_and_preserves_nonreferences(
    codecs, monkeypatch, reference_ids
):
    from contextlib import nullcontext

    encoded, presentations = [], []

    def decode(item, **kwargs):
        return {
            **item,
            "frames": torch.full((3, 1, 32, 32), 7, dtype=torch.uint8),
            "fps": 24.0,
        }

    def encode(frames, *, metric_operation="source"):
        encoded.append(frames.clone())
        return torch.full((1, 2, 1, 2, 2), float(frames.float().mean()))

    def condition(prompt, media):
        presentations.append((prompt, [item["id"] for item in media]))
        return dict(context=torch.ones(1, 4), row_role=torch.zeros(1, dtype=torch.long))

    monkeypatch.setattr(codecs, "_decode_media", decode)
    monkeypatch.setattr(codecs, "encode_visual", encode)
    monkeypatch.setattr(codecs, "encode_condition", condition)
    active = codecs.prepare(
        make_request("video", references=1, source=False),
        lambda _: nullcontext(),
        retain_condition_media=True,
    )
    s0 = dict(
        id="<Video 1>",
        type="video",
        frames=torch.zeros(3, 1, 32, 32, dtype=torch.uint8),
        fps=24.0,
    )
    display = dict(
        id="display", type="image", frames=torch.zeros(3, 1, 32, 32, dtype=torch.uint8)
    )
    active.condition_media = [s0, *active.condition_media, display]
    active.visual["ref-0"].zero_()
    encoded.clear()
    references = [
        dict(id=name, type="image", uri=f"{name}.png") for name in reference_ids
    ]
    candidate = codecs.prepare_reference_update(
        active,
        references,
        "same prompt",
        lambda _: nullcontext(),
        conditioning=condition,
    )
    assert presentations[-1] == (
        "same prompt",
        ["<Video 1>", "display", *reference_ids],
    )
    assert len(encoded) == len(reference_ids)
    assert list(candidate.visual) == reference_ids
    assert [item["content_id"] for item in candidate.conditions] == reference_ids
    assert (
        candidate.condition_media[0] is s0 and candidate.condition_media[1] is display
    )
    assert all(tensor.sum() > 0 for tensor in candidate.visual.values())
    assert list(active.visual) == ["ref-0"] and active.visual["ref-0"].sum() == 0
    assert candidate.visual_shape == active.visual_shape


@pytest.mark.parametrize("remote_failure", [False, True])
def test_reference_decode_failure_consensus_precedes_encoders(
    codecs, monkeypatch, remote_failure
):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.conditioning import worker
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

    def decode(*args, **kwargs):
        if not remote_failure:
            raise OSError("missing upload")
        return dict(
            id="new", type="image", frames=torch.zeros(3, 1, 32, 32, dtype=torch.uint8)
        )

    monkeypatch.setattr(codecs, "_decode_media", decode)
    receipts = []
    monkeypatch.setattr(
        worker, "coordinate_validation", lambda ok: receipts.append(ok) or [ok, False]
    )
    active = SimpleNamespace(
        conditions=[],
        condition_media=[],
        output_type="video",
        pixel_frame_count=17,
        visual_shape=(5, 2, 2),
    )
    with pytest.raises(ConditionError):
        codecs.prepare_reference_update(
            active,
            [dict(id="new", type="image", uri="missing")],
            "prompt",
            lambda _: pytest.fail("VAE must not run"),
            conditioning=lambda *args: pytest.fail("Qwen must not run"),
        )
    assert receipts == [remote_failure]


@pytest.mark.parametrize("mode", ["keep", "clear", "api_rejected", "encode_failed"])
def test_worker_newest_update_uses_active_media_and_discards_pending(mode):
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )

    state = ZingSourceStreamState()
    state.inputs = SimpleNamespace(condition_media=[dict(id="active")])
    seen = []
    pending = ["older candidate"]

    def encode(batch, prompt, media):
        assert not pending
        seen.append(media)
        if mode == "encode_failed":
            raise ConditionError("CONDITION_INVALID", "injected failure")
        return {"context": torch.ones(1, 4)}

    def prepare(active, references, prompt, use, *, conditioning):
        assert active is state.inputs and references == []
        return SimpleNamespace(positive=conditioning(prompt, []), condition_media=[])

    def update(condition, event_id, version, **kwargs):
        pending.append((event_id, kwargs.get("inputs")))

    state.sampler = SimpleNamespace(
        discard_pending_prompt=pending.clear, update_prompt=update
    )
    stage = SimpleNamespace(
        encode_condition_for_batch=encode,
        kv_pool=None,
        prompt_switch_prefix_latent_frames=5,
        _use=None,
        codecs=SimpleNamespace(prepare_reference_update=prepare),
    )
    command = dict(prompt="same prompt", event_id=2)
    if mode == "clear":
        command["references"] = []
    if mode == "api_rejected":
        command["rejected"] = True
    batch = SimpleNamespace(
        extra={"zing_prompt_update": command},
        sampling_params=SimpleNamespace(prompt="initial"),
        realtime_event_id=None,
    )
    notice = state._update_prompt(stage, batch)
    if mode in ("api_rejected", "encode_failed"):
        assert not pending and state.failed_update == ("same prompt", 2)
        assert state.prompt_event is None
        assert notice == (
            {"code": "CONDITION_INVALID", "event_id": 2}
            if mode == "encode_failed"
            else None
        )
    else:
        assert pending[0][0] == 2
        assert seen == ([[]] if mode == "clear" else [state.inputs.condition_media])
        assert (pending[0][1] is not None) == (mode == "clear")
    assert state.inputs.condition_media == [dict(id="active")]


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((736, 1280), (480, 832)),
        ((480, 832), (480, 832)),
        ((384, 640), (384, 640)),
        ((1024, 832), (608, 480)),
    ],
)
def test_source_resize_preserves_frames_and_pixels_at_noop(shape, expected):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.media_geometry import (
        resolve_source_geometry,
    )

    config = SimpleNamespace(
        vae_config=SimpleNamespace(
            arch_config=SimpleNamespace(spatial_compression_ratio=16)
        ),
        dit_config=SimpleNamespace(arch_config=SimpleNamespace(patch_size=(1, 2, 2))),
    )
    geometry = resolve_source_geometry(
        shape,
        shape,
        config,
        dict(
            max_short_edge=1024,
            max_area_config={"source": 1280 * 1024},
            max_source_target_aspect_ratio_factor=2.0,
        ),
    )
    pixels = (
        torch.arange(3 * 2 * shape[0] * shape[1], dtype=torch.int64)
        .to(torch.uint8)
        .reshape(3, 2, *shape)
    )
    resized = geometry.resize(pixels)
    assert resized.shape == (3, 2, *expected)
    assert resized.dtype == torch.uint8
    # Network packet boundaries must not change per-frame filtering.
    assert torch.equal(
        resized,
        torch.cat([geometry.resize(pixels[:, :1]), geometry.resize(pixels[:, 1:])], 1),
    )
    if shape == expected:
        assert resized is pixels
    with pytest.raises(ValueError, match="input geometry"):
        geometry.resize(pixels[:, :, :-1])


@pytest.mark.parametrize(
    "value",
    [
        {},
        True,
        {"max_area": 399360},
        {"max_short_edge": True, "max_area": 399360},
        {"max_short_edge": 480, "max_area": 0},
        {"max_short_edge": 16, "max_area": 1024},
    ],
)
def test_resize_config_rejects_invalid_limits(value):
    with pytest.raises(ValueError):
        ZingResizeLimits.parse(value, multiple=32)


@pytest.mark.parametrize("external", [False, True], ids=["local", "remote-rpc"])
@pytest.mark.parametrize("resize", [False, True], ids=["original", "480p"])
def test_reference_480p_shared_by_qwen_vae_and_prompt_refresh(
    codecs, monkeypatch, tmp_path, external, resize
):
    """Decode real PNGs; remote transport hashes the same RGB that VAE sees."""
    codecs.reference_resize = ZingResizeLimits(480, 832 * 480) if resize else None
    from contextlib import nullcontext

    import numpy as np
    from PIL import Image

    from sglang.multimodal_gen.runtime.conditioning.inputs import (
        PREPROCESS_VERSION,
        content_identity,
    )
    from sglang.multimodal_gen.runtime.conditioning.resources import ResourceLimits
    from sglang.multimodal_gen.runtime.conditioning.wire import (
        decode_request,
        encode_request,
    )
    from sglang.multimodal_gen.runtime.conditioning.worker import (
        RemoteConditioner,
        WorkerConditioningState,
    )

    codecs.max_short_edge = 1024
    codecs.max_area_config = dict(reference_image=1280 * 720, source=1280 * 720)
    codecs.conditioning_mode = "remote" if external else "local"
    codecs.remote_read_fs = None
    request = make_request("image", references=2)
    # The source and its Qwen presentation retain the requested spatial bucket.
    source_path = tmp_path / "source.png"
    Image.new("RGB", (1280, 704), (11, 22, 33)).save(source_path)
    request["messages"][1]["source"]["uri"] = str(source_path)
    request["messages"][1]["content"][0]["output"].update(height=704, width=1280)
    originals = []
    for i, item in enumerate(request["messages"][0]["content"][1:]):
        width, height = (1280, 720) if i == 0 else (1024, 832)
        rgb = np.random.default_rng(i).integers(0, 256, (height, width, 3), np.uint8)
        path = tmp_path / f"reference-{i}.png"
        Image.fromarray(rgb).save(path)
        item["uri"] = str(path)
        originals.append(
            {**item, "frames": torch.from_numpy(rgb).permute(2, 0, 1)[:, None]}
        )
    request["messages"][0]["content"].append(
        dict(type="image", id="qwen-only", uri=str(source_path))
    )
    preview = dict(
        id="preview",
        type="video",
        fps=24.0,
        frames=torch.zeros(3, 1, 704, 1280, dtype=torch.uint8),
    )
    encoded, vae, calls = [], [], []
    profile = SimpleNamespace(
        profile_id="a" * 64, document={"preprocess_version": PREPROCESS_VERSION}
    )

    def encode(prompt, media):
        encoded.append(media)
        calls.append("condition")
        return {
            "context": torch.zeros(1, 4, dtype=torch.bfloat16),
            "row_role": torch.zeros(1, dtype=torch.long),
        }

    def rpc(prompt, media, *, input_digest):
        body = encode_request(
            prompt, media, profile.profile_id, input_digest=input_digest
        )
        request = decode_request(body, profile, ResourceLimits())
        header, decoded = request.header, request.media
        assert header["input_digest"] == content_identity(
            prompt, media, profile.profile_id
        )
        assert all(
            torch.equal(a["frames"], b["frames"]) for a, b in zip(media, decoded)
        )
        return encode(prompt, media)

    def visual(pixels, *, metric_operation="source"):
        calls.append("vae")
        vae.append(pixels)
        return torch.zeros(1, 2, 1, pixels.shape[-2] // 16, pixels.shape[-1] // 16)

    monkeypatch.setattr(codecs, "encode_condition", encode)
    monkeypatch.setattr(codecs, "encode_visual", visual)
    conditioner = RemoteConditioner(
        SimpleNamespace(encode=rpc), profile.profile_id, rank=0
    )
    state = WorkerConditioningState()

    def consume(prompt, media):
        return conditioner(prompt, media, state=state, device="cpu")

    result = codecs.prepare(
        request,
        lambda name: nullcontext(),
        retain_condition_media=True,
        source_preview=preview,
        conditioning=consume if external else None,
    )
    media = result.condition_media
    assert calls == ["condition"] + ["vae"] * 3
    assert media[0]["origin"] == "source_s0"
    assert media[0]["frames"] is preview["frames"]
    assert media[-1]["frames"].shape[-2:] == (704, 1280)
    assert vae[-1].shape[-2:] == (704, 1280)
    assert result.visual_shape[-2:] == (44, 80)  # Output/source stay above 480p.
    refs = media[1:3]
    assert [item["id"] for item in refs] == ["ref-0", "ref-1"]
    expected_shapes = [(480, 832), (480, 608)] if resize else [(704, 1280), (832, 1024)]
    assert [tuple(item["frames"].shape[-2:]) for item in refs] == expected_shapes
    for item, pixels, original, qwen in zip(refs, vae, originals, encoded[0][1:3]):
        # #391 shares normalized reference RGB with Qwen even without the
        # optional resize cap; a 720-high input aligns down to 704.
        assert item["frames"] is pixels
        assert item["frames"] is qwen["frames"]
        assert original["frames"].shape[-2] > 480
    if resize:
        assert content_identity("prompt", refs, profile.profile_id) != content_identity(
            "prompt", originals, profile.profile_id
        )
    # The prompt-refresh callers reuse condition_media, including for remote RPC.
    (consume if external else encode)("new prompt", media)
    assert encoded[-1] is media
    if external:
        consume("new prompt", media)
        assert len(encoded) == 2  # Same normalized input hits the existing cache.


@pytest.mark.parametrize(
    "height,width,area,expected",
    [
        (832, 1024, 1280 * 720, (480, 608)),
        (1024, 832, 1280 * 720, (608, 480)),
        (256, 320, 1280 * 720, (256, 320)),
        (832, 1024, 256 * 320, (256, 320)),
    ],
)
def test_reference_480p_geometry_preserves_stricter_limits(
    codecs, height, width, area, expected
):
    codecs.reference_resize = ZingResizeLimits(480, 832 * 480)
    codecs.max_short_edge = 1024
    pixels = torch.zeros(3, 1, height, width, dtype=torch.uint8)
    media = [dict(id="ref-0", type="image", frames=pixels)]
    request = make_request("image", references=1, source=False)
    target_message = request["messages"][1]
    refs, *_ = codecs._normalize_inference_media(
        media,
        None,
        target_message["conditions"],
        target_message["content"][0],
        max_area_config={"reference_image": area},
    )
    assert refs[0]["frames"].shape[-2:] == expected
    assert media[0]["frames"] is pixels
    assert pixels.shape[-2:] == (height, width)


@pytest.mark.parametrize(
    "kind,references,source",
    [
        ("image", 0, True),
        ("image", 3, True),
        ("video", 0, True),
        ("video", 3, True),
        ("video_audio", 0, True),
    ],
    ids=[
        "image-edit",
        "multi-reference-image",
        "video-edit",
        "reference-source-video",
        "video-audio-edit",
    ],
)
@pytest.mark.parametrize("retain_media", [False, True])
def test_prepare_routes_reference_and_source_media(
    codecs, monkeypatch, kind, references, source, retain_media
):
    loaded, decoded, encoded_media, audio_samples = [], [], [], []

    @contextmanager
    def use_component(name):
        loaded.append(name)
        yield

    def decode(item, *, need_audio):
        decoded.append((item["uri"], need_audio))
        frames = 1 if item["type"] == "image" else 125
        return {
            **item,
            "frames": torch.zeros(3, frames, 32, 32, dtype=torch.uint8),
            "fps": 24.0,
        }

    def condition(prompt, media):
        encoded_media.extend(item["id"] for item in media)
        return {
            "context": torch.zeros(1, 4),
            "row_role": torch.zeros(1, dtype=torch.long),
        }

    def audio(waveform):
        audio_samples.append(tuple(waveform.shape))
        return torch.zeros(1, 2, 10, 4)

    monkeypatch.setattr(codecs, "_decode_media", decode)
    monkeypatch.setattr(codecs, "encode_condition", condition)
    monkeypatch.setattr(
        codecs,
        "encode_visual",
        lambda frames, *, metric_operation="source": torch.zeros(1, 2, 1, 2, 2),
    )
    monkeypatch.setattr(codecs, "encode_audio", audio)
    result = codecs.prepare(
        make_request(kind, references=references, source=source),
        use_component,
        retain_condition_media=retain_media,
    )
    assert (result.condition_media is not None) is retain_media
    if retain_media:
        assert [item["id"] for item in result.condition_media] == encoded_media
    assert result.output_type == kind
    assert result.pixel_frame_count == (1 if kind == "image" else 125)
    assert result.visual_shape == ((1 if kind == "image" else 38), 2, 2)
    assert (result.source_visual is not None) is source
    assert list(result.visual) == [f"ref-{i}" for i in range(references)]
    assert [item["id"] for item in result.media] == encoded_media == list(result.visual)
    assert len(result.conditions) == references
    assert result.audio == {}
    assert all(not need_audio for _, need_audio in decoded)
    assert loaded == ["text_encoder", "video_vae"] + (
        ["audio_vae"] if kind == "video_audio" else []
    )
    if kind == "video_audio":
        # 125 frames / 24 fps, truncated to whole samples at 32 kHz.
        assert result.sample_count == 166666
        assert result.sample_rate == 32000
        assert result.audio_shape == (2, 10, 4)
        assert audio_samples == [(2, 166666)]
    else:
        assert result.sample_count is None
        assert result.audio_shape is None
        assert audio_samples == []


def test_source_video_length_must_match_long_target(codecs, monkeypatch):
    monkeypatch.setattr(
        codecs,
        "_decode_media",
        lambda item, **kwargs: {
            **item,
            "frames": torch.zeros(3, 124, 32, 32, dtype=torch.uint8),
            "fps": 24.0,
        },
    )
    with pytest.raises(ValueError, match="source frames must equal"):
        codecs.prepare(
            make_request("video"), lambda name: pytest.fail("No model should load")
        )


@pytest.mark.parametrize(
    "source_fps,expected",
    [(1.0, [0.5, 2.5, 4.0]), (3.0, [1 / 3, 7 / 6])],
    ids=["deduplicated-low-fps", "rounded-noninteger-stride"],
)
def test_condition_video_timestamps_follow_source_frames(
    codecs, monkeypatch, source_fps, expected
):
    seen = {}

    class Processor:
        temporal_patch_size = 2

        def __call__(self, *, videos, do_sample_frames, return_tensors):
            assert not do_sample_frames
            assert return_tensors == "pt"
            seen["frames"] = len(videos[0])
            return {
                "pixel_values_videos": torch.zeros(1),
                "video_grid_thw": torch.tensor([[len(videos[0]) // 2, 2, 2]]),
            }

    def presentation(tokenizer, **kwargs):
        seen.update(kwargs)
        return torch.tensor([1]), torch.tensor([0])

    codecs.modules = {
        "tokenizer": object(),
        "processor": SimpleNamespace(
            video_processor=Processor(), image_processor=SimpleNamespace(merge_size=2)
        ),
        "text_encoder": SimpleNamespace(
            encode_ids=lambda ids, **kwargs: torch.zeros(1, 4)
        ),
    }
    codecs.runtime = {"condition_encoder_config": {"video_sample_fps": 2.0}}
    monkeypatch.setattr(
        codecs_module, "minimax_h3_ref2va_video_presentation", presentation
    )
    codecs.encode_condition(
        "A video",
        [
            {
                "type": "video",
                "fps": source_fps,
                "frames": torch.zeros(3, 5, 2, 2, dtype=torch.uint8),
            }
        ],
    )
    assert seen["video_block_timestamps"][0] == pytest.approx(expected)
    assert seen["frames"] == len(expected) * 2


@pytest.fixture
def temporal_codecs(codecs):
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
        AutoencoderKL,
    )

    # Exercise the actual H3 temporal assembly/blending, replacing only the
    # expensive learned per-window decoder with a context-sensitive CPU kernel.
    vae = AutoencoderKL.__new__(AutoencoderKL)
    torch.nn.Module.__init__(vae)
    vae.register_parameter("anchor", torch.nn.Parameter(torch.zeros(())))
    vae.vae_ratio_t = 4
    vae.clip_length = 17
    vae.token_drop = 3
    vae.tokens_chunk_size = 5
    vae.token_overlap = 2
    vae.frame_pre_padding = 3
    vae.frame_overlap = 5
    vae.isolated_first_frame = vae.isolated_last_frame = False
    vae.use_3d_conv = True
    vae._blend_weight_cache = {}
    vae.eval()
    calls = []

    def decode(z):
        calls.append(z.detach().clone())
        return z.repeat_interleave(4, dim=2) + z.mean(dim=2, keepdim=True)

    vae._adaptive_decode = decode
    codecs.modules = {"video_vae": vae}
    codecs.va = SimpleNamespace(
        vae_token_drop=3, latents_mean=[0, 0, 0], latents_std=[1, 1, 1]
    )
    return codecs, calls


@pytest.mark.parametrize("stream_cat", ["0", "1"])
@pytest.mark.parametrize("frames", [1, 2, 17, 18, 22, 34, 39, 243])
def test_decode_range_matches_full_h3_temporal_assembly(
    temporal_codecs, monkeypatch, stream_cat, frames
):
    import math

    codecs, calls = temporal_codecs
    monkeypatch.setenv("MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT", stream_cat)
    full, tail = divmod(frames - 1, 17)
    tokens = full * 5 + 1 + math.ceil(tail / 4)
    latent = torch.linspace(-0.4, 0.4, tokens).view(1, 1, tokens, 1, 1)
    latent = latent.expand(1, 3, tokens, 1, 1)
    expected = codecs.decode_visual(latent, frames)
    # Every start offset includes clip boundaries and the blended overlap.
    for start in range(frames):
        actual = codecs.decode_visual_range(latent, start, frames)
        torch.testing.assert_close(actual, expected[:, :, start:], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["immediate", "stable"])
def test_streaming_ranges_match_previous_prefix_semantics_with_bounded_work(
    temporal_codecs, mode
):
    codecs, calls = temporal_codecs
    latent = torch.linspace(-0.4, 0.4, 72).view(1, 1, 72, 1, 1).expand(1, 3, 72, 1, 1)
    emitted = 0
    for end in [*range(5, 72, 5), 72]:
        full, tail = divmod(end - 1, 5)
        frame_end = min(243, full * 17 + 1 + tail * 4)
        if mode == "stable" and end != 72:
            frame_end = max(0, (end - 2) // 5) * 17
        if frame_end <= emitted:
            continue
        expected = codecs.decode_visual(latent[:, :, :end], frame_end)[:, :, emitted:]
        calls.clear()
        actual = codecs.decode_visual_range(latent[:, :, :end], emitted, frame_end)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Work per output remains bounded even late in a 10-second sequence.
        assert len(calls) <= 3
        assert all(z.shape[2] <= 7 for z in calls)
        emitted = frame_end
    assert emitted == 243


@pytest.mark.parametrize("height,width", [(31, 32), (32, 31), (0, 32), (64, 64)])
def test_source_request_rejects_invalid_target_geometry_before_loading(
    codecs, monkeypatch, height, width
):
    request = make_request("video")
    request["messages"][1]["content"][0]["output"].update(height=height, width=width)
    monkeypatch.setattr(
        codecs,
        "_decode_media",
        lambda item, **kw: {
            **item,
            "frames": torch.zeros(3, 125, 32, 32, dtype=torch.uint8),
            "fps": 24.0,
        },
    )
    with pytest.raises(ValueError, match="geometry"):
        codecs.prepare(request, lambda name: pytest.fail("No model should load"))


def test_source_request_rejects_incompatible_aspect_before_loading(codecs, monkeypatch):
    monkeypatch.setattr(
        codecs,
        "_decode_media",
        lambda item, **kw: {
            **item,
            "frames": torch.zeros(3, 125, 32, 128, dtype=torch.uint8),
            "fps": 24.0,
        },
    )
    with pytest.raises(ValueError, match="aspect ratio"):
        codecs.prepare(
            make_request("video"), lambda name: pytest.fail("No model should load")
        )


@pytest.mark.parametrize(
    "uri", ["image.png", "https://example.test/ref.png", "s3://bucket/ref.png"]
)
def test_h3_and_zing_share_reference_descriptor_validation(codecs, uri):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
        minimax_h3_validate_canonical_request,
    )

    request = make_request("image", references=1, source=False)
    request["messages"][0]["content"][1]["uri"] = uri
    _, media, _, _, _ = codecs._parse_inference_request(request)
    canonical = minimax_h3_validate_canonical_request(
        task="ref2va",
        prompt="Edit the scene",
        conditions=[{"type": "image", "uri": uri, "role": "reference"}],
        target={"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 5},
    )
    assert {key: media[0][key] for key in ("type", "uri")} == {
        key: canonical["conditions"][0][key] for key in ("type", "uri")
    }
    request["messages"][0]["content"][1]["uri"] = ""
    with pytest.raises(ValueError, match="uri must be a non-empty string"):
        codecs._parse_inference_request(request)
    with pytest.raises(ValueError, match="uri must be a non-empty string"):
        minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="Edit the scene",
            conditions=[{"type": "image", "uri": "", "role": "reference"}],
            target={"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 5},
        )


@pytest.mark.parametrize(
    "clip",
    [{"start": -1}, {"start": True}, {"start": float("nan")}, {"start": 2, "end": 1}],
)
def test_source_clip_rejected_before_decoding(codecs, clip):
    request = make_request("video")
    request["messages"][1]["source"]["clip"] = clip
    with pytest.raises(ValueError):
        codecs._parse_inference_request(request)


@pytest.mark.parametrize("mode", ["deterministic", "performance"])
@pytest.mark.parametrize("outer_deterministic", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_visual_encode_policy_restores_backend_state(mode, outer_deterministic, fail):
    def state():
        return (
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )

    original = state()
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = outer_deterministic
        torch.use_deterministic_algorithms(outer_deterministic, warn_only=True)
        before = state()
        try:
            with codecs_module._visual_encode_policy(mode):
                assert state()[:3] == (
                    mode == "performance",
                    mode == "deterministic",
                    mode == "deterministic",
                )
                if fail:
                    raise RuntimeError("encoder failed")
        except RuntimeError as exc:
            assert fail and str(exc) == "encoder failed"
        assert state() == before
        with (
            pytest.raises(ValueError, match="encode_mode"),
            codecs_module._visual_encode_policy("fast"),
        ):
            pytest.fail("invalid mode must not enter encoding")
        assert state() == before
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = original[
            :2
        ]
        torch.use_deterministic_algorithms(original[2], warn_only=original[3])


@pytest.mark.parametrize("mode", ["deterministic", "performance"])
@pytest.mark.parametrize("frames", [1, 17, 22])
def test_visual_encode_forwards_mode_and_preserves_seed(
    codecs, monkeypatch, mode, frames
):
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(1, 1)
    model.quant_conv = torch.nn.Linear(1, 1)
    codecs.modules = {"video_vae": model}
    codecs.runtime = {
        "vae_config": {
            "encode_mode": mode,
            "encode_dtype": "float32",
            "encode_tile_batch_size": 2,
            "posterior_seed": 42,
        }
    }
    codecs.va = SimpleNamespace(latents_mean=[0.0, 0.0], latents_std=[1.0, 1.0])
    calls = []

    def encode_clips(model, clips, batch_size):
        assert torch.backends.cudnn.benchmark == (mode == "performance")
        assert torch.backends.cudnn.deterministic == (mode == "deterministic")
        assert torch.are_deterministic_algorithms_enabled() == (mode == "deterministic")
        assert batch_size == 2
        calls.append([clip.shape[2] for clip in clips])
        return [
            torch.zeros(1, 4, 1 if clip.shape[2] == 1 else 5, 1, 1) for clip in clips
        ]

    monkeypatch.setattr(codecs_module, "encode_clips", encode_clips)
    pixels = torch.zeros(3, frames, 1, 1, dtype=torch.uint8)
    first = codecs.encode_visual(pixels)
    second = codecs.encode_visual(pixels)
    assert torch.equal(first, second)
    expected_frames = {1: 1, 17: 5, 22: 7}[frames]
    assert first.shape == (1, 2, expected_frames, 1, 1)
    assert first.dtype == torch.float16
    assert calls == [[1] if frames == 1 else [17] * ((frames + 16) // 17)] * 2


@pytest.mark.parametrize("count", [5, 9])
def test_stream_prompt_preserves_reference_order_without_extra_video(
    codecs, monkeypatch, count
):
    from contextlib import nullcontext

    seen = []
    monkeypatch.setattr(
        codecs,
        "_decode_media",
        lambda item, **kw: {
            **item,
            "frames": torch.full(
                (3, 1, 32, 32), int(item["id"].split("-")[1]), dtype=torch.uint8
            ),
            "fps": 24,
        },
    )
    monkeypatch.setattr(
        codecs, "encode_condition", lambda prompt, media: seen.extend(media) or {}
    )
    encoded = []
    monkeypatch.setattr(
        codecs,
        "encode_visual",
        lambda pixels, *, metric_operation="source": (
            encoded.append(pixels) or torch.zeros(1, 2, 1, 2, 2)
        ),
    )
    request = make_request("video", references=count, source=False)
    # Use deliberately non-sorted IDs and distinct pixels; preserve wire order.
    request["messages"][0]["content"][1:] = reversed(
        request["messages"][0]["content"][1:]
    )
    request["messages"][1]["conditions"].reverse()
    result = codecs.prepare(
        request,
        lambda name: nullcontext(),
        retain_condition_media=True,
    )
    assert [item["id"] for item in seen] == [f"ref-{i}" for i in reversed(range(count))]
    assert all(item["type"] == "image" for item in seen)
    assert len(seen) == len(encoded) == len(result.visual) == count
    assert seen[0]["frames"].eq(count - 1).all()
    assert encoded[0].eq(count - 1).all()
    assert result.source_visual is None
    assert result.condition_media == seen


def test_startup_prepares_encoder_precision_without_reference_encoding(codecs):
    encoder = torch.nn.Conv3d(3, 4, 1)
    quant = torch.nn.Conv3d(4, 4, 1)
    decoder = torch.nn.Conv3d(4, 3, 1)
    codecs.modules = {
        "video_vae": SimpleNamespace(encoder=encoder, quant_conv=quant, decoder=decoder)
    }
    codecs.runtime = {"vae_config": {"encode_dtype": "bfloat16"}}
    codecs.prepare_visual_encoder_weights()
    assert encoder.weight.dtype == quant.weight.dtype == torch.bfloat16
    assert decoder.weight.dtype == torch.float32


@pytest.mark.parametrize("temporal_patch", [1, 2, 4])
@pytest.mark.parametrize(
    "kinds", [("video",), ("image",), ("video", "image"), ("image", "video", "image")]
)
def test_explicit_single_frame_video_contract(codecs, temporal_patch, kinds):
    """Frame count/list position must not determine modality (issue #167)."""
    import numpy as np

    seen = {"text": [], "video_calls": 0, "image_calls": 0}

    class Tokenizer:
        def __call__(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            seen["text"].append(text)
            return {"input_ids": [100 + len(seen["text"])]}

        def convert_tokens_to_ids(self, token):
            return {
                "<|vision_start|>": 10,
                "<|vision_end|>": 11,
                "<|image_pad|>": 12,
                "<|video_pad|>": 13,
            }[token]

    class VideoProcessor:
        temporal_patch_size = temporal_patch

        def __call__(self, *, videos, do_sample_frames, return_tensors):
            seen["video_calls"] += 1
            assert do_sample_frames is False
            assert return_tensors == "pt"
            for video in videos:
                assert video.shape == (temporal_patch, 8, 8, 3)
                for frame in video:
                    np.testing.assert_array_equal(frame, pixels[:, 0].permute(1, 2, 0))
            return {
                "pixel_values_videos": torch.ones(len(videos), 3),
                "video_grid_thw": torch.tensor([[1, 4, 4]] * len(videos)),
            }

    class ImageProcessor:
        merge_size = 2

        def __call__(self, *, images, return_tensors):
            seen["image_calls"] += 1
            assert len(images) == kinds.count("image")
            assert all(image.shape == (8, 8, 3) for image in images)
            return {
                "pixel_values": torch.ones(len(images), 3),
                "image_grid_thw": torch.tensor([[1, 4, 4]] * len(images)),
            }

    def encode(ids, **features):
        seen["ids"], seen["features"] = ids, features
        return torch.zeros(len(ids), 4, dtype=torch.bfloat16)

    codecs.modules = {
        "tokenizer": Tokenizer(),
        "processor": SimpleNamespace(
            video_processor=VideoProcessor(), image_processor=ImageProcessor()
        ),
        "text_encoder": SimpleNamespace(encode_ids=encode),
    }
    codecs.runtime = {"condition_encoder_config": {"video_sample_fps": 2.0}}
    pixels = torch.arange(3 * 8 * 8, dtype=torch.uint8).reshape(3, 1, 8, 8)
    media = [{"type": kind, "fps": 24.0, "frames": pixels.clone()} for kind in kinds]
    result = codecs.encode_condition("Use Video 1 and Picture 1.", media)
    expected_text, counts = [], {"video": 0, "image": 0}
    for kind in kinds:
        counts[kind] += 1
        expected_text.append(
            f"<{'Video' if kind == 'video' else 'Picture'} {counts[kind]}>: "
        )
        if kind == "video":
            expected_text.append("<0.0 seconds>")
    assert seen["text"] == [*expected_text, "Use Video 1 and Picture 1."]
    assert seen["video_calls"] == int("video" in kinds)
    assert seen["image_calls"] == int("image" in kinds)
    assert (seen["ids"] == 13).sum() == 4 * kinds.count("video")
    assert (seen["ids"] == 12).sum() == 4 * kinds.count("image")
    assert (result["row_role"] == 1).sum() == 6 * len(kinds)
    assert result["context"].dtype == torch.bfloat16
    assert ("pixel_values_videos" in seen["features"]) == ("video" in kinds)
    assert ("pixel_values" in seen["features"]) == ("image" in kinds)
    # Qwen temporal padding is local and cannot modify incoming/source pixels.
    assert all(torch.equal(item["frames"], pixels) for item in media)


@pytest.mark.parametrize("preview_uri", ["source", "auto"])
def test_explicit_preview_does_not_enter_source_or_reference_vae(
    codecs, monkeypatch, preview_uri
):
    request = make_request("video", references=1)
    if preview_uri == "auto":
        codecs.runtime = {"causal": True}
    else:
        request["messages"][0]["content"].insert(
            0,
            {
                "id": "preview",
                "type": "video",
                "uri": preview_uri,
                "clip": {"start": 0, "end": 0},
            },
        )
    seen, vae = {}, []

    def decode(item, **kwargs):
        count = 1 if item.get("clip") or item["type"] == "image" else 125
        value = 7 if item.get("clip") else 23
        return {
            **item,
            "fps": 24.0,
            "frames": torch.full((3, count, 32, 32), value, dtype=torch.uint8),
        }

    def condition(prompt, media):
        seen["media"] = media
        return {}

    def visual(frames, *, metric_operation="source"):
        vae.append(frames.clone())
        return torch.zeros(1, 2, 1, 2, 2)

    monkeypatch.setattr(codecs, "_decode_media", decode)
    monkeypatch.setattr(codecs, "encode_condition", condition)
    monkeypatch.setattr(codecs, "encode_visual", visual)

    @contextmanager
    def use(name):
        yield

    result = codecs.prepare(request, use, retain_condition_media=True)
    preview_id = "<Video 1>" if preview_uri == "auto" else "preview"
    assert [item["id"] for item in seen["media"]] == [preview_id, "ref-0"]
    assert seen["media"][0]["type"] == "video"
    assert seen["media"][0]["frames"].shape[1] == 1
    assert len(vae) == 2 and [frames.shape[1] for frames in vae] == [1, 125]
    assert all(torch.all(frames == 23) for frames in vae)
    assert list(result.visual) == ["ref-0"]
    assert [item["content_id"] for item in result.conditions] == ["ref-0"]


def test_ar_source_preview_is_explicit_and_preserves_full_source(codecs):
    codecs.runtime = {"causal": True}
    request = make_request("video", references=6)
    _, media, _, conditions, source = codecs._parse_inference_request(request)
    assert media[0] == dict(
        type="video", id="<Video 1>", uri="source", clip=dict(start=0, end=0)
    )
    assert len(media) == 7 and len(conditions) == 6
    assert source == dict(type="video", uri="source")
    assert len(request["messages"][0]["content"]) == 7  # caller unmodified
    codecs.runtime["causal"] = False
    assert len(codecs._parse_inference_request(request)[1]) == 6


@pytest.mark.parametrize("variant", ["bidirectional", "image", "no_source", "explicit"])
def test_automatic_presentation_preserves_other_request_contracts(codecs, variant):
    codecs.runtime = {"causal": variant != "bidirectional"}
    request = make_request(
        "image" if variant == "image" else "video", source=variant != "no_source"
    )
    if variant == "explicit":
        request["messages"][0]["content"].insert(
            0, dict(type="video", id="v", uri="independent", clip=dict(start=0, end=0))
        )
    _, media, _, _, _ = codecs._parse_inference_request(request)
    assert len(media) == int(variant == "explicit")
    if media:
        assert media[0]["id"] == "v" and media[0]["uri"] == "independent"


def test_removed_preview_field_and_ambiguous_clipped_source_are_rejected(codecs):
    codecs.runtime = {"causal": True}
    request = make_request("video")
    request["source_preview"] = "ar-first-frame"
    with pytest.raises(ValueError, match="automatically"):
        codecs._parse_inference_request(request)
    request.pop("source_preview")
    request["messages"][1]["source"]["clip"] = dict(start=0, end=1)
    with pytest.raises(ValueError, match="explicit video"):
        codecs._parse_inference_request(request)


def test_still_video_preserves_declared_fps_and_pixels(codecs, tmp_path):
    from PIL import Image

    path = tmp_path / "frame.png"
    Image.new("RGB", (32, 48), (17, 42, 91)).save(path)
    codecs.remote_read_fs = None
    item = codecs._decode_media(
        dict(
            type="video",
            id="v",
            uri=str(path),
            clip=dict(start=0, end=0),
            presentation_fps=23.976,
        ),
        need_audio=False,
    )
    assert item["fps"] == 23.976
    assert tuple(item["frames"].shape) == (3, 1, 48, 32)
    assert item["frames"][:, 0, 0, 0].tolist() == [17, 42, 91]


def test_real_decode_explicit_auto_and_stream_snapshot_agree(codecs, tmp_path):
    """Real lossless decode, not a learned-feature or real-processor claim."""
    import av
    import numpy as np
    from PIL import Image

    video = tmp_path / "source.mkv"
    original = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    with av.open(str(video), "w") as container:
        stream = container.add_stream("ffv1", rate=24)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "bgr0"
        for rgb in (original, 255 - original, np.zeros_like(original)):
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(rgb, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    codecs.remote_read_fs = None
    codecs.runtime = {"causal": True}
    request = make_request("video")
    request["messages"][1]["source"]["uri"] = str(video)
    _, automatic, _, _, source = codecs._parse_inference_request(request)
    explicit = dict(
        type="video", id="<Video 1>", uri=str(video), clip=dict(start=0, end=0)
    )
    auto = codecs._decode_media(automatic[0], need_audio=False)
    native_descriptor = codecs._decode_media(explicit, need_audio=False)
    full = codecs._decode_media(source, need_audio=False)
    snapshot = tmp_path / "s0.png"
    Image.fromarray(original).save(snapshot)
    streamed = codecs._decode_media(
        dict(explicit, uri=str(snapshot), presentation_fps=24), need_audio=False
    )
    assert full["frames"].shape[1] == 3
    for candidate in (auto, native_descriptor, streamed):
        assert candidate["fps"] == 24
        assert candidate["type"] == "video"
        assert candidate["clip"] == dict(start=0, end=0)
        assert torch.equal(candidate["frames"], full["frames"][:, :1])


@pytest.mark.parametrize("references", [0, 6])
def test_in_memory_source_presentation_is_not_a_vae_reference(
    codecs, monkeypatch, references
):
    request = make_request("video", references=references, source=False)
    seen = {}
    preview = dict(
        type="video",
        id="<Video 1>",
        clip=dict(start=0, end=0),
        frames=torch.full((3, 1, 32, 32), 17, dtype=torch.uint8),
        fps=24.0,
    )
    monkeypatch.setattr(
        codecs,
        "_decode_media",
        lambda item, **kw: dict(
            item, frames=torch.full((3, 1, 32, 32), 23, dtype=torch.uint8), fps=1.0
        ),
    )

    def condition(prompt, media):
        seen["media"] = media
        return {}

    vae = []

    def visual(frames, *, metric_operation="source"):
        vae.append(frames.clone())
        return torch.zeros(1, 2, 1, 2, 2)

    monkeypatch.setattr(codecs, "encode_condition", condition)
    monkeypatch.setattr(codecs, "encode_visual", visual)

    @contextmanager
    def use(name):
        yield

    result = codecs.prepare(
        request, use, source_preview=preview, retain_condition_media=True
    )
    assert seen["media"] is result.condition_media
    assert torch.equal(result.condition_media[0]["frames"], preview["frames"])
    assert [x["type"] for x in seen["media"]] == ["video"] + ["image"] * references
    assert len(vae) == references and all(torch.all(x == 23) for x in vae)
    assert len(result.conditions) == references and result.source_visual is None
    assert torch.all(preview["frames"] == 17) and preview["frames"].shape[1] == 1


@pytest.mark.parametrize("fail", [False, True])
def test_explicit_encoder_benchmark_limit_applies_inside_scope_and_restores(
    monkeypatch, fail
):
    backend = SimpleNamespace(benchmark=False, deterministic=False, benchmark_limit=10)
    monkeypatch.setattr(torch.backends, "cudnn", backend)
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        with codecs_module._visual_encode_policy("performance", 1) as receipt:
            assert backend.benchmark_limit == 1
            assert receipt == dict(
                encode_mode="performance",
                cudnn_benchmark_limit=1,
                cudnn_benchmark=True,
                cudnn_deterministic=False,
                deterministic_algorithms=False,
            )
            if fail:
                raise RuntimeError("simulated encoder failure")
    except RuntimeError as exc:
        assert fail and str(exc) == "simulated encoder failure"
    assert vars(backend) == dict(
        benchmark=False, deterministic=False, benchmark_limit=10
    )
    assert torch.are_deterministic_algorithms_enabled() == original_deterministic


def test_default_encoder_limit_and_nested_scope_preserve_existing_policy(monkeypatch):
    backend = SimpleNamespace(benchmark=False, deterministic=False, benchmark_limit=10)
    monkeypatch.setattr(torch.backends, "cudnn", backend)
    with codecs_module._visual_encode_policy("performance") as receipt:
        assert receipt["cudnn_benchmark_limit"] == 10
        with codecs_module._visual_encode_policy("deterministic", 1) as nested:
            assert nested["cudnn_benchmark_limit"] == 1
            assert nested["deterministic_algorithms"] is True
        assert backend.benchmark_limit == 10 and backend.benchmark
    assert backend.benchmark_limit == 10 and not backend.benchmark


@pytest.mark.parametrize("invalid", [True, False, 0, 2, "1", 1.0])
def test_encoder_limit_only_admits_explicit_validated_candidate(invalid):
    with pytest.raises(ValueError, match="null or 1"):
        with codecs_module._visual_encode_policy("performance", invalid):
            pytest.fail("Invalid policy must not enter encoding")


def test_unsupported_explicit_limit_fails_before_mutating_flags(monkeypatch):
    backend = SimpleNamespace(benchmark=False, deterministic=True, benchmark_limit=None)
    monkeypatch.setattr(torch.backends, "cudnn", backend)
    with pytest.raises(RuntimeError, match="does not support"):
        with codecs_module._visual_encode_policy("performance", 1):
            pytest.fail("Unsupported override must not enter encoding")
    assert vars(backend) == dict(
        benchmark=False, deterministic=True, benchmark_limit=None
    )


def test_visual_policy_contexts_are_serialized_across_encoder_threads(monkeypatch):
    import threading

    backend = SimpleNamespace(benchmark=False, deterministic=False, benchmark_limit=10)
    monkeypatch.setattr(torch.backends, "cudnn", backend)
    held, release, waiting, entered = (threading.Event() for _ in range(4))
    failures = []
    observed = []

    def first():
        try:
            with codecs_module._visual_encode_policy("performance", 1):
                held.set()
                assert release.wait(5)
        except BaseException as error:
            failures.append(error)

    def second():
        try:
            assert held.wait(5)
            waiting.set()
            with codecs_module._visual_encode_policy("performance") as receipt:
                observed.append(receipt["cudnn_benchmark_limit"])
                entered.set()
        except BaseException as error:
            failures.append(error)

    a, b = threading.Thread(target=first), threading.Thread(target=second)
    a.start()
    b.start()
    try:
        assert waiting.wait(5)
        assert not entered.wait(0.05)
    finally:
        release.set()
        a.join(timeout=5)
        b.join(timeout=5)
    assert not a.is_alive() and not b.is_alive() and not failures
    assert observed == [10]
    assert backend.benchmark_limit == 10


@pytest.mark.parametrize("mode", ["deterministic", "performance"])
def test_replicated_visual_latent_preserves_non_distributed_result(monkeypatch, mode):
    latent = torch.randn(1, 24, 1, 2, 2)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    assert codecs_module._replicated_visual_latent(latent, mode) is latent


@pytest.mark.parametrize("entrypoint", ["reference", "streaming_source"])
def test_performance_encoding_uses_sp_leader_latent(codecs, monkeypatch, entrypoint):
    """Different encoder results must not become different replicated SP inputs."""
    from sglang.multimodal_gen.runtime import distributed

    leader, calls = None, []

    def broadcast(value, src):
        nonlocal leader
        assert src == 0
        calls.append(value.clone())
        if leader is None:
            leader = value.clone()
        return leader.clone()

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        distributed, "get_sp_group", lambda: SimpleNamespace(broadcast=broadcast)
    )
    codecs.runtime = {
        "vae_config": {
            "encode_mode": "performance",
            "encode_dtype": "float32",
            "encode_tile_batch_size": 1,
            "posterior_seed": 42,
        }
    }
    codecs.va = SimpleNamespace(latents_mean=[0.0] * 24, latents_std=[1.0] * 24)
    anchor = torch.nn.Parameter(torch.zeros(()))
    codecs.modules = {
        "video_vae": SimpleNamespace(
            encoder=torch.nn.Identity(),
            quant_conv=torch.nn.Identity(),
            parameters=lambda: iter([anchor]),
        )
    }
    offset = 0

    def moments(*args):
        nonlocal offset
        offset += 1
        value = torch.zeros(1, 48, 1, 2, 2)
        value[:, :24] = offset
        return [value]

    monkeypatch.setattr(codecs_module, "encode_clips", moments)
    if entrypoint == "reference":
        pixels = torch.zeros(3, 1, 32, 32, dtype=torch.uint8)
        first = codecs.encode_visual(pixels)
        second = codecs.encode_visual(pixels)
    else:

        def sample():
            return codecs.sample_streaming_visual_moments(
                moments()[0], latent_start=0, noise_version="indexed-cpu-v1"
            )

        first, second = sample(), sample()
    assert len(calls) == 2 and not torch.equal(calls[0], calls[1])
    assert torch.equal(first, second)


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("reference_kind", ["image", "video"])
def test_qwen_and_vae_share_normalized_reference_and_source_bucket(
    codecs, monkeypatch, external, reference_kind
):
    from contextlib import nullcontext

    codecs.runtime = {"causal": True}
    codecs.max_short_edge = 128
    request = make_request("video", references=1)
    request["h3_max_area_config"] = dict(reference_image=4096, reference_video=4096)
    request["messages"][0]["content"][1]["type"] = reference_kind
    if reference_kind == "video":
        # An explicit reference video suppresses automatic source presentation.
        request["messages"][0]["content"].insert(
            0,
            dict(type="video", id="<Video 1>", uri="source", clip=dict(start=0, end=0)),
        )
    raw, qwen, vae = {}, [], []

    def decode(item, **kwargs):
        count = 1 if item.get("clip") or item["type"] == "image" else 125
        decoded = dict(
            item,
            frames=torch.arange(3 * count * 128 * 128, dtype=torch.int64)
            .remainder(256)
            .to(torch.uint8)
            .reshape(3, count, 128, 128),
            fps=24.0,
        )
        raw[item.get("id", "source")] = decoded
        return decoded

    def condition(prompt, media):
        qwen.append((prompt, media))
        return {}

    monkeypatch.setattr(codecs, "_decode_media", decode)
    monkeypatch.setattr(codecs, "encode_condition", condition)
    monkeypatch.setattr(
        codecs,
        "encode_visual",
        lambda frames, *, metric_operation="source": (
            vae.append(frames) or torch.zeros(1, 2, 1, 2, 2)
        ),
    )
    active = codecs.prepare(
        request,
        lambda _: nullcontext(),
        retain_condition_media=True,
        conditioning=condition if external else None,
    )
    media = qwen[0][1]
    assert media is active.condition_media
    assert media[0]["frames"].shape == (3, 1, 32, 32)
    assert media[0]["fps"] == 24.0
    assert media[1]["frames"].shape[-2:] == (64, 64)
    assert media[1]["frames"] is vae[0]
    assert vae[1].shape == (3, 125, 32, 32)
    assert torch.equal(media[0]["frames"], vae[1][:, :1])
    assert all(item["frames"].shape[-2:] == (128, 128) for item in raw.values())

    # Prompt refresh consumes the exact same normalized tensors, no preparation.
    monkeypatch.setattr(codecs, "_decode_media", lambda *a, **kw: pytest.fail("decode"))
    monkeypatch.setattr(
        codecs, "_normalize_inference_media", lambda *a, **kw: pytest.fail("normalize")
    )
    codecs.encode_condition("new prompt", active.condition_media)
    assert qwen[-1][1] is media and len(vae) == 2


def test_stream_preview_and_reference_replacement_reuse_normalized_media(
    codecs, monkeypatch
):
    from contextlib import nullcontext

    codecs.max_short_edge = 128
    request = make_request("video", references=1, source=False)
    request["h3_max_area_config"] = dict(reference_image=4096, source=4096)
    preview = dict(
        id="<Video 1>",
        type="video",
        origin="source_s0",
        frames=torch.full((3, 1, 128, 128), 17, dtype=torch.uint8),
        fps=23.976,
    )
    qwen, vae = [], []
    monkeypatch.setattr(
        codecs,
        "_decode_media",
        lambda item, **kw: dict(
            item, frames=torch.full((3, 1, 128, 128), 23, dtype=torch.uint8), fps=1.0
        ),
    )
    monkeypatch.setattr(codecs, "encode_condition", lambda p, m: qwen.append(m) or {})
    monkeypatch.setattr(
        codecs,
        "encode_visual",
        lambda f, *, metric_operation="source": (
            vae.append(f) or torch.zeros(1, 2, 1, 2, 2)
        ),
    )
    active = codecs.prepare(
        request,
        lambda _: nullcontext(),
        source_preview=preview,
        retain_condition_media=True,
    )
    normalized_preview = active.condition_media[0]
    assert normalized_preview["frames"].shape == (3, 1, 64, 64)
    assert normalized_preview["fps"] == 23.976
    assert preview["frames"].shape == (3, 1, 128, 128)
    assert len(vae) == 1 and active.source_visual is None
    candidate = codecs.prepare_reference_update(
        active,
        [dict(id="replacement", type="image", uri="new.png")],
        "new prompt",
        lambda _: nullcontext(),
        conditioning=codecs.encode_condition,
    )
    assert candidate.condition_media[0] is normalized_preview
    assert candidate.condition_media[1]["frames"].shape == (3, 1, 64, 64)
    assert qwen[-1] is candidate.condition_media
    assert candidate.condition_media[1]["frames"] is vae[-1]
    assert candidate.max_area_config == active.max_area_config
    assert codecs.max_area_config["reference_image"] == 1024
    assert len(vae) == 2 and candidate.source_visual is None


def test_initial_normalization_failure_agrees_before_qwen(codecs, monkeypatch):
    from sglang.multimodal_gen.runtime.conditioning import worker
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

    monkeypatch.setattr(
        codecs,
        "_normalize_inference_media",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad geometry")),
    )
    receipts = []
    monkeypatch.setattr(
        worker, "coordinate_validation", lambda ok: receipts.append(ok) or [ok, False]
    )
    with pytest.raises(ConditionError, match="Media normalization failed"):
        codecs.prepare(
            make_request("video", source=False),
            lambda _: pytest.fail("No model may run"),
            conditioning=lambda *a: pytest.fail("Qwen must not run"),
        )
    assert receipts == [False]


def test_prepare_decode_failure_and_compute_validation_agree_before_qwen(
    codecs, monkeypatch
):
    from sglang.multimodal_gen.runtime.conditioning import worker
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

    monkeypatch.setattr(
        codecs,
        "_decode_media",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("corrupt source media")),
    )
    receipts = []
    monkeypatch.setattr(
        worker, "coordinate_validation", lambda ok: receipts.append(ok) or [ok, True]
    )
    with pytest.raises(ConditionError, match="Media normalization failed") as exc_info:
        codecs.prepare(
            make_request("video", source=True),
            lambda _: pytest.fail("No model may run"),
            conditioning=lambda *a: pytest.fail("Qwen must not run"),
        )
    assert isinstance(exc_info.value.__cause__, OSError)
    assert receipts == [False]

    receipts.clear()
    monkeypatch.setattr(
        codecs,
        "_decode_media",
        lambda item, **kw: dict(
            item,
            frames=torch.zeros(3, 1, 32, 32, dtype=torch.uint8),
            fps=24.0,
        ),
    )
    monkeypatch.setattr(
        codecs,
        "validate_compute_inputs",
        lambda *a, **kw: (_ for _ in ()).throw(
            ConditionError("CONDITION_INVALID", "This video size is not supported")
        ),
    )
    with pytest.raises(ConditionError, match="This video size is not supported"):
        codecs.prepare(
            make_request("video", source=False),
            lambda _: pytest.fail("No model may run"),
            conditioning=lambda *a: pytest.fail("Qwen must not run"),
        )
    assert receipts == [False]


@pytest.mark.parametrize("kind", ["image", "video", "video_audio"])
@pytest.mark.parametrize("external", [False, True])
def test_unassigned_visual_media_rejected_before_qwen_or_vae(
    codecs, monkeypatch, kind, external
):
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

    request = make_request("video", source=False)
    request["messages"][0]["content"].append(
        dict(id="unassigned", type=kind, uri="other")
    )
    monkeypatch.setattr(
        codecs,
        "_decode_media",
        lambda item, **kw: dict(
            item,
            frames=torch.zeros(3, 1, 32, 32, dtype=torch.uint8),
            fps=24.0,
        ),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid media must fail before any model call")

    monkeypatch.setattr(codecs, "encode_condition", forbidden)
    with pytest.raises(
        ConditionError if external else ValueError,
        match=(
            "Media normalization failed"
            if external
            else "target reference or source presentation"
        ),
    ):
        codecs.prepare(request, forbidden, conditioning=forbidden if external else None)


@pytest.mark.parametrize("origin", ["source_s0", None, "reference"])
def test_request_cannot_claim_internal_source_origin(codecs, origin):
    request = make_request("video", references=1, source=False)
    request["messages"][0]["content"][1]["origin"] = origin
    with pytest.raises(ValueError, match="reserved for internal source presentation"):
        codecs._parse_inference_request(request)


def test_native_960x540_s0_normalizes_only_condition_media_for_480_output(
    codecs, monkeypatch
):
    from contextlib import nullcontext

    codecs.max_short_edge = 480
    codecs.max_area_config["source"] = 832 * 480
    request = make_request("video", references=0, source=False)
    request["messages"][1]["content"][0]["output"].update(width=832, height=480)
    pixels = torch.full((3, 1, 540, 960), 17, dtype=torch.uint8)
    preview = dict(
        type="video", id="<Video 1>", frames=pixels, fps=24.0, clip=dict(start=0, end=0)
    )
    seen = []
    monkeypatch.setattr(
        codecs, "encode_condition", lambda prompt, media: seen.append(media) or {}
    )
    monkeypatch.setattr(
        codecs,
        "encode_visual",
        lambda *_: pytest.fail("S0 must not enter reference/source VAE"),
    )
    result = codecs.prepare(request, lambda _: nullcontext(), source_preview=preview)
    assert result.visual_shape[-2:] == (480 // 16, 832 // 16)
    assert not result.conditions and result.source_visual is None
    normalized = seen[0][0]
    assert normalized["origin"] == "source_s0" and normalized["frames"].shape[1] == 1
    assert normalized["frames"].shape[-2] * normalized["frames"].shape[-1] <= 832 * 480
    assert pixels.shape == (3, 1, 540, 960) and torch.all(pixels == 17)


@pytest.mark.parametrize(
    "target,optimized",
    [
        ((480, 832), False),
        ((736, 1280), True),
        ((720, 1280), False),
        ((1280, 736), False),
    ],
)
def test_resize_uses_target_and_preserves_pixels(monkeypatch, target, optimized):
    # Input/encoded sizes deliberately do not identify the target bucket.
    geometry = media_geometry.SourceGeometry(12, 16, 6, 8, *target)
    pixels = (
        torch.arange(3 * 2 * 12 * 16, dtype=torch.int64)
        .to(torch.uint8)
        .reshape(3, 2, 12, 16)
    )
    baseline = media_geometry._resize_frames(pixels, 6, 8)
    layouts = []
    original = Image.fromarray

    def fromarray(array, *args, **kwargs):
        layouts.append(array.flags.c_contiguous)
        return original(array, *args, **kwargs)

    monkeypatch.setattr(Image, "fromarray", fromarray)
    result = geometry.resize(pixels)
    assert geometry.use_target_736_media_path is optimized
    assert layouts == [optimized, optimized]
    assert result.is_contiguous() and result.dtype == torch.uint8
    assert torch.equal(result, baseline)
    assert geometry.receipt() == dict(
        width=8, height=6, resize_applied=True, algorithm="h3-lanczos-v1"
    )


@pytest.mark.parametrize("target", [(480, 832), (736, 1280)])
def test_source_736_does_not_select_target_path(target):
    geometry = media_geometry.SourceGeometry(736, 1280, 480, 832, *target)
    assert geometry.use_target_736_media_path == (target == (736, 1280))


@pytest.mark.parametrize("target", [None, (480, 832), (736, 1280), (720, 1280)])
@pytest.mark.parametrize("resize", [False, True])
def test_admission_copy_boundary_and_wire_pixels(monkeypatch, target, resize):
    geometry = (
        None
        if target is None
        else media_geometry.SourceGeometry(
            12, 16, 6 if resize else 12, 8 if resize else 16, *target
        )
    )
    optimized = target == (736, 1280)
    hwc = np.arange(2 * 12 * 16 * 3, dtype=np.uint8).reshape(2, 12, 16, 3)
    expected = torch.from_numpy(hwc.copy()).permute(3, 0, 1, 2).contiguous()
    if geometry is not None:
        expected = media_geometry._resize_frames(expected, *geometry.encoded_shape)
    session = SimpleNamespace(
        peer=remote_encoder.EncoderPeer(),
        source_geometry=geometry,
        timeout=1,
        port=1,
        admission_endpoint="inproc://test",
        capacity_frames=17,
        capacity_bytes=100000,
        token="test",
        profile_dir="test",
        stage_timing_dir=None,
        device="cpu",
        stop=threading.Event(),
        prefetched_frames=0,
        _consumed=0,
        accepted_frames=2,
        resize_host_ms=0,
        failure=None,
        remote_usage={},
        _check_admission=lambda *args: None,
    )
    copy_boundaries = []
    original_contiguous = torch.Tensor.contiguous

    def contiguous(tensor, *args, **kwargs):
        copy_boundaries.append(session.prefetched_frames)
        return original_contiguous(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "contiguous", contiguous)
    payloads = []

    class Socket:
        def setsockopt(self, *args):
            pass

        def connect(self, *args):
            pass

        def close(self):
            pass

        def poll(self, *args):
            return True

        def recv_multipart(self):
            return [
                json.dumps(
                    dict(
                        op="append",
                        source_frame_start=0,
                        num_frames=2,
                        width=16,
                        height=12,
                    )
                ).encode(),
                hwc.tobytes(),
            ]

        def send_multipart(self, parts):
            self.request = json.loads(parts[0])
            if self.request["op"] == "append":
                payloads.append(parts[1])
                session.stop.set()

        def recv_json(self):
            return dict(
                ok=True,
                source_encoding=None if geometry is None else geometry.receipt(),
            )

    class Context:
        def socket(self, *args):
            return Socket()

        def term(self):
            pass

    monkeypatch.setattr(remote_encoder.zmq, "Context", Context)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(
        remote_encoder.dist, "TCPStore", lambda *args, **kwargs: SimpleNamespace(port=1)
    )
    monkeypatch.setattr(remote_encoder, "bridge_group", lambda *args, **kwargs: None)
    remote_encoder.RemoteEncoderSession._produce(session)
    assert session.failure is None
    assert session.prefetched_frames == 2
    assert payloads == [expected.numpy().tobytes()]
    # The first CHW copy occurs before admission accounting for all old paths,
    # including unknown geometry and the no-resize path.
    assert copy_boundaries[0] == (2 if optimized else 0)
