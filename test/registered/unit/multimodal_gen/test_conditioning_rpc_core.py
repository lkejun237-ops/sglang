# SPDX-License-Identifier: Apache-2.0
"""Issue #256 RPC v6 CPU contracts, using real ASGI, framing and presentation."""

if __name__ == "__main__":
    import runpy
    import sys
    from pathlib import Path

    test_file = Path(__file__).resolve()
    runner = (
        test_file.parents[4] / "benchmark/zing_h3_causal/run_conditioning_cpu_tests.py"
    )
    arguments = ["-x" if value == "-f" else value for value in sys.argv[1:]]
    sys.argv = [str(runner), str(test_file), *arguments]
    runpy.run_path(str(runner), run_name="__main__")


import asyncio
import hashlib
import inspect
import json
import struct
import threading
import uuid
from dataclasses import fields, replace
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import torch
from safetensors.torch import load, save

from sglang.multimodal_gen.runtime.conditioning import cli, inputs
from sglang.multimodal_gen.runtime.conditioning.contract import DeploymentProfile
from sglang.multimodal_gen.runtime.conditioning.encoder import encode_condition
from sglang.multimodal_gen.runtime.conditioning.resources import ResourceLimits
from sglang.multimodal_gen.runtime.conditioning.service import ConditioningService
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

PREFIX = "/v1/conditioning"


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    ).encode()


def profile():
    return DeploymentProfile(
        dict(
            schema_version=1,
            model={"sha256": "1" * 64},
            projection=None,
            tokenizer={"sha256": "2" * 64},
            processor={"sha256": "3" * 64},
            runtime={"kind": "zing"},
            implementation_version="test-v1",
            preprocess_version="rgb-v1",
            numerical_environment={
                "encoder_tp": 1,
                "gpu": "fixture",
                "cc": "fixture",
                "backend": "fixture",
                "precision": "bf16",
                "software": {"fixture": "1"},
            },
        )
    )


def media(value=1, kind="image", **metadata):
    return dict(
        type=kind,
        fps=24.0 if kind == "video" else 1.0,
        frames=torch.full((3, 1, 2, 4), value, dtype=torch.uint8),
        **metadata,
    )


def reference_digest(prompt, items, profile_id):
    identities = [
        dict(
            type=item["type"],
            shape=list(item["frames"].shape),
            sha256=hashlib.sha256(
                item["frames"].contiguous().numpy().tobytes()
            ).hexdigest(),
        )
        for item in items
    ]
    return hashlib.sha256(
        canonical(
            dict(
                profile_id=profile_id,
                preprocess_version="rgb-v1",
                prompt=prompt,
                items=identities,
            )
        )
    ).hexdigest()


def request_body(
    prompt,
    items,
    profile_id,
    *,
    deadline_ms=10000,
    digest=None,
    overrides=None,
    allow_nonfinite=False,
):
    header = dict(
        schema_version=1,
        profile_id=profile_id,
        input_digest=digest or reference_digest(prompt, items, profile_id),
        prompt=prompt,
        media=[
            dict(
                type=item["type"],
                **({"fps": item["fps"]} if "fps" in item else {}),
                tensor=f"media_{i}",
            )
            for i, item in enumerate(items)
        ],
        deadline_ms=deadline_ms,
    )
    if overrides:
        header.update(overrides)
    encoded = json.dumps(
        header, sort_keys=True, separators=(",", ":"), allow_nan=allow_nonfinite
    ).encode()
    return (
        struct.pack(">I", len(encoded))
        + encoded
        + save(
            {
                f"media_{i}": item["frames"].contiguous().clone()
                for i, item in enumerate(items)
            }
        )
    )


def unpack(body):
    n = struct.unpack(">I", body[:4])[0]
    return json.loads(body[4 : 4 + n]), load(body[4 + n :])


class Tokenizer:
    """The same CPU tokenizer boundary as existing conditioning input tests."""

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}

    def convert_tokens_to_ids(self, token):
        return {
            "<|vision_start|>": 1,
            "<|vision_end|>": 2,
            "<|image_pad|>": 3,
            "<|video_pad|>": 4,
        }[token]


class Images:
    merge_size = 1

    def __call__(self, images, **kwargs):
        return {
            "pixel_values": torch.from_numpy(np.stack(images).copy()),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
        }


class Videos:
    temporal_patch_size = 2

    def __call__(self, videos, **kwargs):
        assert kwargs["do_sample_frames"] is False
        return {
            "pixel_values_videos": torch.from_numpy(np.stack(videos).copy()),
            "video_grid_thw": torch.tensor([[len(v) // 2, 2, 2] for v in videos]),
        }


class Encoder:
    """Replace only encode_ids; every id and processor feature affects output."""

    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.modules = dict(
            tokenizer=Tokenizer(),
            processor=SimpleNamespace(
                image_processor=Images(), video_processor=Videos()
            ),
            text_encoder=self,
        )
        self.runtime = {"condition_encoder_config": {"video_sample_fps": 8}}

    def encode_ids(self, ids, **features):
        self.started.set()
        assert self.release.wait(5), "test encoder was not released"
        digest = hashlib.sha256(ids.contiguous().numpy().tobytes())
        for name, tensor in sorted(features.items()):
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(canonical(list(tensor.shape)))
            digest.update(tensor.contiguous().numpy().tobytes())
        values = torch.tensor(list(digest.digest()), dtype=torch.bfloat16)
        return values.repeat((ids.numel() * 5120 + 31) // 32)[
            : ids.numel() * 5120
        ].reshape(-1, 5120)

    def __call__(self, value):
        self.calls.append(value.prompt)
        return encode_condition(
            self.modules, self.runtime, value.prompt, value.media, max_rows=512
        )


def limits(**changes):
    # Old-only fixture values make /encode reach a real 404 before implementation.
    names = {f.name for f in fields(ResourceLimits)}
    baseline = dict(
        max_condition_rows=512,
        max_input_bytes=1 << 20,
        max_condition_bytes=8 << 20,
        max_inflight_bytes=128 << 20,
        max_inflight_bytes_per_tenant=128 << 20,
        cache_bytes=16 << 20,
        cache_bytes_per_tenant=16 << 20,
        max_ram_bytes=128 << 20,
        max_ram_bytes_per_tenant=128 << 20,
        cpu_workspace_bytes=16 << 20,
        max_total_disk_bytes=128 << 20,
        max_total_disk_bytes_per_tenant=128 << 20,
        max_header_bytes=65536,
    )
    selected = ResourceLimits(**{k: v for k, v in baseline.items() if k in names})
    aliases = {
        "max_queued_encodes": "max_queued_requests",
        "max_request_bytes": "max_input_bytes",
    }
    changes = {
        key if key in names else aliases.get(key, key): value
        for key, value in changes.items()
    }
    return replace(selected, **changes)


def components(*, encoder=None, configured_limits=None):
    p, encoder = profile(), encoder or Encoder()
    kwargs = dict(
        profile=p,
        encoder=encoder,
        credentials={
            "secret": {"tenant": "tenant", "owner": "api"},
            "second": {"tenant": "second", "owner": "api"},
            "admin": {"tenant": "admin", "owner": "api", "admin": True},
        },
        limits=configured_limits or limits(),
    )
    # Transitional fixture only: no disk/store behavior assertions are relaxed.
    parameters = inspect.signature(ConditioningService).parameters
    if "store" in parameters:
        kwargs.update(store=SimpleNamespace(), trusted_issuers={})
    app = ConditioningService(**kwargs)
    client = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.ASGITransport(app=app),
        headers={"Authorization": "Bearer secret"},
    )
    return p, app, client, encoder


async def encode(client, p, items=None, prompt="complete prompt", **kwargs):
    body = request_body(prompt, items or [], p.profile_id, **kwargs)
    return await client.post(
        PREFIX + "/encode",
        content=body,
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(body)),
        },
    )


def assert_ok(response):
    assert response.status_code == 200, response.text
    header, tensors = unpack(response.content)
    assert header["schema_version"] == 1
    assert header["cache"] in {"miss", "joined", "hit"}
    assert "qwen_prefix" not in header and "result_source" not in header
    assert set(tensors) == {"context", "row_role"}
    assert tensors["context"].dtype == torch.bfloat16
    assert tensors["row_role"].dtype == torch.int64
    return header, tensors


def assert_idle(app):
    snapshot = app.ledger.snapshot()["global"]
    assert snapshot["slots"] == snapshot["inflight"] == 0


@pytest.mark.parametrize("method", ["POST", "DELETE"])
@pytest.mark.parametrize("leave", ["deadline", "disconnect"])
def test_task_response_releases_slot_on_stalled_send(method, leave):
    class TaskEncoder(Encoder):
        def create_prefix_cache(self, max_bytes):
            return {}

    async def run():
        _, app, client, _ = components(
            encoder=TaskEncoder(), configured_limits=limits(request_timeout_s=0.03)
        )
        task_id = uuid.uuid4().hex
        if method == "DELETE":
            task_id = app.prefix_tasks.open("tenant", task_id)
        started, disconnected = asyncio.Event(), asyncio.Event()
        consumed = False

        async def receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            started.set()
            await asyncio.Future()

        request = asyncio.create_task(
            app(
                {
                    "type": "http",
                    "method": method,
                    "path": PREFIX + "/tasks/" + task_id,
                    "headers": [(b"authorization", b"Bearer secret")],
                },
                receive,
                send,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 1)
            assert app.ledger.snapshot()["global"]["slots"] == 1
            if leave == "disconnect":
                disconnected.set()
            await asyncio.wait_for(request, 1)
            assert_idle(app)
            assert not app._requests
        finally:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
            await client.aclose()
            await app.close()

    asyncio.run(run())


def test_task_rpc_hashes_pixels_once_through_native_and_rejects_forgery(monkeypatch):
    from sglang.multimodal_gen.runtime.conditioning import native, wire

    class TextEncoder(Encoder):
        identities = []

        def encode_ids(self, ids, *, prefix_cache, media_identity, **features):
            assert prefix_cache is not None
            prefix_cache.reused_rows, prefix_cache.encoded_rows = 0, ids.numel()
            self.identities.append(media_identity)
            return super().encode_ids(ids, **features)

    text_encoder = TextEncoder()
    encoder = object.__new__(native.NativeConditionEncoder)
    encoder.gpu_id, encoder.max_rows, encoder.max_prompt_tokens = 0, 512, None
    encoder.pipeline = SimpleNamespace(
        modules=text_encoder.modules, zing_runtime=text_encoder.runtime
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    items = [media(1), media(2)]
    variants = [
        items,
        items,  # Only the prompt changes.
        [media(3), items[1]],  # Same shape, different pixels.
        list(reversed(items)),
        [dict(items[0], type="video", fps=24), items[1]],
        [dict(item, frames=item["frames"].reshape(3, 1, 4, 2)) for item in items],
    ]
    expected = [inputs.media_identity(value) for value in variants]
    assert expected[0] == expected[1] and len(set(expected)) == 5
    hashes = []
    original = inputs.hash_frame

    def counted(frames):
        hashes.append(tuple(frames.shape))
        return original(frames)

    monkeypatch.setattr(inputs, "hash_frame", counted)
    monkeypatch.setattr(wire, "hash_frame", counted)

    async def run():
        p, app, client, _ = components(encoder=encoder)
        try:
            opened = await client.post(PREFIX + "/tasks/" + uuid.uuid4().hex)
            assert opened.status_code == 200
            handle = opened.json()["task_handle"]
            client.headers["Content-Type"] = "application/octet-stream"
            for i, variant in enumerate(variants):
                response = await client.post(
                    PREFIX + "/encode",
                    content=request_body(
                        f"event {i}",
                        variant,
                        p.profile_id,
                        overrides={"task_handle": handle},
                    ),
                )
                assert_ok(response)
                assert len(hashes) == (i + 1) * 2
            assert text_encoder.identities == expected
            # Changed pixels with an old request digest cannot enter native.
            tampered = await client.post(
                PREFIX + "/encode",
                content=request_body(
                    "event 0",
                    variants[2],
                    p.profile_id,
                    digest=reference_digest("event 0", items, p.profile_id),
                    overrides={"task_handle": handle},
                ),
            )
            assert tampered.status_code == 400
            forged = await client.post(
                PREFIX + "/encode",
                content=request_body(
                    "event 0",
                    items,
                    p.profile_id,
                    overrides={"task_handle": handle, "media_identity": expected[0]},
                ),
            )
            assert forged.status_code == 400
            assert text_encoder.identities == expected
        finally:
            await client.aclose()
            await app.close()
        assert_idle(app)

    asyncio.run(run())


def test_task_rpc_owns_prefix_state_and_rejects_stale_or_foreign_handles():
    class TaskEncoder(Encoder):
        def __init__(self):
            super().__init__()
            self.states = []

        def create_prefix_cache(self, max_bytes):
            return {}

        def __call__(self, value):
            if value.task is not None:
                self.states.append(value.task.cache)
                value.task.cache["last_prompt"] = value.prompt
            return super().__call__(value)

    async def run():
        p, app, client, encoder = components(encoder=TaskEncoder())
        try:
            nonce = uuid.uuid4().hex
            first = await client.post(PREFIX + "/tasks/" + nonce)
            assert first.status_code == 200
            handle = first.json()["task_handle"]
            assert (
                await client.post(PREFIX + "/tasks/" + nonce)
            ).json() == first.json()
            assert (
                await client.post(PREFIX + "/tasks/" + uuid.uuid4().hex)
            ).status_code == 429
            for prompt in ("event: na", "event: rain"):
                assert_ok(
                    await encode(
                        client, p, prompt=prompt, overrides={"task_handle": handle}
                    )
                )
            assert encoder.states[0] is encoder.states[1]
            assert (
                app.cache.current_bytes == 0
            )  # task outputs never enter the shared CPU LRU
            other = await client.post(
                PREFIX + "/encode",
                content=request_body(
                    "event: rain", [], p.profile_id, overrides={"task_handle": handle}
                ),
                headers={
                    "Authorization": "Bearer second",
                    "Content-Type": "application/octet-stream",
                },
            )
            assert other.status_code == 410
            assert (await client.delete(PREFIX + "/tasks/" + handle)).status_code == 200
            assert encoder.states[0] == {}
            stale = await encode(client, p, overrides={"task_handle": handle})
            assert stale.status_code == 410
            second = (await client.post(PREFIX + "/tasks/" + uuid.uuid4().hex)).json()[
                "task_handle"
            ]
            assert second != handle
            await client.delete(PREFIX + "/tasks/" + handle)
            assert_ok(await encode(client, p, overrides={"task_handle": second}))
            assert encoder.states[-1] is not encoder.states[0]
            assert_idle(app)
        finally:
            await client.aclose()
            await app.close()

    asyncio.run(run())


@pytest.mark.parametrize("count", [0, 1, 10, 22, 33, 100])
def test_rgb_framing_budget_bounds_real_safetensors(count):
    from sglang.multimodal_gen.runtime.conditioning.contract import (
        rgb_request_framing_bytes,
    )
    from sglang.multimodal_gen.runtime.conditioning.wire import encode_request

    items = [
        dict(
            type="image",
            frames=torch.zeros((3, 1, 1 + i % 3, 1 + i % 7), dtype=torch.uint8),
        )
        for i in range(count)
    ]
    body = encode_request("prompt", items, "a" * 64)
    header_size = struct.unpack(">I", body[:4])[0]
    rgb = sum(item["frames"].numel() for item in items)
    assert len(body) - header_size - rgb <= rgb_request_framing_bytes(
        count, 7, max(1, rgb)
    )


def test_full_rpc_header_and_rgb_budget_reach_real_asgi_encode():
    from sglang.multimodal_gen.runtime.conditioning.wire import encode_request

    async def run():
        calls = []

        def encoder(value):
            calls.append(value.prompt)
            return dict(
                context=torch.zeros(1, 5120, dtype=torch.bfloat16),
                row_role=torch.zeros(1, dtype=torch.int64),
            )

        p, app, client, _ = components(
            encoder=encoder,
            configured_limits=limits(
                max_media=22,
                max_input_bytes=66,
                max_decoded_pixels=22,
                max_side=1,
                max_request_bytes=None,
            ),
        )
        items = [
            dict(type="image", frames=torch.zeros(3, 1, 1, 1, dtype=torch.uint8))
            for _ in range(22)
        ]
        first = encode_request("x", items, p.profile_id)
        prompt = "x" * (65536 - struct.unpack(">I", first[:4])[0] + 1)
        body = encode_request(prompt, items, p.profile_id)
        assert struct.unpack(">I", body[:4])[0] == 65536
        assert len(body) > 66 + 65536 + 1024
        assert len(body) <= app.limits.max_request_bytes
        try:
            # Actual wire size equal to an explicit request limit is legal.
            app.limits = replace(app.limits, max_request_bytes=len(body))
            response = await client.post(
                PREFIX + "/encode",
                content=body,
                headers={"Content-Type": "application/octet-stream"},
            )
            assert_ok(response)
            assert calls == [prompt]
            assert_idle(app)
            app.limits = replace(app.limits, max_request_bytes=len(body) - 1)
            response = await client.post(
                PREFIX + "/encode",
                content=body,
                headers={"Content-Type": "application/octet-stream"},
            )
            assert response.status_code == 413
            assert calls == [prompt]
            assert_idle(app)
        finally:
            await client.aclose()
            await app.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "items",
    [
        [],
        [media()],
        [media(0, "video", origin="source_s0")] + [media(i) for i in range(1, 7)],
        [media(0, "video", origin="explicit", clip={"start": 3, "end": 3})]
        + [media(i) for i in range(1, 7)],
        [media(0, "video", origin="source_s0")] + [media(i) for i in range(1, 22)],
    ],
)
def test_t1_real_encode_route_matches_complete_local_forward(items):
    async def run():
        p, app, client, encoder = components()
        try:
            expected = encode_condition(
                encoder.modules, encoder.runtime, "  prompt é  ", items, max_rows=512
            )
            header, tensors = assert_ok(
                await encode(client, p, items, prompt="  prompt é  ")
            )
            assert header["profile_id"] == p.profile_id
            assert header["input_digest"] == reference_digest(
                "  prompt é  ", items, p.profile_id
            )
            for key in expected:
                assert torch.equal(expected[key], tensors[key])
            assert len(encoder.calls) == 1
            assert_idle(app)
        finally:
            await client.aclose()
            await app.close()

    asyncio.run(run())


def test_t1b_wire_preserves_every_byte_type_fps_and_order():
    from sglang.multimodal_gen.runtime.conditioning import wire

    items = [media(3), media(8, "video"), media(5)]
    p = profile()
    body = wire.encode_request("prompt", items, p.profile_id, deadline_ms=10000)
    header, tensors = unpack(body)
    for i, item in enumerate(items):
        assert header["media"][i]["type"] == item["type"]
        assert header["media"][i]["fps"] == item["fps"]
        assert torch.equal(tensors[header["media"][i]["tensor"]], item["frames"])


def test_t1c_identity_ignores_only_non_effective_metadata_and_hits_cache():
    async def run():
        p, app, client, encoder = components()
        a = [media(1, "video", id="a", origin="source_s0", clip={"start": 0, "end": 0})]
        b = [
            {
                **a[0],
                "id": "b",
                "origin": "explicit",
                "clip": {"start": 7, "end": 7},
                "fps": 60,
            }
        ]
        try:
            assert inputs.content_identity(
                "p", a, p.profile_id
            ) == inputs.content_identity("p", b, p.profile_id)
            assert inputs.content_identity(
                "p", a, p.profile_id
            ) != inputs.content_identity("p", [media(2, "video")], p.profile_id)
            assert inputs.content_identity(
                "p", a, p.profile_id
            ) != inputs.content_identity("p", [media(1)], p.profile_id)
            assert_ok(await encode(client, p, a, prompt="p"))
            header, _ = assert_ok(await encode(client, p, b, prompt="p"))
            assert header["cache"] == "hit" and len(encoder.calls) == 1
        finally:
            await client.aclose()
            await app.close()

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["digest", "profile", "empty_prompt"])
def test_t3_rejects_identity_and_profile_mismatch_before_encoder(kind):
    async def run():
        p, app, client, encoder = components()
        try:
            kwargs = (
                {"digest": "f" * 64}
                if kind == "digest"
                else {"overrides": {"profile_id": "f" * 64}}
                if kind == "profile"
                else {"prompt": ""}
            )
            response = await encode(client, p, [media()], **kwargs)
            expected = 409 if kind == "profile" else 400
            assert response.status_code == expected
            assert response.json()["code"] == (
                "CONDITION_PROFILE_MISMATCH"
                if kind == "profile"
                else "CONDITION_INVALID"
            )
            assert encoder.calls == []
            assert_idle(app)
        finally:
            await client.aclose()
            await app.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "change",
    [
        "zero_fps",
        "negative_fps",
        "nan_fps",
        "inf_fps",
        "multi_frame",
        "dtype",
        "type",
        "too_many",
        "pixels",
        "empty_side",
    ],
)
def test_t3b_bad_media_never_reaches_encode_ids(change):
    async def run():
        p, app, client, encoder = components()
        item = media()
        if change.endswith("fps"):
            item["fps"] = {
                "zero_fps": 0,
                "negative_fps": -1,
                "nan_fps": float("nan"),
                "inf_fps": float("inf"),
            }[change]
        elif change == "multi_frame":
            item["frames"] = torch.zeros(3, 2, 2, 4, dtype=torch.uint8)
        elif change == "dtype":
            item["frames"] = item["frames"].float()
        elif change == "type":
            item["type"] = "audio"
        elif change == "pixels":
            item["frames"] = torch.zeros(3, 1, 8193, 1, dtype=torch.uint8)
        elif change == "empty_side":
            item["frames"] = torch.zeros(3, 1, 0, 4, dtype=torch.uint8)
        items = [item] * 23 if change == "too_many" else [item]
        try:
            response = await encode(
                client, p, items, allow_nonfinite=change in {"nan_fps", "inf_fps"}
            )
            assert response.status_code == 400
            assert response.json()["code"] == "CONDITION_INVALID"
            assert encoder.calls == [] and not encoder.started.is_set()
            assert_idle(app)
        finally:
            await client.aclose()
            await app.close()

    asyncio.run(run())


def test_t3b_image_missing_fps_and_whitespace_prompt_are_accepted():
    async def run():
        p, app, client, _ = components()
        item = media()
        del item["fps"]
        try:
            assert inputs.validate_condition_media([item])[0]["fps"] == 1.0
            assert_ok(await encode(client, p, [item], prompt="   "))
        finally:
            await client.aclose()
            await app.close()

    asyncio.run(run())


async def wait_until(predicate):
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0.001)
    assert False, "observable test state did not arrive"


def test_t5_t6_t18_eight_waiters_encode_once_and_stats_match():
    async def run():
        p, app, client, encoder = components()
        encoder.release.clear()
        tasks = []
        try:
            tasks = [
                asyncio.create_task(encode(client, p, [media()])) for _ in range(8)
            ]
            await wait_until(
                lambda: encoder.started.is_set() or all(t.done() for t in tasks)
            )
            if not encoder.started.is_set():
                assert_ok(await tasks[0])
            await asyncio.sleep(0.03)
            encoder.release.set()
            results = [assert_ok(r) for r in await asyncio.gather(*tasks)]
            states = [h["cache"] for h, _ in results]
            assert states.count("miss") == 1 and states.count("joined") == 7
            assert len(encoder.calls) == 1
            assert all(
                torch.equal(results[0][1]["context"], x[1]["context"]) for x in results
            )
            header, _ = assert_ok(await encode(client, p, [media()]))
            assert header["cache"] == "hit" and len(encoder.calls) == 1
            stats = await client.get(PREFIX + "/stats")
            assert stats.status_code == 200
            assert {k: stats.json()[k] for k in ("miss", "joined", "hit")} == {
                "miss": 1,
                "joined": 7,
                "hit": 1,
            }
            assert stats.json()["queue_depth"] == 0
            assert stats.json()["response_bytes"] > 0
            registry = app.metrics.registry
            assert (
                registry.get_sample_value(
                    "world_model_conditioning_encodes_total",
                    {"prefix_outcome": "unknown"},
                )
                == 1
            )
            for source, count in (("encoding", 1), ("joined", 7), ("result_cache", 1)):
                assert (
                    registry.get_sample_value(
                        "world_model_conditioning_results_total", {"source": source}
                    )
                    == count
                )
            assert_idle(app)
        finally:
            encoder.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.aclose()
            await app.close()

    asyncio.run(run())


def test_t7_lru_byte_bound_and_recency():
    from sglang.multimodal_gen.runtime.conditioning.cache import ConditioningCache

    a = {
        "context": torch.ones(1, 5120, dtype=torch.bfloat16),
        "row_role": torch.ones(1, dtype=torch.int64),
    }
    size = sum(t.numel() * t.element_size() for t in a.values())
    cache = ConditioningCache(size * 2)
    cache.put("a", a)
    cache.put("b", a)
    assert cache.get("a") is not None
    cache.put("c", a)
    assert (
        cache.get("b") is None
        and cache.get("a") is not None
        and cache.get("c") is not None
    )
    assert cache.current_bytes == size * 2
    cache.put(
        "oversize",
        {"context": a["context"].repeat(3, 1), "row_role": a["row_role"].repeat(3)},
    )
    assert cache.get("oversize") is None and cache.current_bytes <= size * 2


async def raw_request(
    app, body, headers, *, path=PREFIX + "/encode", receive_counter=None, gate=None
):
    sent, consumed = [], False

    async def receive():
        nonlocal consumed
        if receive_counter is not None:
            receive_counter.append(1)
        if not consumed:
            consumed = True
            return {"type": "http.request", "body": body, "more_body": False}
        if gate is not None:
            await gate.wait()
            return {"type": "http.disconnect"}
        await asyncio.Future()

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [
                (b"authorization", b"Bearer secret"),
                (b"content-type", b"application/octet-stream"),
                *headers,
            ],
        },
        receive,
        send,
    )
    start = next((m for m in sent if m["type"] == "http.response.start"), None)
    return start["status"] if start else None, b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )


@pytest.mark.parametrize(
    "case,expected",
    [
        ("declared_oversize", 413),
        ("transfer", 400),
        ("actual_oversize", 400),
        ("header_oversize", 400),
        ("missing_length", 400),
    ],
)
def test_t9_framing_rejections_release_admission_without_reading_oversize(
    case, expected
):
    async def run():
        p, app, client, encoder = components()
        body = request_body("p", [], p.profile_id)
        headers = [(b"content-length", str(len(body)).encode())]
        if case == "declared_oversize":
            headers = [(b"content-length", str(2 << 20).encode())]
        elif case == "transfer":
            headers.append((b"transfer-encoding", b"chunked"))
        elif case == "actual_oversize":
            headers = [(b"content-length", str(len(body) - 1).encode())]
        elif case == "header_oversize":
            body = struct.pack(">I", 65537) + b"{"
            headers = [(b"content-length", str(len(body)).encode())]
        elif case == "missing_length":
            headers = []
        reads = []
        try:
            status, data = await raw_request(app, body, headers, receive_counter=reads)
            assert status == expected, data
            assert json.loads(data)["code"] == "CONDITION_INVALID"
            if case == "declared_oversize":
                assert reads == []
            assert encoder.calls == []
            assert_idle(app)
        finally:
            await client.aclose()
            await app.close()

    asyncio.run(run())


@pytest.mark.parametrize("token", [None, "wrong"])
def test_t13_unauthorized_token(token):
    async def run():
        p, app, client, encoder = components()
        client.headers.clear()
        if token:
            client.headers["Authorization"] = "Bearer " + token
        try:
            response = await encode(client, p)
            assert response.status_code == 401
            assert response.json()["code"] == "CONDITION_UNAUTHORIZED"
            assert encoder.calls == []
            assert_idle(app)
        finally:
            await client.aclose()
            await app.close()

    asyncio.run(run())


def test_t13_stats_tenant_isolation_and_admin_global():
    async def run():
        p, app, client, encoder = components()
        try:
            assert_ok(await encode(client, p, prompt="tenant one"))
            client.headers["Authorization"] = "Bearer second"
            assert_ok(await encode(client, p, prompt="tenant two"))
            assert_ok(await encode(client, p, prompt="tenant two"))
            own = await client.get(PREFIX + "/stats")
            assert (
                own.status_code == 200
                and own.json()["miss"] == 1
                and own.json()["hit"] == 1
            )
            assert "tenants" not in own.json()
            own_cache = own.json()["cache"]
            assert own_cache["entries"] == 1
            assert own_cache["current_bytes"] > 0
            assert own_cache["current_bytes"] <= own_cache["max_bytes"]
            client.headers["Authorization"] = "Bearer secret"
            first = await client.get(PREFIX + "/stats")
            assert first.json()["hit"] == 0
            client.headers["Authorization"] = "Bearer admin"
            global_stats = await client.get(PREFIX + "/stats")
            assert global_stats.json()["miss"] == 2 and global_stats.json()["hit"] == 1
            assert global_stats.json()["cache"]["entries"] == 2
            assert global_stats.json()["cache"]["current_bytes"] == (
                own_cache["current_bytes"] + first.json()["cache"]["current_bytes"]
            )
        finally:
            await client.aclose()
            await app.close()

    asyncio.run(run())


@pytest.mark.parametrize("resource", ["slots", "inflight"])
def test_t8_running_request_admission_bounds_and_release(resource):
    async def run():
        selected = (
            limits(max_active_requests=1, max_active_requests_per_tenant=1)
            if resource == "slots"
            else limits(
                max_inflight_bytes=6 << 20, max_inflight_bytes_per_tenant=6 << 20
            )
        )
        p, app, client, encoder = components(configured_limits=selected)
        encoder.release.clear()
        task = asyncio.create_task(encode(client, p, prompt="first"))
        try:
            await wait_until(lambda: encoder.started.is_set() or task.done())
            if not encoder.started.is_set():
                assert_ok(await task)
            snapshot = app.ledger.snapshot()["global"]
            assert snapshot["slots"] == 1
            assert snapshot["inflight"] >= 512 * (5120 * 2 + 8)
            denied = await encode(
                client, p, prompt="first" if resource == "slots" else "second"
            )
            assert denied.status_code == 429
            assert denied.json()["code"] == "CONDITION_BUSY"
            assert encoder.calls == ["first"]
            encoder.release.set()
            assert_ok(await task)
            assert_idle(app)
        finally:
            encoder.release.set()
            await asyncio.gather(task, return_exceptions=True)
            await client.aclose()
            await app.close()

    asyncio.run(run())


@pytest.mark.parametrize("cached", [False, True])
def test_t8_hit_slot_and_shrunk_inflight_are_retained_until_response_sent(cached):
    async def run():
        p, app, client, encoder = components(
            configured_limits=limits(
                max_active_requests=1, max_active_requests_per_tenant=1
            )
        )
        body = request_body("p", [], p.profile_id)
        sent = asyncio.Event()
        finish = asyncio.Event()
        messages = []
        consumed = False

        async def receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Future()

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.body":
                sent.set()
                await finish.wait()

        task = None
        try:
            if cached:
                assert_ok(await encode(client, p, prompt="p"))
            task = asyncio.create_task(
                app(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": PREFIX + "/encode",
                        "headers": [
                            (b"authorization", b"Bearer secret"),
                            (b"content-type", b"application/octet-stream"),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    },
                    receive,
                    send,
                )
            )
            await wait_until(lambda: sent.is_set() or task.done())
            assert sent.is_set()
            response = next(m for m in messages if m["type"] == "http.response.body")[
                "body"
            ]
            header, _ = unpack(response)
            assert header["cache"] == ("hit" if cached else "miss")
            snapshot = app.ledger.snapshot()["global"]
            assert snapshot["slots"] == 1
            assert 0 < snapshot["inflight"] <= len(body) + len(response)
            assert (await encode(client, p, prompt="p")).status_code == 429
            finish.set()
            await task
            assert_idle(app)
            assert len(encoder.calls) == 1
        finally:
            finish.set()
            if task:
                await asyncio.gather(task, return_exceptions=True)
            await client.aclose()
            await app.close()

    asyncio.run(run())


@pytest.mark.parametrize("leave", ["disconnect", "deadline"])
def test_t10_t12_last_queued_waiter_leaves_without_gpu_work(leave):
    async def run():
        p, app, client, encoder = components()
        encoder.release.clear()
        first = asyncio.create_task(encode(client, p, prompt="running"))
        queued = None
        disconnect = asyncio.Event()
        try:
            await wait_until(lambda: encoder.started.is_set() or first.done())
            if not encoder.started.is_set():
                assert_ok(await first)
            body = request_body(
                "queued",
                [],
                p.profile_id,
                deadline_ms=100 if leave == "deadline" else 10000,
            )
            queued = asyncio.create_task(
                raw_request(
                    app,
                    body,
                    [(b"content-length", str(len(body)).encode())],
                    gate=disconnect,
                )
            )
            for _ in range(100):
                stats = await client.get(PREFIX + "/stats")
                assert stats.status_code == 200
                if stats.json()["queue_depth"] == 1:
                    break
                await asyncio.sleep(0.001)
            assert stats.json()["queue_depth"] == 1
            if leave == "disconnect":
                disconnect.set()
            status, payload = await asyncio.wait_for(queued, 1)
            if leave == "deadline":
                assert (
                    status == 504 and json.loads(payload)["code"] == "CONDITION_TIMEOUT"
                )
            encoder.release.set()
            assert_ok(await first)
            await asyncio.sleep(0.02)
            assert encoder.calls == ["running"]
            assert_idle(app)
            assert (await client.get(PREFIX + "/stats")).json()["queue_depth"] == 0
        finally:
            disconnect.set()
            encoder.release.set()
            await asyncio.gather(
                *(t for t in (first, queued) if t), return_exceptions=True
            )
            await client.aclose()
            await app.close()

    asyncio.run(run())


def test_t11_running_disconnect_finishes_and_populates_cache():
    async def run():
        p, app, client, encoder = components()
        encoder.release.clear()
        body = request_body("running", [], p.profile_id)
        disconnect = asyncio.Event()
        task = asyncio.create_task(
            raw_request(
                app,
                body,
                [(b"content-length", str(len(body)).encode())],
                gate=disconnect,
            )
        )
        try:
            await wait_until(lambda: encoder.started.is_set() or task.done())
            assert encoder.started.is_set()
            disconnect.set()
            await asyncio.wait_for(task, 1)
            encoder.release.set()
            await wait_until(lambda: not app.flights)
            header, _ = assert_ok(await encode(client, p, prompt="running"))
            assert header["cache"] == "hit" and encoder.calls == ["running"]
            assert_idle(app)
        finally:
            disconnect.set()
            encoder.release.set()
            await asyncio.gather(task, return_exceptions=True)
            await client.aclose()
            await app.close()

    asyncio.run(run())


def test_t8_encode_queue_full_rejects_new_work():
    async def run():
        p, app, client, encoder = components(
            configured_limits=limits(max_queued_encodes=1)
        )
        encoder.release.clear()
        first = asyncio.create_task(encode(client, p, prompt="first"))
        second = None
        try:
            await wait_until(lambda: encoder.started.is_set() or first.done())
            if not encoder.started.is_set():
                assert_ok(await first)
            second = asyncio.create_task(encode(client, p, prompt="second"))
            for _ in range(100):
                stats = await client.get(PREFIX + "/stats")
                assert stats.status_code == 200
                if stats.json()["queue_depth"] == 1:
                    break
                await asyncio.sleep(0.001)
            assert stats.json()["queue_depth"] == 1
            third = await encode(client, p, prompt="third")
            assert third.status_code == 429 and third.json()["code"] == "CONDITION_BUSY"
            encoder.release.set()
            assert_ok(await first)
            assert_ok(await second)
            assert encoder.calls == ["first", "second"]
            assert_idle(app)
        finally:
            encoder.release.set()
            await asyncio.gather(
                *(t for t in (first, second) if t), return_exceptions=True
            )
            await client.aclose()
            await app.close()

    asyncio.run(run())


def test_warmup_rgb_s0_matches_online_pixels_and_type(tmp_path):
    from PIL import Image

    a = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    b = np.flip(a, axis=1).copy()
    Image.fromarray(a).save(tmp_path / "ref.png")
    Image.fromarray(b).save(tmp_path / "s0.png")
    prompt, items = cli.load_warmup_input(
        {
            "prompt": "p",
            "images": [str(tmp_path / "ref.png")],
            "s0_image": str(tmp_path / "s0.png"),
        }
    )
    online = [
        dict(
            type="video",
            fps=30,
            origin="source_s0",
            frames=torch.from_numpy(b).permute(2, 0, 1).unsqueeze(1),
        ),
        dict(type="image", frames=torch.from_numpy(a).permute(2, 0, 1).unsqueeze(1)),
    ]
    assert [m["type"] for m in items] == ["video", "image"]
    assert inputs.content_identity(
        prompt, items, profile().profile_id
    ) == inputs.content_identity(prompt, online, profile().profile_id)


def test_configuration_requires_every_limit_and_no_artifact_secrets(tmp_path):
    p = profile()
    (tmp_path / "profile.json").write_bytes(canonical(p.document))
    configured = {f.name: getattr(limits(), f.name) for f in fields(ResourceLimits)}
    config = {
        "tokens": {"secret": {"tenant": "tenant"}},
        "limits": configured,
        "native": {"model_path": "/trusted/model", "gpu_id": 0},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    args = SimpleNamespace(
        config=str(path), profile=str(tmp_path / "profile.json"), command="serve"
    )
    result = cli.load_configuration(args)
    assert len(result) == 3  # config, profile, limits; no artifact store
    for omitted in configured:
        bad = {
            **config,
            "limits": {k: v for k, v in configured.items() if k != omitted},
        }
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="explicitly"):
            cli.load_configuration(args)


def test_warmup_cli_and_removed_artifact_commands():
    parser = cli.parser()
    subparsers = next(
        a
        for a in parser._actions
        if hasattr(a, "choices") and isinstance(a.choices, dict)
    )
    assert set(subparsers.choices) == {"build-profile", "serve", "warmup"}
    args = parser.parse_args(
        [
            "warmup",
            "--config",
            "config.json",
            "--profile",
            "profile.json",
            "--input",
            "inputs.json",
        ]
    )
    assert args.command == "warmup"
    assert not hasattr(args, "cache_root")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))


def test_response_disconnect_while_sender_blocked_releases_request_ownership():
    async def run():
        p, app, client, _ = components()
        body = request_body("p", [], p.profile_id)
        sending, disconnect, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
        read = False

        async def receive():
            nonlocal read
            if not read:
                read = True
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                sending.set()
                await finish.wait()

        task = asyncio.create_task(
            app(
                {
                    "type": "http",
                    "method": "POST",
                    "path": PREFIX + "/encode",
                    "headers": [
                        (b"authorization", b"Bearer secret"),
                        (b"content-type", b"application/octet-stream"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                },
                receive,
                send,
            )
        )
        try:
            await wait_until(sending.is_set)
            assert app.ledger.snapshot()["global"]["slots"] == 1
            disconnect.set()
            await asyncio.sleep(0.03)
            assert task.done(), (
                "a disconnected sender must not retain request ownership"
            )
            await task
            assert_idle(app)
        finally:
            disconnect.set()
            finish.set()
            await asyncio.gather(task, return_exceptions=True)
            await client.aclose()
            await app.close()

    asyncio.run(run())


def test_sender_exception_never_starts_a_second_response_and_releases_ledger():
    async def run():
        p, app, client, _ = components()
        body = request_body("p", [], p.profile_id)
        messages, read = [], False

        async def receive():
            nonlocal read
            if not read:
                read = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Future()

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.body":
                raise OSError("transport closed")

        try:
            with pytest.raises(OSError, match="transport closed"):
                await app(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": PREFIX + "/encode",
                        "headers": [
                            (b"authorization", b"Bearer secret"),
                            (b"content-type", b"application/octet-stream"),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    },
                    receive,
                    send,
                )
            assert sum(m["type"] == "http.response.start" for m in messages) == 1
            assert_idle(app)
        finally:
            await client.aclose()
            await app.close()

    asyncio.run(run())


def test_error_response_retains_slot_until_sender_finishes():
    async def run():
        p, app, client, encoder = components()
        body = request_body("p", [], p.profile_id, digest="f" * 64)
        sending, finish = asyncio.Event(), asyncio.Event()
        read, messages = False, []

        async def receive():
            nonlocal read
            if not read:
                read = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Future()

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.body":
                sending.set()
                await finish.wait()

        task = asyncio.create_task(
            app(
                {
                    "type": "http",
                    "method": "POST",
                    "path": PREFIX + "/encode",
                    "headers": [
                        (b"authorization", b"Bearer secret"),
                        (b"content-type", b"application/octet-stream"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                },
                receive,
                send,
            )
        )
        try:
            await wait_until(sending.is_set)
            assert (
                next(m for m in messages if m["type"] == "http.response.start")[
                    "status"
                ]
                == 400
            )
            assert app.ledger.snapshot()["global"]["slots"] == 1
            assert encoder.calls == []
            finish.set()
            await task
            assert_idle(app)
        finally:
            finish.set()
            await asyncio.gather(task, return_exceptions=True)
            await client.aclose()
            await app.close()

    asyncio.run(run())


def test_last_running_waiter_disconnect_transfers_retained_bytes_until_native_drain():
    async def run():
        p, app, client, encoder = components()
        encoder.release.clear()
        item = media(9)
        body = request_body("running", [item], p.profile_id)
        disconnect = asyncio.Event()
        task = asyncio.create_task(
            raw_request(
                app,
                body,
                [(b"content-length", str(len(body)).encode())],
                gate=disconnect,
            )
        )
        try:
            await wait_until(encoder.started.is_set)
            disconnect.set()
            await asyncio.wait_for(task, 1)
            snapshot = app.ledger.snapshot()["global"]
            assert snapshot["slots"] == 0
            assert snapshot["inflight"] >= item["frames"].numel() + 512 * (
                5120 * 2 + 8
            ), (
                "uninterruptible native buffers must remain charged after the last waiter leaves"
            )
            encoder.release.set()
            await wait_until(lambda: not app.flights)
            assert_idle(app)
            header, _ = assert_ok(await encode(client, p, [item], prompt="running"))
            assert header["cache"] == "hit" and encoder.calls == ["running"]
        finally:
            disconnect.set()
            encoder.release.set()
            await asyncio.gather(task, return_exceptions=True)
            await client.aclose()
            await app.close()

    asyncio.run(run())
