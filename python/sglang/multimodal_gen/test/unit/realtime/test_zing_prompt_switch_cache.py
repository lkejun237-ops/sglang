# SPDX-License-Identifier: Apache-2.0
"""Shared startup/switch cache: real CPU KV arithmetic and transport stand-ins."""

import time
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.conditioning.worker import WorkerConditioningState
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    source_session,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
    preparation_key,
)
from sglang.multimodal_gen.test.unit.realtime.test_zing_preparation_cache import inputs
from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import (  # noqa: F401
    cpu_cache,
    cpu_dma,
    dense_kv,
    disk_commit,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import model as model_fixture


@pytest.fixture
def model():
    return model_fixture.__wrapped__()


def test_lru_reuses_preallocated_slots_and_waits_for_fill(cpu_dma):
    stores = [NS(k=torch.zeros(8, 2, 3), v=torch.zeros(8, 2, 3)) for _ in range(2)]
    owner = cpu_cache(stores, entries=4, rows=8)
    pointers = [owner.pages[:, :, i].data_ptr() for i in range(4)]

    def fill(key, value, rows=3):
        cache = NS(
            prefix=NS(
                positions=torch.zeros(rows, 3),
                layers=[
                    (torch.full((rows, 2, 3), value), torch.full((rows, 2, 3), -value))
                    for _ in stores
                ],
            ),
            _model_signature="model",
        )
        owner.fill(key, inputs(), "condition", "request", cache)
        return owner.lookup(key)

    for index, key in enumerate("ABCD"):
        entry = fill(key, float(index + 1))
        assert entry.page_ids == (index,)
        assert cpu_dma[0][-1].synchronized
    assert dense_kv(owner.lookup("A").data)[0, 0, 0, 0, 0] == 1
    entry = fill("E", 5.0, rows=7)
    assert entry.page_ids == (1,)  # B is oldest after A was accessed.
    assert owner.lookup("B") is None
    assert owner.lookup("A") is not None
    assert owner.evictions == 1
    assert torch.equal(
        dense_kv(entry.data)[:, 0], torch.full_like(dense_kv(entry.data)[:, 0], 5)
    )
    assert torch.equal(
        dense_kv(entry.data)[:, 1], torch.full_like(dense_kv(entry.data)[:, 1], -5)
    )
    assert [owner.pages[:, :, i].data_ptr() for i in range(4)] == pointers
    # Repeated fill neither allocates nor evicts an existing ready entry.
    fill("E", 99.0)
    assert dense_kv(owner.lookup("E").data)[0, 0, 0, 0, 0] == 99
    assert owner.evictions == 1


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_reject_invalid_byte_capacity(budget):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        Block0PreparationCache,
    )

    with pytest.raises(ValueError, match="positive integer"):
        Block0PreparationCache(None, max_bytes=budget, identity={}, disk_cache=None)


def test_partial_sp_hit_falls_back(cpu_dma, monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        preparation_cache,
    )

    owner = cpu_cache([NS(k=torch.zeros(8, 2, 3))])
    owner.entries["A"] = NS(generation="A-generation")
    monkeypatch.setattr(
        preparation_cache, "coordinate_validation", lambda value: [value, ("A", False)]
    )
    assert owner.lookup("A") is None
    assert owner.hits == 0 and owner.misses == 1


def test_prefetch_cache_hint_has_no_cuda_collective_or_lru_side_effect(cpu_dma):
    owner = cpu_cache([NS(k=torch.zeros(8, 2, 3))])
    owner.entries["A"] = object()
    owner.entries["B"] = object()
    owner.pending = object()  # Not yet published; the hint must not finish it.
    owner.finish_fill = Mock(side_effect=AssertionError("CUDA fill forbidden"))
    owner.disk_cache = NS(may_contain=Mock(return_value=False))
    assert owner.may_contain("A")
    assert not owner.may_contain("C")
    assert list(owner.entries) == ["A", "B"]
    assert owner.hits == owner.misses == 0
    owner.finish_fill.assert_not_called()


@pytest.mark.parametrize("scenario", ["hit", "miss", "evicted", "partial-rank"])
def test_source_prefetch_checks_cache_before_rpc(
    cpu_dma, tmp_path, monkeypatch, scenario
):
    import msgspec.msgpack
    import zmq

    from sglang.multimodal_gen.runtime.conditioning.worker import RemoteConditioner
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        preparation_cache,
    )

    state = source_session.ZingSourceStreamState()
    state.inputs = inputs()
    state.inputs.condition_media = []
    state.block0_input_request = dict(prompt="A", media=[], conditions=[])
    codecs = NS(remote_read_fs=None)
    state.block0_key_args = (codecs, None, dict(width=32, height=32, fps=24), {})
    owner = state.block0_cache = cpu_cache([NS(k=torch.zeros(8, 2, 3))])
    client = NS(
        clock=time.monotonic,
        timeout_s=2,
        validate_input=Mock(),
        encode=Mock(return_value=state.inputs.positive),
    )
    provider = RemoteConditioner(
        client,
        "a" * 64,
        rank=0,
        coordinate=lambda value: [value],
        broadcast_receipt=lambda value: value,
        broadcast_tensor=lambda value, name: value,
    )
    condition = WorkerConditioningState()
    stage = NS(codecs=codecs, conditioning_provider=provider)
    stage.encode_condition_for_batch = Mock(
        side_effect=lambda batch, prompt, media: provider(
            prompt, media, device="cpu", state=condition
        )
    )
    key, _ = state._block0_update_key(stage, "B", {})
    if scenario != "miss":
        owner.entries[key] = NS(
            inputs=state.inputs,
            conditioning_identity="cached-condition",
            generation="fixture",
        )
    state.sampler = NS(discard_pending_prompt=Mock(), update_prompt=Mock())
    state._admit_prompt_update = Mock(return_value="pool-request")
    command = dict(prompt="B", event_id=1)
    batch = NS(
        extra={"zing_prompt_update": command},
        sampling_params=NS(prompt="B"),
        realtime_event_id=1,
        realtime_prompt_version=1,
        session=NS(get_or_create_state=lambda _: condition),
    )
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.LINGER, 0)
    endpoint = "ipc://" + str(tmp_path / "prefetch.sock")
    socket.bind(endpoint)
    try:
        state._start_conditioning_prefetch(
            stage, batch, endpoint, {"conditioning_prefetch_max_bytes": 1 << 20}
        )
        prefetch = state.conditioning_prefetch
        socket.send(msgspec.msgpack.encode(command))
        deadline = time.monotonic() + 5
        while not (prefetch.encodes if scenario == "miss" else prefetch.cache_skips):
            assert time.monotonic() < deadline, "Prefetch did not inspect the cache"
            time.sleep(0.01)
        assert client.encode.call_count == int(scenario == "miss")
        if scenario == "evicted":
            owner.entries.clear()
        elif scenario == "partial-rank":
            monkeypatch.setattr(
                preparation_cache,
                "coordinate_validation",
                lambda value: [value, (value[0], False)],
            )
        assert state._update_prompt(stage, batch) is None
        assert client.encode.call_count == int(scenario != "hit")
        assert stage.encode_condition_for_batch.call_count == int(scenario != "hit")
        state.sampler.update_prompt.assert_called_once()
    finally:
        if state.conditioning_prefetch is not None:
            state.conditioning_prefetch.close()
        socket.close()
        context.term()


@pytest.mark.parametrize("enabled", [False, True])
def test_post_switch_fill_waits_for_new_epoch_output(enabled):
    state = source_session.ZingSourceStreamState()
    prefix = object()
    state.sampler = NS(cache=NS(prefix=prefix), committed_pixel=21)
    state.block0_cache = NS(fill=Mock(), register=Mock())
    state.cache_write = enabled
    state.block0_key = "B"
    state.block0_inputs = inputs()
    state.block0_condition_identity = "B-condition"
    state.block0_request = object()
    state.ledger = NS(output_frames=17)
    state._remember_block0(prefix)
    state._fill_block0_after_output()
    state.ledger.output_frames = 21  # Draining only old prompt output.
    state._fill_block0_after_output()
    state.block0_cache.fill.assert_not_called()
    state.ledger.output_frames = 22
    state._fill_block0_after_output()
    state._fill_block0_after_output()
    assert state.block0_cache.fill.call_count == int(enabled)
    assert state.block0_cache.register.call_count == int(enabled)
    if enabled:
        assert state.block0_cache.fill.call_args.args[0] == "B"


def test_key_reuses_accepted_media_after_uri_changes(tmp_path):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        preparation_request,
    )

    path = tmp_path / "ref.png"
    path.write_bytes(b"accepted")
    codecs = NS(remote_read_fs=None)
    raw = dict(prompt="A", media=[dict(type="image", uri=str(path))])
    accepted = preparation_request(codecs, raw)
    config = dict(width=32, height=32, fps=24)
    a = preparation_key(codecs, accepted, None, config, {})
    assert a == preparation_key(codecs, raw, None, config, {})
    path.write_bytes(b"replaced")
    assert preparation_key(codecs, raw, None, config, {}) != a
    assert preparation_key(codecs, {**accepted, "prompt": "A"}, None, config, {}) == a
    assert preparation_key(codecs, {**accepted, "prompt": "B"}, None, config, {}) != a


@torch.no_grad()
@pytest.mark.parametrize("use_drain", [False, True])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("tier", ["l1", "l2"])
def test_startup_a_switch_b_a_reuses_kv_and_preserves_outputs(
    model, cpu_dma, use_drain, joint, tier, tmp_path, monkeypatch
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
        ZingH3Packer,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
        ZingH3Sampler,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import fixture_provenance
    from sglang.multimodal_gen.test.unit.test_zing_h3_kv_pool_cache import (
        _pointers,
        _storage,
    )

    base = inputs()
    base.visual = {}
    base.positive = dict(
        context=torch.randn(3, 12).bfloat16(), row_role=torch.zeros(3, dtype=torch.long)
    )
    positive_b = dict(
        context=torch.cat(
            (base.positive["context"] + 0.25, base.positive["context"][:2] - 0.1)
        ),
        row_role=torch.zeros(5, dtype=torch.long),
    )
    base.visual_shape = (30, 2, 2)
    source = torch.randn(1, 2, 30, 2, 2)
    noise = torch.randn_like(source)
    conditions = {"A": base.positive, "B": positive_b}
    if tier == "l2":
        monkeypatch.setenv("SGLANG_ZING_DISK_CACHE_DIR", str(tmp_path / "disk"))
    else:
        monkeypatch.delenv("SGLANG_ZING_DISK_CACHE_DIR", raising=False)

    def run(cached):
        packer = ZingH3Packer(
            model.arch, num_frame_per_block=[3, 2], split_first_latent=True
        )
        sampler = ZingH3Sampler(
            model,
            packer,
            solver="few_step",
            denoising_step_list=[1000, 500],
            window_chunks=6,
            sink_chunks=3,
            joint_forward="cross_chunk" if joint else "off",
        )
        stores = _storage(model, 256)
        pointers = _pointers(stores)
        cache = cpu_cache(stores, entries=1 if tier == "l2" else 6) if cached else None
        _, _, prompt_joint = source_session._source_key_policy(
            NS(block0_cache=cache, codecs=NS(video_decoder_name="lynn")),
            sampler,
            1,
            {"prompt_joint": joint},
        )
        ready = SourceReadySession(
            sampler,
            base,
            seed=123,
            max_pending_chunks=1,
            max_pending_bytes=1 << 20,
            acquire_storage=lambda *_: stores,
            initial_prompt_event_id=0,
            prompt_joint=prompt_joint,
        )
        state = source_session.ZingSourceStreamState()
        state.sampler, state.inputs = ready, base
        state.ledger = NS(output_frames=0)
        state.block0_cache = cache
        state.cache_write = cached
        state.block0_inputs = base
        state.block0_condition_identity = "A-condition"
        state.block0_request = "pool-request"
        state.block0_input_request = dict(prompt="A", media=[], conditions=[])
        codecs = NS(remote_read_fs=None)
        config = dict(width=32, height=32, fps=24)
        state.block0_key_args = (codecs, None, config, {})
        state.block0_key = preparation_key(
            codecs, state.block0_input_request, None, config, {}
        )
        if cached:
            ready.capture_initial_prefix = state._remember_block0
            ready.prepare_prompt_prefix = state._prepare_block0_switch
            if joint:
                ready.prefix_loader = state._load_block0_prefix
        condition_state = WorkerConditioningState()
        stage = NS(
            codecs=codecs,
            encode_condition_for_batch=Mock(
                side_effect=lambda batch, prompt, media: conditions[prompt]
            ),
            kv_pool=NS(acquire=Mock()),
        )
        state._admit_prompt_update = Mock(return_value="pool-request")
        results, snapshots, histories = [], [], []
        prefix_calls = 0
        forward = sampler._forward
        switch_branches = []
        predict_joint = sampler._predict_joint

        def counted_joint(packs, phases, *args, **kwargs):
            if len(packs) in (5, 6):
                switch_branches.append(len(packs))
            return predict_joint(packs, phases, *args, **kwargs)

        sampler._predict_joint = counted_joint

        def counted(packed, cache, phase):
            nonlocal prefix_calls
            prefix_calls += phase == "prefix"
            return forward(packed, cache, phase)

        sampler._forward = counted
        for i in range(12):
            state._fill_block0_after_output()
            if cached and tier == "l2":
                # One L1 slot forces every A/B return through actual disk IO.
                # Await background writes only in this CPU correctness test.
                cache.finish_fill()
                cache.wait_for_disk()
            applied = ready.applied_prompt
            if applied is not None and applied.inputs is not None:
                state.inputs = applied.inputs
            if ready.prompt_switch_ready:
                # Switch after complete codec cycles so every candidate applies.
                prompt = "B" if (ready.committed_latent // 5) % 2 else "A"
                batch = NS(
                    extra={},
                    sampling_params=NS(prompt=prompt),
                    realtime_event_id=i,
                    realtime_prompt_version=i,
                    session=NS(get_or_create_state=lambda _: condition_state),
                )
                if cached and tier == "l2":
                    # Explicit preparation is outside the online lookup path.
                    key, _ = state._block0_update_key(stage, prompt, {})
                    cache.manage("preload", [key])
                assert state._update_prompt(stage, batch) is None
            start, end = ready.next_latent, ready.next_latent + ready.next_chunk_size
            ready.append(
                source[:, :, start:end],
                noise[:, :, start:end],
                provenance=fixture_provenance(1, end, start, end),
            )
            rng = torch.get_rng_state().clone()
            result = list(ready.drain_ready())[0] if use_drain else ready.step()
            assert torch.equal(rng, torch.get_rng_state())
            results.append(
                (result.visual.clone(), result.prompt_event_id, result.prompt_version)
            )
            snapshots.append(
                [(k.clone(), v.clone()) for k, v in ready.cache.prefix.layers]
            )
            histories.append(
                [
                    tuple(value.clone() for value in ready.cache.history(layer))
                    for layer in range(len(stores))
                ]
            )
            assert _pointers(stores) == pointers
            state.ledger.output_frames = ready.committed_pixel
        assert [row[1] for row in results] == [0, 0, 0, 3, 3, 5, 5, 7, 7, 9, 9, 11]
        state._fill_block0_after_output()
        if cached:
            cache.finish_fill()
            if tier == "l2":
                cache.wait_for_disk()
                assert len(cache.entries) == 1 and cache.evictions >= 4
                assert next(iter(cache.entries.values())).model_signature is None
            else:
                assert len(cache.entries) == 2 and cache.evictions == 0
            assert stage.encode_condition_for_batch.call_count == 1  # Only first B.
            # Startup A is separate. Cold B joins six branches only when joint
            # execution is enabled; subsequent A/B returns use restored KV.
            assert prefix_calls == (1 if joint else 2)
            if joint:
                assert ready.metrics["cached_prefix_hits"] == 4
                assert ready.metrics["cached_prefix_misses"] == 2
                assert ready.metrics["forward_counts"]["prompt_joint"] == 5
                assert switch_branches == [6, 5, 5, 5, 5]
            assert ready.cache.stats["store_allocations"] == 0
        else:
            assert prefix_calls == (1 if joint else 6)
            if joint:
                assert ready.metrics["forward_counts"]["prompt_joint"] == 5
                assert switch_branches == [6, 6, 6, 6, 6]
        if not joint:
            assert switch_branches == []
        stage.kv_pool.acquire.assert_not_called()  # Switch keeps the original slot.
        ready.dispose()
        return results, snapshots, histories

    expected, actual = run(False), run(True)
    for left, right in zip(expected[0], actual[0]):
        torch.testing.assert_close(
            left[0], right[0], rtol=0.02 if joint else 0, atol=0.02 if joint else 0
        )
        assert left[1:] == right[1:]
    for expected_layers, actual_layers in zip(
        expected[1] + expected[2], actual[1] + actual[2]
    ):
        for expected_kv, actual_kv in zip(expected_layers, actual_layers):
            for a, b in zip(expected_kv, actual_kv):
                torch.testing.assert_close(
                    a, b, rtol=0.02 if joint else 0, atol=0.02 if joint else 0
                )


def test_message_request_and_prompt_switch_use_the_same_key():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        preparation_request,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_media_contract import (
        codecs as media_codecs,
    )

    codecs = media_codecs.__wrapped__()
    raw = dict(
        messages=[
            dict(role="user", content=[dict(type="text", text="A")]),
            dict(
                role="target",
                content=[
                    dict(type="video", output=dict(frames=17, height=32, width=32))
                ],
                conditions=[],
            ),
        ]
    )
    config = dict(width=32, height=32, fps=24)
    initial = preparation_key(codecs, raw, None, config, {})
    accepted = preparation_request(codecs, raw)
    assert initial == preparation_key(
        codecs, {**accepted, "prompt": "A"}, None, config, {}
    )
    assert initial != preparation_key(
        codecs, {**accepted, "prompt": "B"}, None, config, {}
    )


@pytest.mark.parametrize("failure", ["media", "admission", "encode"])
def test_failed_update_drops_pending_but_preserves_active_cache(tmp_path, failure):
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

    state = source_session.ZingSourceStreamState()
    active = inputs()
    state.inputs = active
    state.block0_key = "active-A"
    state.block0_input_request = dict(prompt="A", media=[], conditions=[])
    state.pending_block0 = "older-pending"
    state.sampler = NS(discard_pending_prompt=Mock(), update_prompt=Mock())
    codecs = NS(remote_read_fs=None)
    state.block0_key_args = (codecs, None, dict(width=32, height=32, fps=24), {})
    state.block0_cache = NS(
        lookup=Mock(
            return_value=NS(inputs=active, conditioning_identity="cached")
            if failure == "admission"
            else None
        )
    )
    stage = NS(
        codecs=codecs,
        encode_condition_for_batch=Mock(
            side_effect=ConditionError("CONDITION_INVALID", "Encoding failed")
        ),
    )
    state._admit_prompt_update = Mock(
        side_effect=ConditionError("CONDITION_INVALID", "This update is too large")
    )
    command = dict(prompt="B", event_id=3)
    if failure == "media":
        command["references"] = [
            dict(type="image", id="new", uri=str(tmp_path / "missing.png"))
        ]
    batch = NS(
        extra={"zing_prompt_update": command},
        sampling_params=NS(prompt="A"),
        realtime_event_id=0,
    )
    notice = state._update_prompt(stage, batch)
    assert notice == dict(code="CONDITION_INVALID", event_id=3)
    state.sampler.discard_pending_prompt.assert_called_once()
    state.sampler.update_prompt.assert_not_called()
    assert state.pending_block0 is None
    assert state.inputs is active and state.block0_key == "active-A"
    if failure == "media":
        state.block0_cache.lookup.assert_not_called()
    if failure == "admission":
        stage.encode_condition_for_batch.assert_not_called()


def disk_cache(tmp_path, monkeypatch, **env):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
        Block0DiskCache,
    )

    root = tmp_path / "disk"
    root.mkdir(parents=True, exist_ok=True)
    return Block0DiskCache(
        root,
        max_bytes=1 << 30,
        identity={"model": {"fixture": "model"}, "layout": {"sp": 1}},
    )


def test_disk_cache_roundtrip_and_usage_snapshot(tmp_path, monkeypatch):
    cache = disk_cache(tmp_path, monkeypatch)
    key = "abcdef012345"
    kv, _ = disk_commit(cache, key, value=2.5)
    folder = cache._entry_dir(key)
    assert (folder / "rank-0/ready.json").is_file()
    assert not any(cache.tmp_root.iterdir())
    usage = cache.collect()
    assert usage["root_entries"] == 1
    assert usage["root_bytes"] > kv.numel() * kv.element_size()
    hit = cache.lookup(key, 0)
    restored = torch.zeros_like(kv)
    assert hit.load_kv_into(restored) is restored
    assert torch.equal(restored, kv)
    assert hit.conditions["conditioning_identity"] == f"{key}-condition"
    assert hit.manifest["early_key"] == key


def test_disk_prefetch_hint_is_local_read_only_and_not_a_validated_hit(
    tmp_path, monkeypatch
):
    cache = disk_cache(tmp_path, monkeypatch)
    key = "abcdef012345"
    assert not cache.may_contain(key)
    disk_commit(cache, key)
    cache.collect()
    index = cache.usage_path.read_bytes()
    folder = cache._entry_dir(key)
    assert cache.may_contain(key)
    assert cache.usage_path.read_bytes() == index
    cache.local_rank = 1
    assert not cache.may_contain(key)  # Another rank's KV is not this shard.
    cache.local_rank = 0
    (folder / "rank-0/conditions.pt").write_bytes(b"corrupt")
    assert cache.may_contain(key)  # A hint must still be validated by the owner.
    assert cache.lookup(key) is None
    (folder / "rank-0/ready.json").unlink()
    assert not cache.may_contain(key)
    assert not cache.may_contain("../escape")


def test_usage_snapshot_corruption_does_not_affect_hits_and_gc_repairs_it(
    tmp_path, monkeypatch
):
    cache = disk_cache(tmp_path, monkeypatch)
    key = "012345abcdef"
    disk_commit(cache, key)
    assert cache.lookup(key) is not None
    assert cache.list_entries()["root_bytes"] is None
    cache.usage_path.write_text("broken")
    assert cache.lookup(key) is not None
    cache.collect()
    assert cache.list_entries()["total_entries"] == 1


def test_disk_lru_evicts_only_during_gc(tmp_path, monkeypatch):
    cache = disk_cache(tmp_path, monkeypatch)
    for key in ("a1", "b2", "c3"):
        disk_commit(cache, key)
    usage = cache.collect()
    assert len(usage["entries"]) == 3
    cache.max_bytes = int(usage["root_bytes"] * 0.6)
    disk_commit(cache, "d4")
    assert len(cache._scan()) == 4
    usage = cache.collect()
    assert usage["root_bytes"] <= cache.max_bytes * cache.gc_low_watermark
    assert not cache._entry_dir("a1").exists()
    assert cache.lookup("d4") is not None


def test_l2_disk_hit_promotes_into_l1_and_matches_values(
    tmp_path, monkeypatch, cpu_dma
):
    monkeypatch.setenv("SGLANG_ZING_DISK_CACHE_DIR", str(tmp_path / "disk"))
    stores = [NS(k=torch.zeros(8, 2, 3), v=torch.zeros(8, 2, 3)) for _ in range(2)]
    owner = cpu_cache(stores, entries=2, rows=8)
    assert owner.disk_cache.is_enabled()
    prefix = NS(
        positions=torch.zeros(3, 3),
        layers=[
            (torch.full((3, 2, 3), 7.0), torch.full((3, 2, 3), -7.0)) for _ in stores
        ],
    )
    live = NS(prefix=prefix, _model_signature="model")
    owner.fill("L2", inputs(), "L2-condition", "L2-request", live)
    owner.finish_fill()
    owner.wait_for_disk()
    old = owner.entries["L2"]

    fresh = cpu_cache(stores, entries=2, rows=8)
    assert fresh.lookup("L2") is None
    assert fresh.manage("preload", ["L2"])["all_ready"]
    promoted = fresh.entries["L2"]
    assert fresh.hits == 1 and fresh.misses == 1  # Online miss, then explicit preload.
    assert promoted.page_ids == (0,)
    assert promoted.conditioning_identity == "L2-condition"
    assert torch.equal(dense_kv(promoted.data), dense_kv(old.data))


@torch.no_grad()
@pytest.mark.parametrize("legacy_signature", [False, True])
@pytest.mark.parametrize("reuse_cache", [False, True])
def test_l2_restore_rebinds_process_local_signature(
    model, cpu_dma, tmp_path, monkeypatch, legacy_signature, reuse_cache
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
    from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy
    from sglang.multimodal_gen.test.unit.test_zing_h3_kv_pool_cache import _storage

    monkeypatch.setenv("SGLANG_ZING_DISK_CACHE_DIR", str(tmp_path / "disk"))
    stores = _storage(model, 32)
    positions = torch.zeros(3, 3)
    blocks = torch.zeros(3, dtype=torch.long)
    layers = [(torch.randn_like(s.k[:3]), torch.randn_like(s.v[:3])) for s in stores]
    source = ZingH3KVCache(
        len(stores),
        storage=stores,
        window_policy=ZingWindowPolicy(window_chunks=6, sink_chunks=3),
    )
    source.update(positions, blocks, "prefix", 0, layers=layers)
    model._bind_cache(source, positions)
    old_signature = source._model_signature
    owner = cpu_cache(stores, entries=2)
    owner.fill("restart", inputs(), "condition", "pool-request", source)
    owner.finish_fill()
    owner.wait_for_disk()
    path = owner.disk_cache._entry_dir("restart") / "rank-0" / "conditions.pt"
    payload = torch.load(path, weights_only=True)
    conditions = payload["conditions"]
    if legacy_signature:
        # Runtime signatures are no longer serializable objects. Even a plain
        # representation supplied in a new-format file must never be imported.
        conditions["model_signature"] = repr(old_signature)
        torch.save(payload, path)
    else:
        assert "model_signature" not in conditions

    restarted = model_fixture.__wrapped__()  # Same weights, new parameter identities.
    assert all(
        torch.equal(a, b) for a, b in zip(model.parameters(), restarted.parameters())
    )
    destination = ZingH3KVCache(
        len(stores),
        storage=_storage(restarted, 32),
        window_policy=ZingWindowPolicy(window_chunks=6, sink_chunks=3),
    )
    if reuse_cache:
        restarted._bind_cache(destination, torch.empty(0, 3))
        destination.clear()  # Existing prompt-switch boundary.
    fresh = cpu_cache(destination.storage, entries=2)
    assert fresh.lookup("restart") is None
    assert fresh.manage("preload", ["restart"])["all_ready"]
    promoted = fresh.lookup("restart")
    assert promoted is not None
    fresh.restore(destination, promoted).commit(destination)
    restarted._bind_cache(destination, positions)
    assert destination._model_signature != old_signature
    for expected, actual in zip(layers, destination.prefix.layers):
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    # A later in-process model mutation must still be rejected with live KV.
    next(restarted.parameters()).add_(1)
    with pytest.raises(ValueError, match="model changed within a cached session"):
        restarted._bind_cache(destination, positions)


def test_disk_cache_rejects_invalid_root_at_startup(tmp_path, monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
        Block0DiskCache,
    )

    blocked = tmp_path / "file"
    blocked.write_bytes(b"not a directory")
    with pytest.raises(ValueError, match="already exist"):
        Block0DiskCache(
            blocked, max_bytes=1 << 20, identity={"model": {}, "layout": {}}
        )


def test_dispose_drains_transfers_before_releasing_gpu_slot():
    state = source_session.ZingSourceStreamState()
    actions = []
    state.block0_cache = NS(
        finish_fill=lambda: actions.append("fill"),
        mode="overlap",
        data=torch.empty(0),
        entries={},
        hits=0,
        misses=0,
        evictions=0,
    )
    state.block0_restore = NS(
        end=NS(synchronize=lambda: actions.append("restore")),
        begin=NS(elapsed_time=lambda _: 0),
        submit_ms=0,
        ready_at_consumer=False,
        lead_ms=0,
    )
    state._dispose_states = lambda: actions.append("states")
    state.kv_lease = NS(release=lambda: actions.append("release"))
    state.dispose()
    assert actions == ["fill", "restore", "states", "release"]
    assert state.kv_lease is None and state.block0_cache is None
