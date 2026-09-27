# SPDX-License-Identifier: Apache-2.0
"""With host row metadata, Zing forwards and the sampler must not read device tensors back.

CPU tests cannot observe CUDA synchronization, so device-side metadata is
wrapped in a tensor subclass that raises on every operation that would read a
CUDA tensor back to the host.
"""

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.zing_h3 import ZingH3DiTModel
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    _sampling_helpers as helpers,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_host_rows import _run_session
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import model as model_fixture
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    request,
)

# Operations that copy a CUDA tensor's contents to the host.
_SYNCING = {
    "tolist",
    "item",
    "__bool__",
    "__int__",
    "__float__",
    "__index__",
    "unique",
    "_unique",
    "_unique2",
    "unique_consecutive",
    "unique_dim",
    "equal",
    "nonzero",
    "nonzero_static",
    "cpu",
    "numpy",
}


class HostSync(AssertionError):
    pass


class DeviceOnly(torch.Tensor):
    """Behaves like a device tensor whose contents must never reach the host."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        if name in _SYNCING:
            raise HostSync(f"device tensor read back through {name}")
        return super().__torch_function__(func, types, args, kwargs or {})


def device_only(tensor):
    return tensor.as_subclass(DeviceOnly)


@pytest.fixture
def model():
    return model_fixture.__wrapped__()


def _trap_device_metadata(monkeypatch):
    """Wrap device row metadata of every host-metadata forward in DeviceOnly."""
    forward = ZingH3DiTModel.forward
    seen = []

    def record(self, *args, **kwargs):
        if kwargs.get("host_layout") is not None:
            for name in (
                "t",
                "seq_lens",
                "clean_seq_lens",
                "block_idx",
                "position_ids",
                "row_role",
            ):
                kwargs[name] = device_only(kwargs[name])
        seen.append((kwargs.get("self_cache_update"), kwargs.get("cache")))
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(ZingH3DiTModel, "forward", record)
    return seen


@torch.no_grad()
@pytest.mark.parametrize("window", [False, True])
def test_host_metadata_forwards_never_read_device_rows(model, monkeypatch, window):
    seen = _trap_device_metadata(monkeypatch)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        _run_session(model, window, guidance=1)
    phases = {phase for phase, _ in seen}
    assert {"source", "active", "final"} <= phases


@torch.no_grad()
def test_host_path_matches_device_path_rope_reuse(model, monkeypatch):
    """#185 reuse accounting is identical whether coordinates map on the host or device."""
    monkeypatch.setenv("SGLANG_ZING_CHUNK_ROPE_REUSE", "1")
    traces = {}
    forward = ZingH3DiTModel.forward
    for label in ("host", "device"):
        trace = []

        def record(self, *args, **kwargs):
            result = forward(self, *args, **kwargs)
            cache = kwargs.get("cache")
            if cache is not None:
                stats = dict(cache.rope_reuse_stats)
                stats.pop("map_syncs")
                trace.append((kwargs.get("self_cache_update"), stats))
            return result

        with monkeypatch.context() as patch:
            patch.setattr(ZingH3DiTModel, "forward", record)
            if label == "device":
                patch.setattr(helpers, "_host_metadata", lambda packed: None)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(5)
                _run_session(model, window=True, guidance=1)
        traces[label] = trace
    assert traces["host"] == traces["device"]
    final = traces["host"][-1][1]
    assert final["reused_rows"] > 0 and final["rebuilt_rows"] > 0


def _chunk_rows(start, frames, width=2):
    rows = [
        [float(t), 0.0, float(w)]
        for t in range(start, start + frames)
        for w in range(width)
    ]
    return torch.tensor(rows)


@torch.no_grad()
def test_host_mapping_matches_device_mapping_across_eviction():
    """Host and device coordinate maps are equal through growth, merge and compaction."""
    from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy

    def session(host):
        cache = ZingH3KVCache(
            1,
            window_policy=ZingWindowPolicy(2, 1),
            rope_position_mode="block_relative",
            rope_max_frame_gap=2,
        )
        prefix = torch.tensor([[0.0, 0, 0], [0.0, 0, 1], [1.0, 1, 0]])
        maps = []
        layer = lambda rows: [(torch.zeros(rows, 1, 2), torch.zeros(rows, 1, 2))]
        cache.update(
            prefix,
            torch.zeros(3, dtype=torch.long),
            "prefix",
            0,
            layer(3),
            validation=(prefix.clone(), torch.zeros(3, dtype=torch.long))
            if host
            else None,
        )
        time = 3
        for block, frames in zip(range(2, 12, 2), (1, 2, 2, 1, 2)):
            cache.prepare_chunk(block)
            source = _chunk_rows(time, frames)
            target = _chunk_rows(time + 20, frames)
            for phase, rows, number in (
                ("source", source, block - 1),
                ("active", target, block),
                ("final", target, block),
            ):
                ids = torch.full((rows.shape[0],), number)
                current = device_only(rows) if host else rows
                maps.append(
                    cache.attention_positions(
                        current, host_current=rows.clone() if host else None
                    )
                )
                cache.update(
                    current,
                    device_only(ids) if host else ids,
                    phase,
                    frames,
                    None if phase == "active" else layer(rows.shape[0]),
                    validation=(rows.clone(), ids.clone()) if host else None,
                )
                for chunk in cache._stored_chunks():
                    if host:
                        device = chunk.positions.as_subclass(torch.Tensor)
                        assert torch.equal(chunk.host_positions, device)
            time += frames + 3
        return maps

    host, device = session(True), session(False)
    # Raw chunk times advance by frames + 3 >= 4, above rope_max_frame_gap=2.
    # Gaps of 4 frames exceed rope_max_frame_gap=2, so the clamp must shape the map.
    clamped = host[-1][1][3:, 0].unique()
    assert (clamped[1:] - clamped[:-1]).max() <= 2
    for (h_current, h_history), (d_current, d_history) in zip(
        host, device, strict=True
    ):
        assert torch.equal(h_current, d_current)
        assert (h_history is None) == (d_history is None)
        if h_history is not None:
            assert torch.equal(h_history, d_history)


@torch.no_grad()
def test_missing_host_copy_falls_back_to_device_mapping():
    cache = ZingH3KVCache(1, rope_position_mode="block_relative")
    history = torch.tensor([[1.0, 0, 0], [4.0, 0, 0]])
    cache.update(
        history,
        torch.zeros(2, dtype=torch.long),
        "prefix",
        0,
        [(torch.zeros(2, 1, 2), torch.zeros(2, 1, 2))],
    )
    current = torch.tensor([[9.0, 0, 0]])
    mapped = cache.attention_positions(current, host_current=current.clone())
    expected = cache._map_attention_positions(current, False)
    assert torch.equal(mapped[0], expected[0])
    assert torch.equal(mapped[1], expected[1])


@torch.no_grad()
def test_sampler_moves_packed_rows_through_async_host_copies(model, monkeypatch):
    packer, inputs, visual, audio = request(model)
    sampler = ZingH3Sampler(model, packer, sigma_points=2)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, audio)
    packed = sampler._select_block(layout, 2, phase="target", video_t=0.2, audio_t=0.3)
    copied = []
    original = helpers.host_to_device

    def spy(tensor, device):
        copied.append(tensor)
        return original(tensor, device)

    expected = sampler._forward(packed, None, None)
    monkeypatch.setattr(helpers, "host_to_device", spy)
    result = sampler._forward(packed, None, None)
    names = (
        "x",
        "audio_x",
        "context",
        "t",
        "seq_lens",
        "clean_seq_lens",
        "block_idx",
        "position_ids",
        "row_role",
    )
    assert all(any(packed[name] is tensor for tensor in copied) for name in names)
    assert torch.equal(result[0], expected[0]) and torch.equal(result[1], expected[1])


def _layers(rows):
    return [(torch.zeros(rows, 1, 2), torch.zeros(rows, 1, 2))]


def _prefixed_cache(rope_position_mode="block_relative"):
    cache = ZingH3KVCache(1, rope_position_mode=rope_position_mode)
    prefix = torch.tensor([[0.0, 0, 0], [0.0, 0, 1], [1.0, 1, 0]])
    ids = torch.zeros(3, dtype=torch.long)
    cache.update(
        prefix, ids, "prefix", 0, _layers(3), validation=(prefix.clone(), ids.clone())
    )
    return cache


@torch.no_grad()
def test_replaced_prefix_keeps_its_host_copy():
    """A prefix adopted from the prefix workspace must not force the device path."""
    donor = _prefixed_cache()
    session = ZingH3KVCache(1, rope_position_mode="block_relative")
    session.replace_prefix(donor.prefix)
    assert torch.equal(session.prefix.host_positions, donor.prefix.host_positions)
    assert session.prefix.host_positions is not donor.prefix.host_positions
    current = torch.tensor([[9.0, 0, 0]])
    mapped, history = session.attention_positions(
        device_only(current), host_current=current.clone()
    )
    assert type(mapped) is torch.Tensor and type(history) is torch.Tensor


@torch.no_grad()
def test_owned_tensors_include_host_position_copies():
    cache = _prefixed_cache()
    owned = [id(t) for t in cache.owned_tensors()]
    assert id(cache.prefix.host_positions) in owned
