# SPDX-License-Identifier: Apache-2.0
"""Denoise CUDA graphs must reproduce the eager forward and cache state bitwise.

On CPU a recording backend stands in for CUDA graphs: capture records every
aten op the forward dispatches, and replay re-executes those ops on the
captured tensors without running any Python. A replay therefore sees exactly
what a CUDA graph sees: frozen host decisions, fixed addresses and whatever
the runner copied into the static inputs.
"""

import collections
import os
import pathlib
import subprocess
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import sglang
from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
from sglang.multimodal_gen.runtime.models.dits import zing_h3
from sglang.multimodal_gen.runtime.models.dits.zing_h3 import ZingH3DiTModel
from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
    ROUTE_CAPTURE,
    ROUTE_INELIGIBLE,
    ROUTE_REPLAY,
    ROUTE_WARMUP,
    CudaGraphBackend,
    GraphWarmupError,
    ZingDenoiseGraphs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    _sampling_helpers,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
    ZingH3Packer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import allocate_layer_stores
from sglang.multimodal_gen.test.unit.realtime.test_zing_prompt_switch_cache import (  # noqa: F401
    cpu_cache,
    cpu_dma,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_kv_pool_cache import _inputs
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401  # noqa: F401
    cpu_execution_options,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import model as model_fixture
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401  # noqa: F401
    verify_synthetic_table,
)

# 17 latent frames in 2-frame chunks: enough chunks after the window
# saturates for every steady key to warm up, capture and replay.
FRAMES = 17
STEPS = [1000, 750, 500, 250]

# Ops whose result reaches the host; a capture must never contain one.
_HOST_READS = {
    torch.ops.aten._local_scalar_dense.default,
    torch.ops.aten.nonzero.default,
    torch.ops.aten.is_nonzero.default,
    torch.ops.aten.equal.default,
    torch.ops.aten._unique2.default,
    torch.ops.aten.unique_consecutive.default,
}
_NO_KERNEL = {
    torch.ops.aten.empty.memory_format,
    torch.ops.aten.empty_strided.default,
    torch.ops.aten.empty_like.default,
}


class _Recorder(TorchDispatchMode):
    def __init__(self, ops):
        super().__init__()
        self.ops = ops

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _HOST_READS:
            raise AssertionError(f"graph capture read a tensor back via {func}")
        out = func(*args, **kwargs)
        self.ops.append((func, args, kwargs, out))
        return out


def _replay_op(func, args, kwargs, out):
    if func in _NO_KERNEL or func.is_view:
        return
    result = func(*args, **kwargs)
    if func._schema.is_mutable:
        return
    pairs = zip(result, out) if isinstance(out, (tuple, list)) else [(result, out)]
    for fresh, recorded in pairs:
        if isinstance(recorded, torch.Tensor):
            recorded.copy_(fresh)


class RecordingGraphBackend:
    """CPU stand-in with CUDA graph replay semantics."""

    def __init__(self):
        self.capturing = False
        self.captures = 0

    def supports(self, device):
        return True

    def capture(self, fn):
        ops = []
        self.capturing = True
        try:
            with _Recorder(ops):
                outputs = fn()
        finally:
            self.capturing = False
        self.captures += 1
        return ops, outputs

    def replay(self, graph):
        for op in graph:
            _replay_op(*op)


@pytest.fixture(autouse=True)
def _eager_by_default(monkeypatch):
    # Graphs default on; eager references stay eager and graph runs opt in.
    monkeypatch.setenv("SGLANG_ZING_DENOISE_CUDA_GRAPH", "0")


@pytest.fixture
def model():
    return model_fixture.__wrapped__()


def _storage(model, device="cpu"):
    arch = model.arch
    return allocate_layer_stores(
        256,
        num_layers=len(model.blocks),
        local_heads=arch.num_attention_heads,
        head_dim=arch.attention_head_dim,
        dtype=torch.bfloat16,
        device=torch.device(device),
    )


def _snapshot(cache, storage):
    projections = cache.adaln_projection_cache
    live = cache._length
    return dict(
        length=live,
        active=None if cache.active is None else cache.active[0],
        stats=dict(cache.stats),
        rope=dict(cache.rope_reuse_stats),
        last_rope_reuse=cache.last_rope_reuse,
        published={k: len(v) for k, v in cache._rotated_store_signatures.items()},
        coefficients=len(cache._rope_coefficients),
        raw_write_epochs=dict(cache._raw_write_epochs),
        adaln=(
            None
            if projections is None
            else {
                k: projections.stats()[k]
                for k in ("bytes", "hits", "misses", "evictions", "bypasses", "entries")
            }
        ),
        store=[
            (layer.k[:live].cpu(), layer.v[:live].cpu()) for layer in (storage or ())
        ],
    )


def _run(model, monkeypatch, *, storage, device="cpu", perturb=None, **session):
    """One windowed few-step session; returns final latents and one trace per forward."""
    (outcome,) = _run_sessions(
        model,
        monkeypatch,
        storages=[storage],
        device=device,
        perturb=perturb,
        **session,
    )
    return outcome


def _run_sessions(
    model,
    monkeypatch,
    *,
    storages,
    seeds=(11,),
    device="cpu",
    perturb=None,
    rope_position_mode="block_relative",
):
    """Sessions stepped round-robin through one model; (latents, trace) per session."""
    traces = [[] for _ in storages]
    current = [0]
    forward = ZingH3DiTModel.forward

    def record(self, *args, **kwargs):
        trace, storage = traces[current[0]], storages[current[0]]
        cache = kwargs.get("cache")
        phase = kwargs.get("self_cache_update")
        if perturb is not None:
            perturb(phase, sum(1 for t in trace if t["phase"] == "active"), cache)
        graphs = self.denoise_graphs
        before = collections.Counter() if graphs is None else graphs.routes.copy()
        video, audio = forward(self, *args, **kwargs)
        routes = collections.Counter() if graphs is None else graphs.routes - before
        trace.append(
            dict(
                phase=phase,
                routes=routes,
                video=video.cpu(),
                audio=None if audio is None else audio.cpu(),
                cache=None if cache is None else _snapshot(cache, storage),
            )
        )
        return video, audio

    states = []
    with monkeypatch.context() as patch, torch.no_grad():
        patch.setattr(ZingH3DiTModel, "forward", record)
        try:
            for index, (storage, seed) in enumerate(zip(storages, seeds)):
                current[0] = index
                states.append(
                    _initialize(
                        model,
                        storage=storage,
                        seed=seed,
                        device=device,
                        rope_position_mode=rope_position_mode,
                    )
                )
            while any(state.index < len(state.spans) for _, state in states):
                for index, (sampler, state) in enumerate(states):
                    if state.index < len(state.spans):
                        current[0] = index
                        sampler.step(state)
            results = [state.visual.cpu() for _, state in states]
        finally:
            for _, state in states:
                state.close()
    return list(zip(results, traces))


def _initialize(model, *, storage, seed, device, rope_position_mode):
    packer = ZingH3Packer(model.arch, num_frame_per_block=2)
    sampler = ZingH3Sampler(
        model,
        packer,
        solver="few_step",
        denoising_step_list=STEPS,
        window_chunks=3,
        sink_chunks=1,
        window_eviction_timing="before_source",
        rope_position_mode=rope_position_mode,
    )
    torch.manual_seed(seed)
    inputs = _inputs(FRAMES)
    visual = torch.randn(1, 2, FRAMES, 2, 2)
    if device != "cpu":
        inputs.source_visual = inputs.source_visual.to(device)
        visual = visual.to(device)
    return sampler, sampler.initialize(inputs, visual, storage=storage)


def _assert_same(actual, expected):
    result, trace = actual
    reference, eager = expected
    assert len(trace) == len(eager)
    for index, (got, want) in enumerate(zip(trace, eager)):
        context = f"forward {index} ({want['phase']})"
        assert got["phase"] == want["phase"], context
        assert torch.equal(got["video"], want["video"]), context
        if want["audio"] is None:
            assert got["audio"] is None, context
        else:
            assert torch.equal(got["audio"], want["audio"]), context
        got_cache, want_cache = got["cache"], want["cache"]
        assert (got_cache is None) == (want_cache is None), context
        if want_cache is not None:
            for (gk, gv), (wk, wv) in zip(
                got_cache.pop("store"), want_cache.pop("store")
            ):
                assert torch.equal(gk, wk) and torch.equal(gv, wv), context
            assert got_cache == want_cache, context
    assert torch.equal(result, reference)


def _graph_mode(model, monkeypatch, backend, max_graphs=32):
    trap_host_to_device(monkeypatch, backend)
    model.denoise_graphs = ZingDenoiseGraphs(backend=backend, max_graphs=max_graphs)
    return model.denoise_graphs


def trap_host_to_device(monkeypatch, backend):
    copy = zing_h3.host_to_device

    def guarded(tensor, device):
        if backend.capturing:
            raise AssertionError("host-to-device copy inside a graph capture")
        return copy(tensor, device)

    monkeypatch.setattr(zing_h3, "host_to_device", guarded)


def _active_routes(trace, *, allow_disabled=False):
    total = collections.Counter()
    for entry in trace:
        if entry["phase"] == "active":
            assert sum(entry["routes"].values()) == 1, entry["routes"]
            total += entry["routes"]
        else:
            assert not entry["routes"], entry["routes"]
    # Disabled graphs run eager, which would hide the capture-time host-read
    # and host-to-device guards; only the failure test expects that.
    if not allow_disabled:
        assert not total["capture_failed"] and not total["graphs_disabled"], total
    return total


def test_denoise_graph_env_switch(monkeypatch):
    """On by default; an invalid value is rejected rather than ignored."""
    monkeypatch.delenv("SGLANG_ZING_DENOISE_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("SGLANG_ZING_DENOISE_CUDA_GRAPH_MAX", raising=False)
    graphs = model_fixture.__wrapped__().denoise_graphs
    assert isinstance(graphs, ZingDenoiseGraphs) and graphs.max_graphs == 32
    monkeypatch.setenv("SGLANG_ZING_DENOISE_CUDA_GRAPH", "0")
    assert model_fixture.__wrapped__().denoise_graphs is None
    monkeypatch.setenv("SGLANG_ZING_DENOISE_CUDA_GRAPH", "1")
    monkeypatch.setenv("SGLANG_ZING_DENOISE_CUDA_GRAPH_MAX", "3")
    graphs = model_fixture.__wrapped__().denoise_graphs
    assert isinstance(graphs, ZingDenoiseGraphs) and graphs.max_graphs == 3
    monkeypatch.setenv("SGLANG_ZING_DENOISE_CUDA_GRAPH_MAX", "many")
    with pytest.raises(ValueError, match="SGLANG_ZING_DENOISE_CUDA_GRAPH_MAX"):
        model_fixture.__wrapped__()
    monkeypatch.setenv("SGLANG_ZING_DENOISE_CUDA_GRAPH_MAX", "3")
    monkeypatch.setenv("SGLANG_ZING_DENOISE_CUDA_GRAPH", "yes")
    with pytest.raises(ValueError):
        model_fixture.__wrapped__()


def test_denoise_graph_sp_env_switch(monkeypatch):
    """Sequence-parallel graphs are on by default; invalid values are rejected."""
    monkeypatch.delenv("SGLANG_ZING_DENOISE_CUDA_GRAPH_SP", raising=False)
    assert model_fixture.__wrapped__().denoise_graph_sp is True
    monkeypatch.setenv("SGLANG_ZING_DENOISE_CUDA_GRAPH_SP", "0")
    assert model_fixture.__wrapped__().denoise_graph_sp is False
    monkeypatch.setenv("SGLANG_ZING_DENOISE_CUDA_GRAPH_SP", "yes")
    with pytest.raises(ValueError, match="SGLANG_ZING_DENOISE_CUDA_GRAPH_SP"):
        model_fixture.__wrapped__()


# A late step-2 denoise forward: in steady state it would replay.
_LATE_FORWARD = (FRAMES // 2 - 1) * len(STEPS) + 2


def _write_history_k(phase, active_forwards, cache):
    if phase == "active" and active_forwards == _LATE_FORWARD:
        cache.storage[0].k[0].add_(1)


def _weight_swap(model):
    """Moves one block weight to new storage mid-session, as offload would."""
    param = model.blocks[0].mlp.fc2.weight
    base = param.detach().clone()

    def perturb(phase, active_forwards, cache):
        if phase == "prefix":
            param.data = base.clone()
        if phase == "active" and active_forwards == _LATE_FORWARD:
            param.data = base + 1

    return perturb


@pytest.mark.parametrize(
    "case",
    [
        "steady",
        "history_k_write",
        "weight_swap",
        "absolute_rope",
        "adaln_evictions",
        "one_graph",
    ],
)
@pytest.mark.parametrize("count", [4, 5])
def test_graph_replay_matches_eager_bitwise(model, monkeypatch, case, count):
    """Replays must refresh every input that changes under one key, and a key
    must change whenever a captured address (weights, projections) moves."""
    if count == 5:
        arch = replace(
            model.arch,
            num_modalities=5,
            adaln_out_features=6 * 5 * model.arch.hidden_size,
            modality_tags=dict(
                video=4, text=0, audio=3, source_visual=1, condition_visual=2
            ),
        )
        model = ZingH3DiTModel(ZingH3DiTConfig(arch_config=arch), {}).eval()
        torch.manual_seed(7)
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(
                    torch.ones_like(param)
                    if "norm" in name and name.endswith("weight")
                    else torch.randn_like(param) * 0.08
                )
            model.rope.inv_freq.fill_(1)
    perturb = {"history_k_write": _write_history_k, "weight_swap": _weight_swap(model)}
    perturb = perturb.get(case)
    session = {"rope_position_mode": "absolute"} if case == "absolute_rope" else {}
    if case == "adaln_evictions":
        # Room for half of the session's projections: entries are evicted and
        # rebuilt at new addresses while graphs are replayed.
        _, probe = _run(model, monkeypatch, storage=_storage(model))
        peak = max(
            t["cache"]["adaln"]["bytes"]
            for t in probe
            if t["cache"] and t["cache"]["adaln"]
        )
        model.adaln_projection_cache_bytes = peak // 2
    expected = _run(
        model, monkeypatch, storage=_storage(model), perturb=perturb, **session
    )
    backend = RecordingGraphBackend()
    graphs = _graph_mode(
        model, monkeypatch, backend, max_graphs=1 if case == "one_graph" else 32
    )
    actual = _run(
        model, monkeypatch, storage=_storage(model), perturb=perturb, **session
    )
    _assert_same(actual, expected)

    routes = _active_routes(actual[1])
    assert backend.captures == routes["capture"]
    assert len(graphs.graphs) <= graphs.max_graphs
    # Every steady key repeats: denoise steps of one chunk share a key.
    assert routes["replay"] > 0 and routes["capture"] > 0
    late = [t for t in actual[1] if t["phase"] == "active"][_LATE_FORWARD]
    if case == "history_k_write":
        assert late["routes"] == {"rope_rebuild": 1}
    if case == "weight_swap":
        assert late["routes"] == {"warmup": 1}
    if case == "adaln_evictions":
        assert any(t["cache"]["adaln"]["evictions"] for t in expected[1] if t["cache"])
    if case == "one_graph":
        # One slot: earlier geometries were captured and evicted.
        assert routes["capture"] > 1
    if case == "steady":
        # Each slot's AdaLN parts tile one dense block, refreshed by one copy.
        for entry in graphs.graphs.values():
            parts = [n for n in entry.static if n.startswith("projection.")]
            assert parts and all(n.count(".") == 1 for n in parts), parts


@pytest.mark.parametrize(
    "case, reason",
    [
        ("unpooled", "unpooled"),
        ("adaln_cache_off", "adaln_cache"),
        ("history_rope_plan", "history_rope_plan"),
        ("chunk_rope_reuse_off", "chunk_rope_reuse_off"),
        ("unsupported_device", "device"),
        ("module_hooks", "module_hooks"),
        ("device_metadata", "device_metadata"),
    ],
)
def test_ineligible_denoise_forwards_stay_eager(model, monkeypatch, case, reason):
    """Each mode a captured forward cannot honour must fall back, never capture."""
    storage = None if case == "unpooled" else _storage(model)
    if case == "adaln_cache_off":
        model.adaln_projection_cache_enabled = False
    if case == "history_rope_plan":
        monkeypatch.setenv("SGLANG_ZING_HISTORY_ROPE_PLAN", "1")
    if case == "chunk_rope_reuse_off":
        monkeypatch.setenv("SGLANG_ZING_CHUNK_ROPE_REUSE", "0")
    if case == "module_hooks":
        # Offload hooks move weights inside the forward, behind a capture's back.
        model.blocks[0].register_forward_pre_hook(lambda module, args: None)
    if case == "device_metadata":
        monkeypatch.setattr(_sampling_helpers, "_host_metadata", lambda packed: None)
    expected = _run(model, monkeypatch, storage=storage)
    backend = RecordingGraphBackend()
    if case == "unsupported_device":
        backend.supports = lambda device: False
    _graph_mode(model, monkeypatch, backend)
    actual = _run(model, monkeypatch, storage=storage)
    _assert_same(actual, expected)
    routes = _active_routes(actual[1])
    assert set(routes) == {reason} and backend.captures == 0


class _FailingCaptureBackend(RecordingGraphBackend):
    """Runs the forward's Python like a real capture; the fail_at-th one fails."""

    def __init__(self, fail_at):
        super().__init__()
        self.fail_at = fail_at
        self.attempts = 0

    def capture(self, fn):
        self.attempts += 1
        captured = super().capture(fn)
        if self.attempts == self.fail_at:
            raise RuntimeError("capture failed")
        return captured


@pytest.mark.parametrize("fail_at", [1, 2])
def test_failed_capture_raises_and_disables_graphs(model, monkeypatch, fail_at):
    """A failed real capture can leave CUDA RNG and allocator state unusable, so
    the failing forward must raise; later sessions run eager without graphs."""
    expected = _run(model, monkeypatch, storage=_storage(model))
    backend = _FailingCaptureBackend(fail_at)
    graphs = _graph_mode(model, monkeypatch, backend)
    with pytest.raises(RuntimeError, match="capture failed"):
        _run(model, monkeypatch, storage=_storage(model))
    assert backend.attempts == fail_at and graphs.disabled and not graphs.graphs
    assert graphs.routes["capture_failed"] == 1
    assert graphs.routes["capture"] == fail_at - 1
    actual = _run(model, monkeypatch, storage=_storage(model))
    _assert_same(actual, expected)
    routes = _active_routes(actual[1], allow_disabled=True)
    assert set(routes) == {"graphs_disabled"} and backend.attempts == fail_at


class _FailingWarmupBackend(RecordingGraphBackend):
    """The first capture's warm-up runs part of the forward, then fails."""

    def __init__(self):
        super().__init__()
        self.warmup_failures = 0

    def capture(self, fn):
        if not self.warmup_failures:
            self.warmup_failures += 1
            fn()
            raise GraphWarmupError("warm-up failed")
        return super().capture(fn)


def test_failed_warmup_runs_eager_and_keeps_graphs(model, monkeypatch):
    """A warm-up failure (e.g. OOM before any capture) leaves CUDA state intact,
    so that forward must run eager, keep host state exact and keep graphs on."""
    expected = _run(model, monkeypatch, storage=_storage(model))
    backend = _FailingWarmupBackend()
    graphs = _graph_mode(model, monkeypatch, backend)
    actual = _run(model, monkeypatch, storage=_storage(model))
    _assert_same(actual, expected)
    routes = _active_routes(actual[1])
    assert routes["warmup_failed"] == 1 and not graphs.disabled
    assert routes["capture"] > 0 and routes["replay"] > 0


# Runs in a child process: a failed CUDA capture leaves the default generator
# in capture mode, which would break every later CUDA test in this process.
_INVALIDATED_CAPTURE = """
import gc
import torch
from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
    CudaGraphBackend,
)

backend = CudaGraphBackend()
x = torch.ones(8, device="cuda")
calls = []


def forward():
    calls.append(None)
    y = x * 2
    if len(calls) == 2:
        torch.cuda.synchronize()
    return (y,)


before = torch.cuda.current_stream()
try:
    backend.capture(forward)
except RuntimeError:
    pass
else:
    raise AssertionError("an invalidated capture must raise")
assert len(calls) == 2, calls
assert torch.cuda.current_stream() == before
assert backend.capturing is False
assert gc.isenabled(), "Automatic GC must be restored after a failed capture"
print("INVALIDATED_CAPTURE_OK")
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalidated_cuda_capture_restores_the_callers_stream():
    """A capture invalidated by a host sync must raise and leave the caller on
    its own stream with the backend no longer capturing."""
    env = dict(os.environ)
    root = str(pathlib.Path(sglang.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (root, env.get("PYTHONPATH")) if p)
    result = subprocess.run(
        [sys.executable, "-c", _INVALIDATED_CAPTURE],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "INVALIDATED_CAPTURE_OK" in result.stdout, result.stdout


def test_interleaved_sessions_replay_their_own_slots(model, monkeypatch):
    """Two sessions of one geometry on different slots must never share a graph."""
    seeds = (11, 12)
    # Re-noising draws from the global RNG, so the eager reference must step
    # the sessions in the same interleaved order.
    expected = _run_sessions(
        model, monkeypatch, storages=[_storage(model), _storage(model)], seeds=seeds
    )
    backend = RecordingGraphBackend()
    _graph_mode(model, monkeypatch, backend)
    actual = _run_sessions(
        model,
        monkeypatch,
        storages=[_storage(model), _storage(model)],
        seeds=seeds,
    )
    routes = collections.Counter()
    for got, want in zip(actual, expected):
        _assert_same(got, want)
        routes += _active_routes(got[1])
    assert routes["replay"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_replay_matches_eager_bitwise(monkeypatch):
    """Real CUDA graphs, with the default CUDA kernels (FA4 when available)."""
    try:
        import flash_attn.cute  # noqa: F401

        attention = "fa4"
    except (ImportError, OSError):
        attention = "torch_sdpa"
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_ATTENTION", attention)
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN", "1")
    model = _cuda_model()
    assert model.experimental_attention_backend == attention
    expected = _run(model, monkeypatch, storage=_storage(model, "cuda"), device="cuda")
    _graph_mode(model, monkeypatch, CudaGraphBackend())
    actual = _run(model, monkeypatch, storage=_storage(model, "cuda"), device="cuda")
    _assert_same(actual, expected)
    routes = _active_routes(actual[1])
    assert routes["replay"] > 0 and routes["capture"] > 0


def _cuda_model():
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
        ZingH3DiTArchConfig,
        ZingH3DiTConfig,
    )
    from sglang.multimodal_gen.test.unit.test_minimax_h3_dit_contract import (
        _ensure_single_process_parallel_runtime,
    )

    _ensure_single_process_parallel_runtime()
    # Smallest geometry the fused CUDA kernels and FA4 accept: 128-dim heads.
    hidden = 256
    arch = ZingH3DiTArchConfig(
        num_layers=2,
        token_refiner_num_layers=1,
        hidden_size=hidden,
        num_attention_heads=2,
        attention_head_dim=128,
        ffn_hidden_size=512,
        latents_dim=2,
        audio_latents_dim=2,
        text_dim=12,
        timestep_input_dim=8,
        time_embed_hidden_size=16,
        time_embed_dim=8,
        adaln_out_features=24 * hidden,
        final_adaln_out_features=2 * hidden,
        rope_inv_freq_len=16,
    )
    model = ZingH3DiTModel(ZingH3DiTConfig(arch_config=arch), {}).eval()
    torch.manual_seed(7)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                param.fill_(1)
            else:
                param.copy_(torch.randn(param.shape) * 0.08)
        model.rope.inv_freq.fill_(1)
    return model.cuda()


@torch.no_grad()
def _weights_free_model(model, tmp_path, plan):
    """Build a real table and construct a model with no projection weights."""
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
        ZingAdalnTable,
        write_table,
    )

    path = tmp_path / "table"
    write_table(
        path,
        arch=model.arch,
        tags=model.modality_tags,
        embedder=model.time_embedder,
        projections=[
            *((i, b.adaln_proj) for i, b in enumerate(model.blocks)),
            ("final", model.final_layer.adaln_proj),
        ],
        plan=plan,
        identity={"fixture": True},
        provenance={"synthetic": True},
    )
    table = ZingAdalnTable(
        path, arch=model.arch, tags=model.modality_tags, max_bytes=1 << 20
    )
    verify_synthetic_table(table, tmp_path)
    candidate = ZingH3DiTModel(
        ZingH3DiTConfig(arch_config=replace(model.arch)), {}, adaln_table=table
    ).eval()
    candidate.load_state_dict(
        {k: v for k, v in model.state_dict().items() if "adaln_proj" not in k}
    )
    candidate.initialize_adaln_table()
    receipt = candidate.adaln_deployment_receipt()
    assert (
        receipt["projection_parameters"] == []
        and receipt["projection_storage_bytes"] == 0
    )
    return candidate


@torch.no_grad()
def test_weights_free_table_replays_interleaved_sessions_exactly(
    model, tmp_path, monkeypatch
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import video_plan
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_zing_h3_causal_renoise import (
        ZingH3CausalRenoiseSchedule,
    )

    seeds = (11, 12)
    expected = _run_sessions(
        model, monkeypatch, storages=[_storage(model), _storage(model)], seeds=seeds
    )
    schedule = ZingH3CausalRenoiseSchedule(STEPS, num_train_timesteps=1000, shift=5.0)
    plan = video_plan("few_step", schedule.timesteps, sigmas=schedule.sigmas)
    candidate = _weights_free_model(model, tmp_path, plan)
    backend = RecordingGraphBackend()
    _graph_mode(candidate, monkeypatch, backend)
    actual = _run_sessions(
        candidate,
        monkeypatch,
        storages=[_storage(candidate), _storage(candidate)],
        seeds=seeds,
    )
    routes = collections.Counter()
    for got, want in zip(actual, expected):
        for entry in want[1]:
            if entry["cache"] is not None:
                entry["cache"]["adaln"] = (
                    None  # Resident LRU counters differ by design.
                )
        _assert_same(got, want)
        routes += _active_routes(got[1])
    assert routes["capture"] > 0 and routes["replay"] > 0
    assert candidate.adaln_table.active_plans == 0
    assert candidate.adaln_table.stats()["fallback"] == 0


# Sequence-parallel coordination: a scripted coordinator stands in for the peer
# rank, recording every vote, so each rank's side of the protocol is checked
# without a process group.


class _ScriptedBackend:
    """Follows the real backend's order: warm-up, warm-up vote, then capture."""

    def __init__(self, fail=None, warmup_fail=None):
        self.fail = fail
        self.warmup_fail = warmup_fail
        self.captures = 0
        self.replays = 0
        self.events = []

    def supports(self, device):
        return True

    def capture(self, fn, warmed=None):
        self.events.append("warmup")
        agreed = True if warmed is None else warmed(self.warmup_fail is None)
        if self.warmup_fail is not None:
            raise GraphWarmupError("warm-up failed") from self.warmup_fail
        if not agreed:
            raise GraphWarmupError("warm-up failed on a peer rank")
        self.captures += 1
        self.events.append("capture")
        if self.fail is not None:
            raise self.fail
        return object(), fn()

    def replay(self, graph):
        self.replays += 1


class _ScriptedCoordinator:
    def __init__(self, agree=True, peers_ok=True):
        self.verdict = agree
        self.peers_ok = peers_ok
        self.routes = []
        self.oks = []
        self.fail_vote = None  # 1-based index of the all_ok vote a peer fails

    def agree(self, route):
        self.routes.append(route)
        return self.verdict

    def all_ok(self, ok):
        self.oks.append(ok)
        if self.fail_vote == len(self.oks):
            return False
        return ok and self.peers_ok


def _sp_graphs(coordinator, backend=None):
    graphs = ZingDenoiseGraphs(backend=backend or _ScriptedBackend())
    graphs.coordinator = coordinator
    return graphs


def _sp_run(graphs, key="k"):
    tensors = {"x": torch.ones(2)}
    return graphs.run(
        key,
        tensors=tensors,
        capture=lambda static: (static["x"] * 2,),
        replay_host=lambda: None,
    )


def test_every_forward_votes_its_route_once():
    coordinator = _ScriptedCoordinator()
    graphs = _sp_graphs(coordinator)
    assert _sp_run(graphs) is None
    assert _sp_run(graphs) is not None
    assert _sp_run(graphs) is not None
    assert coordinator.routes == [ROUTE_WARMUP, ROUTE_CAPTURE, ROUTE_REPLAY]
    # One vote after the warm-up, one after the capture.
    assert coordinator.oks == [True, True]
    assert graphs.backend.captures == 1 and graphs.backend.replays == 2


def test_ineligible_forward_votes_without_a_key():
    coordinator = _ScriptedCoordinator()
    _sp_graphs(coordinator).vote_eager()
    assert coordinator.routes == [ROUTE_INELIGIBLE]


def test_vote_eager_is_free_without_a_coordinator():
    _sp_graphs(None).vote_eager()


def test_disagreement_runs_eager_and_forgets_the_key():
    coordinator = _ScriptedCoordinator()
    graphs = _sp_graphs(coordinator)
    _sp_run(graphs)
    _sp_run(graphs)
    coordinator.verdict = False
    assert _sp_run(graphs) is None
    assert graphs.routes["route_mismatch"] == 1
    assert not graphs.graphs and "k" not in graphs._seen
    coordinator.verdict = True
    assert _sp_run(graphs) is None  # warms up again, together with its peers
    assert coordinator.routes[-1] == ROUTE_WARMUP


def test_capture_failure_is_reported_to_peers_before_raising():
    coordinator = _ScriptedCoordinator()
    graphs = _sp_graphs(coordinator, _ScriptedBackend(fail=RuntimeError("boom")))
    _sp_run(graphs)
    with pytest.raises(RuntimeError, match="boom"):
        _sp_run(graphs)
    assert coordinator.oks == [True, False] and graphs.disabled


def test_peer_capture_failure_fails_this_rank_too():
    coordinator = _ScriptedCoordinator()
    graphs = _sp_graphs(coordinator)
    _sp_run(graphs)
    coordinator.fail_vote = 2  # the peer's capture failed: the second vote
    with pytest.raises(RuntimeError, match="peer rank"):
        _sp_run(graphs)
    assert graphs.disabled and graphs.backend.replays == 0


def test_warmup_failure_is_voted_before_any_capture():
    coordinator = _ScriptedCoordinator()
    graphs = _sp_graphs(coordinator, _ScriptedBackend(warmup_fail=RuntimeError("oom")))
    _sp_run(graphs)
    with pytest.raises(GraphWarmupError):
        _sp_run(graphs)
    # Exactly one vote (the peers match it) and no capture collective follows it.
    assert coordinator.oks == [False] and graphs.disabled
    assert graphs.backend.events == ["warmup"] and graphs.backend.captures == 0


def test_peer_warmup_failure_stops_this_rank_before_any_capture():
    coordinator = _ScriptedCoordinator(peers_ok=False)
    graphs = _sp_graphs(coordinator)
    _sp_run(graphs)
    with pytest.raises(GraphWarmupError, match="peer rank"):
        _sp_run(graphs)
    # This rank warmed up fine, yet it must not enter a capture its peer skips.
    assert coordinator.oks == [True] and graphs.disabled
    assert graphs.backend.events == ["warmup"] and graphs.backend.captures == 0
    assert graphs.backend.replays == 0


@pytest.mark.parametrize(
    "enabled, size, ring, tp, rows, reason",
    [
        (False, 2, 1, 1, 8, "sequence_parallel"),
        (True, 4, 2, 1, 8, "sp_topology"),
        (True, 2, 2, 1, 8, "sp_topology"),
        (True, 2, 1, 2, 8, "sp_topology"),
        (True, 2, 1, 1, 9, None),
        (True, 2, 1, 1, 8, None),
        (True, 4, 1, 1, 8, None),
        (True, 4, 1, 1, 10, None),
        (True, 8, 1, 1, 8, None),
        (True, 4, 1, 1, 1345, None),
        (True, 8, 1, 1, 1345, None),
    ],
)
def test_sequence_parallel_admission(
    monkeypatch, enabled, size, ring, tp, rows, reason
):
    monkeypatch.setattr(
        zing_h3, "get_sp_group", lambda: SimpleNamespace(ring_world_size=ring)
    )
    monkeypatch.setattr(zing_h3, "get_tp_world_size", lambda: tp)
    model = SimpleNamespace(denoise_graph_sp=enabled)
    actual = ZingH3DiTModel._sequence_parallel_graph_rejection(
        model, SimpleNamespace(rows=rows), size
    )
    assert actual == reason


@pytest.mark.parametrize(
    "enabled, size, ring, tp, coordinated",
    [
        (True, 2, 1, 1, True),
        (True, 1, 1, 1, False),
        (True, 4, 1, 1, True),
        (True, 4, 2, 1, False),
        (True, 8, 1, 1, True),
        (True, 2, 2, 1, False),
        (True, 2, 1, 2, False),
        (False, 2, 1, 1, False),
    ],
)
def test_only_topologies_that_can_use_graphs_vote_on_routes(
    monkeypatch, enabled, size, ring, tp, coordinated
):
    """A topology that always runs eager must not pay a CPU/Gloo vote on every
    denoise forward; one that can use graphs keeps voting when a forward falls
    back (RoPE rebuild, cache eligibility), because its peers may not."""
    cpu_group = object()
    monkeypatch.setattr(
        zing_h3,
        "get_sp_group",
        lambda: SimpleNamespace(ring_world_size=ring, cpu_group=cpu_group),
    )
    monkeypatch.setattr(zing_h3, "get_tp_world_size", lambda: tp)
    model = SimpleNamespace(denoise_graph_sp=enabled)
    graphs = ZingDenoiseGraphs(backend=_ScriptedBackend())
    ZingH3DiTModel._coordinate_sequence_parallel_routes(
        model, graphs, device=torch.device("cpu"), ulysses_size=size
    )
    assert (graphs.coordinator is not None) is coordinated
    if coordinated:
        assert graphs.coordinator.group is cpu_group


def test_sp_graph_key_distinguishes_global_splits_with_same_local_shape():
    """Rank 0 owns three rows for both 10 and 11 global rows at SP4.

    The all-to-all split lists differ, so the two forwards must not share a
    captured graph even though this rank's input tensor layout is identical.
    """
    from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
        compute_sequence_splits,
    )

    assert compute_sequence_splits(10, 4) == [3, 3, 2, 2]
    assert compute_sequence_splits(11, 4) == [3, 3, 3, 2]
    model = SimpleNamespace(
        _cache_weight_generation=0,
        experimental_compiled_adaln=False,
        experimental_attention_backend="torch_sdpa",
    )
    cache = SimpleNamespace(store_layout_signature=lambda: ())
    tensors = {"x": torch.ones(3, 4)}
    keys = [
        ZingH3DiTModel._denoise_graph_key(
            model,
            tensors=tensors,
            cache=cache,
            history=20,
            rows=rows,
            weight_addresses=(),
            ulysses_size=4,
        )
        for rows in (10, 11)
    ]
    assert keys[0] != keys[1]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fixed_kv_workspace_records_dynamic_history_without_python(dtype):
    """One graph must select live rows, ignore padding and scatter clean KV.

    The fixed workspace is the execution primitive used by chunk graphs, not
    a second cache: its authoritative K/V tensors are the supplied pool store.
    Replay changes device metadata while keeping every storage address fixed.
    """
    from sglang.multimodal_gen.runtime.models.dits import zing_h3_cache

    workspace_type = getattr(zing_h3_cache, "ZingFixedKVWorkspace", None)
    assert workspace_type is not None, "whole-chunk graphs require fixed KV workspace"
    torch.manual_seed(37)
    capacity, current_rows, heads, dim = 16, 3, 2, 8
    k = torch.randn(capacity, heads, dim).to(dtype)
    v = torch.randn_like(k)
    workspace = workspace_type(k, v, current_rows=current_rows)
    q = torch.randn(current_rows, heads, dim).to(dtype)
    current_k, current_v = torch.randn_like(q), torch.randn_like(q)
    identities = tuple(t.data_ptr() for t in workspace.owned_tensors())
    backend = RecordingGraphBackend()
    python_calls = []

    def execute():
        python_calls.append(True)
        keys, values, mask = workspace.attention_view(current_k, current_v)
        result = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            keys.transpose(0, 1)[None],
            values.transpose(0, 1)[None],
            attn_mask=mask[None, None, None],
            dropout_p=0.0,
        )[0].transpose(0, 1)
        workspace.commit(current_k, current_v)
        return result

    for trial, history in enumerate(([0, 1], [0, 1, 4, 5, 6], [0, 1, 8, 9], [])):
        # Duplicate sink/history positions, window filling and eviction all
        # use the same static graph. Unselected rows contain adversarial data.
        k.copy_(torch.randn_like(k))
        v.copy_(torch.randn_like(v))
        q.copy_(torch.randn_like(q))
        current_k.copy_(torch.randn_like(q))
        current_v.copy_(torch.randn_like(q))
        before_k, before_v = k.clone(), v.clone()
        destinations = [11, 12, 13]
        workspace.prepare(history, destinations)
        ref_k = torch.cat((before_k[history], current_k))
        ref_v = torch.cat((before_v[history], current_v))
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            ref_k.transpose(0, 1)[None],
            ref_v.transpose(0, 1)[None],
            dropout_p=0.0,
        )[0].transpose(0, 1)
        if trial == 0:
            graph, result = backend.capture(execute)
        else:
            backend.replay(graph)
        torch.testing.assert_close(
            result,
            expected,
            rtol=2e-5 if dtype == torch.float32 else 0.02,
            atol=2e-6 if dtype == torch.float32 else 0.02,
        )
        padded_k = torch.zeros(capacity + current_rows, heads, dim, dtype=dtype)
        padded_v = torch.zeros_like(padded_k)
        padded_k[: len(history)] = before_k[history]
        padded_v[: len(history)] = before_v[history]
        padded_k[len(history) : len(history) + current_rows] = current_k
        padded_v[len(history) : len(history) + current_rows] = current_v
        valid = torch.arange(capacity + current_rows) < len(history) + current_rows
        fixed_expected = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            padded_k.transpose(0, 1)[None],
            padded_v.transpose(0, 1)[None],
            attn_mask=valid[None, None, None],
            dropout_p=0.0,
        )[0].transpose(0, 1)
        assert torch.equal(result, fixed_expected)
        before_k[destinations] = current_k
        before_v[destinations] = current_v
        assert torch.equal(k, before_k) and torch.equal(v, before_v)
        assert tuple(t.data_ptr() for t in workspace.owned_tensors()) == identities
        assert int(workspace.cu_seqlens_k[-1]) == len(history) + current_rows
    assert len(python_calls) == 1 and backend.captures == 1


def test_fixed_kv_workspace_refuses_invalid_plan_without_writes():
    from sglang.multimodal_gen.runtime.models.dits import zing_h3_cache

    workspace_type = getattr(zing_h3_cache, "ZingFixedKVWorkspace", None)
    assert workspace_type is not None, "whole-chunk graphs require fixed KV workspace"
    k, v = torch.randn(8, 2, 8), torch.randn(8, 2, 8)
    workspace = workspace_type(k, v, current_rows=2)
    workspace.prepare([0, 1], [5, 6])
    before = [t.clone() for t in workspace.owned_tensors()]
    for history, writes in (([8], [5, 6]), ([0], [5]), ([0], [5, 5]), ([0], [-1, 6])):
        with pytest.raises(ValueError):
            workspace.prepare(history, writes)
        assert all(
            torch.equal(t, old) for t, old in zip(workspace.owned_tensors(), before)
        )


@torch.no_grad()
def _run_ready_chunks(
    model,
    monkeypatch,
    *,
    backend=None,
    batches=(4, 4, 4, 5),
    graph_owner=None,
    storage=None,
    input_seed=83,
    initial_steps=None,
    steps=None,
    prompt_updates=None,
    yield_hook=None,
    allow_cancel=False,
    defer_final=False,
    known_eos=True,
    prompt_joint=False,
    joint_forward="cross_chunk",
    output_limit=None,
    prefix_loader_factory=None,
    use_step=False,
    reference=False,
    split_first_latent=True,
    sink_updates=None,
    sink_trace=None,
    reuse_updates=False,
):
    """Exercise the production streaming iterator, including source pauses."""
    from sglang.multimodal_gen.runtime.models.dits import zing_h3_denoise_graph
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import fixture_provenance

    packer = ZingH3Packer(
        model.arch, num_frame_per_block=[3, 2], split_first_latent=split_first_latent
    )
    # Prompt changes preserve the global native chunk sequence.
    frames = packer.chunk_plan.chunk(sum(batches) - 1).end
    torch.manual_seed(input_seed)
    inputs, noise = _inputs(frames), torch.randn(1, 2, frames, 2, 2)
    if reference:
        inputs.visual = {"ref": torch.arange(8).reshape(1, 2, 1, 2, 2).float() / 9}
        inputs.media = [{"id": "ref", "type": "image"}]
        inputs.conditions = [{"content_id": "ref", "role": "reference"}]
    extra = {}
    sampler_type = ZingH3Sampler
    if initial_steps is not None:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.realtime_sampling import (
            ZingH3RealtimeSampler,
        )

        sampler_type = ZingH3RealtimeSampler
        extra = dict(initial_denoising_step_list=initial_steps, initial_block_count=3)
    if reference:
        extra.update(ar_condition_timestep=0.875, reference_visual_timestep=0.999)
    sampler = sampler_type(
        model,
        packer,
        solver="few_step",
        denoising_step_list=(
            steps
            if steps is not None
            else [1000, 500]
            if initial_steps is not None
            else STEPS
        ),
        window_chunks=6 if sink_updates else 3,
        sink_chunks=2 if sink_updates else 1,
        window_eviction_timing="after_step0",
        rope_position_mode="block_relative",
        joint_forward=joint_forward,
        **extra,
    )
    storage = _storage(model) if storage is None else storage
    graphs = None
    if backend is not None or graph_owner is not None:
        graph_type = getattr(zing_h3_denoise_graph, "ZingChunkGraphs", None)
        assert graph_type is not None, (
            "SourceReadySession requires whole-chunk graph execution"
        )
        graphs = graph_owner or graph_type(
            backend=None if backend == "eager" else backend,
            max_graphs=16,
            capture_enabled=backend != "eager",
        )
        model.chunk_graphs = graphs
    else:
        model.chunk_graphs = None
    session = SourceReadySession(
        sampler,
        inputs,
        seed=97,
        max_pending_chunks=max(batches),
        max_pending_bytes=1000000,
        total_latent_frames=frames,
        acquire_storage=lambda *_: storage,
        defer_final=defer_final,
        prompt_joint=prompt_joint,
        prefix_loader=None
        if prefix_loader_factory is None
        else prefix_loader_factory(sampler),
    )
    outputs, snapshots, rng_states, boundary = [], [], [], []
    cursor = 0
    cancelled = False
    events = []
    try:
        for batch_index, count in enumerate(batches):
            if sink_updates and batch_index in sink_updates:
                for operation in sink_updates[batch_index]:
                    session.update_sink(operation, 200 + batch_index)
            if prompt_updates and batch_index in prompt_updates:
                size = prompt_updates[batch_index]
                session.update_prompt(
                    session.prompt.condition
                    if reuse_updates and size is None
                    else inputs.positive
                    if size is None
                    else {
                        "context": torch.randn(size, 12).bfloat16(),
                        "row_role": torch.tensor([0, 1, 0] * (size // 3)),
                    },
                    100 + batch_index,
                    **(
                        {"reuse_prefix": True} if reuse_updates and size is None else {}
                    ),
                )
            # Drain whenever a pending prompt blocks lookahead, including the
            # old cycle's final chunk before the new epoch can be planned.
            while count:
                while count and session.next_chunk_size is not None:
                    start = session.next_latent
                    end = min(frames, start + session.next_chunk_size)
                    session.append(
                        inputs.source_visual[:, :, start:end],
                        noise[:, :, start:end],
                        provenance=fixture_provenance(1, end, start, end),
                        final=known_eos and end == frames,
                    )
                    cursor += 1
                    count -= 1

                def bounded_drains():
                    while session.pending:
                        yield from session.drain_ready(max_outputs=output_limit)

                drain = (
                    (session.step() for _ in range(len(session.pending)))
                    if use_step
                    else bounded_drains()
                    if output_limit is not None
                    else session.drain_ready()
                )
                for result in drain:
                    if sink_trace is not None:
                        sink_trace.append(
                            (
                                result.chunk.start,
                                result.chunk.end,
                                result.prompt_event_id,
                                result.sink_event_id,
                                result.sink_chunks,
                                result.sink_chunk_ids,
                            )
                        )
                    outputs.append(result.visual.clone())
                    events.append(result.prompt_event_id)
                    cache = session.cache
                    live = cache._length
                    snapshots.append(
                        (
                            live,
                            tuple(
                                (c.block, c.offset, c.positions.clone())
                                for c in cache.committed
                            ),
                            tuple(
                                (s.k[:live].clone(), s.v[:live].clone())
                                for s in storage
                            ),
                        )
                    )
                    rng_states.append(
                        None
                        if session.rng.cpu_rng is None
                        else session.rng.cpu_rng.clone()
                    )
                    # A lookahead prediction is raw: its boundary is still open
                    # and its scheduler0 eviction belongs to the next iteration.
                    boundary.append(
                        (
                            cache.eviction_target_block,
                            None if cache.active is None else cache.active[0],
                        )
                    )
                    if yield_hook is not None:
                        yield_hook(session, drain, len(outputs))
                    if session.closed and not session.complete:
                        cancelled = True
                        break
                if cancelled:
                    break
            if cancelled:
                break
            assert not session.pending
        if not cancelled and not session.ended:
            session.finish()
        assert (session.complete or (allow_cancel and cancelled)) and session.closed
        assert all(value == 0 for value in session.memory_snapshot().values())
    finally:
        session.dispose()
    return torch.cat(outputs, 2), snapshots, rng_states, boundary, graphs, events


@pytest.mark.parametrize("defer_final", [False, True])
def test_sink_controls_mix_with_prompt_cross_chunk_and_graph_replay(
    model, monkeypatch, defer_final
):
    options = dict(
        batches=(1,) * 18,
        known_eos=False,
        split_first_latent=False,
        defer_final=defer_final,
        sink_updates={
            3: ["off"],
            5: ["refresh"],
            8: ["refresh", "off"],
            10: ["refresh"],
            13: ["off"],
        },
        prompt_updates={3: 9, 8: 9, 13: 9},
    )
    trace = []
    serial = _run_ready_chunks(model, monkeypatch, sink_trace=trace, **options)
    recorded = []
    fixed = _run_ready_chunks(
        model, monkeypatch, backend="eager", sink_trace=recorded, **options
    )
    backend = RecordingGraphBackend()
    replay_trace = []
    replay = _run_ready_chunks(
        model, monkeypatch, backend=backend, sink_trace=replay_trace, **options
    )
    assert trace == recorded == replay_trace
    torch.testing.assert_close(serial[0], fixed[0], rtol=0.02, atol=0.02)
    assert torch.equal(fixed[0], replay[0])
    for a, b in zip(fixed[2], replay[2], strict=True):
        assert (a is None and b is None) or torch.equal(a, b)
    # At latent 8 the prompt is not ready, but off must already take effect.
    assert trace[3][0:5] == (8, 10, None, 203, 0)
    assert trace[4][2] == 103
    # Refresh at the next complete boundary includes the latest generated block.
    assert trace[5][3:] == (205, 2, (3, 4))
    # Latest-wins off remains off through the following prompt epoch reset.
    assert trace[8][3:] == (208, 0, ())
    assert trace[9][4:] == (0, ())
    assert trace[10][3:] == (210, 2, (8, 9))
    assert trace[13][3:] == (213, 0, ())


@pytest.mark.parametrize("operation", ["off", "refresh"])
@pytest.mark.parametrize("use_step", [False, True])
def test_sink_before_first_chunk_preserves_initial_prefix_joint(
    model, monkeypatch, operation, use_step
):
    options = dict(
        batches=(1, 1, 1),
        split_first_latent=False,
        sink_updates={0: [operation]},
        use_step=use_step,
    )
    monkeypatch.setenv("SGLANG_ZING_INITIAL_PREFIX_JOINT", "0")
    expected_trace = []
    expected = _run_ready_chunks(
        model, monkeypatch, sink_trace=expected_trace, **options
    )
    monkeypatch.setenv("SGLANG_ZING_INITIAL_PREFIX_JOINT", "1")
    metrics = []

    def observe(session, *_):
        if session.sampler is not None:
            metrics.append(dict(session.sampler._forward_counts))

    actual_trace = []
    actual = _run_ready_chunks(
        model,
        monkeypatch,
        sink_trace=actual_trace,
        yield_hook=observe,
        **options,
    )
    assert actual_trace == expected_trace
    assert actual_trace[0][3:] == (
        200,
        0 if operation == "off" else 2,
        () if operation == "off" else (0,),
    )
    assert metrics[-1]["initial_joint"] == 1 and metrics[-1]["prefix"] == 0
    torch.testing.assert_close(actual[0], expected[0], rtol=0.02, atol=0.02)
    for got, want in zip(actual[1], expected[1], strict=True):
        assert got[0] == want[0]
        assert [(c[0], c[1]) for c in got[1]] == [(c[0], c[1]) for c in want[1]]
        for (gk, gv), (wk, wv) in zip(got[2], want[2], strict=True):
            torch.testing.assert_close(gk, wk, rtol=0.02, atol=0.02)
            torch.testing.assert_close(gv, wv, rtol=0.02, atol=0.02)


def test_sink_cross_chunk_queue_drains_before_refresh(model, monkeypatch):
    trace = []
    _run_ready_chunks(
        model,
        monkeypatch,
        batches=(2,) * 9,
        known_eos=False,
        split_first_latent=False,
        sink_updates={2: ["refresh"], 4: ["off"]},
        sink_trace=trace,
    )
    assert trace[4][3:] == (202, 2, (2, 3))
    assert trace[8][3:] == (204, 0, ())


def test_source_ready_whole_chunk_three_routes_replay_and_match_eager(
    model, monkeypatch
):
    original = _run_ready_chunks(model, monkeypatch)
    expected = _run_ready_chunks(model, monkeypatch, backend="eager")
    torch.testing.assert_close(expected[0], original[0], rtol=0.02, atol=0.02)
    for fixed, reference in zip(expected[1], original[1], strict=True):
        assert fixed[0] == reference[0]
        for (fk, fv), (rk, rv) in zip(fixed[2], reference[2], strict=True):
            torch.testing.assert_close(fk, rk, rtol=0.02, atol=0.02)
            torch.testing.assert_close(fv, rv, rtol=0.02, atol=0.02)
    backend = RecordingGraphBackend()
    actual = _run_ready_chunks(model, monkeypatch, backend=backend)
    assert torch.equal(actual[0], expected[0])
    assert actual[3] == expected[3]
    for got, want in zip(actual[1], expected[1], strict=True):
        assert got[0] == want[0]
        assert [(c[0], c[1]) for c in got[1]] == [(c[0], c[1]) for c in want[1]]
        for gc, wc in zip(got[1], want[1], strict=True):
            assert torch.equal(gc[2], wc[2])
        for (gk, gv), (wk, wv) in zip(got[2], want[2], strict=True):
            assert torch.equal(gk, wk) and torch.equal(gv, wv)
    for got, want in zip(actual[2], expected[2], strict=True):
        assert (got is None and want is None) or torch.equal(got, want)
    graphs = actual[4]
    assert backend.captures > 0
    for kind in ("steady", "finish", "recovery"):
        assert graphs.routes[(kind, "replay")] > 0, (kind, graphs.routes)
    assert len(graphs.graphs) <= 16
    assert not graphs.routes[("steady", "capture_failed")]
    assert not graphs.routes[("steady", "unsupported")]


@pytest.mark.parametrize("device", ["cpu", "meta", "cuda"])
def test_deferred_final_matches_layout_devices(model, device):
    """GPU payloads do not imply GPU metadata at the joint packing boundary."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device required for the production device layout")
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )

    packer = ZingH3Packer(model.arch, num_frame_per_block=2)
    sampler = ZingH3Sampler(model, packer, solver="few_step", denoising_step_list=STEPS)
    inputs = _inputs(3)
    layout = packer.pack_inference_layout(
        inputs, inputs.positive, torch.zeros(1, 2, 3, 2, 2), None
    )
    for name in ("clean_visual", "clean_audio", "context"):
        layout[name] = layout[name].to(device)
    ordinary = sampler._select_block(
        layout, 2, phase="target", video_t=1.0, audio_t=None, final=True
    )
    host = ordinary["_zing_host_layout"]
    cache = SimpleNamespace(
        active=(2, ordinary["position_ids"]), _active_reference=host.position_ids
    )
    frames = int(host.position_ids[:, 0].unique().numel())
    final = sampler._deferred_final(
        cache, torch.zeros(1, 2, frames, 2, 2, device=device), layout
    )
    for name in ("x", "audio_x", "context", *_sampling_helpers._GUARDED_ROWS):
        assert final[name].device == ordinary[name].device, name
    # Exercise the failing downstream concatenation, not just helper output.
    joined = ZingChunkGraphs._joint_packed([final, ordinary], ("final", "active"))
    assert joined["x"].device.type == device
    assert joined["t"].device.type == "cpu"


@pytest.mark.parametrize("switch", [False, True])
def test_deferred_final_across_single_chunk_dispatches(model, monkeypatch, switch):
    options = dict(
        batches=(1,) * 24,
        initial_steps=STEPS,
        prompt_updates={13: 9} if switch else None,
    )
    baseline = _run_ready_chunks(model, monkeypatch, **options)
    eager = _run_ready_chunks(model, monkeypatch, defer_final=True, **options)
    fixed = _run_ready_chunks(
        model, monkeypatch, defer_final=True, backend="eager", **options
    )
    backend = RecordingGraphBackend()
    replay = _run_ready_chunks(
        model, monkeypatch, defer_final=True, backend=backend, **options
    )
    for result in (eager, fixed, replay):
        torch.testing.assert_close(result[0], baseline[0], rtol=0.02, atol=0.02)
        assert result[5] == baseline[5]
        for actual, expected in zip(result[2], baseline[2], strict=True):
            assert (actual is None and expected is None) or torch.equal(
                actual, expected
            )
    assert torch.equal(replay[0], fixed[0])
    for actual, expected in zip(replay[1], fixed[1], strict=True):
        assert actual[0] == expected[0]
        for (ak, av), (ek, ev) in zip(actual[2], expected[2], strict=True):
            assert torch.equal(ak, ek) and torch.equal(av, ev)
    for kind in ("resume", "deferred"):
        assert replay[4].routes[(kind, "replay")] > 0
    # Output was delivered while the last generated block remained active;
    # final EOS clears it, and a prompt rebuild restarts the local block phase.
    assert all(active is not None for _, active in replay[3][:-1])
    assert replay[3][-1] == (None, None)
    if switch:
        assert replay[5][13:] == [113] * 11


@pytest.mark.parametrize("cross_graph", [False, True])
def test_single_step_deferred_final_captures_without_kv_writes(
    model, monkeypatch, cross_graph
):
    monkeypatch.setenv("SGLANG_ZING_HISTORY_ROPE_CROSS_GRAPH", str(int(cross_graph)))
    options = dict(batches=(1,) * 12, steps=[1000], defer_final=True)
    fixed = _run_ready_chunks(model, monkeypatch, backend="eager", **options)
    replay = _run_ready_chunks(
        model, monkeypatch, backend=WarmupRecordingGraphBackend(), **options
    )
    assert torch.equal(replay[0], fixed[0])
    assert replay[5] == fixed[5]
    assert replay[4].routes[("deferred", "capture")] > 0
    assert replay[4].routes[("deferred", "replay")] > 0
    assert not replay[4].disabled


@pytest.mark.parametrize("local_failure", [False, True])
def test_deferred_prepare_failure_votes_before_joint_and_releases_session(
    model, monkeypatch, local_failure
):
    original = ZingH3Sampler._agree_sp_boundary
    sessions, votes = [], []

    def boundary(sampler, *args, **kwargs):
        if kwargs.get("phase") == "source_deferred_prepare":
            votes.append(kwargs["precheck_error"])
            if not local_failure:
                raise ValueError("injected peer preparation failure")
        return original(sampler, *args, **kwargs)

    def fail(*args):
        raise ValueError("injected local preparation failure")

    def unexpected_joint(*args):
        pytest.fail("Joint eligibility must not run after a failed prepare vote")

    monkeypatch.setattr(ZingH3Sampler, "_agree_sp_boundary", boundary)
    monkeypatch.setattr(ZingH3Sampler, "_resume_cross_packs", unexpected_joint)
    if local_failure:
        monkeypatch.setattr(ZingH3Sampler, "_deferred_final", fail)
    with pytest.raises(ValueError, match="preparation failure"):
        _run_ready_chunks(
            model,
            monkeypatch,
            batches=(1,) * 8,
            defer_final=True,
            yield_hook=lambda session, *_: sessions.append(session),
        )
    assert votes == [local_failure]
    assert sessions and all(s.closed and s.recent_visual is None for s in sessions)


def test_deferred_final_cancel_releases_history_and_cache(model, monkeypatch):
    def cancel(session, drain, count):
        if count == 3:
            drain.close()
            assert session.closed and not session.pending
            assert session.recent_visual is None

    _run_ready_chunks(
        model,
        monkeypatch,
        batches=(1,) * 8,
        defer_final=True,
        backend=RecordingGraphBackend(),
        yield_hook=cancel,
        allow_cancel=True,
    )


def test_deferred_final_separate_eos_and_sequential_fallback(model, monkeypatch):
    options = dict(batches=(1,) * 8, known_eos=False)
    baseline = _run_ready_chunks(model, monkeypatch, **options)
    monkeypatch.setattr(ZingH3Sampler, "_resume_cross_packs", lambda *_: None)
    actual = _run_ready_chunks(model, monkeypatch, defer_final=True, **options)
    assert torch.equal(actual[0], baseline[0])
    # finish() releases this last active block without an unnecessary forward.
    assert actual[3][-1][1] is not None


@pytest.mark.parametrize("backend", ["eager", "recording"])
def test_whole_chunk_execution_unpublishes_rotated_k_and_shares_rotations(
    model, monkeypatch, backend
):
    """Executions rewrite the rotated-K scratch, so reuse is never published.

    Within one execution only the first forward that sees a history rotates it.
    """
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )

    monkeypatch.setenv("SGLANG_ZING_HISTORY_ROPE_CROSS_GRAPH", "0")
    rotations = []
    original = ZingChunkGraphs._forward_plan

    def spy(sampler, shadow, packed, phase, after=None, **kwargs):
        plan = original(sampler, shadow, packed, phase, after=after, **kwargs)
        rotations.append((phase, plan.kv_plan.rotates_history))
        return plan

    monkeypatch.setattr(ZingChunkGraphs, "_forward_plan", staticmethod(spy))
    published = []

    def hook(session, drain, count):
        published.append(dict(session.cache._rotated_store_signatures))

    _run_ready_chunks(
        model,
        monkeypatch,
        backend="eager" if backend == "eager" else RecordingGraphBackend(),
        yield_hook=hook,
    )
    assert published and not any(published)
    active = [rotated for phase, rotated in rotations if phase == "active"]
    # Each steady chunk runs steps - 1 active forwards over one history.
    assert True in active and False in active
    assert active.count(True) * (len(STEPS) - 2) <= active.count(False)


def test_discarding_rotated_k_covers_failed_and_published_executions():
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )

    cache = ZingH3KVCache(2, rope_position_mode="block_relative")
    cache._rotated_store_signatures[0] = ("published",)
    shadow = cache.planning_copy()
    shadow._rotated_store_signatures[1] = ("planned",)
    ZingChunkGraphs._discard_rotated(cache, shadow)
    assert not cache._rotated_store_signatures
    cache.publish_plan(shadow)
    assert not cache._rotated_store_signatures


class WarmupRecordingGraphBackend(RecordingGraphBackend):
    """Match the production backend's eager warmup before operator capture."""

    def __init__(self):
        super().__init__()
        self.warmups = 0

    def capture(self, fn, warmed=None):
        error = None
        try:
            fn()
            self.warmups += 1
        except Exception as failure:
            error = failure
        all_ok = warmed(error is None) if warmed is not None else error is None
        if error is not None or not all_ok:
            raise GraphWarmupError("recording warmup failed") from error
        return super().capture(fn)


@pytest.mark.parametrize("prompt_updates", [None, {2: 6}])
def test_cross_graph_history_reuse_matches_rebuild_across_eviction_and_recovery(
    model, monkeypatch, prompt_updates
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )

    monkeypatch.setenv("SGLANG_ZING_HISTORY_ROPE_CROSS_GRAPH", "0")
    expected = _run_ready_chunks(
        model, monkeypatch, backend="eager", prompt_updates=prompt_updates
    )
    monkeypatch.delenv("SGLANG_ZING_HISTORY_ROPE_CROSS_GRAPH", raising=False)
    plans, states = [], []
    original = ZingChunkGraphs._forward_plan

    def spy(sampler, shadow, packed, phase, after=None, **kwargs):
        plan = original(sampler, shadow, packed, phase, after=after, **kwargs)
        if after is None:
            plans.append(
                (phase, plan.kv_plan.write_start, plan.kv_plan.reused_history_rows)
            )
        return plan

    def observe(session, drain, count):
        cache = session.cache
        states.append(
            (dict(cache._rotated_store_signatures), dict(cache._raw_write_epochs))
        )

    monkeypatch.setattr(ZingChunkGraphs, "_forward_plan", staticmethod(spy))
    backend = WarmupRecordingGraphBackend()
    actual = _run_ready_chunks(
        model,
        monkeypatch,
        backend=backend,
        prompt_updates=prompt_updates,
        yield_hook=observe,
    )
    assert torch.equal(actual[0], expected[0])
    assert actual[3] == expected[3] and actual[5] == expected[5]
    for got, want in zip(actual[1], expected[1], strict=True):
        assert got[0] == want[0]
        assert [(c[0], c[1]) for c in got[1]] == [(c[0], c[1]) for c in want[1]]
        for (gk, gv), (wk, wv) in zip(got[2], want[2], strict=True):
            assert torch.equal(gk, wk) and torch.equal(gv, wv)
    assert any(0 < reused < live for _, live, reused in plans)
    assert any(live == 0 and reused == 0 for _, live, reused in plans)
    assert any(published for published, _ in states)
    # Owned raw writes and compaction must not masquerade as external mutations.
    assert all(not any(epochs.values()) for _, epochs in states)
    assert backend.captures > 0 and backend.warmups == backend.captures
    assert actual[4].routes["steady", "replay"] > 0


def test_cross_graph_unpublishes_before_execution_and_publishes_only_on_success(
    monkeypatch,
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy

    monkeypatch.delenv("SGLANG_ZING_HISTORY_ROPE_CROSS_GRAPH", raising=False)
    stores = allocate_layer_stores(
        32,
        num_layers=1,
        local_heads=2,
        head_dim=8,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    cache = ZingH3KVCache(
        1,
        rope_position_mode="block_relative",
        storage=stores,
        window_policy=ZingWindowPolicy(3, 1),
    )
    cache._rotated_store_signatures[0] = ("previous",)
    shadow = cache.planning_copy()
    shadow._rotated_store_signatures[0] = ("planned",)
    ZingChunkGraphs._discard_rotated(cache, shadow)
    # Failure leaves the real cache invalid; the uncommitted shadow is private.
    assert not cache._rotated_store_signatures
    assert shadow._rotated_store_signatures == {0: ("planned",)}
    cache.publish_plan(shadow)
    assert cache._rotated_store_signatures == {0: ("planned",)}
    cache.clear()
    assert not cache._rotated_store_signatures


@torch.no_grad()
def test_cross_graph_capture_restores_reused_rows_overwritten_by_later_forward():
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )

    k, v, rotated = (torch.zeros(1024, 2, 8) for _ in range(3))
    rotated[:8] = 10
    plans = [
        SimpleNamespace(
            signature=(index,),
            kv_plan=SimpleNamespace(
                write_start=8,
                rows=2,
                clean_rows=0,
                rotates_history=True,
                reused_history_rows=reused,
            ),
        )
        for index, reused in enumerate((8, 3))
    ]
    views = ZingChunkGraphs._capture_store_views(plans, [(k, v)], (rotated,))
    assert len(views) == 2  # V tail plus only five threatened historical rows.
    assert views[-1].data_ptr() == rotated[3:8].data_ptr() and views[-1].shape[0] == 5
    backend = WarmupRecordingGraphBackend()
    graphs = ZingChunkGraphs(backend=backend)

    def execute(static):
        result = rotated[:8].clone()  # First forward reuses entry-time rotations.
        rotated[3:10].add_(static["delta"])  # Later map overwrites part of them.
        v[8:10].add_(static["delta"])
        return (result,)

    for step in range(3):
        (out,) = graphs._run(
            "steady", plans, {"delta": torch.tensor(1.0)}, execute, [(k, v)], (rotated,)
        )
        assert torch.all(out[:3] == 10) and torch.all(out[3:8] == 10 + step)
        assert torch.all(rotated[:3] == 10) and torch.all(rotated[3:8] == 11 + step)
    assert backend.captures == 1 and graphs.routes["steady", "replay"] == 1


def test_source_ready_whole_chunk_warmup_capture_do_not_double_apply(
    model, monkeypatch
):
    expected = _run_ready_chunks(model, monkeypatch, backend="eager")
    backend = WarmupRecordingGraphBackend()
    actual = _run_ready_chunks(model, monkeypatch, backend=backend)
    assert backend.warmups == backend.captures and backend.captures > 0
    assert torch.equal(actual[0], expected[0])
    assert actual[3] == expected[3]
    for got, want in zip(actual[2], expected[2], strict=True):
        assert (got is None and want is None) or torch.equal(got, want)
    for got, want in zip(actual[1], expected[1], strict=True):
        assert got[0] == want[0]
        for (gk, gv), (wk, wv) in zip(got[2], want[2], strict=True):
            assert torch.equal(gk, wk) and torch.equal(gv, wv)


@pytest.mark.parametrize("failed_phase", ["warmup", "capture"])
def test_chunk_capture_peer_failure_votes_once_and_restores_store(failed_phase):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )

    class PeerFailure:
        def __init__(self):
            self.flags = []

        def agree(self, route):
            return True

        def all_ok(self, flag):
            self.flags.append(flag)
            return failed_phase == "capture" and len(self.flags) == 1

    backend = WarmupRecordingGraphBackend()
    graphs = ZingChunkGraphs(backend=backend)
    k, v = torch.zeros(1024, 2, 8), torch.zeros(1024, 2, 8)
    plans = [
        SimpleNamespace(
            signature=("test",),
            kv_plan=SimpleNamespace(write_start=100, rows=3, clean_rows=3),
        )
    ]
    tensors = {"delta": torch.tensor(1.0)}

    def execute(static):
        k[100:103].add_(static["delta"])
        v[100:103].add_(static["delta"])
        return (k.clone(),)

    graphs._run("steady", plans, tensors, execute, [(k, v)])
    before = k.clone(), v.clone()
    peer = PeerFailure()
    graphs.coordinator = peer
    expected_error = GraphWarmupError if failed_phase == "warmup" else RuntimeError
    with pytest.raises(expected_error):
        graphs._run("steady", plans, tensors, execute, [(k, v)])
    # A rank that receives a failed postcapture vote must leave directly;
    # voting again would block forever after the failing peer already left.
    assert peer.flags == ([True] if failed_phase == "warmup" else [True, True])
    assert torch.equal(k, before[0]) and torch.equal(v, before[1])
    assert graphs.disabled and not graphs.graphs


@pytest.mark.parametrize("clean", [False, True])
def test_chunk_capture_snapshot_is_bounded_by_writes_not_pool_capacity(clean):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )

    stores = [(torch.zeros(1024, 2, 8), torch.zeros(1024, 2, 8)) for _ in range(2)]
    plans = [
        SimpleNamespace(
            kv_plan=SimpleNamespace(write_start=start, rows=rows, clean_rows=count)
        )
        for start, rows, count in ((100, 5, 2 if clean else 0), (107, 4, int(clean)))
    ]
    views = ZingChunkGraphs._capture_store_views(plans, stores)
    expected = ([pair[0][100:108] for pair in stores] if clean else []) + [
        pair[1][100:111] for pair in stores
    ]
    assert len(views) == len(expected)
    for got, want in zip(views, expected, strict=True):
        assert got.data_ptr() == want.data_ptr() and got.shape == want.shape
    assert sum(view.numel() for view in views) == (38 if clean else 22) * 2 * 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_chunk_cuda_capture_restores_write_spans_and_preserves_history():
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )

    k = torch.zeros(1024, 2, 8, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    plans = [
        SimpleNamespace(
            signature=("bounded",),
            kv_plan=SimpleNamespace(write_start=100, rows=3, clean_rows=2),
        )
    ]
    graphs = ZingChunkGraphs()
    tensors = {"delta": torch.ones((), device="cuda", dtype=k.dtype)}

    def execute(static):
        k[100:102].add_(static["delta"])
        v[100:103].add_(static["delta"])
        return (k[100:102] + v[100:102],)

    for step in range(1, 4):  # eager warm-up, capture, replay
        (result,) = graphs._run("steady", plans, tensors, execute, [(k, v)])
        expected_k, expected_v = torch.zeros_like(k), torch.zeros_like(v)
        expected_k[100:102] = step
        expected_v[100:103] = step
        assert torch.equal(k, expected_k) and torch.equal(v, expected_v)
        assert torch.equal(result, torch.full_like(result, 2 * step))
    assert graphs.routes["steady", "capture"] == 1
    assert graphs.routes["steady", "replay"] == 1


def test_source_ready_init_replays_prefix_and_isolates_reused_session_slot(
    model, monkeypatch
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )

    eager_owner = ZingChunkGraphs(capture_enabled=False)
    backend = WarmupRecordingGraphBackend()
    graph_owner = ZingChunkGraphs(backend=backend)
    eager_storage, graph_storage = _storage(model), _storage(model)
    prefixes = []
    forward = model.forward
    for seed in (41, 42, 43, 44):
        expected = _run_ready_chunks(
            model,
            monkeypatch,
            batches=(3,),
            graph_owner=eager_owner,
            storage=eager_storage,
            input_seed=seed,
        )
        called = []

        def observe(*args, **kwargs):
            if kwargs.get("self_cache_update") == "prefix":
                called.append(True)
            return forward(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(model, "forward", observe)
            actual = _run_ready_chunks(
                model,
                monkeypatch,
                batches=(3,),
                graph_owner=graph_owner,
                storage=graph_storage,
                input_seed=seed,
            )
        prefixes.append(len(called))
        assert torch.equal(actual[0], expected[0])
        assert actual[3] == expected[3]
        for got, want in zip(actual[2], expected[2], strict=True):
            assert (got is None and want is None) or torch.equal(got, want)
        for got, want in zip(actual[1], expected[1], strict=True):
            assert got[0] == want[0]
            for (gk, gv), (wk, wv) in zip(got[2], want[2], strict=True):
                assert torch.equal(gk, wk) and torch.equal(gv, wv)
    assert graph_owner.routes[("init", "replay")] > 0, graph_owner.routes
    assert prefixes[-1] == 0, "init replay must include prefix model work"
    assert len(graph_owner.graphs) <= 16


def test_source_ready_graph_eligibility_is_agreed_before_execution(model, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from copy import copy
    from threading import Barrier

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        sampling,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import fixture_provenance
    from sglang.multimodal_gen.test.unit.test_zing_h3_sp_cpu import _CPUCollectives

    comm = _CPUCollectives(4)
    comm.barrier = Barrier(4, timeout=3)
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (4, comm.thread.rank))
    monkeypatch.setattr(
        zing_h3, "get_sp_group", lambda: SimpleNamespace(cpu_group=object())
    )

    def gather_objects(output, value, group=None):
        output[:] = comm.exchange(value, lambda values, _: list(values))

    monkeypatch.setattr(sampling.dist, "all_gather_object", gather_objects)
    # Prefix model kernels are outside this routing contract. No model body
    # may run before the four CPU ranks resolve graph eligibility.
    monkeypatch.setattr(SourceReadySession, "_apply_prefix", lambda *_: None)
    inputs, noise = _inputs(3), torch.randn(1, 2, 3, 2, 2)
    entered = []

    class GraphTrap:
        def eligible(self, *args):
            return True

        def recovery(self, *args):
            entered.append(comm.thread.rank)
            raise AssertionError("graph executed before unanimous eligibility")

    rank_models = [copy(model) for _ in range(4)]
    for rank, rank_model in enumerate(rank_models):
        rank_model.chunk_graphs = None if rank == 3 else GraphTrap()

    def worker(rank):
        comm.thread.rank = rank
        rank_model = rank_models[rank]
        packer = ZingH3Packer(rank_model.arch, num_frame_per_block=2)
        packer.num_frame_per_block = [3, 2]
        packer.split_first_latent = True
        sampler = ZingH3Sampler(
            rank_model,
            packer,
            solver="few_step",
            denoising_step_list=STEPS,
            window_chunks=3,
            sink_chunks=1,
            joint_forward="cross_chunk",
            window_eviction_timing="after_step0",
        )
        session = SourceReadySession(
            sampler,
            inputs,
            seed=5,
            max_pending_chunks=1,
            max_pending_bytes=1000000,
            total_latent_frames=3,
            acquire_storage=lambda *_: _storage(rank_model),
        )
        chunk = packer.chunk_plan.chunk(0, 3)
        session.append(
            inputs.source_visual[:, :, chunk.start : chunk.end],
            noise[:, :, chunk.start : chunk.end],
            provenance=fixture_provenance(1, chunk.end, chunk.start, chunk.end),
        )
        try:
            list(session.drain_ready())
        except Exception as error:
            return type(error).__name__, str(error), session.closed
        finally:
            session.dispose()
        return "NO_ERROR", "", False

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(worker, range(4)))
    assert all(
        kind == "ValueError" and "mismatch" in message and closed
        for kind, message, closed in results
    ), results
    assert len({message for _, message, _ in results}) == 1
    assert not entered, entered


def test_chunk_local_capture_failure_and_peer_leave_cpu_barrier(model, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_sp_cpu import _CPUCollectives

    comm = _CPUCollectives(2)
    comm.barrier = Barrier(2, timeout=3)

    class Coordinator:
        def __init__(self):
            self.flags = []

        def agree(self, route):
            return comm.exchange(route, lambda values, _: len(set(values)) == 1)

        def all_ok(self, flag):
            self.flags.append(flag)
            return comm.exchange(flag, lambda values, _: all(values))

    class InjectedBackend(WarmupRecordingGraphBackend):
        def capture(self, fn, warmed=None):
            result = super().capture(fn, warmed=warmed)
            if comm.thread.rank == 0:
                raise RuntimeError("injected local capture failure")
            return result

    def worker(rank):
        comm.thread.rank = rank
        graphs = ZingChunkGraphs(backend=InjectedBackend())
        coordinator = Coordinator()
        graphs.coordinator = coordinator
        k, v = torch.zeros(3, 2, 8), torch.zeros(3, 2, 8)
        plans, tensors = (
            [
                SimpleNamespace(
                    signature=("barrier",),
                    kv_plan=SimpleNamespace(write_start=0, rows=3, clean_rows=3),
                )
            ],
            {"delta": torch.tensor(1.0)},
        )

        def execute(static):
            k.add_(static["delta"])
            v.add_(static["delta"])
            return (k.clone(),)

        graphs._run("steady", plans, tensors, execute, [(k, v)])
        before = k.clone(), v.clone()
        try:
            graphs._run("steady", plans, tensors, execute, [(k, v)])
        except Exception as error:
            return (
                type(error).__name__,
                str(error),
                coordinator.flags,
                graphs.disabled,
                torch.equal(k, before[0]) and torch.equal(v, before[1]),
            )
        return "NO_ERROR", "", coordinator.flags, graphs.disabled, False

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, rank) for rank in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert all(
        kind == "RuntimeError" and disabled and restored
        for kind, _, _, disabled, restored in results
    ), results
    assert all(len(flags) == 2 for _, _, flags, _, _ in results), results
    assert "local capture failure" in results[0][1] and "peer rank" in results[1][1]


def test_low_level_kv_cache_defaults_to_after_step0():
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache

    assert ZingH3KVCache(1).window_eviction_timing == "after_step0"


@pytest.mark.parametrize("stable_shapes", [False, True])
def test_init_prefix_prepare_failure_agrees_before_next_collective(
    model, monkeypatch, stable_shapes
):
    from concurrent.futures import ThreadPoolExecutor
    from copy import copy
    from threading import Barrier

    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        sampling,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import fixture_provenance
    from sglang.multimodal_gen.test.unit.test_zing_h3_sp_cpu import _CPUCollectives

    comm = _CPUCollectives(2)
    comm.barrier = Barrier(2, timeout=3)
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (2, comm.thread.rank))
    monkeypatch.setattr(
        zing_h3,
        "get_sp_group",
        lambda: SimpleNamespace(cpu_group=object(), ring_world_size=1),
    )

    def gather_objects(output, value, group=None):
        output[:] = comm.exchange(value, lambda values, _: list(values))

    monkeypatch.setattr(sampling.dist, "all_gather_object", gather_objects)
    original = ZingH3DiTModel.prepare_fixed_forward

    def fail_prefix(self, **kwargs):
        prefix = (
            kwargs["phase"] == "prefix"
            or kwargs["packed"].get("_zing_joint_phases", (None,))[0] == "prefix"
        )
        if comm.thread.rank == 0 and prefix:
            raise RuntimeError("injected prefix plan failure")
        return original(self, **kwargs)

    monkeypatch.setattr(ZingH3DiTModel, "prepare_fixed_forward", fail_prefix)

    class Coordinator:
        def all_ok(self, flag):
            return comm.exchange(flag, lambda values, _: all(values))

    inputs, noise = _inputs(5), torch.randn(1, 2, 5, 2, 2)
    rank_models = [copy(model), copy(model)]
    # This fault injection exercises admitted SP graphs, independent of env defaults.
    for rank_model in rank_models:
        rank_model.denoise_graph_sp = True
        rank_model.experimental_stable_shapes = stable_shapes
    monkeypatch.setattr(zing_h3, "get_tp_world_size", lambda: 1)

    @torch.no_grad()
    def worker(rank):
        comm.thread.rank = rank
        backend = RecordingGraphBackend()
        graphs = ZingChunkGraphs(backend=backend)
        graphs.coordinator = Coordinator()
        rank_models[rank].chunk_graphs = graphs
        packer = ZingH3Packer(model.arch, num_frame_per_block=2)
        packer.num_frame_per_block = [3, 2]
        packer.split_first_latent = True
        sampler = ZingH3Sampler(
            rank_models[rank],
            packer,
            solver="few_step",
            denoising_step_list=STEPS,
            window_chunks=3,
            sink_chunks=1,
            joint_forward="cross_chunk",
            window_eviction_timing="after_step0",
        )
        storage = _storage(model)
        for store in storage:
            store.k.fill_(2)
            store.v.fill_(3)
        before = [(s.k.clone(), s.v.clone()) for s in storage]
        session = SourceReadySession(
            sampler,
            inputs,
            seed=5,
            max_pending_chunks=1,
            max_pending_bytes=1000000,
            total_latent_frames=5,
            acquire_storage=lambda *_: storage,
        )
        chunk = packer.chunk_plan.chunk(0, 5)
        session.append(
            inputs.source_visual[:, :, chunk.start : chunk.end],
            noise[:, :, chunk.start : chunk.end],
            provenance=fixture_provenance(1, chunk.end, chunk.start, chunk.end),
        )
        try:
            current, _ = session._prepare(session.pending[0], session.prompt, None)
            graphs.initialization(sampler, current)
        except Exception as error:
            session.dispose()
            restored = all(
                torch.equal(s.k, k) and torch.equal(s.v, v)
                for s, (k, v) in zip(storage, before)
            )
            return (
                type(error).__name__,
                str(error),
                session.closed,
                restored,
                backend.captures,
            )
        finally:
            session.dispose()
        return "NO_ERROR", "", False, False, backend.captures

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, rank) for rank in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert results[0][:2] == ("RuntimeError", "injected prefix plan failure"), results
    assert results[1][0] == "ValueError" and "peer rank" in results[1][1], results
    assert all(
        closed and restored and captures == 0
        for _, _, closed, restored, captures in results
    ), results


def test_chunk_session_rejects_live_weight_mutation(model, monkeypatch):
    observed = []

    def mutate(session, drain, count):
        if count == 1:
            observed.append((session, torch.get_rng_state().clone()))
            with torch.no_grad():
                model.blocks[0].mlp.fc2.weight.add_(0.25)

    with pytest.raises(ValueError, match="model changed"):
        _run_ready_chunks(
            model, monkeypatch, backend=WarmupRecordingGraphBackend(), yield_hook=mutate
        )
    session, caller_rng = observed[0]
    assert session.closed and not session.pending
    assert all(value == 0 for value in session.memory_snapshot().values())
    assert torch.equal(torch.get_rng_state(), caller_rng)


@pytest.mark.parametrize(
    "options",
    [{"initial_steps": [1000, 750, 500, 250]}, {"prompt_updates": {1: 6, 2: 3, 3: 9}}],
)
def test_chunk_replay_schedule_and_prompt_transitions(model, monkeypatch, options):
    original = _run_ready_chunks(model, monkeypatch, **options)
    expected = _run_ready_chunks(model, monkeypatch, backend="eager", **options)
    torch.testing.assert_close(expected[0], original[0], rtol=0.02, atol=0.02)
    actual = _run_ready_chunks(
        model, monkeypatch, backend=WarmupRecordingGraphBackend(), **options
    )
    assert torch.equal(actual[0], expected[0])
    assert actual[3] == expected[3] and actual[5] == expected[5]
    for got, want in zip(actual[2], expected[2], strict=True):
        assert (got is None and want is None) or torch.equal(got, want)
    for got, want in zip(actual[1], expected[1], strict=True):
        assert got[0] == want[0]
        for (gk, gv), (wk, wv) in zip(got[2], want[2], strict=True):
            assert torch.equal(gk, wk) and torch.equal(gv, wv)
    assert len(actual[4].graphs) <= 16
    assert actual[4].routes[("steady", "replay")] > 0
    if "prompt_updates" in options:
        assert set(actual[5]) == {None, 101, 102, 103}


def test_chunk_cancellation_releases_state_and_next_session_reuses_slot(
    model, monkeypatch
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )

    eager = ZingChunkGraphs(capture_enabled=False)
    recorded = ZingChunkGraphs(backend=WarmupRecordingGraphBackend())
    eager_store, recorded_store = _storage(model), _storage(model)
    closed = []

    def cancel(session, drain, count):
        if count == 2:
            drain.close()
            closed.append(session)

    expected = _run_ready_chunks(
        model,
        monkeypatch,
        graph_owner=eager,
        storage=eager_store,
        yield_hook=cancel,
        allow_cancel=True,
    )
    actual = _run_ready_chunks(
        model,
        monkeypatch,
        graph_owner=recorded,
        storage=recorded_store,
        yield_hook=cancel,
        allow_cancel=True,
    )
    assert torch.equal(actual[0], expected[0])
    assert actual[3] == expected[3]
    assert all(
        s.closed
        and not s.complete
        and not s.pending
        and all(v == 0 for v in s.memory_snapshot().values())
        for s in closed
    )
    expected = _run_ready_chunks(
        model, monkeypatch, graph_owner=eager, storage=eager_store, input_seed=104
    )
    actual = _run_ready_chunks(
        model, monkeypatch, graph_owner=recorded, storage=recorded_store, input_seed=104
    )
    assert torch.equal(actual[0], expected[0])
    for got, want in zip(actual[1], expected[1], strict=True):
        for (gk, gv), (wk, wv) in zip(got[2], want[2], strict=True):
            assert torch.equal(gk, wk) and torch.equal(gv, wv)
    assert len(recorded.graphs) <= 16


def test_whole_chunk_forward_hooks_keep_streaming_eager(model, monkeypatch):
    calls = []

    def change_output(module, args, output):
        calls.append(None)
        return output + 0.125

    handle = model.blocks[0].mlp.register_forward_hook(change_output)
    try:
        expected = _run_ready_chunks(model, monkeypatch)
        expected_calls = len(calls)
        calls.clear()
        backend = RecordingGraphBackend()
        actual = _run_ready_chunks(model, monkeypatch, backend=backend)
        assert backend.captures == 0, actual[4].routes
        assert not actual[4].graphs
        assert not any(route == "replay" for _, route in actual[4].routes)
        assert len(calls) == expected_calls > 0
        assert torch.equal(actual[0], expected[0])
        assert actual[3] == expected[3]
        for got, want in zip(actual[1], expected[1], strict=True):
            assert got[0] == want[0]
            for (gk, gv), (wk, wv) in zip(got[2], want[2], strict=True):
                assert torch.equal(gk, wk) and torch.equal(gv, wv)
    finally:
        handle.remove()


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("enabled,ring,tp", [(False, 1, 1), (True, 2, 1), (True, 1, 2)])
def test_whole_chunk_respects_sp_graph_topology(
    model, monkeypatch, world, enabled, ring, tp
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )

    model.denoise_graph_sp = enabled
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (world, 0))
    monkeypatch.setattr(
        zing_h3,
        "get_sp_group",
        lambda: SimpleNamespace(cpu_group=object(), ring_world_size=ring),
    )
    monkeypatch.setattr(zing_h3, "get_tp_world_size", lambda: tp)
    sampler = SimpleNamespace(
        generator=model,
        window_eviction_timing="after_step0",
        joint_forward="cross_chunk",
        solver="few_step",
        do_cfg=False,
        window_policy=SimpleNamespace(window_chunks=3),
    )
    cache = SimpleNamespace(pooled=True, raw_store=True)
    current = SimpleNamespace(audio=None, visual=torch.zeros(1))
    backend = RecordingGraphBackend()
    graphs = ZingChunkGraphs(backend=backend)
    for _ in range(4):
        assert not graphs.eligible(sampler, cache, current)
    assert backend.captures == 0 and not graphs.graphs and not graphs.routes


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("cached_prefix", [False, True])
def test_whole_chunk_sp_opt_out_streaming_never_captures(
    model, monkeypatch, world, cached_prefix
):
    from concurrent.futures import ThreadPoolExecutor

    from sglang.multimodal_gen.runtime.layers import usp
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        sampling,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_sp_cpu import _CPUCollectives

    arch = replace(
        model.arch,
        num_attention_heads=4,
        hidden_size=32,
        adaln_out_features=6 * model.arch.num_modalities * 32,
        final_adaln_out_features=64,
    )
    rank_models = [
        ZingH3DiTModel(ZingH3DiTConfig(arch_config=arch), {}).eval()
        for _ in range(world)
    ]
    for rank_model in rank_models:
        rank_model.denoise_graph_sp = False
        if cached_prefix:
            rank_model.experimental_stable_shapes = True
        with torch.no_grad():
            for name, value in rank_model.named_parameters():
                value.copy_(
                    torch.ones_like(value)
                    if "norm" in name and name.endswith("weight")
                    else torch.randn_like(value) * 0.08
                )
            rank_model.rope.inv_freq.fill_(1)
    comm = _CPUCollectives(world)
    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (world, comm.thread.rank))
    monkeypatch.setattr(
        zing_h3,
        "get_sp_group",
        lambda: SimpleNamespace(ulysses_group=None, cpu_group=comm, ring_world_size=1),
    )
    monkeypatch.setattr(zing_h3, "get_tp_world_size", lambda: 1)
    monkeypatch.setattr(zing_h3, "gather_sequence_varlen", comm.gather)
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: world)
    monkeypatch.setattr(usp, "get_ulysses_parallel_rank", lambda: comm.thread.rank)
    monkeypatch.setattr(usp, "_ipc_ready_group", lambda: None)
    monkeypatch.setattr(usp, "_usp_all_to_all_single_varlen", comm.all_to_all)
    monkeypatch.setattr(sampling.dist, "all_gather_object", comm.gather_objects)

    def all_reduce(value, op, group=None):
        reduced = comm.exchange(
            value.clone(),
            lambda values, _: (
                torch.stack(values).amin(0)
                if op == sampling.dist.ReduceOp.MIN
                else torch.stack(values).amax(0)
            ),
        )
        value.copy_(reduced)

    monkeypatch.setattr(sampling.dist, "all_reduce", all_reduce)

    def all_gather(output, value, group=None):
        values = comm.exchange(value.clone(), lambda values, _: list(values))
        for destination, source in zip(output, values, strict=True):
            destination.copy_(source)

    monkeypatch.setattr(sampling.dist, "all_gather", all_gather)

    @torch.no_grad()
    def worker(rank):
        comm.thread.rank = rank
        backend = WarmupRecordingGraphBackend()
        storage = allocate_layer_stores(
            256,
            num_layers=len(model.blocks),
            local_heads=arch.num_attention_heads // world,
            head_dim=arch.attention_head_dim,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        actual = _run_ready_chunks(
            rank_models[rank],
            monkeypatch,
            backend=backend,
            storage=storage,
            batches=(1,) * 17 if cached_prefix else (4, 4, 4, 5),
            prompt_updates={4: 9} if cached_prefix else None,
            prefix_loader_factory=_cached_prefix_writer([]) if cached_prefix else None,
        )
        assert torch.isfinite(actual[0]).all()
        assert backend.captures == 0, actual[4].routes
        assert not actual[4].graphs
        assert not any(route == "replay" for _, route in actual[4].routes)
        assert actual[0].shape[2] == 40  # Seventeen native 1,2,2,3,2,... chunks.

    with ThreadPoolExecutor(max_workers=world) as executor:
        futures = [executor.submit(worker, rank) for rank in range(world)]
        for future in futures:
            future.result(timeout=60)


@pytest.mark.parametrize("defer_final", [False, True])
@pytest.mark.parametrize("weights_free", [False, True])
@pytest.mark.parametrize("split_first_latent", [False, True])
def test_online_prompt_joint_matches_serial_and_replays(
    model, monkeypatch, defer_final, weights_free, tmp_path, split_first_latent
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_prefix_graph import (
        PrefixJointGraphs,
    )

    options = dict(
        batches=(1,) * 16,
        initial_steps=STEPS,
        prompt_updates={4: 9, 7: 9, 10: 9, 13: 9},
        defer_final=defer_final,
        split_first_latent=split_first_latent,
    )
    baseline = _run_ready_chunks(model, monkeypatch, **options)
    if weights_free:
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
            initial_blocks_plan,
            video_plan,
        )
        from sglang.multimodal_gen.runtime.models.schedulers.scheduling_zing_h3_causal_renoise import (
            ZingH3CausalRenoiseSchedule,
        )

        plans = []
        for steps in ([1000, 500], STEPS):
            schedule = ZingH3CausalRenoiseSchedule(
                steps, num_train_timesteps=1000, shift=5.0
            )
            plans.append(
                video_plan("few_step", schedule.timesteps, sigmas=schedule.sigmas)
            )
        model = _weights_free_model(
            model, tmp_path, initial_blocks_plan(*plans, count=3)
        )
    eager = _run_ready_chunks(model, monkeypatch, prompt_joint=True, **options)
    model.prefix_graphs = PrefixJointGraphs(capture_enabled=False)
    fixed = _run_ready_chunks(
        model, monkeypatch, prompt_joint=True, backend="eager", **options
    )
    prefix_backend = WarmupRecordingGraphBackend()
    model.prefix_graphs = PrefixJointGraphs(backend=prefix_backend)
    replay = _run_ready_chunks(
        model,
        monkeypatch,
        prompt_joint=True,
        backend=WarmupRecordingGraphBackend(),
        **options,
    )
    for actual in (eager, fixed, replay):
        torch.testing.assert_close(actual[0], baseline[0], atol=0.02, rtol=0.02)
        assert actual[5] == baseline[5]
        for got, want in zip(actual[2], baseline[2]):
            assert (got is None and want is None) or torch.equal(got, want)
    assert torch.equal(fixed[0], replay[0])
    assert model.prefix_graphs.routes[("prefix_separate", "replay")] == 2
    if split_first_latent:
        assert replay[5][:5] == [None] * 5
        assert replay[5][5:7] == [104] * 2
        assert replay[5][7:11] == [107] * 4
        assert replay[5][11:13] == [110] * 2
        assert replay[5][13:] == [113] * 3
    for actual, expected in zip(replay[1], fixed[1]):
        assert actual[0] == expected[0]
        for (ak, av), (ek, ev) in zip(actual[2], expected[2]):
            assert torch.equal(ak, ek) and torch.equal(av, ev)
    if weights_free:
        assert model.adaln_table.active_plans == 0
        assert model.adaln_table.hits > 0
        assert model.adaln_table.stats()["fallback"] == 0


@pytest.mark.parametrize("split_first_latent", [False, True])
def test_online_prompt_joint_failure_releases_session_and_reuses_slot(
    model, monkeypatch, split_first_latent
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_prefix_graph import (
        PrefixJointGraphs,
    )

    storage = _storage(model)
    options = dict(
        batches=(1,) * 10,
        prompt_updates={4: 9},
        defer_final=True,
        prompt_joint=True,
        storage=storage,
        backend="eager",
        split_first_latent=split_first_latent,
    )
    model.prefix_graphs = PrefixJointGraphs(capture_enabled=False)
    owned = []

    def inject(session, drain, count):
        if count == 4:
            owned.append(session)

            def fail(*args):
                raise RuntimeError("injected prefix failure")

            monkeypatch.setattr(model.prefix_graphs, "run", fail)

    with pytest.raises(RuntimeError, match="injected prefix failure"):
        _run_ready_chunks(model, monkeypatch, yield_hook=inject, **options)
    assert owned[0].closed and owned[0]._pending_prefix is None
    assert all(v == 0 for v in owned[0].memory_snapshot().values())
    model.prefix_graphs = PrefixJointGraphs(capture_enabled=False)
    result = _run_ready_chunks(model, monkeypatch, **options)
    assert result[5][-1] == 104
    if split_first_latent:
        assert result[5][:5] == [None] * 5
        assert result[5][5:] == [104] * 5


def test_online_prompt_joint_unsupported_route_keeps_serial_result(model, monkeypatch):
    options = dict(batches=(1,) * 9, prompt_updates={4: 9}, joint_forward="off")
    baseline = _run_ready_chunks(model, monkeypatch, **options)
    metrics = []
    actual = _run_ready_chunks(
        model,
        monkeypatch,
        prompt_joint=True,
        yield_hook=lambda session, drain, count: metrics.append(session.metrics),
        **options,
    )
    assert torch.equal(actual[0], baseline[0])
    assert actual[5] == baseline[5]
    assert metrics[-1]["prompt_joint_fallbacks"] == {"unsupported_layout": 1}
    assert metrics[-1]["forward_counts"].get("prompt_joint", 0) == 0


@pytest.mark.parametrize("switch", [False, True])
def test_bounded_pair_drain_defers_joint_lookahead_until_next_output(
    model, monkeypatch, switch
):
    options = dict(
        batches=(1,) + (2,) * 12,
        initial_steps=STEPS,
        steps=[1000, 500],
        prompt_updates={7: 9} if switch else None,
    )
    baseline = _run_ready_chunks(model, monkeypatch, **options)
    eager = _run_ready_chunks(
        model, monkeypatch, defer_final=True, output_limit=1, **options
    )
    fixed = _run_ready_chunks(
        model, monkeypatch, defer_final=True, output_limit=1, backend="eager", **options
    )
    backend = RecordingGraphBackend()
    replay = _run_ready_chunks(
        model, monkeypatch, defer_final=True, output_limit=1, backend=backend, **options
    )
    for actual in (eager, fixed, replay):
        torch.testing.assert_close(actual[0], baseline[0], rtol=0.02, atol=0.02)
        assert actual[5] == baseline[5]
        for current, expected in zip(actual[2], baseline[2], strict=True):
            assert (current is None and expected is None) or torch.equal(
                current, expected
            )
    assert torch.equal(replay[0], fixed[0])
    assert all(active is not None for _, active in replay[3][:-1])
    assert replay[3][-1] == (None, None)
    assert replay[4].routes[("resume", "replay")] > 0
    assert replay[4].routes[("deferred", "replay")] > 0


def _cached_prefix_writer(receipts, *, fail=None):
    """Fixture producer: compute donor KV, then publish it into the leased pool.

    This is a correctness oracle, not a cache lookup/performance measurement.
    Donor model work is deliberately excluded from consumer forward counters.
    """
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache

    def factory(sampler):
        def load(cache, inputs, prompt, prefix):
            if fail is not None:
                return fail(cache, inputs, prompt, prefix)
            donor = ZingH3KVCache(
                cache.num_layers,
                window_policy=cache.window_policy,
                window_eviction_timing=cache.window_eviction_timing,
                rope_position_mode=cache.rope_position_mode,
                rope_max_frame_gap=cache.rope_max_frame_gap,
            )
            counts, sampler._forward_counts = sampler._forward_counts, None
            try:
                sampler._forward(prefix, donor, "prefix")
            finally:
                sampler._forward_counts = counts
            layers = [(k.clone(), v.clone()) for k, v in donor.prefix.layers]
            cache.clear()
            host = sampler._layout_index(prefix).host
            cache.update(
                prefix["position_ids"],
                prefix["block_idx"],
                "prefix",
                0,
                layers,
                validation=(host["position_ids"], host["block_idx"]),
            )
            cache._model_signature = donor._model_signature
            receipts.append((prompt.event_id, layers, counts))
            return True

        return load

    return factory


@pytest.mark.parametrize("backend", [None, "eager", "recording"])
@pytest.mark.parametrize("adaln_mode", ["dynamic", "table"])
@pytest.mark.parametrize("switch_miss", [False, True])
@pytest.mark.parametrize("split_first_latent", [False, True])
def test_cached_prefix_start_and_switch_match_serial(
    model, monkeypatch, tmp_path, backend, adaln_mode, switch_miss, split_first_latent
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_prefix_graph import (
        PrefixJointGraphs,
    )

    monkeypatch.setattr(model.arch, "ar_condition_timestep", 0.875)
    monkeypatch.setattr(model.arch, "reference_visual_timestep", 0.999)
    options = dict(
        batches=(1,) * 17,
        prompt_updates={4: 9, 7: 6, 10: 9, 13: 9},
        reference=True,
        split_first_latent=split_first_latent,
    )
    baseline = _run_ready_chunks(model, monkeypatch, joint_forward="off", **options)
    if adaln_mode == "table":
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
            ZingAdalnTable,
            video_plan,
            write_table,
        )
        from sglang.multimodal_gen.runtime.models.schedulers.scheduling_zing_h3_causal_renoise import (
            ZingH3CausalRenoiseSchedule,
        )

        schedule = ZingH3CausalRenoiseSchedule(
            STEPS, num_train_timesteps=1000, shift=5.0
        )
        path = tmp_path / "cached-prefix-table"
        write_table(
            path,
            arch=model.arch,
            tags=model.modality_tags,
            embedder=model.time_embedder,
            projections=[
                *((i, b.adaln_proj) for i, b in enumerate(model.blocks)),
                ("final", model.final_layer.adaln_proj),
            ],
            plan=video_plan(
                "few_step",
                schedule.timesteps,
                condition=0.875,
                reference=0.999,
                sigmas=schedule.sigmas,
            ),
            identity={"fixture": True},
            provenance={"synthetic": True},
        )
        table = ZingAdalnTable(
            path, arch=model.arch, tags=model.modality_tags, max_bytes=1 << 20
        )
        verify_synthetic_table(table, tmp_path)
        candidate = ZingH3DiTModel(
            ZingH3DiTConfig(arch_config=replace(model.arch)), {}, adaln_table=table
        ).eval()
        candidate.load_state_dict(
            {k: v for k, v in model.state_dict().items() if "adaln_proj" not in k}
        )
        candidate.initialize_adaln_table()
        assert candidate.adaln_deployment_receipt()["projection_storage_bytes"] == 0
        model = candidate
    else:
        assert model.adaln_projection_cache_enabled
    cache_stats = []
    prefix_branch_projections = []
    forward_body = model._forward_body

    def inspect_modulation(*args, **kwargs):
        options = kwargs["options"]
        if options.joint_ends is not None and len(options.joint_ends) in (4, 5, 6):
            if options.fixed_kv is not None:
                # Captured execution consumes resolved inputs, with no resident
                # projection GEMMs or Python LRU calls inside the graph body.
                assert kwargs["projections"] is not None
            else:
                assert options.projection_cache is not None
                assert options.branch_projection_keys is not None
            prefix_branch_projections.append(len(options.joint_ends))
        return forward_body(*args, **kwargs)

    monkeypatch.setattr(model, "_forward_body", inspect_modulation)

    def observe(session, *_):
        metrics.append(session.metrics)
        cache = session.cache.adaln_projection_cache
        if cache is not None:
            cache_stats.append(cache.stats())

    receipts, metrics = [], []

    def loader(sampler):
        writer = _cached_prefix_writer(receipts)(sampler)
        return lambda cache, inputs, prompt, prefix: (
            False
            if switch_miss and prompt.event_id == 107
            else writer(cache, inputs, prompt, prefix)
        )

    prefix_backend = WarmupRecordingGraphBackend() if backend == "recording" else None
    model.prefix_graphs = PrefixJointGraphs(
        backend=prefix_backend, capture_enabled=backend == "recording"
    )
    actual = _run_ready_chunks(
        model,
        monkeypatch,
        backend=WarmupRecordingGraphBackend() if backend == "recording" else backend,
        prompt_joint=True,
        prefix_loader_factory=loader,
        yield_hook=observe,
        **options,
    )
    assert (5 if split_first_latent else 4) in prefix_branch_projections
    if switch_miss:
        assert (6 if split_first_latent else 5) in prefix_branch_projections
    if adaln_mode == "dynamic":
        assert cache_stats and max(s["hits"] for s in cache_stats) > 0
    else:
        assert table.hits > 0
    torch.testing.assert_close(actual[0], baseline[0], rtol=0.02, atol=0.02)
    assert actual[5] == baseline[5]
    assert [r[0] for r in receipts] == (
        [None, 104, 110, 113] if switch_miss else [None, 104, 107, 110, 113]
    )
    assert metrics[-1]["cached_prefix_hits"] == 5 - int(switch_miss)
    assert metrics[-1]["forward_counts"]["prefix"] == 0
    assert metrics[-1]["forward_counts"]["prompt_joint"] == 4
    assert all(r[2]["prefix"] == 0 for r in receipts)
    for got, want in zip(actual[1], baseline[1], strict=True):
        assert got[0] == want[0]
        assert [(c[0], c[1]) for c in got[1]] == [(c[0], c[1]) for c in want[1]]
        for (gk, gv), (wk, wv) in zip(got[2], want[2], strict=True):
            torch.testing.assert_close(gk, wk, rtol=0.02, atol=0.02)
            torch.testing.assert_close(gv, wv, rtol=0.02, atol=0.02)
    # Every snapshot keeps the producer's prefix bytes exactly, even as the
    # rolling window evicts dynamic history or the graph changes shape.
    by_event = {event: layers for event, layers, _ in receipts}
    for event, snapshot in zip(actual[5], actual[1], strict=True):
        if snapshot[0] == 0 or event not in by_event:
            # EOS disposal already released the logical cache before yielding.
            continue
        for (k, v), (pk, pv) in zip(snapshot[2], by_event[event], strict=True):
            assert torch.equal(k[: pk.shape[0]], pk)
            assert torch.equal(v[: pv.shape[0]], pv)
    for got, want in zip(actual[2], baseline[2], strict=True):
        assert (got is None and want is None) or torch.equal(got, want)
    if prefix_backend is not None:
        assert prefix_backend.captures > 0
        assert model.prefix_graphs.routes[("prefix_separate", "replay")] > 0


@pytest.mark.parametrize("prompt_joint", [False, True])
def test_cached_prefix_miss_preserves_existing_path(model, monkeypatch, prompt_joint):
    options = dict(batches=(1,) * 8, prompt_updates={4: 9}, prompt_joint=prompt_joint)
    expected = _run_ready_chunks(model, monkeypatch, **options)
    actual = _run_ready_chunks(
        model,
        monkeypatch,
        prefix_loader_factory=_cached_prefix_writer([], fail=lambda *_: False),
        **options,
    )
    assert torch.equal(actual[0], expected[0])
    assert actual[5] == expected[5]


@pytest.mark.parametrize("failure", ["raise", "unpublished", "return_type"])
def test_cached_prefix_bad_writer_releases_session(model, monkeypatch, failure):
    owned = []

    def fail(cache, *_):
        owned.append(cache)
        if failure == "raise":
            raise RuntimeError("injected writer failure")
        return True if failure == "unpublished" else None

    with pytest.raises(
        (ValueError, RuntimeError), match="writer failure|prefix-only|boolean"
    ):
        _run_ready_chunks(
            model,
            monkeypatch,
            batches=(1,) * 4,
            prefix_loader_factory=_cached_prefix_writer([], fail=fail),
        )
    assert owned and owned[0].prefix is None
    assert owned[0]._length == 0 and not owned[0].committed


def test_cached_prefix_step_also_joins_switch(model, monkeypatch):
    options = dict(batches=(1,) * 8, prompt_updates={4: 9}, use_step=True)
    expected = _run_ready_chunks(model, monkeypatch, **options)
    metrics = []
    actual = _run_ready_chunks(
        model,
        monkeypatch,
        prefix_loader_factory=_cached_prefix_writer([]),
        yield_hook=lambda session, *_: metrics.append(session.metrics),
        **options,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0.02, atol=0.02)
    assert actual[5] == expected[5]
    assert metrics[-1]["forward_counts"]["prompt_joint"] == 1
    assert metrics[-1]["forward_counts"]["prefix"] == 0


def test_cached_prefix_switch_failure_releases_and_reuses_slot(model, monkeypatch):
    owned, receipts = [], []
    storage = _storage(model)

    def factory(sampler):
        load = _cached_prefix_writer(receipts)(sampler)

        def writer(cache, inputs, prompt, prefix):
            if prompt.event_id is not None:
                # The writer may already have replaced old logical state.
                cache.clear()
                raise RuntimeError("injected switch writer failure")
            return load(cache, inputs, prompt, prefix)

        return writer

    options = dict(batches=(1,) * 8, prompt_updates={4: 9}, storage=storage)
    with pytest.raises(RuntimeError, match="switch writer failure"):
        _run_ready_chunks(
            model,
            monkeypatch,
            prefix_loader_factory=factory,
            yield_hook=lambda session, *_: owned.append(session),
            **options,
        )
    assert owned and owned[-1].closed
    assert owned[-1].cache.prefix is None and owned[-1].cache._length == 0
    assert owned[-1].prefix_loader is None
    assert all(value == 0 for value in owned[-1].memory_snapshot().values())
    actual = _run_ready_chunks(
        model, monkeypatch, prefix_loader_factory=_cached_prefix_writer([]), **options
    )
    assert actual[5][-1] == 104


def test_cached_prefix_only_loads_latest_candidate(model, monkeypatch):
    receipts = []
    actual = _run_ready_chunks(
        model,
        monkeypatch,
        batches=(1,) * 8,
        prompt_updates={4: 9, 5: 6},
        prefix_loader_factory=_cached_prefix_writer(receipts),
    )
    assert [event for event, *_ in receipts] == [None, 105]
    assert 104 not in actual[5] and actual[5][-1] == 105


@pytest.mark.parametrize("backend", [None, "eager", "recording"])
@pytest.mark.parametrize("warm_start", [False, True])
@pytest.mark.parametrize("split_first_latent", [False, True])
@pytest.mark.usefixtures("cpu_dma")
def test_block0_provider_routes_start_and_prompt_cycle(
    model, monkeypatch, backend, warm_start, split_first_latent
):
    from types import SimpleNamespace as NS

    from sglang.multimodal_gen.runtime.models.dits.zing_h3_prefix_graph import (
        PrefixJointGraphs,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )

    options = dict(
        batches=(1,) * 11,
        prompt_updates={4: 6, 7: None},
        split_first_latent=split_first_latent,
    )
    baseline = _run_ready_chunks(model, monkeypatch, joint_forward="off", **options)
    metrics, owners, states = [], [], []

    def prepared(inputs, condition):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
            H3InferenceInputs,
        )

        return H3InferenceInputs(
            **{**vars(inputs), "positive": condition},
            output_type="video",
            audio_shape=None,
            pixel_frame_count=1,
            sample_count=None,
            sample_rate=24000,
        )

    def provider(sampler):
        stores = _storage(model)
        owner = cpu_cache(stores, entries=2, rows=stores[0].rows)
        state = ZingSourceStreamState()
        state.block0_cache = owner
        state.sampler = NS(sampler=sampler, applied_prompt=None)
        writer = _cached_prefix_writer([])(sampler)
        calls = 0
        owners.append(owner)
        states.append(state)

        def load(cache, inputs, prompt, prefix):
            nonlocal calls
            if calls == 0:
                if warm_start:
                    assert writer(cache, inputs, prompt, prefix)
                    owner.fill(
                        "case-A",
                        prepared(inputs, prompt.condition),
                        "A",
                        {},
                        cache,
                    )
                    owner.finish_fill()
                    cache.clear()
                    state.block0_entry = owner.entries["case-A"]
                    state.block0_restore = owner.restore(cache, state.block0_entry)
            calls += 1
            return state._load_block0_prefix(cache, inputs, prompt, prefix)

        return load

    def observe(session, _, count):
        metrics.append(session.metrics)
        state = states[0]
        state.sampler = session

        def prepare(candidate):
            # Real prompt snapshots clone condition tensors. Cache keys belong
            # to the provider, independent of the snapshot's Python identity.
            key = "case-A" if candidate.event_id == 107 else "case-B"
            state.pending_block0 = (
                key,
                {"prompt": key},
                prepared(session.inputs, candidate.condition),
                key,
                {},
            )
            return state._prepare_block0_switch(candidate)

        session.prepare_prompt_prefix = prepare
        if count == 1 and not warm_start:
            # Match the provider's post-output backfill; retain its one A entry.
            owners[0].fill(
                "case-A",
                prepared(session.inputs, session.prompt.condition),
                "A",
                {},
                session.cache,
            )
            owners[0].finish_fill()

    recording = WarmupRecordingGraphBackend() if backend == "recording" else None
    model.prefix_graphs = PrefixJointGraphs(
        backend=recording, capture_enabled=backend == "recording"
    )
    actual = _run_ready_chunks(
        model,
        monkeypatch,
        backend=recording if recording else backend,
        prompt_joint=True,
        prefix_loader_factory=provider,
        yield_hook=observe,
        **options,
    )
    torch.testing.assert_close(actual[0], baseline[0], rtol=0.02, atol=0.02)
    assert actual[5] == baseline[5]
    for got, want in zip(actual[1], baseline[1], strict=True):
        assert got[0] == want[0]
        assert [(c[0], c[1]) for c in got[1]] == [(c[0], c[1]) for c in want[1]]
        for (gk, gv), (wk, wv) in zip(got[2], want[2], strict=True):
            torch.testing.assert_close(gk, wk, rtol=0.02, atol=0.02)
            torch.testing.assert_close(gv, wv, rtol=0.02, atol=0.02)
    for got, want in zip(actual[2], baseline[2], strict=True):
        assert (got is None and want is None) or torch.equal(got, want)
    assert metrics[-1]["cached_prefix_hits"] == 1 + int(warm_start)
    assert metrics[-1]["cached_prefix_misses"] == 1 + int(not warm_start)
    assert metrics[-1]["forward_counts"]["prompt_joint"] == 2
    assert metrics[-1]["forward_counts"]["prefix"] == 0
    assert metrics[-1]["forward_counts"].get("initial_joint", 0) == int(not warm_start)
    assert "case-A" in owners[0].entries


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("route", ["step", "drain", "graph"])
@pytest.mark.parametrize("initial_joint", [False, True])
@pytest.mark.parametrize("split_first_latent", [False, True])
def test_initial_prefix_joint_matches_serial_cache(
    model, monkeypatch, reference, route, initial_joint, split_first_latent
):
    """Compare the real session entry, latent and every live raw KV row."""
    if reference:
        monkeypatch.setattr(model.arch, "ar_condition_timestep", 0.875)
        monkeypatch.setattr(model.arch, "reference_visual_timestep", 0.999)
    options = dict(
        batches=(1, 1, 1),
        reference=reference,
        use_step=route == "step",
        split_first_latent=split_first_latent,
    )
    expected = _run_ready_chunks(model, monkeypatch, joint_forward="off", **options)
    monkeypatch.setenv(
        "SGLANG_ZING_INITIAL_PREFIX_JOINT", "1" if initial_joint else "0"
    )
    metrics = []

    def observe(session, *_):
        if session.sampler is not None:
            metrics.append(dict(session.sampler._forward_counts))

    owner = None
    storage = _storage(model)
    if route == "graph":
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
            ZingChunkGraphs,
        )

        owner = ZingChunkGraphs(backend=RecordingGraphBackend(), max_graphs=16)
    first_fixed = None
    # Repeat a fresh session to cover initial warmup, capture and replay.
    for _ in range(3 if owner is not None else 1):
        actual = _run_ready_chunks(
            model,
            monkeypatch,
            graph_owner=owner,
            storage=storage,
            yield_hook=observe,
            **options,
        )
        torch.testing.assert_close(actual[0], expected[0], atol=0.02, rtol=0.02)
        if first_fixed is None:
            first_fixed = actual[0].clone()
        else:
            assert torch.equal(actual[0], first_fixed)
        for got, want in zip(actual[1], expected[1], strict=True):
            assert got[0] == want[0]
            assert [(c[0], c[1]) for c in got[1]] == [(c[0], c[1]) for c in want[1]]
            for (gk, gv), (wk, wv) in zip(got[2], want[2], strict=True):
                torch.testing.assert_close(gk, wk, atol=0.02, rtol=0.02)
                torch.testing.assert_close(gv, wv, atol=0.02, rtol=0.02)
    assert metrics and all(
        m["prefix"] == (0 if initial_joint else 1)
        and m.get("initial_joint", 0) == int(initial_joint)
        for m in metrics
    )
    if owner is not None:
        assert owner.routes["init", "replay"] > 0, owner.routes


@pytest.mark.parametrize(
    "shapes,expected",
    [
        ([(46, 80)], (23, 40)),
        ([(30, 52)], (15, 26)),
        ([(80, 46)], (40, 23)),
        ([(46, 80), (46, 80)], (23, 40)),
        ([(30, 52), (46, 80)], None),
        ([(45, 80)], None),
    ],
)
def test_fixed_target_grid_uses_latent_shape_only(monkeypatch, shapes, expected):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("Target geometry must not scan tensor coordinates")

    monkeypatch.setattr(torch, "unique", unexpected_scan)
    sampler = SimpleNamespace(processor=SimpleNamespace(patch_size=(1, 2, 2)))
    states = [
        SimpleNamespace(visual=SimpleNamespace(shape=(1, 16, 1, *s))) for s in shapes
    ]
    assert ZingChunkGraphs._target_grid(sampler, *states) == expected


def test_chunk_plans_carry_actual_target_grid(model, monkeypatch):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )

    original = ZingChunkGraphs._forward_plan
    observed = []
    grid = (23, 40)
    monkeypatch.setattr(
        ZingChunkGraphs, "_target_grid", staticmethod(lambda *args: grid)
    )

    def spy(sampler, shadow, packed, phase, after=None, *, target_grid=None):
        observed.append((phase, target_grid))
        return original(
            sampler, shadow, packed, phase, after=after, target_grid=target_grid
        )

    monkeypatch.setattr(ZingChunkGraphs, "_forward_plan", staticmethod(spy))
    _run_ready_chunks(model, monkeypatch, backend=RecordingGraphBackend())
    assert {phase for phase, _ in observed} >= {"active", "joint", "final"}
    assert all(
        value == (None if phase == "prefix" else grid) for phase, value in observed
    )


@pytest.mark.parametrize("cached", [False, True])
def test_unsplit_prompt_prefix_joins_both_history_blocks(model, monkeypatch, cached):
    options = dict(
        batches=(1,) * 8, prompt_updates={3: 9, 5: 6}, split_first_latent=False
    )
    baseline = _run_ready_chunks(model, monkeypatch, **options)
    metrics, receipts = [], []
    actual = _run_ready_chunks(
        model,
        monkeypatch,
        prompt_joint=True,
        prefix_loader_factory=(_cached_prefix_writer(receipts) if cached else None),
        yield_hook=lambda session, drain, count: metrics.append(session.metrics),
        **options,
    )
    torch.testing.assert_close(actual[0], baseline[0], rtol=0.02, atol=0.02)
    assert actual[5] == baseline[5]
    assert not metrics[-1].get("prompt_joint_fallbacks")
    assert metrics[-1]["forward_counts"]["prompt_joint"] == 2
    if cached:
        assert receipts


@pytest.fixture
def private_capture_stream_driver(monkeypatch):
    from sglang.multimodal_gen.runtime.utils import zing_capture_stream as module

    calls = SimpleNamespace(created=[], destroyed=[], error=0)

    def create(flags):
        assert flags == 1
        handle = 1000 + len(calls.created)
        calls.created.append(handle)
        return calls.error, handle

    driver = SimpleNamespace(
        CUstream_flags=SimpleNamespace(CU_STREAM_NON_BLOCKING=1),
        CUresult=SimpleNamespace(CUDA_SUCCESS=0),
        cuStreamCreate=create,
        cuStreamDestroy=lambda handle: calls.destroyed.append(handle),
    )
    cuda = ModuleType("cuda")
    bindings = ModuleType("cuda.bindings")
    cuda.bindings = bindings
    bindings.driver = driver
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.driver", driver)
    monkeypatch.setattr(module, "_streams", {})
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "ExternalStream",
        lambda handle, device: SimpleNamespace(cuda_stream=handle, device=device),
    )

    def pooled_stream_is_not_private(*args, **kwargs):
        raise AssertionError("A pooled stream cannot be reserved for capture")

    monkeypatch.setattr(torch.cuda, "Stream", pooled_stream_is_not_private)
    monkeypatch.setattr(module, "_ORIG_CUDA_STREAM", pooled_stream_is_not_private)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "device",
        lambda dev: collections.abc.Callable and __import__("contextlib").nullcontext(),
    )
    return module, calls


def test_background_io_private_streams_and_peer_isolation(
    private_capture_stream_driver, monkeypatch
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_transport import (
        EncoderDiTChannel,
    )

    module, calls = private_capture_stream_driver
    enc_stream = module.remote_encoder_stream("cuda:0")
    cond_stream = module.conditioning_nccl_stream("cuda:0")
    io_peer0 = module.dit_io_stream("cuda:0", key="peer0")
    io_peer1 = module.dit_io_stream("cuda:0", key="peer1")
    assert (
        len(
            {
                enc_stream.cuda_stream,
                cond_stream.cuda_stream,
                io_peer0.cuda_stream,
                io_peer1.cuda_stream,
            }
        )
        == 4
    )
    assert module.remote_encoder_stream("cuda:0") is enc_stream
    assert module.conditioning_nccl_stream("cuda:0") is cond_stream
    assert module.dit_io_stream("cuda:0", key="peer0") is io_peer0
    assert module.dit_io_stream("cuda:0", key="peer1") is io_peer1

    # Verify two EncoderDiTChannel instances with rank=0 on different ports get distinct streams in io_context()
    activated_streams = []
    monkeypatch.setattr(
        torch.cuda,
        "stream",
        lambda s: activated_streams.append(s) or __import__("contextlib").nullcontext(),
    )
    ch1 = EncoderDiTChannel(
        object(), object(), rank=0, device="cuda:0", stream_key="127.0.0.1:29501:0"
    )
    ch2 = EncoderDiTChannel(
        object(), object(), rank=0, device="cuda:0", stream_key="127.0.0.1:29502:0"
    )
    with ch1.io_context():
        pass
    with ch2.io_context():
        pass
    assert len(activated_streams) == 2
    assert activated_streams[0].cuda_stream != activated_streams[1].cuda_stream

    # Verify fallback when torch.cuda.Stream is monkeypatched by CPU unit tests
    sentinel = object()
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **kw: sentinel)
    assert module.remote_encoder_stream("cuda:0") is sentinel
    assert module.dit_io_stream("cuda:0", key="peer2") is sentinel
    assert module.conditioning_nccl_stream("cuda:0") is sentinel


@pytest.mark.parametrize("fail_capture", [False, True])
def test_cuda_graph_backend_logs_capture_event_json(monkeypatch, fail_capture):
    import contextlib

    from sglang.multimodal_gen.runtime.models.dits import (
        zing_h3_denoise_graph as graph_mod,
    )

    events = []

    class DummyLogger:
        def info(self, msg, *args):
            events.append(("info", msg % args if args else msg))

        def error(self, msg, *args, **kwargs):
            events.append(("error", msg % args if args else msg))

        def exception(self, msg, *args, **kwargs):
            events.append(("exception", msg % args if args else msg))

    caller = SimpleNamespace(cuda_stream=1111, device=0, wait_stream=lambda other: None)
    cap_stream = SimpleNamespace(
        cuda_stream=2222, device=0, wait_stream=lambda other: None
    )
    monkeypatch.setattr(graph_mod, "logger", DummyLogger())
    monkeypatch.setattr(graph_mod, "capture_stream", lambda: cap_stream)
    monkeypatch.setattr(graph_mod, "nccl_during_capture", contextlib.nullcontext)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: caller)
    monkeypatch.setattr(torch.cuda, "set_stream", lambda s: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: "pool")
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: SimpleNamespace())
    monkeypatch.setattr(torch.cuda, "graph", lambda *a, **kw: contextlib.nullcontext())

    backend = CudaGraphBackend()
    calls = []

    def fn():
        calls.append(1)
        if fail_capture and len(calls) == 2:
            raise RuntimeError("injected stream capture failure")
        return ("ok",)

    if fail_capture:
        with pytest.raises(RuntimeError, match="injected stream capture failure"):
            backend.capture(fn)
        assert any(
            "ZING_GRAPH_CAPTURE_EVENT_JSON" in m and '"phase": "start"' in m
            for _, m in events
        )
        assert any(
            "ZING_GRAPH_CAPTURE_EVENT_JSON" in m
            and '"phase": "failed"' in m
            and "2222" in m
            for _, m in events
        )
    else:
        backend.capture(fn)
        assert any(
            "ZING_GRAPH_CAPTURE_EVENT_JSON" in m and '"phase": "start"' in m
            for _, m in events
        )
        assert any(
            "ZING_GRAPH_CAPTURE_EVENT_JSON" in m
            and '"phase": "end"' in m
            and "2222" in m
            for _, m in events
        )


def test_private_capture_stream_is_bounded_and_not_a_pooled_wrapper(
    private_capture_stream_driver, monkeypatch
):
    module, calls = private_capture_stream_driver
    first = module.capture_stream()
    assert all(module.capture_stream() is first for _ in range(256))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    second = module.capture_stream()
    assert second is not first
    assert (first.device, second.device) == (0, 1)
    assert calls.created == [1000, 1001]
    assert not calls.destroyed  # Graph/workspace lifetime is the process lifetime.


def test_private_capture_stream_creation_failure_does_not_publish_a_handle(
    private_capture_stream_driver,
):
    module, calls = private_capture_stream_driver
    calls.error = 2
    with pytest.raises(RuntimeError, match="Cannot create private Zing capture stream"):
        module.capture_stream()
    assert not module._streams
    calls.error = 0
    assert module.capture_stream().cuda_stream == 1001
    assert not calls.destroyed


def test_decoder_stream_reuses_its_own_nonpooled_device_stream(
    private_capture_stream_driver, monkeypatch
):
    module, calls = private_capture_stream_driver
    capture = module.capture_stream()
    decoder = module.decoder_stream()
    assert decoder is not capture
    assert all(module.decoder_stream() is decoder for _ in range(256))
    assert module.capture_stream() is capture
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert module.decoder_stream().device == 1
    assert calls.created == [1000, 1001, 1002]
    assert not calls.destroyed


def test_private_capture_stream_wrapper_failure_releases_unpublished_handle(
    private_capture_stream_driver, monkeypatch
):
    module, calls = private_capture_stream_driver

    def fail(*args, **kwargs):
        raise RuntimeError("External stream wrapper failed")

    monkeypatch.setattr(torch.cuda, "ExternalStream", fail)
    with pytest.raises(RuntimeError, match="External stream wrapper failed"):
        module.capture_stream()
    assert not module._streams
    assert calls.destroyed == [1000]


def test_sink_controls_preserve_cached_prompt_prefix(model, monkeypatch):
    options = dict(
        batches=(1,) * 8,
        prompt_updates={3: 9, 5: 6},
        sink_updates={1: ["off"], 3: ["refresh"], 5: ["off"]},
        split_first_latent=False,
        defer_final=True,
        prompt_joint=True,
    )
    expected_trace = []
    expected = _run_ready_chunks(
        model, monkeypatch, backend="eager", sink_trace=expected_trace, **options
    )
    actual_trace, receipts, metrics = [], [], []
    actual = _run_ready_chunks(
        model,
        monkeypatch,
        backend=RecordingGraphBackend(),
        prefix_loader_factory=_cached_prefix_writer(receipts),
        sink_trace=actual_trace,
        yield_hook=lambda session, drain, count: metrics.append(session.metrics),
        **options,
    )
    assert actual_trace == expected_trace
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-6)
    assert [row[0] for row in receipts] == [None, 103, 105]
    assert metrics[-1]["cached_prefix_hits"] == 3
    assert metrics[-1]["forward_counts"]["prefix"] == 0
    assert actual_trace[3][3:] == (203, 2, (1, 2))
    assert actual_trace[6][2:5] == (105, 205, 0)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_same_condition_resident_prefix_rebuild_matches_recomputed_prefix(
    model, monkeypatch, graph, split, deferred
):
    monkeypatch.setattr(model.arch, "ar_condition_timestep", 0.875)
    monkeypatch.setattr(model.arch, "reference_visual_timestep", 0.999)
    options = dict(
        batches=(1,) * 12,
        prompt_updates={4: None, 8: None},
        split_first_latent=split,
        defer_final=deferred,
        reference=True,
    )
    baseline = _run_ready_chunks(model, monkeypatch, **options)
    metrics = []

    def forbidden_restore(*_):
        raise AssertionError("Same-condition update must not restore CPU prefix")

    def observe(session, *_):
        metrics.append(session.metrics)
        session.prepare_prompt_prefix = lambda c: (
            None if c.reuse_prefix else forbidden_restore()
        )

    actual = _run_ready_chunks(
        model,
        monkeypatch,
        backend=RecordingGraphBackend() if graph else None,
        reuse_updates=True,
        yield_hook=observe,
        **options,
    )
    torch.testing.assert_close(actual[0], baseline[0], rtol=1e-5, atol=1e-6)
    assert actual[5] == baseline[5]
    assert metrics[-1]["gpu_prefix_reuses"] == 2
    assert metrics[-1]["forward_counts"]["prefix"] <= 1
