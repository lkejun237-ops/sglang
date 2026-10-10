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

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import sglang
from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
from sglang.multimodal_gen.runtime.models.dits import zing_h3
from sglang.multimodal_gen.runtime.models.dits.zing_h3 import ZingH3DiTModel
from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
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
from sglang.multimodal_gen.test.unit.test_zing_h3_kv_pool_cache import _inputs
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
    verify_synthetic_table,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import model as model_fixture

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
def test_weights_free_table_replays_interleaved_sessions_exactly(
    model, tmp_path, monkeypatch
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
        ZingAdalnTable,
        video_plan,
        write_table,
    )
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_zing_h3_causal_renoise import (
        ZingH3CausalRenoiseSchedule,
    )

    seeds = (11, 12)
    expected = _run_sessions(
        model, monkeypatch, storages=[_storage(model), _storage(model)], seeds=seeds
    )
    schedule = ZingH3CausalRenoiseSchedule(STEPS, num_train_timesteps=1000, shift=5.0)
    plan = video_plan("few_step", schedule.timesteps, sigmas=schedule.sigmas)
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
    assert table.active_plans == 0 and table.stats()["fallback"] == 0
