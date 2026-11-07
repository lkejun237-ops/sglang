# SPDX-License-Identifier: Apache-2.0
"""Untimed first-three-drain input/RNG observations; never install in formal rounds."""

import dataclasses
import functools
import hashlib
import os
import threading
from contextlib import contextmanager
from pathlib import Path


def parameter_sha256(value):
    """Hash every loaded parameter byte with bounded host copies, untimed only."""
    import torch

    digest = hashlib.sha256()
    for chunk in value.detach().reshape(-1).split(16 << 20):
        raw = chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def install():
    import torch

    from sglang.multimodal_gen.runtime.layers.linear import (
        ColumnParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
    )
    from sglang.multimodal_gen.runtime.models.dits.zing_h3 import (
        FixedForwardPlan,
        ZingH3DiTModel,
        _numerics_mode,
    )
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
        ZingChunkGraphs,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.realtime import (
        ZingH3RealtimeState,
    )

    local = threading.local()
    graph_run = ZingChunkGraphs._run
    fixed_prepare = ZingH3DiTModel.prepare_fixed_forward
    execute = FixedForwardPlan.execute
    forward = ZingH3DiTModel.forward
    embed_inputs = ZingH3DiTModel._embed_inputs
    block_forward = ZingH3DiTModel._block_forward
    fuse_pre_linear = ZingH3DiTModel._should_fuse_pre_linear_quant
    linear_forwards = {
        cls: cls.forward
        for cls in (ColumnParallelLinear, ReplicatedLinear, RowParallelLinear)
    }
    drain = SourceReadySession.drain_ready
    stream = ZingH3RealtimeState.random_stream
    randn = torch.randn_like
    prepare = SourceReadySession._prepare

    def snapshot(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if dataclasses.is_dataclass(value):
            # KV identity and backing-store views are process-local owners, not
            # model inputs. Preserve the transaction geometry without pickling them.
            if type(value).__name__ == "ZingKVChunk":
                return {
                    key: snapshot(getattr(value, key))
                    for key in (
                        "positions",
                        "host_positions",
                        "block",
                        "frames",
                        "offset",
                    )
                }
            return {
                field.name: snapshot(getattr(value, field.name))
                for field in dataclasses.fields(value)
            }
        if isinstance(value, dict):
            return {key: snapshot(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(snapshot(item) for item in value)
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise TypeError(f"Unsupported diagnostic snapshot type: {type(value).__name__}")

    def fingerprint(value, *, retain=False):
        if isinstance(value, torch.Tensor):
            host = value.detach().to("cpu", copy=True).contiguous()
            result = dict(
                shape=list(host.shape),
                dtype=str(host.dtype),
                stride=list(value.stride()),
                storage_offset=value.storage_offset(),
                inference=torch.is_inference(value),
                requires_grad=value.requires_grad,
                sha256=parameter_sha256(host),
                samples=host.reshape(-1)[:32].clone(),
            )
            if retain:
                result["value"] = host
            return result
        if isinstance(value, (list, tuple)):
            return [fingerprint(item, retain=retain) for item in value]
        if value is None:
            return None
        raise TypeError("Unexpected first-forward diagnostic output")

    @functools.wraps(embed_inputs)
    def observed_embed_inputs(self, *args, **kwargs):
        result = embed_inputs(self, *args, **kwargs)
        if getattr(local, "trace_first_execute", False):
            local.observation["first_body_trace"].append(
                dict(stage="embedding", output=fingerprint(result, retain=True))
            )
        return result

    @functools.wraps(block_forward)
    def observed_block_forward(self, *args, **kwargs):
        previous = getattr(local, "block_index", None)
        if getattr(local, "trace_first_execute", False):
            local.block_index = kwargs["index"]
        try:
            result = block_forward(self, *args, **kwargs)
        finally:
            local.block_index = previous
        if getattr(local, "trace_first_execute", False):
            local.observation["first_body_trace"].append(
                dict(
                    stage="block", index=kwargs["index"], output=fingerprint(result[0])
                )
            )
        return result

    def tracing_first_block():
        return (
            getattr(local, "trace_first_execute", False)
            and getattr(local, "block_index", None) == 0
        )

    @functools.wraps(fuse_pre_linear)
    def observed_fuse_pre_linear(self, value):
        result = fuse_pre_linear(self, value)
        if tracing_first_block():
            local.observation["first_block_fused_pre_linear"] = result
        return result

    def method_settings(method):
        def settings(value):
            if value is None:
                return None
            return dict(
                type=f"{type(value).__module__}.{type(value).__qualname__}",
                scalars={
                    key: item
                    for key, item in vars(value).items()
                    if item is None or type(item) in (str, int, float, bool)
                },
            )

        return {
            "method": settings(method),
            "backend": settings(getattr(method, "backend", None)),
            "policy": settings(getattr(method, "policy", None)),
        }

    def observe_linear(original):
        @functools.wraps(original)
        def observed(self, *args, **kwargs):
            if not tracing_first_block():
                return original(self, *args, **kwargs)
            row = dict(
                prefix=self.prefix,
                input=fingerprint(args[0], retain=True),
                weight_stride=list(self.weight.stride()),
                weight_storage_offset=self.weight.storage_offset(),
                weight_requires_grad=self.weight.requires_grad,
                weight_inference=torch.is_inference(self.weight),
                numerics_mode=list(_numerics_mode()),
                grad_enabled=torch.is_grad_enabled(),
                inference_mode=torch.is_inference_mode_enabled(),
                quant_method=method_settings(self.quant_method),
            )
            local.observation["first_block_linear_trace"].append(row)
            result = original(self, *args, **kwargs)
            row["output"] = fingerprint(result[0], retain=True)
            return result

        return observed

    @functools.wraps(fixed_prepare)
    def observed_fixed_prepare(self, **kwargs):
        plan = fixed_prepare(self, **kwargs)
        row = getattr(local, "observation", None)
        if row is not None and row["index"] == 0:
            row.setdefault("fixed_plan_preparations", []).append(
                dict(
                    phase=kwargs["phase"],
                    target_grid=kwargs.get("target_grid"),
                    rows=plan.kv_plan.rows,
                    ends=plan.kv_plan.ends,
                    clean_rows=plan.kv_plan.clean_rows,
                    write_start=plan.kv_plan.write_start,
                )
            )
            if "model_buffers_after_prepare" not in row:
                row["model_buffers_after_prepare"] = snapshot(
                    {
                        **dict(self.named_buffers()),
                        "rope.inv_freq": self.rope.inv_freq,
                        "time_embedder._frequency_cache": self.time_embedder._frequency_cache,
                    }
                )
        return plan

    @functools.wraps(graph_run)
    def observed_graph_run(self, kind, plans, tensors, execute, stores, rotated=()):
        row = getattr(local, "observation", None)
        if row is None or row["index"] != 0:
            return graph_run(self, kind, plans, tensors, execute, stores, rotated)
        observation = dict(
            kind=kind,
            plan_count=len(plans),
            graphs_before=len(self.graphs),
            seen_before=len(self._seen),
            tensors=snapshot(tensors),
            strides={key: list(value.stride()) for key, value in tensors.items()},
            routes_before=[
                dict(kind=k[0], route=k[1], count=v)
                for k, v in sorted(self.routes.items())
            ],
        )
        row.setdefault("graph_runs", []).append(observation)
        result = graph_run(self, kind, plans, tensors, execute, stores, rotated)
        observation.update(
            graphs_after=len(self.graphs),
            seen_after=len(self._seen),
            routes_after=[
                dict(kind=k[0], route=k[1], count=v)
                for k, v in sorted(self.routes.items())
            ],
            output=snapshot(result),
        )
        return result

    @functools.wraps(execute)
    def observed_execute(self, tensors=None):
        row = getattr(local, "observation", None)
        if (
            row is None
            or row["index"] != 0
            or "first_fixed_execute" in row
            or torch.compiler.is_compiling()
            or torch.cuda.is_current_stream_capturing()
        ):
            return execute(self, tensors)
        actual = self.tensors if tensors is None else tensors
        kv = self.kv_plan
        observation = dict(
            tensors=snapshot(actual),
            strides={key: list(value.stride()) for key, value in actual.items()},
            kv={
                key: snapshot(getattr(kv, key))
                for key in (
                    "capacity",
                    "rows",
                    "clean_rows",
                    "ends",
                    "write_start",
                    "rotates_history",
                    "history_positions",
                )
            },
        )
        row["first_fixed_execute"] = observation
        row["first_body_trace"] = []
        row["first_block_linear_trace"] = []
        # Trace only the first uncaptured execution. No Module hooks are added,
        # and CPU hashing never runs while recording or replaying a CUDA Graph.
        local.trace_first_execute = True
        try:
            result = execute(self, tensors)
        finally:
            local.trace_first_execute = False
        observation["output"] = snapshot(result)
        return result

    @functools.wraps(forward)
    def observed_forward(self, *args, **kwargs):
        row = getattr(local, "observation", None)
        if row is None or row["index"] != 0 or "first_forward" in row:
            return forward(self, *args, **kwargs)
        # A method wrapper leaves Module forward hooks empty and preserves the
        # existing CUDA Graph eligibility. All copies belong to untimed probes.
        observation = {
            "args": snapshot(args),
            "kwargs": {
                key: snapshot(value) for key, value in kwargs.items() if key != "cache"
            },
            "tensor_strides": {
                key: list(value.stride())
                for key, value in kwargs.items()
                if isinstance(value, torch.Tensor)
            },
        }
        cache = kwargs.get("cache")
        if cache is not None:
            observation["cache"] = snapshot(
                {
                    key: getattr(cache, key)
                    for key in (
                        "prefix",
                        "pending_source",
                        "committed",
                        "active",
                        "last_block",
                        "_length",
                        "window_policy",
                        "rope_position_mode",
                        "rope_max_frame_gap",
                        "_window_eviction_timing",
                    )
                }
            )
        row["first_forward"] = observation
        result = forward(self, *args, **kwargs)
        observation["output"] = snapshot(result)
        return result

    @functools.wraps(prepare)
    def observed_prepare(self, *args, **kwargs):
        current, origin = prepare(self, *args, **kwargs)
        row = getattr(local, "observation", None)
        if row is not None and row["index"] == 0:
            row.setdefault("prepared", []).append(
                snapshot(
                    dict(
                        positive=current.positive,
                        negative=current.negative,
                        visual=current.visual,
                        audio=current.audio,
                        spans=current.spans,
                        origin=origin,
                        generation_block_index=current.generation_block_index,
                    )
                )
            )
        return current, origin

    @functools.wraps(drain)
    def observed_drain(self):
        root = getattr(self, "_stall_rng_probe_root", None)
        index = getattr(self, "_stall_rng_probe_index", 0)
        if root is None or index >= 3:
            yield from drain(self)
            return
        self._stall_rng_probe_index = index + 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        destination = Path(root) / f"rank{rank}"
        destination.mkdir(parents=True, exist_ok=True)
        previous = getattr(local, "observation", None)
        row = dict(
            index=index,
            ready_chunks=[ready.chunk.chunk_id for ready in self.pending],
            streams=[],
            noise=[],
            model_parameter_dtypes={},
        )
        for parameter in self.sampler.generator.parameters():
            key = str(parameter.dtype)
            row["model_parameter_dtypes"][key] = (
                row["model_parameter_dtypes"].get(key, 0) + parameter.numel()
            )
        model = self.sampler.generator
        if index == 0:
            row["model_parameter_sha256"] = {
                name: parameter_sha256(value)
                for name, value in model.named_parameters()
            }
            row["model_module_settings"] = {
                name: dict(
                    module_type=f"{type(module).__module__}.{type(module).__qualname__}",
                    scalars={
                        key: value
                        for key, value in vars(module).items()
                        if value is None or type(value) in (str, int, float, bool)
                    },
                )
                for name, module in model.named_modules()
            }
            row["session_state"] = dict(
                ended=self.ended,
                defer_final=self.defer_final,
                prompt_joint=self.prompt_joint,
                has_applied_prompt=self.applied_prompt is not None,
                has_prefix_loader=self.prefix_loader is not None,
                has_restore_prefix=self.restore_initial_prefix is not None,
                has_pending_prefix=self._pending_prefix is not None,
                cache_present=self.cache is not None,
                cache_last_block=None if self.cache is None else self.cache.last_block,
                cache_has_prefix=self.cache is not None
                and self.cache.prefix is not None,
                cache_has_active=self.cache is not None
                and self.cache.active is not None,
            )
            row["model_parameter_samples"] = {
                name: dict(
                    shape=list(value.shape),
                    dtype=str(value.dtype),
                    samples=torch.cat(
                        (
                            value.detach().reshape(-1)[:8],
                            value.detach().reshape(-1)[-8:],
                        )
                    )
                    .cpu()
                    .clone(),
                )
                for name, value in model.named_parameters()
            }
        row["numerics_mode"] = list(_numerics_mode())
        row["execution_settings"] = dict(
            default_dtype=str(torch.get_default_dtype()),
            cuda_autocast_enabled=torch.is_autocast_enabled("cuda"),
            cuda_autocast_dtype=str(torch.get_autocast_dtype("cuda")),
            flash_sdp_enabled=torch.backends.cuda.flash_sdp_enabled(),
            mem_efficient_sdp_enabled=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math_sdp_enabled=torch.backends.cuda.math_sdp_enabled(),
            cudnn_sdp_enabled=torch.backends.cuda.cudnn_sdp_enabled(),
            cudnn_enabled=torch.backends.cudnn.enabled,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
        )
        row["model_arch"] = dataclasses.asdict(model.arch)
        row["sampler_scalars"] = {
            key: value
            for key, value in vars(self.sampler).items()
            if value is None or type(value) in (str, int, float, bool)
        }
        row["model_scalars"] = {
            key: value
            for key, value in vars(model).items()
            if value is None or type(value) in (str, int, float, bool)
        }
        row["environment"] = {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("SGLANG_ZING_", "NCCL_"))
        }
        row["tf32"] = torch.backends.cuda.matmul.allow_tf32
        row["inference_mode"] = torch.is_inference_mode_enabled()
        iterator = drain(self)
        try:
            while True:
                local.observation = row
                try:
                    result = next(iterator)
                except StopIteration:
                    break
                finally:
                    local.observation = previous
                yield result
        finally:
            local.observation = previous
            iterator.close()
            # Exclusive write prevents an accidental second run from replacing evidence.
            with (destination / f"drain-{index:05d}.pt").open("xb") as output:
                torch.save(row, output)

    def states(device):
        return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device))

    @contextmanager
    def observed_stream(self, device, seed):
        with stream(self, device, seed):
            row = getattr(local, "observation", None)
            previous = getattr(local, "in_rng", False)
            local.in_rng = True
            sample = None
            if row is not None:
                sample = dict(seed=int(seed), before=states(device))
                row["streams"].append(sample)
            try:
                yield
            finally:
                if sample is not None:
                    sample["after"] = states(device)
                local.in_rng = previous

    @functools.wraps(randn)
    def observed_randn(*args, **kwargs):
        if torch.compiler.is_compiling():
            return randn(*args, **kwargs)
        row = getattr(local, "observation", None)
        if row is None or not getattr(local, "in_rng", False):
            return randn(*args, **kwargs)
        if torch.cuda.is_current_stream_capturing():
            return randn(*args, **kwargs)
        value = randn(*args, **kwargs)
        row["noise"].append(
            dict(value=value.detach().cpu().clone(), stride=list(value.stride()))
        )
        return value

    ZingChunkGraphs._run = observed_graph_run
    ZingH3DiTModel.prepare_fixed_forward = observed_fixed_prepare
    FixedForwardPlan.execute = observed_execute
    ZingH3DiTModel.forward = observed_forward
    ZingH3DiTModel._embed_inputs = observed_embed_inputs
    ZingH3DiTModel._block_forward = observed_block_forward
    ZingH3DiTModel._should_fuse_pre_linear_quant = observed_fuse_pre_linear
    for cls, original in linear_forwards.items():
        cls.forward = observe_linear(original)
    SourceReadySession._prepare = observed_prepare
    SourceReadySession.drain_ready = observed_drain
    ZingH3RealtimeState.random_stream = observed_stream
    torch.randn_like = observed_randn

    def restore():
        ZingChunkGraphs._run = graph_run
        ZingH3DiTModel.prepare_fixed_forward = fixed_prepare
        FixedForwardPlan.execute = execute
        ZingH3DiTModel.forward = forward
        ZingH3DiTModel._embed_inputs = embed_inputs
        ZingH3DiTModel._block_forward = block_forward
        ZingH3DiTModel._should_fuse_pre_linear_quant = fuse_pre_linear
        for cls, original in linear_forwards.items():
            cls.forward = original
        SourceReadySession._prepare = prepare
        SourceReadySession.drain_ready = drain
        ZingH3RealtimeState.random_stream = stream
        torch.randn_like = randn

    return restore


def install_native(capture_root):
    """Bind each rank to the real source-session identity; capture requests only."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )

    install()
    generate = ZingSourceStreamState._generate_ready

    @functools.wraps(generate)
    def observed(self, stage, batch, *args, **kwargs):
        self.sampler._stall_rng_probe_root = (
            Path(capture_root) / batch.realtime_session_id / "rng"
        )
        return generate(self, stage, batch, *args, **kwargs)

    ZingSourceStreamState._generate_ready = observed
