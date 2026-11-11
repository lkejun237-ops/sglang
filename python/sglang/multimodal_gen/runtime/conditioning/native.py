# SPDX-License-Identifier: Apache-2.0
"""Single-process TP1 native Qwen loader, independent from VAE and DiT."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


def build_encoder_pipeline(model_path, server_args, *, pipeline_cls=None):
    if pipeline_cls is None:
        from sglang.multimodal_gen.runtime.pipelines.zing_h3_causal_pipeline import (
            ZingH3CausalPipeline,
        )

        pipeline_cls = ZingH3CausalPipeline

    class EncoderOnlyPipeline(pipeline_cls):
        _required_config_modules = ["text_encoder", "tokenizer", "processor"]

        def create_pipeline_stages(self, server_args):
            # In particular, do not construct codecs, KV pools, VAE or DiT.
            pass

    return EncoderOnlyPipeline(
        model_path,
        server_args,
        required_config_modules=["text_encoder", "tokenizer", "processor"],
    )


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _component_identity(root, name):
    from .contract import ConditionError

    source = root / name
    if not source.is_dir():
        raise ConditionError(
            "CONDITION_PROFILE_MISMATCH", "Native component directory is missing"
        )
    files = {}
    visited = set()
    # Exported model components are administrator-owned symlinks to pinned
    # shared assets. Hash their actual complete content without copying weights.
    for current, directories, names in os.walk(source, followlinks=True):
        info = os.stat(current)
        identity = info.st_dev, info.st_ino
        if identity in visited:
            raise ConditionError(
                "CONDITION_PROFILE_MISMATCH",
                "Native component contains a directory cycle",
            )
        visited.add(identity)
        for filename in sorted(names):
            path = Path(current) / filename
            if not path.is_file() or len(files) >= 4096:
                raise ConditionError(
                    "CONDITION_PROFILE_MISMATCH",
                    "Native component inventory is invalid",
                )
            files[str(path.relative_to(root))] = _file_hash(path)
    if not files:
        raise ConditionError(
            "CONDITION_PROFILE_MISMATCH", "Native component inventory is empty"
        )
    return {"files": files}


def _code_identity():
    from .contract import canonical_json, sha256

    root = Path(__file__).parent
    files = {
        name: _file_hash(root / name)
        for name in ("contract.py", "inputs.py", "encoder.py", "native.py")
    }
    presentation = (
        root.parent
        / "pipelines_core/stages/model_specific_stages/minimax_h3/presentation.py"
    )
    files["native_presentation"] = _file_hash(presentation)
    for name in ("minimax_h3_qwen3vl.py", "qwen3vl.py", "qwen_prefix.py"):
        files[name] = _file_hash(root.parent / "models/encoders" / name)
    return "conditioning-code-sha256:" + sha256(canonical_json(files))


def _execution_options():
    from dataclasses import asdict

    from sglang.multimodal_gen.configs.models.encoders.minimax_h3_qwen3vl import (
        h3_qwen_execution_options,
        h3_qwen_fp8_enabled,
    )

    return asdict(h3_qwen_execution_options()) | {"fp8": h3_qwen_fp8_enabled()}


def _require_profiled_execution(environment):
    from .contract import ConditionError

    actual = _execution_options()
    # An old saved profile only describes the explicit all-off baseline.
    expected = environment.get("execution_options", dict.fromkeys(actual, False))
    if (
        not isinstance(expected, dict)
        or set(expected) != set(actual)
        or any(type(value) is not bool for value in expected.values())
        or expected != actual
    ):
        raise ConditionError(
            "CONDITION_PROFILE_MISMATCH",
            "H3 execution options differ from the frozen conditioning profile",
        )
    return actual


def _attention_backend(options):
    return "+".join(
        sorted(
            {
                "fa4" if options[key] else "torch_sdpa"
                for key in ("text_fa4", "vision_fa4")
            }
        )
    )


def _measure_execution(encoder):
    """Inspect loaded modules once; never run these checks per request."""
    from dataclasses import asdict

    import torch

    from .contract import ConditionError

    options = asdict(encoder.execution_options)
    fp8 = False
    for name, param in encoder.named_parameters():
        if not param.is_floating_point() or param.dtype == torch.bfloat16:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "weight" and str(param.dtype).startswith("torch.float8"):
            fp8 = True
        elif (
            leaf in {"weight_scale", "weight_scale_inv", "input_scale"}
            and param.dtype == torch.float32
        ):
            continue
        else:
            raise ConditionError(
                "CONDITION_PROFILE_MISMATCH",
                "Unexpected native encoder weight precision",
            )
    options["fp8"] = fp8
    for part, key in (
        (encoder.model.language_model, "text_fa4"),
        (encoder.model.visual, "vision_fa4"),
    ):
        backends = set()
        for module in part.modules():
            if type(module).__name__ == "Qwen3VLFA4Attention":
                backends.add("fa4")
            elif (
                type(module).__name__ == "QwenVLVisionAttention"
                and module._attention_impl is None
            ):
                # Vision SDPA calls F.scaled_dot_product_attention directly;
                # unlike text LocalAttention it has no backend enum member.
                backends.add("torch_sdpa")
            elif hasattr(module, "backend") and hasattr(module.backend, "name"):
                backends.add(module.backend.name.lower())
        if backends != {"fa4" if options[key] else "torch_sdpa"}:
            raise ConditionError(
                "CONDITION_PROFILE_MISMATCH",
                "Loaded Qwen attention differs from execution options",
            )
    for layer in encoder.model.language_model.layers:
        attention = layer.self_attn
        if (
            (attention.fused_rope is not None) != options["fused_mrope"]
            or layer.mlp.fused_gate_up != options["fused_gate_up"]
            or any(
                (type(module).__name__ == "Qwen3VLHFRMSNorm") != options["hf_rmsnorm"]
                for module in (
                    layer.input_layernorm,
                    layer.post_attention_layernorm,
                    layer.self_attn.q_norm,
                    layer.self_attn.k_norm,
                )
            )
        ):
            raise ConditionError(
                "CONDITION_PROFILE_MISMATCH",
                "Loaded Qwen operators differ from execution options",
            )
    return options


def build_profile(model_path, numerical_environment, *, projection=None):
    """CPU-only asset inventory; the supplied hardware environment is declared.

    This does not attest GPU execution. The native loader independently measures
    and checks its actual numerical environment before serving requests.
    """
    from .contract import ConditionError, DeploymentProfile, safe_json
    from .inputs import PREPROCESS_VERSION

    numerical_environment = dict(numerical_environment)
    numerical_environment.setdefault("execution_options", _execution_options())
    options = _require_profiled_execution(numerical_environment)
    if numerical_environment.get("backend") != _attention_backend(options):
        raise ConditionError(
            "CONDITION_PROFILE_MISMATCH",
            "The native attention backend differs from the H3 execution options",
        )
    root = Path(model_path).absolute()
    runtime = safe_json((root / "zing_config.json").read_bytes())
    projection_identity = None
    if projection is not None:
        relative = Path(projection)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not (root / relative).is_file()
        ):
            raise ValueError("Projection must be an administrator model-relative file")
        projection_identity = {"files": {str(relative): _file_hash(root / relative)}}
    return DeploymentProfile(
        {
            "schema_version": 1,
            "model": _component_identity(root, "text_encoder"),
            "tokenizer": _component_identity(root, "tokenizer"),
            "processor": _component_identity(root, "processor"),
            "projection": projection_identity,
            "runtime": {
                "condition_encoder_config": runtime["condition_encoder_config"]
            },
            "implementation_version": _code_identity(),
            "preprocess_version": PREPROCESS_VERSION,
            "numerical_environment": numerical_environment,
        }
    )


def validate_runtime_receipt(profile, receipt):
    from .contract import ConditionError

    expected = profile.document["numerical_environment"]
    if receipt.get("folding_mode") is not None or any(
        receipt.get(name) != value for name, value in expected.items()
    ):
        raise ConditionError(
            "CONDITION_PROFILE_MISMATCH",
            "Measured native numerical receipt differs from the frozen profile",
        )


def runtime_receipt(encoder, profile):
    """Return observed encoder state; never substitute the declared profile."""
    import copy

    actual = getattr(encoder, "receipt", None)
    validation = getattr(encoder, "receipt_validation", "UNVALIDATED")
    if actual is None or validation not in {"NATIVE_MEASURED", "CPU_TEST_ONLY"}:
        actual, validation = None, "UNVALIDATED"
    return {
        "profile_id": profile.profile_id,
        "validation": validation,
        "actual": copy.deepcopy(actual),
    }


def verify_runtime_receipt(profile, receipt, *, allow_test=False):
    from .contract import ConditionError

    if (
        not isinstance(receipt, dict)
        or set(receipt) != {"profile_id", "validation", "actual"}
        or receipt["profile_id"] != profile.profile_id
        or receipt["validation"]
        not in (
            {"NATIVE_MEASURED", "CPU_TEST_ONLY"} if allow_test else {"NATIVE_MEASURED"}
        )
        or not isinstance(receipt["actual"], dict)
        or "folding_mode" not in receipt["actual"]
    ):
        raise ConditionError(
            "CONDITION_PROFILE_MISMATCH",
            "A validated actual native runtime receipt is required",
        )
    validate_runtime_receipt(profile, receipt["actual"])


class NativeConditionEncoder:
    receipt_validation = "NATIVE_MEASURED"

    def __init__(
        self,
        model_path,
        profile,
        *,
        max_rows=16384,
        gpu_id=0,
        max_prompt_tokens=None,
    ):
        if max_prompt_tokens is not None and (
            type(max_prompt_tokens) is not int or max_prompt_tokens <= 0
        ):
            raise ValueError("Prompt token budget must be a positive integer")
        # Check again when loading a saved profile: the process environment may
        # differ from the one that built it. Reject before CUDA/weight loading.
        _require_profiled_execution(profile.document["numerical_environment"])
        import torch

        from .contract import ConditionError

        if (
            not torch.cuda.is_available()
            or gpu_id >= torch.cuda.device_count()
            or gpu_id < 0
        ):
            raise ConditionError(
                "CONDITION_UNAVAILABLE",
                "A verified NVIDIA device is required for native conditioning",
            )
        root = Path(model_path).resolve()
        document = profile.document
        projection = document["projection"]
        projection_relative = None
        if projection is not None:
            files = projection.get("files", {})
            if len(files) != 1:
                raise ConditionError(
                    "CONDITION_PROFILE_MISMATCH",
                    "Native projection profile requires exactly one safetensors asset",
                )
            projection_relative = next(iter(files))
        actual = build_profile(
            root, document["numerical_environment"], projection=projection_relative
        )
        if actual.profile_id != profile.profile_id:
            raise ConditionError(
                "CONDITION_PROFILE_MISMATCH",
                "Native asset or implementation identity differs from the frozen profile",
            )
        env = document["numerical_environment"]
        receipt = {
            "encoder_tp": 1,
            "gpu": torch.cuda.get_device_name(gpu_id),
            "cc": ".".join(map(str, torch.cuda.get_device_capability(gpu_id))),
            "precision": "bf16",
        }
        import transformers

        software = {
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "cuda": str(torch.version.cuda),
        }
        if any(
            software.get(name) != value
            for name, value in env.get("software", {}).items()
        ) or not {"torch", "transformers", "cuda"} <= set(env.get("software", {})):
            raise ConditionError(
                "CONDITION_PROFILE_MISMATCH",
                "Native software receipt differs from the frozen profile",
            )
        if any(
            env.get(name) != value for name, value in receipt.items()
        ) or not isinstance(env.get("backend"), str):
            raise ConditionError(
                "CONDITION_PROFILE_MISMATCH",
                "Native hardware or precision receipt differs from the frozen profile",
            )
        from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
            ZingH3CausalPipelineConfig,
        )
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            maybe_init_distributed_environment_and_model_parallel,
        )
        from sglang.multimodal_gen.runtime.server_args import (
            ServerArgs,
            set_global_server_args,
        )

        args = ServerArgs.from_kwargs(
            model_path=str(root),
            pipeline_class_name="ZingH3CausalPipeline",
            pipeline_config=ZingH3CausalPipelineConfig(video_decoder_backend="native"),
            num_gpus=1,
            tp_size=1,
            sp_degree=1,
            ulysses_degree=1,
            ring_degree=1,
            encoder_parallel="replicate",
            # Explicit H3 choices select FA4 per text/vision module. The generic
            # loader fallback stays SDPA, including mixed FA4/SDPA profiles.
            attention_backend="torch_sdpa",
            text_encoder_cpu_offload=False,
            dit_cpu_offload=False,
            vae_cpu_offload=False,
            performance_mode="speed",
            warmup_mode="off",
            zing_conditioning_mode="local",
            component_paths={"conditioning_projection": str(root / projection_relative)}
            if projection_relative
            else {},
        )
        set_global_server_args(args)
        torch.cuda.set_device(gpu_id)
        # File rendezvous has no external scheduler or dependency on a DiT group.
        self._rendezvous = tempfile.TemporaryDirectory(prefix="conditioning-tp1-")
        os.environ["RANK"], os.environ["WORLD_SIZE"], os.environ["LOCAL_RANK"] = (
            "0",
            "1",
            str(gpu_id),
        )
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=1,
            sp_size=1,
            distributed_init_method="file://" + self._rendezvous.name + "/group",
        )
        self.pipeline = build_encoder_pipeline(str(root), args)
        if set(self.pipeline.modules) != {"text_encoder", "tokenizer", "processor"}:
            raise ConditionError(
                "CONDITION_INVALID",
                "Native conditioning loaded an unexpected component",
            )
        expected_runtime = document["runtime"].get("condition_encoder_config")
        if expected_runtime != self.pipeline.zing_runtime["condition_encoder_config"]:
            raise ConditionError(
                "CONDITION_PROFILE_MISMATCH",
                "Conditioning runtime differs from the frozen profile",
            )
        encoder = self.pipeline.modules["text_encoder"]
        # Actual loader receipt, rather than trusting a CLI parallelism flag.
        from sglang.multimodal_gen.runtime.distributed import get_tp_world_size

        measured_options = _measure_execution(encoder)
        if get_tp_world_size() != 1:
            raise ConditionError(
                "CONDITION_PROFILE_MISMATCH",
                "Native encoder TP or resident weight precision differs from the profile",
            )
        folding = getattr(encoder.config, "parallel_folding_mode", None)
        self.receipt = receipt | {
            "encoder_tp": get_tp_world_size(),
            "backend": _attention_backend(measured_options),
            "execution_options": measured_options,
            "software": software,
            "folding_mode": folding,
        }
        validate_runtime_receipt(profile, self.receipt)
        self.receipt["loaded_modules"] = sorted(self.pipeline.modules)
        # Detect changed assets between inventory and native loading.
        if (
            build_profile(
                root, document["numerical_environment"], projection=projection_relative
            ).profile_id
            != profile.profile_id
        ):
            raise ConditionError(
                "CONDITION_PROFILE_MISMATCH", "Native assets changed while loading"
            )
        self.max_rows, self.gpu_id = max_rows, gpu_id
        self.max_prompt_tokens = max_prompt_tokens

    @staticmethod
    def create_prefix_cache(max_bytes):
        from sglang.multimodal_gen.runtime.models.encoders.qwen_prefix import (
            QwenPrefixCache,
        )

        return QwenPrefixCache(max_bytes)

    def __call__(self, condition_input):
        return {
            name: tensor.to("cpu")
            for name, tensor in self.encode_device(condition_input).items()
        }

    def encode_device(self, condition_input):
        """Return owned CUDA tensors; transport chooses where they are consumed."""
        import torch

        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )

        from .encoder import encode_condition
        from .inputs import media_identity

        task = getattr(condition_input, "task", None)
        # RPC validation already hashes the pixels. Local callers without that
        # metadata still compute identity from content, never shape/address alone.
        media_digest = None
        if task is not None:
            media_digest = getattr(condition_input, "media_identity", None)
            if media_digest is None:
                media_digest = media_identity(condition_input.media)

        torch.cuda.set_device(self.gpu_id)
        try:
            # Encoder attention layers read the forward context; no DiT step here.
            with (
                torch.inference_mode(),
                set_forward_context(current_timestep=0, attn_metadata=None),
            ):
                result = encode_condition(
                    self.pipeline.modules,
                    self.pipeline.zing_runtime,
                    condition_input.prompt,
                    condition_input.media,
                    self.max_rows,
                    max_prompt_tokens=self.max_prompt_tokens,
                    **(
                        {"prefix_cache": task.cache, "media_identity": media_digest}
                        if task
                        else {}
                    ),
                )
                return {
                    name: tensor.detach().contiguous()
                    for name, tensor in result.items()
                }
        except BaseException:
            if task is not None:
                task.cache.clear()
            raise
        finally:
            # A failed Python forward can still leave kernels enqueued. Always
            # drain this encoder's selected device before the future completes.
            torch.cuda.synchronize(self.gpu_id)
