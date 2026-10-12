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
    return "conditioning-code-sha256:" + sha256(canonical_json(files))


def build_profile(model_path, numerical_environment, *, projection=None):
    """CPU-only asset inventory; the supplied hardware environment is declared.

    This does not attest GPU execution. The native loader independently measures
    and checks its actual numerical environment before serving requests.
    """
    from .contract import DeploymentProfile, safe_json
    from .inputs import PREPROCESS_VERSION

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

    def __init__(self, model_path, profile, *, max_rows=16384, gpu_id=0):
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
            attention_backend=env["backend"],
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

        if get_tp_world_size() != 1 or any(
            param.is_floating_point() and param.dtype != torch.bfloat16
            for param in encoder.parameters()
        ):
            raise ConditionError(
                "CONDITION_PROFILE_MISMATCH",
                "Native encoder TP or resident weight precision differs from the profile",
            )
        backends = {
            module.backend.name.lower()
            for module in encoder.modules()
            if hasattr(module, "backend") and hasattr(module.backend, "name")
        }
        if len(backends) != 1:
            raise ConditionError(
                "CONDITION_PROFILE_MISMATCH",
                "Native encoder has mixed or unobservable attention backends",
            )
        folding = getattr(encoder.config, "parallel_folding_mode", None)
        self.receipt = receipt | {
            "encoder_tp": get_tp_world_size(),
            "backend": next(iter(backends)),
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

    def __call__(self, condition_input):
        import torch
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )

        from .encoder import encode_condition

        torch.cuda.set_device(self.gpu_id)
        try:
            # Encoder attention layers read the forward context; no DiT step here.
            with torch.inference_mode(), set_forward_context(
                current_timestep=0, attn_metadata=None
            ):
                result = encode_condition(
                    self.pipeline.modules,
                    self.pipeline.zing_runtime,
                    condition_input.prompt,
                    condition_input.media,
                    self.max_rows,
                )
                return {
                    name: tensor.detach().to("cpu") for name, tensor in result.items()
                }
        finally:
            # A failed Python forward can still leave kernels enqueued. Always
            # drain this encoder's selected device before the future completes.
            torch.cuda.synchronize(self.gpu_id)
