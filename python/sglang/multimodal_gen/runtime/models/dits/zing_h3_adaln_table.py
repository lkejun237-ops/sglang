# SPDX-License-Identifier: Apache-2.0
"""Immutable, fixed-version Zing AdaLN tables (no online build or fallback)."""

import functools
import hashlib
import itertools
import json
import threading
from pathlib import Path
from time import perf_counter

import torch
from safetensors.torch import load_file, save_file

SCHEMA = "zing-adaln-table-v1"
STRATEGY = "resident-full-unique-batch-bf16-v1"


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_bytes(value):
    return value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def tensor_description(value):
    return dict(
        dtype=str(value.dtype),
        shape=list(value.shape),
        sha256=hashlib.sha256(tensor_bytes(value)).hexdigest(),
    )


def _validate_batch_values(values):
    if (
        values.dtype != torch.float32
        or values.ndim != 1
        or not values.numel()
        or not bool(torch.isfinite(values).all())
        or not bool(((values >= 0) & (values <= 1)).all())
        or not torch.equal(values, torch.unique(values, sorted=True))
    ):
        raise ValueError(
            "Zing AdaLN requires a sorted unique FP32 timestep batch in [0, 1]"
        )


def batch_description(values):
    _validate_batch_values(values)
    raw = tensor_bytes(values)
    return dict(
        dtype=str(values.dtype),
        shape=list(values.shape),
        sha256=hashlib.sha256(raw).hexdigest(),
        raw_hex=raw.hex(),
    )


def batch_key(values):
    _validate_batch_values(values)
    return (str(values.dtype), tuple(values.shape), tensor_bytes(values).hex())


@functools.lru_cache(maxsize=8)
def _cuda_device_contract(device):
    return (
        list(torch.cuda.get_device_capability(device)),
        torch.cuda.get_device_name(device),
    )


def arithmetic_contract(device):
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Zing AdaLN tables support CPU or CUDA arithmetic only")
    result = dict(
        strategy=STRATEGY,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=device.type,
        float32_precision=torch.get_float32_matmul_precision(),
        tf32=torch.backends.cuda.matmul.allow_tf32,
        fp16_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        deterministic=torch.are_deterministic_algorithms_enabled(),
    )
    if device.type == "cuda":
        capability, gpu = _cuda_device_contract(device)
        result.update(capability=list(capability), gpu=gpu)
    return result


def layout_contract(arch, tags):
    return dict(
        num_layers=arch.num_layers,
        hidden_size=arch.hidden_size,
        time_embed_dim=arch.time_embed_dim,
        timestep_input_dim=arch.timestep_input_dim,
        time_embed_hidden_size=arch.time_embed_hidden_size,
        num_modalities=arch.num_modalities,
        modality_tags=dict(tags),
        block_width=6 * arch.num_modalities * arch.hidden_size,
        final_width=2 * arch.hidden_size,
        block_parts=6,
        final_parts=2,
        output_dtype="torch.bfloat16",
    )


def schedule_contract(solver, timesteps, condition, reference, *, sigmas):
    return dict(
        solver=solver,
        video_timesteps=batch_description(torch.unique(timesteps.cpu(), sorted=True)),
        # The order of solver evaluations is part of admission, too.
        video_order=tensor_description(timesteps.cpu())
        | {"raw_hex": tensor_bytes(timesteps).hex()},
        video_sigmas=tensor_description(sigmas.cpu())
        | {"raw_hex": tensor_bytes(sigmas).hex()},
        condition=float(condition),
        reference=float(reference),
        mode="video",
    )


def video_plan(solver, timesteps, condition=1.0, reference=1.0, *, sigmas):
    """Cover supported causal prefix variants plus source, active and final.

    Each full unique-t batch is built separately; no single-t slices are
    concatenated to synthesize a differently shaped GEMM. Admission also
    enumerates the actual packed request before any KV lease/write.
    """
    values = [torch.tensor([float(t)], dtype=torch.float32) for t in timesteps.cpu()]
    for length in range(1, 4):
        for prefix in itertools.combinations([condition, reference, 1.0], length):
            values.append(
                torch.unique(torch.tensor(prefix, dtype=torch.float32), sorted=True)
            )
    unique = {batch_key(v): v for v in values}
    return dict(
        schedule=schedule_contract(
            solver, timesteps, condition, reference, sigmas=sigmas
        ),
        batches=[v.tolist() for v in unique.values()],
    )


def initial_blocks_plan(regular, initial, count):
    """One weights-free table, with exact admission for both realtime schedules.

    Keep the ordinary schedule fields intact. Old single-schedule manifests
    remain valid for ordinary sampling and fail closed for mixed sampling.
    Deduplicate complete batches, never synthesize a GEMM from singleton rows.
    """
    if type(count) is not int or count != 3:
        raise ValueError("Zing AdaLN initial blocks require count=3")
    for field in ("solver", "condition", "reference", "mode"):
        if regular["schedule"][field] != initial["schedule"][field]:
            raise ValueError(f"Zing AdaLN initial schedule differs in {field}")
    if regular["schedule"]["solver"] != "few_step":
        raise ValueError("Zing AdaLN initial blocks require few_step")
    batches = {}
    for plan in (regular, initial):
        for values in plan["batches"]:
            batch = torch.tensor(values, dtype=torch.float32)
            batches[batch_key(batch)] = values
    return dict(
        schedule=regular["schedule"]
        | {"initial_blocks": {"count": count, "schedule": initial["schedule"]}},
        batches=list(batches.values()),
    )


def checkpoint_identity(files, config_path):
    names = [Path(f).name for f in files]
    if len(set(names)) != len(names):
        raise ValueError("Zing AdaLN checkpoint shard names must be unique")
    return dict(
        config_sha256=file_digest(config_path),
        shards={Path(f).name: file_digest(f) for f in sorted(files)},
    )


def table_key(*, identity, provenance, schedule, arithmetic):
    """Short stable name for one table contract; any input change gives a new key."""
    contract = dict(
        identity=identity,
        provenance=provenance,
        schedule=schedule,
        arithmetic=arithmetic,
    )
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[
        :16
    ]


def table_staleness(table, files, config_path, deployment, schedule=None):
    """Reasons a built table cannot serve this export; empty means current.

    Host-only checks (no torch device needed): artifact files, checkpoint
    identity, effective provenance and, when given, the requested schedule.
    """
    table = Path(table)
    try:
        manifest = json.loads((table / "manifest.json").read_text())
    except (OSError, ValueError):
        return ["table manifest missing or unreadable"]
    reasons = []
    if manifest.get("schema") != SCHEMA:
        reasons.append("table schema mismatch")
    try:
        if file_digest(table / "table.safetensors") != manifest.get("table_sha256"):
            reasons.append("table file checksum mismatch")
    except OSError:
        reasons.append("table file missing")
    identity = checkpoint_identity(files, config_path)
    old = manifest.get("identity", {})
    if old.get("config_sha256") != identity["config_sha256"]:
        reasons.append("export config changed")
    changed = sorted(
        n
        for n in {*old.get("shards", {}), *identity["shards"]}
        if old.get("shards", {}).get(n) != identity["shards"].get(n)
    )
    if changed:
        reasons.append("checkpoint shards changed: " + ", ".join(changed))
    if manifest.get("provenance") != deployment:
        reasons.append("effective checkpoint or LoRA provenance changed")
    if schedule is not None and manifest.get("schedule") != schedule:
        reasons.append("schedule changed")
    return reasons


def time_identity(embedder):
    return {
        name: tensor_description(value) for name, value in embedder.state_dict().items()
    }


def embedder_signature(embedder):
    """Cheap host-only snapshot: storage address and in-place version per tensor."""
    values = [*embedder.state_dict().items()]
    frequencies = getattr(embedder, "_frequency_cache", None)
    if frequencies is not None:
        values.append(("_frequency_cache", frequencies))
    return (
        id(embedder),
        tuple((n, v.data_ptr(), v._version, str(v.device)) for n, v in values),
    )


@torch.no_grad()
def write_table(
    output, *, arch, tags, embedder, projections, plan, identity, provenance
):
    """Stream modules in layer order, keeping only compact outputs resident.

    ``projections`` yields (slot, module); a production builder loads exactly
    one projection's weights at a time. Uses the real module's full-batch GEMM.
    """
    output = Path(output)
    if output.exists():
        raise ValueError("Zing AdaLN output directory already exists")
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        raise ValueError("Zing AdaLN table build requires autocast disabled")
    device = next(embedder.parameters()).device
    arithmetic = arithmetic_contract(device)
    batches = [torch.tensor(v, dtype=torch.float32) for v in plan["batches"]]
    keys = [batch_key(v) for v in batches]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Zing AdaLN plan batches must be nonempty and unique")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    embeddings = [
        torch.nn.functional.silu(embedder(v.to(device))).bfloat16() for v in batches
    ]
    tensors, descriptions, slots = {}, {}, []
    for slot, module in projections:
        slots.append(str(slot))
        for i, embedding in enumerate(embeddings):
            parts = module(embedding)
            # Preserve the compact, modality-interleaved split_output layout.
            value = torch.cat(parts, dim=-1).cpu().contiguous()
            if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
                raise ValueError(
                    "Zing AdaLN projection produced invalid compact parameters"
                )
            name = f"batch.{i}.slot.{slot}"
            tensors[name], descriptions[name] = value, tensor_description(value)
        del module
    if slots != [str(i) for i in range(arch.num_layers)] + ["final"]:
        raise ValueError("Zing AdaLN builder must yield every block and final in order")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    build_receipt = dict(
        scope="embeddings_streamed_projections_and_compact_output_transfer",
        synchronized_seconds=perf_counter() - started,
        device=str(device),
    )
    if device.type == "cuda":
        build_receipt["memory"] = dict(
            allocated_bytes=torch.cuda.memory_allocated(device),
            reserved_bytes=torch.cuda.memory_reserved(device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    manifest = dict(
        schema=SCHEMA,
        build=build_receipt,
        identity=identity,
        provenance=provenance,
        layout=layout_contract(arch, tags),
        arithmetic=arithmetic,
        schedule=plan["schedule"],
        batches=[batch_description(v) for v in batches],
        time_embedder=time_identity(embedder),
        frequencies=tensor_description(embedder._frequency_cache),
        tensors=descriptions,
    )
    output.mkdir(parents=True)
    save_file(tensors, str(output / "table.safetensors"))
    manifest["table_sha256"] = file_digest(output / "table.safetensors")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def keep_zing_weight(name):
    name = name.removeprefix("backbone.")
    return not (
        name.startswith("transformer_blocks.")
        and ".adaln_proj." in name
        or name.startswith("norm_out.linear.")
    )


class _PlanLease:
    def __init__(self, table):
        self.table = table

    def close(self):
        table = self.table
        if table is not None:
            with table._lock:
                if self.table is not None:
                    table.active_plans -= 1
                    self.table = None


class ZingAdalnTable:
    """One immutable worker-owned table; fixed budget, no eviction/rebuild.

    Every session leases a complete plan. Device storage survives all leases,
    so graph references cannot dangle after session close. SP ranks load their
    own copy; lookups have no collectives. CUDA readiness is fenced explicitly.
    """

    def __init__(self, path, *, arch, tags, max_bytes):
        path = Path(path)
        self.manifest = json.loads((path / "manifest.json").read_text())
        m = self.manifest
        if m.get("schema") != SCHEMA or m.get("layout") != layout_contract(arch, tags):
            raise ValueError("Zing AdaLN table schema or projection layout mismatch")
        if file_digest(path / "table.safetensors") != m["table_sha256"]:
            raise ValueError("Zing AdaLN table file checksum mismatch")
        self.values = load_file(str(path / "table.safetensors"), device="cpu")
        expected = {
            f"batch.{i}.slot.{slot}"
            for i in range(len(m["batches"]))
            for slot in [*range(arch.num_layers), "final"]
        }
        if set(self.values) != expected or set(m["tensors"]) != expected:
            raise ValueError("Zing AdaLN table has missing or unexpected projections")
        if not m["batches"]:
            raise ValueError(
                "Zing AdaLN table must contain at least one timestep batch"
            )
        self.keys = {}
        for i, batch in enumerate(m["batches"]):
            raw = bytearray.fromhex(batch["raw_hex"])
            t = torch.frombuffer(raw, dtype=torch.float32).clone()
            if batch_description(t) != batch or batch_key(t) in self.keys:
                raise ValueError("Zing AdaLN table timestep descriptor mismatch")
            self.keys[batch_key(t)] = i
            for slot in [*range(arch.num_layers), "final"]:
                name = f"batch.{i}.slot.{slot}"
                value = self.values[name]
                final = slot == "final"
                shape = (
                    t.numel() * (1 if final else arch.num_modalities),
                    (2 if final else 6) * arch.hidden_size,
                )
                if (
                    tuple(value.shape) != shape
                    or value.dtype != torch.bfloat16
                    or not bool(torch.isfinite(value).all())
                    or tensor_description(value) != m["tensors"][name]
                ):
                    raise ValueError(
                        "Zing AdaLN table tensor layout or checksum mismatch"
                    )
        storages = {
            v.untyped_storage().data_ptr(): v.untyped_storage()
            for v in self.values.values()
        }
        self.bytes = sum(s.nbytes() for s in storages.values())
        if max_bytes < self.bytes:
            raise ValueError("Zing AdaLN table exceeds its storage budget")
        self.checkpoint_verified = False
        self._verified_embedder = None
        self.device = None
        self.ready = None
        self._plan_cache = {}
        self.active_plans = self.hits = 0
        self._lock = threading.RLock()

    def validate_checkpoint(self, files, config_path):
        self.checkpoint_verified = False
        if checkpoint_identity(files, config_path) != self.manifest["identity"]:
            raise ValueError("Zing AdaLN table effective checkpoint identity mismatch")
        self.checkpoint_verified = True

    def prepare(self, batches, *, device, schedule, embedder):
        with self._lock:
            if not self.checkpoint_verified:
                raise ValueError(
                    "Zing AdaLN table requires checkpoint identity validation "
                    "before admission"
                )
            if schedule != self.manifest["schedule"]:
                raise ValueError("Zing AdaLN table schedule mismatch")
            if not batches:
                raise ValueError(
                    "Zing AdaLN admission requires a complete nonempty plan"
                )
            for batch in batches:
                self.lookup(batch)
            device = torch.device(device)
            if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
                raise ValueError(
                    "Zing AdaLN table inference requires autocast disabled"
                )
            if arithmetic_contract(device) != self.manifest["arithmetic"]:
                raise ValueError("Zing AdaLN table arithmetic or build device mismatch")
            self._verify_embedder(embedder, device)
            if self.device is None:
                self.values = {name: v.to(device) for name, v in self.values.items()}
                self.device = device
                if device.type == "cuda":
                    self.ready = torch.cuda.Event()
                    self.ready.record(torch.cuda.current_stream(device))
            elif self.device != device:
                raise ValueError("Zing AdaLN table cannot change its execution device")
            self.wait_ready()
            self.active_plans += 1
            return _PlanLease(self)

    def _verify_embedder(self, embedder, device):
        # Weights are immutable in weights-free mode; hash content once, then
        # only compare host-side storage/version signatures.
        if self._verified_embedder == embedder_signature(embedder):
            return
        if time_identity(embedder) != self.manifest["time_embedder"]:
            raise ValueError("Zing AdaLN table time embedder identity mismatch")
        with torch.no_grad():
            embedder(torch.tensor([1.0], dtype=torch.float32, device=device))
        if (
            tensor_description(embedder._frequency_cache)
            != self.manifest["frequencies"]
        ):
            raise ValueError("Zing AdaLN table frequency constants mismatch")
        self._verified_embedder = embedder_signature(embedder)

    def wait_ready(self):
        if self.ready is not None:
            torch.cuda.current_stream(self.device).wait_event(self.ready)

    def lookup(self, batch):
        key = batch_key(batch)
        if key not in self.keys:
            raise ValueError(
                "Zing AdaLN table does not cover the complete timestep batch"
            )
        return self.keys[key]

    def project(self, slot, module, embedding, batch_index):
        if self.device is None:
            raise ValueError(
                "Zing AdaLN table requires complete-plan admission before inference"
            )
        if module is not None:
            raise ValueError(
                "Zing weights-free mode must not retain projection modules"
            )
        self.wait_ready()
        self.hits += 1
        value = self.values[f"batch.{batch_index}.slot.{slot}"]
        return tuple(value.chunk(2 if slot == "final" else 6, dim=-1))

    def project_plan(self, batch_indices, num_layers):
        """Resolve and cache all slot projections for a fixed timestep batch tuple."""
        if self.device is None:
            raise ValueError(
                "Zing AdaLN table requires complete-plan admission before inference"
            )
        self.wait_ready()
        slots = (*range(num_layers), "final")
        self.hits += len(slots) * len(batch_indices)
        key = (tuple(batch_indices), num_layers)
        cached = self._plan_cache.get(key)
        if cached is not None:
            return cached
        resolved = {}
        for slot in slots:
            chunks = 2 if slot == "final" else 6
            parts = [
                self.values[f"batch.{idx}.slot.{slot}"].chunk(chunks, dim=-1)
                for idx in batch_indices
            ]
            for i, values in enumerate(zip(*parts)):
                resolved[f"projection.{slot}.{i}"] = (
                    values[0] if len(parts) == 1 else torch.cat(values)
                )
        self._plan_cache[key] = resolved
        return resolved

    def stats(self):
        return dict(
            bytes=self.bytes,
            entries=len(self.keys),
            active_plans=self.active_plans,
            hits=self.hits,
            misses=0,
            fallback=0,
            checkpoint_verified=self.checkpoint_verified,
        )
