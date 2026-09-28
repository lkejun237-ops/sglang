# SPDX-License-Identifier: Apache-2.0
"""Bounded, request-owned reuse of resident Zing AdaLN projections.

Only compact projection outputs are retained. The caller still computes the
time embedding and performs the original token indexing and modulation. No
projection is split into smaller GEMMs, and no weights are reconstructed.
"""

from collections import OrderedDict
from time import perf_counter_ns

import torch


def _signature(module):
    # Runtime LoRA has scalar enable/strength state and unregistered delta
    # tensors. Until its full state has a versioned contract, run that module
    # normally (including merged/disabled adapters), never serve stale results.
    if any(hasattr(child, "disable_lora") for child in module.modules()):
        return None
    result = []
    for value in (*module.parameters(), *module.buffers()):
        try:
            version = value._version
        except RuntimeError:
            return None
        result.append(
            (
                id(value),
                version,
                value.data_ptr(),
                tuple(value.shape),
                tuple(value.stride()),
                value.dtype,
                value.device,
            )
        )
    return tuple(result)


class ZingH3AdalnCache:
    """LRU per session and compute stream; unsuitable inputs run uncached.

    ``build_host_ns`` measures host submission, NOT synchronized CUDA latency.
    ``key_host_ns`` includes the small timestep D2H synchronization. Both costs
    must remain in end-to-end A/B timings. Raw bytes avoid float-key rounding
    and include the entire ordered unique-t batch, including its dtype/shape.
    """

    def __init__(self, max_bytes):
        if max_bytes < 0:
            raise ValueError("Zing AdaLN cache budget must be nonnegative")
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = self.evictions = self.bypasses = 0
        self.build_host_ns = self.key_host_ns = 0
        self._owner = None

    def clear(self):
        self.entries.clear()
        self.bytes = 0
        self._owner = None

    def begin(self, unique_t, embedding, embedder, generation):
        start = perf_counter_ns()
        try:
            signature = _signature(embedder)
            if (
                torch.is_grad_enabled()
                or signature is None
                or unique_t.ndim != 1
                or embedding.ndim != 2
                or unique_t.numel() != embedding.shape[0]
                or embedding.device.type not in {"cpu", "cuda"}
                or torch.is_autocast_enabled()
                or torch.is_autocast_enabled("cpu")
            ):
                self.clear()
                return None
            stream = (
                torch.cuda.current_stream(embedding.device).cuda_stream
                if embedding.is_cuda
                else None
            )
            owner = (
                generation,
                signature,
                embedding.device,
                stream,
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                torch.are_deterministic_algorithms_enabled(),
            )
            if owner != self._owner:
                self.clear()
                self._owner = owner
            values = unique_t.detach().contiguous().view(torch.uint8).cpu()
            return (
                unique_t.dtype,
                tuple(unique_t.shape),
                bytes(values.tolist()),
                embedding.dtype,
                tuple(embedding.shape),
                tuple(embedding.stride()),
            )
        finally:
            self.key_host_ns += perf_counter_ns() - start

    def project(self, slot, module, embedding, batch_key):
        signature = _signature(module) if batch_key is not None else None
        if signature is None or not self.max_bytes:
            self.bypasses += 1
            return module(embedding)
        key = (slot, signature, batch_key)
        cached = self.entries.get(key)
        if cached is not None:
            self.hits += 1
            self.entries.move_to_end(key)
            return cached[0]
        self.misses += 1
        start = perf_counter_ns()
        result = module(embedding)
        self.build_host_ns += perf_counter_ns() - start
        # split_output returns views of one compact allocation. Count actual
        # storage once, including any padding, instead of counting aliases.
        storages = {v.untyped_storage().data_ptr(): v.untyped_storage() for v in result}
        size = sum(storage.nbytes() for storage in storages.values())
        if size > self.max_bytes:
            self.bypasses += 1
            return result
        while self.entries and self.bytes + size > self.max_bytes:
            _, (_, removed) = self.entries.popitem(last=False)
            self.bytes -= removed
            self.evictions += 1
        self.entries[key] = (result, size)
        self.bytes += size
        return result

    def stats(self):
        return {
            name: getattr(self, name)
            for name in (
                "max_bytes",
                "bytes",
                "hits",
                "misses",
                "evictions",
                "bypasses",
                "build_host_ns",
                "key_host_ns",
            )
        } | {"entries": len(self.entries)}
