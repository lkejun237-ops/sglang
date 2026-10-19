# SPDX-License-Identifier: Apache-2.0
"""Reuse immutable model signatures without retaining obsolete weight tensors."""

from itertools import chain

import torch


class _GraphModelSignature(tuple):
    def __new__(cls, values):
        result = super().__new__(cls, values)
        # Python tuples recompute their hash on every Graph dictionary lookup.
        # All members are immutable; compute this model-sized hash only once.
        result._hash = tuple.__hash__(result)
        return result

    def __hash__(self):
        return self._hash


class _ModelSignatures:
    def __init__(self, model):
        self.graph = _GraphModelSignature(
            (
                id(value),
                value.data_ptr(),
                None if torch.is_inference(value) else value._version,
                (tuple(value.shape), value.stride(), value.dtype, value.device),
            )
            for value in chain(model.parameters(), model.buffers())
        )
        self.cache = tuple(
            (identity, version, layout[0], layout[2], layout[3])
            for identity, _, version, layout in self.graph
        )
        self.addresses = tuple(entry[1] for entry in self.graph)
        self.versioned = all(entry[2] is not None for entry in self.graph)

    def matches(self, model):
        # Keep scanning the current module tree: parameter/buffer replacement,
        # insertion and removal must remain visible. Temporary per-tensor
        # metadata dies immediately instead of accumulating in every plan key.
        count = 0
        for value in chain(model.parameters(), model.buffers()):
            if count == len(self.graph):
                return False
            identity, address, version, layout = self.graph[count]
            if (
                id(value) != identity
                or value.data_ptr() != address
                or (None if torch.is_inference(value) else value._version) != version
                or value.shape != layout[0]
                or value.stride() != layout[1]
                or value.dtype != layout[2]
                or value.device != layout[3]
            ):
                return False
            count += 1
        return count == len(self.graph)


def model_signatures(model):
    """Validate on every use, allocating a new signature only after a change.

    Graphs and cache owners hold these tuples after plan construction. Returning
    the same tuples for unchanged weights avoids both a large retained object
    graph and repeated deep key comparisons. No tensor contents are read, and
    no parameter references are retained across device/offload replacements.
    """
    previous = getattr(model, "_zing_model_signatures", None)
    if previous is None or not previous.matches(model):
        previous = _ModelSignatures(model)
        model._zing_model_signatures = previous
    return previous
