# SPDX-License-Identifier: Apache-2.0
"""One task's resident Qwen KV and projected token features.

The conditioning service serializes all access and owns the lifetime. This is
not a model-global cache: callers without a task continue to encode in full.
"""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class QwenPrefixCache:
    max_bytes: int
    encoder: Any = None
    media_identity: str | None = None
    ids: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    kv: Any = None
    context: torch.Tensor | None = None
    reused_rows: int = 0
    encoded_rows: int = 0
    reason: str = "cold"

    def clear(self):
        self.encoder = self.media_identity = None
        self.ids = self.positions = self.kv = self.context = None

    def prefix_length(self, encoder, ids, positions, media_identity):
        if (
            self.encoder is not encoder
            or self.kv is None
            or self.media_identity != media_identity
        ):
            return 0
        n = min(ids.shape[-1], self.ids.shape[-1])
        equal = (ids[..., :n] == self.ids[..., :n]).reshape(-1)
        equal &= (positions[..., :n] == self.positions[..., :n]).all(dim=(0, 1))
        changed = (~equal).nonzero()
        return int(changed[0, 0]) if changed.numel() else n
