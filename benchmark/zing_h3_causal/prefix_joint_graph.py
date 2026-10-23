# SPDX-License-Identifier: Apache-2.0
"""Component benchmark uses the same prefix graph implementation as serving."""

from sglang.multimodal_gen.runtime.models.dits.zing_h3_prefix_graph import (
    PrefixGraphBackend,
    PrefixJointGraphs,
)

__all__ = ["PrefixGraphBackend", "PrefixJointGraphs"]
