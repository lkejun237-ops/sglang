# SPDX-License-Identifier: Apache-2.0
"""Logical Block0 KV backed by nonadjacent, startup-pinned CPU pages."""

import math
from dataclasses import dataclass

import torch

PAGE_ROWS = 2048


@dataclass(frozen=True)
class PagedKV:
    pages: torch.Tensor
    page_ids: tuple[int, ...]
    rows: int

    @property
    def shape(self):
        return (*self.pages.shape[:2], self.rows, *self.pages.shape[4:])

    @property
    def dtype(self):
        return self.pages.dtype

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return self.pages.element_size()

    def slices(self):
        """File order: layer, K/V, logical row; never expose tail padding."""
        page_rows = self.pages.shape[3]
        runs = []
        for index, page_id in enumerate(self.page_ids):
            if runs and page_id == runs[-1][1] + runs[-1][2]:
                start, first, count = runs[-1]
                runs[-1] = (start, first, count + 1)
            else:
                runs.append((index * page_rows, page_id, 1))
        for layer in range(self.shape[0]):
            for part in range(2):
                for start, first, count in runs:
                    rows = min(count * page_rows, self.rows - start)
                    value = self.pages[layer, part, first : first + count].view(
                        -1, *self.pages.shape[4:]
                    )
                    yield layer, part, start, value[:rows]


def kv_slices(data):
    """Stream dense legacy files and paged L1 through the same byte layout."""
    if isinstance(data, PagedKV):
        for _, _, _, value in data.slices():
            yield value
    elif data.dim() >= 2 and data.shape[1] == 2:
        for layer in range(data.shape[0]):
            for part in range(2):
                yield data[layer, part]
    else:
        yield data
