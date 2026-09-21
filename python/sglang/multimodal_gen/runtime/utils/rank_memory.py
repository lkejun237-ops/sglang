# SPDX-License-Identifier: Apache-2.0
"""Preserve allocator snapshots per global rank without a MAX reduction."""


def gather_rank_memory(distributed, group, rank, load, request):
    local = (rank, load, request)
    rows = [None] * distributed.get_world_size(group)
    distributed.all_gather_object(rows, local, group=group)
    records = {}
    for global_rank, load_snapshot, request_snapshot in rows:
        for phase, snapshot in (("load", load_snapshot), ("request", request_snapshot)):
            key = f"rank_{global_rank}_{phase}_peak"
            if key in records:
                raise ValueError(f"Duplicate memory record: {key}")
            records[key] = snapshot
    return records
