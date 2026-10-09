# SPDX-License-Identifier: Apache-2.0
"""Supported endpoint layouts, independent of a particular experiment plan."""


def layouts(max_gpus=8):
    result = [(f"shared-sp{sp}", sp, 0, "serial") for sp in (1, 2, 4, 8)]
    result += [
        (f"enc{enc}-sp{sp}-{mode}", sp, enc, mode)
        for sp in (1, 2, 4)
        for enc in (1, 2, 4)
        for mode in ("serial", "overlap")
    ]
    return [row for row in result if row[1] + row[2] <= max_gpus]


def selected_layouts(names=()):
    names = names or ()
    by_name = {row[0]: row for row in layouts()}
    if len(names) != len(set(names)) or any(name not in by_name for name in names):
        raise ValueError("Unknown or duplicate endpoint layout")
    return [by_name[name] for name in names]
