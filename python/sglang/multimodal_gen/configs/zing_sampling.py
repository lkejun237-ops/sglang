# SPDX-License-Identifier: Apache-2.0
"""Deployment sampling contract for Zing's initial three target blocks."""


def parse_noise_grid(value):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = [int(item.strip()) for item in value.split(",")]
        except ValueError as exc:
            raise ValueError(
                "Zing noise grids must be comma-separated integers"
            ) from exc
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(type(item) is not int for item in value)
    ):
        raise ValueError("Zing noise grids must contain integers")
    if (
        value[0] != 1000
        or any(not 0 < item <= 1000 for item in value)
        or any(left <= right for left, right in zip(value, value[1:]))
    ):
        raise ValueError(
            "Zing noise grids must descend strictly from 1000 without zero"
        )
    return list(value)


def validate_initial_blocks(solver, regular, initial, count):
    if type(count) is not int or count not in (0, 3):
        raise ValueError("Zing initial block count must be 0 or 3")
    if count == 0:
        if initial is not None:
            raise ValueError("Zing initial schedule requires initial block count 3")
        return
    if (
        solver != "few_step"
        or regular != [1000, 750]
        or initial != [1000, 750, 500, 250]
    ):
        raise ValueError(
            "Zing initial blocks require few_step, regular grid 1000,750 "
            "and initial grid 1000,750,500,250"
        )


def normalize_source_sampling_schedule(plan):
    """Validate a complete source deployment plan at admission and worker entry."""
    solver = plan.get("solver")
    regular = parse_noise_grid(plan.get("denoising_step_list"))
    initial = parse_noise_grid(plan.get("initial_denoising_step_list"))
    count = plan.get("initial_block_count", 0)
    if solver != "few_step" or regular is None or len(regular) not in (2, 4):
        raise ValueError("The deployed model requires two-step or four-step streaming.")
    validate_initial_blocks(solver, regular, initial, count)
    return dict(
        solver=solver,
        denoising_step_list=regular,
        initial_block_count=count,
        initial_denoising_step_list=initial,
    )
