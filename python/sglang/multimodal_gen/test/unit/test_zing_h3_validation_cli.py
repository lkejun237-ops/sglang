# SPDX-License-Identifier: Apache-2.0
"""Launch options and legacy aliases must select the same worker configuration."""

import os
import runpy
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def cli():
    path = (
        Path(__file__).resolve().parents[5]
        / "benchmark/zing_h3_causal/validate_real_checkpoint.py"
    )
    return runpy.run_path(str(path))


@pytest.mark.parametrize(
    "options,expected",
    [
        ([], ("fa4", "1", "1")),
        (["--dit-attention-backend", "torch_sdpa"], ("torch_sdpa", "1", "1")),
        (["--experimental-attention", "torch_sdpa"], ("torch_sdpa", "1", "1")),
        (["--no-sp-preserve-projection-shapes"], ("fa4", "0", "1")),
        (["--no-experimental-stable-shapes"], ("fa4", "0", "1")),
        (["--no-compile-adaln"], ("fa4", "1", "0")),
        (["--no-experimental-compiled-adaln"], ("fa4", "1", "0")),
        (
            ["--sp-preserve-projection-shapes", "--compile-adaln"],
            ("fa4", "1", "1"),
        ),
        (
            ["--experimental-stable-shapes", "--experimental-compiled-adaln"],
            ("fa4", "1", "1"),
        ),
    ],
)
def test_launch_options_configure_workers_without_reading_checkpoint(
    cli, monkeypatch, options, expected
):
    variables = (
        "SGLANG_ZING_EXPERIMENTAL_ATTENTION",
        "SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES",
        "SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN",
    )
    for variable in variables:
        monkeypatch.setenv(variable, "stale")
    # Nonexistent model/request paths: selecting the implementation must not
    # load checkpoint data or require a GPU.
    args = cli["parser"]().parse_args(
        [
            "sglang",
            "--sglang-root",
            "/unused/source",
            "--model",
            "/unused/model",
            "--request",
            "/unused/request.json",
            "--output",
            "/unused/output",
            *options,
        ]
    )
    cli["configure_execution"](args)
    assert tuple(os.environ[name] for name in variables) == expected
    assert args.dit_attention_backend == expected[0]
    assert args.sp_preserve_projection_shapes is (expected[1] == "1")
    assert args.compile_adaln is (expected[2] == "1")


def test_window_cli_defaults_and_explicit_absolute_reference(cli):
    required = [
        "sglang",
        "--sglang-root",
        "/unused",
        "--model",
        "/unused",
        "--request",
        "/unused",
        "--output",
        "/unused",
    ]
    args = cli["parser"]().parse_args(required)
    assert (args.window_chunks, args.sink_chunks) == (21, 3)
    assert (args.rope_position_mode, args.rope_max_frame_gap) == ("block_relative", 12)
    ref = cli["parser"]().parse_args(
        [
            *required,
            "--full-history",
            "--sink-chunks",
            "0",
            "--rope-position-mode",
            "absolute",
        ]
    )
    assert ref.window_chunks is None and ref.sink_chunks == 0
    assert ref.rope_position_mode == "absolute"
    args = cli["parser"]().parse_args(
        [
            *required,
            "--window-chunks",
            "21",
            "--sink-chunks",
            "3",
            "--solver",
            "few_step",
            "--denoising-step-list",
            "1000",
            "500",
        ]
    )
    assert (args.window_chunks, args.sink_chunks) == (21, 3)
    assert args.denoising_step_list == [1000, 500]
