# SPDX-License-Identifier: Apache-2.0
"""Explicit Disaggregated GPU roles: parsing, CLI validation, and safety invariants."""

import sys
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.server_args.server_args import (
    ServerArgs,
    parse_gpu_list,
)
from sglang.multimodal_gen.utils import FlexibleArgumentParser


def _check_disagg(**overrides):
    values = dict(
        num_gpus=2,
        nnodes=1,
        encoder_gpus=None,
        denoiser_gpus="0,1",
        decoder_gpus="2",
    )
    values.update(overrides)
    validator = getattr(ServerArgs, "_validate_disagg_gpu_roles", None) or getattr(
        ServerArgs, "validate_disagg_gpu_roles", None
    )
    if validator is None:
        pytest.fail("ServerArgs has no _validate_disagg_gpu_roles method implemented")
    validator(SimpleNamespace(**values))


# ==============================================================================
# 1. Verification that deprecated Zing GPU args and CLI flags are removed
# ==============================================================================


def test_deprecated_zing_gpu_args_removed_from_server_args():
    assert not hasattr(ServerArgs, "zing_dit_gpus"), (
        "ServerArgs.zing_dit_gpus must be completely removed"
    )
    assert not hasattr(ServerArgs, "zing_decoder_gpus"), (
        "ServerArgs.zing_decoder_gpus must be completely removed"
    )
    assert not hasattr(ServerArgs, "zing_source_decoder_port"), (
        "ServerArgs.zing_source_decoder_port must be completely removed"
    )


def test_deprecated_zing_gpu_cli_flags_rejected():
    parser = FlexibleArgumentParser()
    ServerArgs.add_cli_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--zing-dit-gpus", "0,1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--zing-decoder-gpus", "2"])


# ==============================================================================
# 2. Verification that unified Disagg GPU role fields exist on ServerArgs
# ==============================================================================


@pytest.mark.parametrize(
    "attr",
    [
        "encoder_gpus",
        "denoiser_gpus",
        "decoder_gpus",
        "disagg_encoder_port",
        "disagg_decoder_port",
    ],
)
def test_unified_disagg_gpu_role_attributes_exist(attr):
    assert hasattr(ServerArgs, attr), f"ServerArgs missing required attribute: {attr}"


def test_validate_disagg_gpu_roles_method_exists():
    assert hasattr(ServerArgs, "_validate_disagg_gpu_roles") or hasattr(
        ServerArgs, "validate_disagg_gpu_roles"
    ), "ServerArgs must define _validate_disagg_gpu_roles"


# ==============================================================================
# 3. Parsing helper (parse_gpu_list) contracts
# ==============================================================================


def test_comma_separated_indices_are_parsed_in_order():
    assert parse_gpu_list("2,0") == [2, 0]


def test_unset_role_parses_to_none():
    assert parse_gpu_list(None) is None


@pytest.mark.parametrize("value", ["", "0,,1", "a", "-1", "1.5"])
def test_malformed_lists_are_rejected(value):
    with pytest.raises(ValueError, match="comma-separated|integer|non-negative"):
        parse_gpu_list(value)


# ==============================================================================
# 4. Happy Paths: Valid disaggregated / monolithic configurations
# ==============================================================================


def test_monolithic_mode_all_none_passes():
    _check_disagg(encoder_gpus=None, denoiser_gpus=None, decoder_gpus=None)


def test_denoiser_and_decoder_disagg_passes():
    # 2 GPUs for DiT (Denoiser), 1 GPU for Decoder
    _check_disagg(
        encoder_gpus=None,
        denoiser_gpus="0,1",
        decoder_gpus="2",
        num_gpus=2,
    )


def test_encoder_denoiser_and_decoder_disagg_passes():
    # 1 GPU for Encoder, 2 GPUs for DiT, 1 GPU for Decoder
    _check_disagg(
        encoder_gpus="0",
        denoiser_gpus="1,2",
        decoder_gpus="3",
        num_gpus=2,
    )


def test_encoder_and_denoiser_disagg_passes():
    # 2 GPUs for Encoder, 2 GPUs for DiT, Decoder shares DiT cards
    _check_disagg(
        encoder_gpus="0,1",
        denoiser_gpus="2,3",
        decoder_gpus=None,
        num_gpus=2,
    )


# ==============================================================================
# 5. Error Paths: Invalid topologies, missing pairings, overlaps, and multi-node
# ==============================================================================


def test_encoder_without_denoiser_is_rejected():
    with pytest.raises(ValueError, match="denoiser"):
        _check_disagg(encoder_gpus="0", denoiser_gpus=None, decoder_gpus=None)


def test_decoder_without_denoiser_is_rejected():
    with pytest.raises(ValueError, match="denoiser"):
        _check_disagg(encoder_gpus=None, denoiser_gpus=None, decoder_gpus="2")


def test_denoiser_only_without_encoder_or_decoder_is_rejected():
    with pytest.raises(ValueError):
        _check_disagg(encoder_gpus=None, denoiser_gpus="0,1", decoder_gpus=None)


def test_denoiser_count_must_match_num_gpus():
    with pytest.raises(ValueError, match="num_gpus"):
        _check_disagg(denoiser_gpus="0,1,2", num_gpus=2)


def test_encoder_and_denoiser_gpu_overlap_rejected():
    with pytest.raises(ValueError, match="overlap|both|shared|same"):
        _check_disagg(
            encoder_gpus="1",
            denoiser_gpus="1,2",
            decoder_gpus="3",
            num_gpus=2,
        )


def test_denoiser_and_decoder_gpu_overlap_rejected():
    with pytest.raises(ValueError, match="overlap|both|shared|same"):
        _check_disagg(
            encoder_gpus="0",
            denoiser_gpus="1,2",
            decoder_gpus="2",
            num_gpus=2,
        )


def test_encoder_and_decoder_gpu_overlap_rejected():
    with pytest.raises(ValueError, match="overlap|both|shared|same"):
        _check_disagg(
            encoder_gpus="3",
            denoiser_gpus="1,2",
            decoder_gpus="3",
            num_gpus=2,
        )


def test_duplicate_gpus_in_encoder_rejected():
    with pytest.raises(ValueError, match="duplicate|more than once|repeat"):
        _check_disagg(
            encoder_gpus="0,0",
            denoiser_gpus="1,2",
            decoder_gpus="3",
            num_gpus=2,
        )


def test_duplicate_gpus_in_denoiser_rejected():
    with pytest.raises(ValueError, match="duplicate|more than once|repeat"):
        _check_disagg(
            encoder_gpus="0",
            denoiser_gpus="1,1",
            decoder_gpus="3",
            num_gpus=2,
        )


def test_duplicate_gpus_in_decoder_rejected():
    with pytest.raises(ValueError, match="duplicate|more than once|repeat"):
        _check_disagg(
            encoder_gpus="0",
            denoiser_gpus="1,2",
            decoder_gpus="3,3",
            num_gpus=2,
        )


def test_multi_node_disagg_rejected():
    with pytest.raises(ValueError, match="single node|nnodes"):
        _check_disagg(
            nnodes=2,
            encoder_gpus=None,
            denoiser_gpus="0,1",
            decoder_gpus="2",
            num_gpus=2,
        )


# ==============================================================================
# 6. SP alignment and pool role list constraints
# ==============================================================================


def test_validate_disagg_gpu_roles_rejects_denoiser_sp_mismatch():
    validator = getattr(ServerArgs, "_validate_disagg_gpu_roles", None) or getattr(
        ServerArgs, "validate_disagg_gpu_roles", None
    )
    if validator is None:
        pytest.fail("ServerArgs has no _validate_disagg_gpu_roles method implemented")
    args = SimpleNamespace(
        model_path="dummy",
        encoder_gpus="0",
        denoiser_gpus="1,2,3,4",
        decoder_gpus="5",
        num_gpus=4,
        ulysses_degree=2,
        ring_degree=1,
        nnodes=1,
    )
    with pytest.raises(ValueError, match="denoiser_gpus"):
        validator(args)


def test_launch_pool_disagg_server_rejects_multiple_roles_in_role_list(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "setproctitle",
        SimpleNamespace(setproctitle=lambda *a, **k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.multimodal_gen.runtime.managers.gpu_worker",
        SimpleNamespace(run_scheduler_process=lambda *a, **k: None),
    )
    from sglang.multimodal_gen.runtime.launch_server import launch_pool_disagg_server

    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(decoder_config={"type": "lynn_streaming"}),
        encoder_gpus=None,
        denoiser_gpus=None,
        decoder_gpus=None,
        role_list=["encoder", "decoder"],
        log_level="info",
        disagg_config_file=None,
        model_path="dummy",
        num_gpus=1,
    )
    with pytest.raises(
        ValueError,
        match="multiple.*instances are not supported|Multiple roles|single active session",
    ):
        launch_pool_disagg_server(
            server_args,
            encoder_gpus=[[0], [1]],
            denoiser_gpus=[[2]],
            decoder_gpus=[[3]],
        )
