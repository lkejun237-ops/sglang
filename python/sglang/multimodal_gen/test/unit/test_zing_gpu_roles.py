# SPDX-License-Identifier: Apache-2.0
"""Explicit Zing GPU roles: parsing and the checks that reject unsafe layouts."""

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.server_args.server_args import (
    ServerArgs,
    parse_gpu_list,
)


def _check(**overrides):
    values = dict(
        num_gpus=2,
        nnodes=1,
        zing_dit_gpus="0,1",
        zing_decoder_gpus="2",
    )
    values.update(overrides)
    ServerArgs._validate_zing_gpu_roles(SimpleNamespace(**values))


def test_comma_separated_indices_are_parsed_in_order():
    assert parse_gpu_list("2,0") == [2, 0]


def test_unset_role_parses_to_none():
    assert parse_gpu_list(None) is None


@pytest.mark.parametrize("value", ["", "0,,1", "a", "-1", "1.5"])
def test_malformed_lists_are_rejected(value):
    with pytest.raises(ValueError, match="comma-separated"):
        parse_gpu_list(value)


def test_valid_roles_pass():
    _check()


def test_no_decoder_needs_no_dit_list():
    _check(zing_dit_gpus=None, zing_decoder_gpus=None)


def test_dit_list_without_decoder_is_rejected():
    with pytest.raises(ValueError, match="requires --zing-decoder-gpus"):
        _check(zing_decoder_gpus=None)


def test_dit_list_must_match_num_gpus():
    with pytest.raises(ValueError, match="exactly --num-gpus"):
        _check(zing_dit_gpus="0,1,3")


def test_a_gpu_cannot_serve_both_roles():
    with pytest.raises(ValueError, match="both the DiT and the decoder"):
        _check(zing_decoder_gpus="1")


def test_a_decoder_gpu_cannot_repeat():
    with pytest.raises(ValueError, match="more than once"):
        _check(zing_decoder_gpus="2,2")


def test_multi_node_decoder_is_rejected():
    with pytest.raises(ValueError, match="single node"):
        _check(nnodes=2)
