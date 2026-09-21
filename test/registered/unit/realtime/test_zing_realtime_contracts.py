# SPDX-License-Identifier: Apache-2.0
"""CPU tests of shared pixel metadata, not a simulated service implementation."""

import importlib.util
import sys
from pathlib import Path

import pytest

# Import the actual standard-library leaf without GPU package initialization.
_path = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/multimodal_gen/runtime/realtime/zing_contracts.py"
)
_spec = importlib.util.spec_from_file_location("zing_shared_contracts", _path)
contracts = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = contracts
_spec.loader.exec_module(contracts)
ContractError = contracts.ContractError
PixelRange = contracts.PixelRange
SourceSpan = contracts.SourceSpan
Provenance = contracts.Provenance


def test_shared_fixture():
    import json

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/zing_source_provenance.json").read_text()
    )
    output = fixture["output"]
    p = Provenance(
        PixelRange(
            output["output_frame_start"],
            output["output_frame_start"] + output["num_frames"],
        ),
        tuple(SourceSpan(**span) for span in output["source_alignment"]),
        output["source_dependency_end"],
    )
    assert p.metadata() == output
    assert p.slice(PixelRange(16, 18)).metadata() == fixture["slice"]


@pytest.mark.parametrize(
    "spans,dep",
    [
        ((SourceSpan(1, 0, 0, 1),), 2),
        ((SourceSpan(1, 0, 0, 2), SourceSpan(2, 1, 1, 1)), 3),
        ((SourceSpan(1, 0, 0, 2),), 1),
    ],
)
def test_invalid_provenance(spans, dep):
    with pytest.raises(ContractError):
        Provenance(PixelRange(0, 2), spans, dep)


def test_provenance_copies_mutable_input():
    spans = [SourceSpan(1, 0, 0, 2)]
    p = Provenance(PixelRange(0, 2), spans, 2)
    spans.clear()
    assert p.source_alignment == (SourceSpan(1, 0, 0, 2),)
    with pytest.raises(ContractError):
        SourceSpan(1, 0, False, 2)


@pytest.mark.parametrize("start,end", [(-1, 1), (0, 0), (2, 1), (True, 2), (0, 1.5)])
def test_invalid_pixel_range(start, end):
    with pytest.raises(ContractError):
        PixelRange(start, end)


@pytest.mark.parametrize("seed", range(10))
def test_rebatch_against_pixel_oracle(seed):
    import random

    rng = random.Random(seed)
    spans, oracle = [], []
    for source_id in range(1, 36):
        count = rng.randint(1, 5)
        start = len(oracle)
        spans.append(SourceSpan(source_id, start, start, count))
        oracle.extend([source_id] * count)
    whole = Provenance(PixelRange(0, len(oracle)), tuple(spans), len(oracle) + 17)
    parts, cursor = [], 0
    while cursor < len(oracle):
        end = min(len(oracle), cursor + rng.randint(1, 7))
        part = whole.slice(PixelRange(cursor, end))
        assert [
            s.source_id for s in part.source_alignment for _ in range(s.num_frames)
        ] == oracle[cursor:end]
        assert part.source_dependency_end == len(oracle) + 17
        parts.append(part)
        cursor = end
    joined = Provenance.concatenate(parts)
    assert joined.pixels == whole.pixels
    assert joined.source_dependency_end == whole.source_dependency_end
    assert [
        s.source_id for s in joined.source_alignment for _ in range(s.num_frames)
    ] == oracle


def test_concatenation_keeps_actual_dependency():
    a = Provenance(PixelRange(0, 2), (SourceSpan(1, 0, 0, 2),), 17)
    b = Provenance(PixelRange(2, 3), (SourceSpan(1, 2, 2, 1),), 34)
    assert Provenance.concatenate([a, b]).source_dependency_end == 34
    with pytest.raises(ContractError):
        Provenance.concatenate([])
    with pytest.raises(ContractError):
        Provenance.concatenate([b, a])
    with pytest.raises(ContractError):
        a.slice(PixelRange(1, 3))


def test_ids_have_distinct_type_identity():
    assert contracts.TemporalChunkId is not contracts.PackedRoleId
    assert contracts.TemporalChunkId(30) == 30
    assert contracts.PackedRoleId(61) == 61
