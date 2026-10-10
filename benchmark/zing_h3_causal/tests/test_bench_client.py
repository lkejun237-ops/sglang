# SPDX-License-Identifier: Apache-2.0
"""Small persistent contracts; protocol fault injection stays out of the suite."""

import copy
import json

import pytest
from bench_client import Delivery, calculate_metrics, prepare_case, summarize


def header(block=0, part=0, parts=1, frames=2):
    return {
        "format": "rgb24",
        "channels": 3,
        "width": 2,
        "height": 1,
        "num_frames": frames,
        "total_size": frames * 6,
        "chunk_index": block,
        "frame_batch_index": part,
        "num_frame_batches": parts,
        "is_final_frame_batch": part == parts - 1,
        "request_id": f"block-{block}",
    }


def test_known_unequal_chunks_use_frames_and_exclude_completion():
    delivery = Delivery(10, 2, 1)
    for block, (frames, at) in enumerate([(2, 2), (3, 5), (5, 9)]):
        delivery.accept(header(block=block, frames=frames), bytes(frames * 6), at)
    delivery.complete()
    metrics = calculate_metrics(delivery.chunks)
    assert metrics["ttff_seconds"] == 2
    assert metrics["e2e_seconds"] == 9
    assert metrics["fps"] == pytest.approx(10 / 9)
    assert metrics["tpoc_seconds"] == 3.5
    assert [row["icl_seconds"] for row in delivery.chunks] == [None, 3, 4]
    assert metrics["icl_seconds"]["median"] == 3.5
    assert metrics["icl_seconds"]["p95"] == pytest.approx(3.95)


def test_empty_and_single_chunk_do_not_invent_intervals():
    assert calculate_metrics([])["ttff_seconds"] is None
    assert calculate_metrics([])["icl_seconds"]["count"] == 0
    metrics = calculate_metrics([{"arrival_seconds": 2, "num_frames": 10}])
    assert metrics["fps"] == 5
    assert metrics["tpoc_seconds"] is None
    assert metrics["icl_seconds"]["median"] is None


def test_split_batches_are_distinct_deliveries_of_one_server_chunk():
    delivery = Delivery(4, 2, 1)
    delivery.accept(header(part=0, parts=2), bytes(12), 2)
    delivery.accept(header(part=1, parts=2), bytes(12), 3)
    delivery.complete()
    assert delivery.frames == 4
    assert len(delivery.chunks) == 2
    assert delivery.chunks[1]["output_frame_start"] == 2
    assert calculate_metrics(delivery.chunks)["tpoc_seconds"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"chunk_index": 1},
        {"frame_batch_index": 1},
        {"total_size": 1},
        {"num_frames": 0},
        {"format": "jpeg"},
        {"output_frame_start": 3},
    ],
)
def test_invalid_delivery_is_not_counted(change):
    delivery = Delivery(4, 2, 1)
    bad = {**header(), **change}
    with pytest.raises(ValueError):
        delivery.accept(bad, bytes(12), 2)
    assert not delivery.chunks
    assert delivery.frames == 0


def test_duplicate_or_incomplete_delivery_cannot_pass():
    delivery = Delivery(4, 2, 1)
    delivery.accept(header(), bytes(12), 2)
    with pytest.raises(ValueError):
        delivery.accept(header(), bytes(12), 3)
    with pytest.raises(ValueError):
        delivery.complete()


def test_summary_excludes_warmup_and_failed_partial_metrics():
    rows = []
    for phase, success, e2e in [
        ("warmup", True, 100),
        ("measured", True, 9),
        ("measured", False, 1),
    ]:
        rows.append(
            {
                "label": "5s",
                "phase": phase,
                "success": success,
                "metrics": {
                    "ttff_seconds": 2,
                    "e2e_seconds": e2e,
                    "fps": 10 / e2e,
                    "tpoc_seconds": 3.5,
                },
                "icl_values_seconds": [3, 4],
            }
        )
    summary = summarize(rows, [{"label": "5s", "case_id": "fixed"}], 3)["5s"]
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["not_run_measured"] == 1
    assert summary["metrics"]["e2e_seconds"]["median"] == 9
    assert summary["icl_seconds"]["count"] == 2


def test_missing_30s_is_not_replaced_with_another_case(tmp_path):
    with pytest.raises(ValueError, match="unavailable"):
        prepare_case("30s", tmp_path / "absent.jsonl", tmp_path, {})


def test_old_five_reference_variant_cannot_be_relabelled_as_fixed_case(tmp_path):
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "whitebox_video_smoke_cases.jsonl"
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    variant = copy.deepcopy(cases[1])
    variant["sample_id"] = cases[0]["sample_id"]
    manifest = tmp_path / "wrong.jsonl"
    manifest.write_text(json.dumps(variant))
    with pytest.raises(ValueError, match="six-reference"):
        prepare_case("5s", manifest, tmp_path, {})
