# SPDX-License-Identifier: Apache-2.0
"""Contracts for real archive identity and replay of effective conditions."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from case_sets import load_case_set
from prepare_whitebox_video_smoke import expected_assets, local_source_path


def test_archive_duration_subtracts_nonzero_source_pts_origin():
    from prepare_product_cases import archive_timing

    manifest = {
        "created_at": "2026-10-29T06:38:53.348530+00:00",
        "finished_at": "2026-10-29T06:39:35.238064+00:00",
        "rendering": {"model_frames": 800},
    }
    timing = archive_timing(
        manifest,
        [
            {"start_time_us": 1379365600, "end_time_us": 1383032280},
            {"start_time_us": 1383032280, "end_time_us": 1413432820},
        ],
    )
    assert timing["source_pts_span_seconds"] == pytest.approx(34.06722)
    assert timing["recording_wall_seconds"] == pytest.approx(41.889534)
    assert timing["replay_duration_seconds"] == pytest.approx(800 / 24)


def test_archive_collection_preserves_real_frames_and_condition_versions():
    members = load_case_set("product-case-20261029")["members"]
    assert [m["request"]["provenance"]["recording_id"] for m in members] == [
        "7a3c0ea9d0da42dca1ad8254b0d9c755",
        "8b63cb6d483c464da388313595788a8e",
        "606af3a572244a18a808bf227426cb6f",
        "07567e68cb4949708690b2a53a5aeb66",
    ]
    assert [
        m["request"]["messages"][1]["content"][0]["output"]["frames"] for m in members
    ] == [403, 6247, 764, 15921]
    expected = [
        ([1088], [1]),
        ([9122, 10736, 11335, 11678, 10799], [3] * 5),
        ([2556, 1675, 2282, 1675, 2239, 1675, 2239], [3, 1, 2, 1, 2, 1, 2]),
        ([5504] * 4, [5] * 4),
    ]
    for member, (lengths, counts) in zip(members, expected):
        case = member["request"]
        assert case["messages"][1]["source"]["uri"].endswith("/whitebox.mp4")
        replay = case["metadata"]["streaming_replay"]
        prompts = [case["messages"][0]["content"][-1]["text"]] + [
            update["prompt"] for update in replay["prompt_updates"]
        ]
        assert [len(p) for p in prompts] == lengths
        assert [v["reference_count"] for v in replay["original_versions"]] == counts
        assert [hashlib.sha256(p.encode()).hexdigest() for p in prompts] == [
            v["prompt_sha256"] for v in replay["original_versions"]
        ]
        assert replay["original_wall_clock_replayed"] is False
        assets = expected_assets(case)
        for update in replay["prompt_updates"]:
            assert all(item["uri"] in assets for item in update["references"])


def test_streaming_prepares_complete_reference_replacements(tmp_path, monkeypatch):
    import bench_streaming_source as streaming
    from continuous_source_media import load_prompt_updates

    member = load_case_set("product-case-20261029")["members"][2]
    case = copy.deepcopy(member["request"])
    for asset in case["provenance"]["assets"]:
        path = local_source_path(asset["uri"], tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asset["uri"].encode()
        path.write_bytes(data)
        asset.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(streaming, "load_cases", lambda _: {member["case_id"]: case})
    prepared = streaming.prepare_case(
        "switch",
        tmp_path,
        tmp_path / "prepared",
        manifest=member["manifest"],
        case_id=member["case_id"],
    )
    updates = load_prompt_updates(prepared["prompt_updates_path"])
    assert prepared["reference_ids"] == ["reference-0", "reference-1", "reference-2"]
    assert [len(u["references"]) for u in updates] == [1, 2, 1, 2, 1, 2]
    assert [u["source_frame_start"] for u in updates] == [34, 289, 374, 544, 561, 629]
    assert all(
        local_source_path(item["path"], tmp_path).is_file()
        for update in updates
        for item in update["references"]
    )
    assert (
        json.loads((tmp_path / "prepared/prompt-updates.json").read_text())
        == prepared["prompt_updates"]
    )
    # This asset is used only after the initial condition; preflight must check it too.
    local_source_path(updates[-1]["references"][-1]["path"], tmp_path).write_bytes(
        b"corrupt"
    )
    with pytest.raises(ValueError, match="differs from frozen manifest"):
        streaming.prepare_case(
            "switch",
            tmp_path,
            tmp_path / "bad",
            manifest=member["manifest"],
            case_id=member["case_id"],
        )


def test_finite_client_does_not_silently_drop_replay_updates():
    import bench_client

    with pytest.raises(ValueError, match="bench_streaming_source"):
        bench_client.select_cases(
            SimpleNamespace(case="all", case_set="product-case-20261029")
        )
    assert len(load_case_set("whitebox-basic")["members"]) == 4


def test_default_collection_is_real_replay_and_finite_requires_legacy_opt_in():
    import bench_client

    assert load_case_set()["definition"]["name"] == "product-case-20261029"
    with pytest.raises(ValueError, match="bench_streaming_source"):
        bench_client.select_cases(SimpleNamespace(case="all", case_set=None))
    collection, selected = bench_client.select_cases(
        SimpleNamespace(case="all", case_set="whitebox-basic")
    )
    assert collection["definition"]["name"] == "whitebox-basic"
    assert len(selected) == 4


@pytest.mark.parametrize("selection", [[], ["--case", "all"]])
def test_streaming_cli_defaults_to_real_collection(tmp_path, monkeypatch, selection):
    import sys

    import bench_streaming_source as streaming

    selected = []

    def prepare(label, media_root, output, **kwargs):
        selected.append(kwargs["case_id"])
        output.mkdir(parents=True)
        return {"label": label}

    monkeypatch.setattr(streaming, "prepare_case", prepare)
    monkeypatch.setattr(streaming, "case_geometry", lambda case, **kwargs: case)
    output = tmp_path / "preflight"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_streaming_source.py",
            "--media-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--preflight-only",
            *selection,
        ],
    )
    streaming.main()
    assert selected == [m["case_id"] for m in load_case_set()["members"]]
    assert (
        json.loads((output / "case-set.json").read_text())["definition"]["name"]
        == "product-case-20261029"
    )
