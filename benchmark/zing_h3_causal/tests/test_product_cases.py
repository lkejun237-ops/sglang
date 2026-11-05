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


def test_product_case_20261103_collection_preserves_capped_switches_and_scenarios():
    members = load_case_set("product-case-20261103")["members"]
    assert [m["request"]["provenance"]["recording_id"] for m in members] == [
        "128bf7642fc445c7b52817d838f7373b",
        "b4594e29b1924a33a3608678df785ac5",
        "80b3d329982e41c0862b495dcd3954b7",
        "a749b4b779104dc7a2516d80c2ae6a87",
        "fb01f4abe7be40c2a1b363e209b4b106",
        "fd42d3d886e24f53bbb3bf7add6701bc",
        "77ae7475ab8f468797e722214d55aaf5",
        "908c06172f5146e095c21760fb26dc4e",
    ]
    assert [
        m["request"]["messages"][1]["content"][0]["output"]["frames"] for m in members
    ] == [2596, 5993, 1097, 1107, 515, 790, 3023, 867]
    expected = [
        ([6429, 6683, 6420, 6407, 6441, 5597, 6341], [3, 4, 4, 4, 4, 3, 4]),
        ([3522, 3758, 3486, 3758, 3486, 2750, 2750], [3, 3, 3, 3, 3, 2, 2]),
        ([11285, 11000, 10743, 10879, 11006], [6, 6, 6, 6, 6]),
        ([5016], [4]),
        ([10971], [5]),
        ([4512, 4510, 4073], [5, 5, 4]),
        ([5291], [3]),
        ([9515], [2]),
    ]
    for member, (lengths, counts) in zip(members, expected):
        case = member["request"]
        assert case["messages"][1]["source"]["uri"].endswith("/whitebox.mp4")
        replay = case["metadata"]["streaming_replay"]
        assert len(replay["prompt_updates"]) <= 6
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


def test_streaming_prepares_explicit_source_geometry_for_736p_case(
    tmp_path, monkeypatch
):
    import bench_streaming_source as streaming

    member = load_case_set("product-case-20261103")["members"][-1]
    assert member["case_id"] == "zing-product-case-20261103-736p-dream-foyer"
    case = copy.deepcopy(member["request"])
    for asset in case["provenance"]["assets"]:
        path = local_source_path(asset["uri"], tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asset["uri"].encode()
        path.write_bytes(data)
        asset.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(streaming, "load_cases", lambda _: {member["case_id"]: case})
    prepared = streaming.prepare_case(
        "736p-dream-foyer",
        tmp_path,
        tmp_path / "prepared-736p",
        manifest=member["manifest"],
        case_id=member["case_id"],
    )
    assert (prepared["source_width"], prepared["source_height"]) == (832, 480)
    assert (prepared["width"], prepared["height"]) == (1280, 736)
