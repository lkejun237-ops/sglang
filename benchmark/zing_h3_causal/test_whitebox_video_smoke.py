# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import re
import shutil
import subprocess

import prepare_whitebox_video_smoke as smoke
import pytest

CASE_ID = "zing-whitebox-video-target-frame0-five-triview-0001"


def test_prompt_subject_reference_mapping():
    case = smoke.load_cases(smoke.DEFAULT_MANIFEST)[CASE_ID]
    content = case["messages"][0]["content"]
    prompt = next(item["text"] for item in content if item["type"] == "text")
    media = {item["id"]: item for item in content if "id" in item}
    definitions = prompt.split("subject_definitions:\n", 1)[1].split("\n\n", 1)[0]
    retention = prompt.split("retention_analysis:\n", 1)[1].split("\n\n", 1)[0]
    entities = ["ranger", "patrol-rover", "field-station", "beacon", "lookout"]
    condition_ids = [c["content_id"] for c in case["messages"][1]["conditions"]]
    assert re.findall(r"^<Picture \d+>", definitions, re.MULTILINE) == condition_ids
    for index, entity in enumerate(entities, 1):
        subject = f"<Subject {index}>"
        picture = f"<Picture {index}>"
        assert condition_ids[index - 1] == picture
        assert media[picture]["uri"].endswith(
            f"Picture_{index + 1}_{entity}_source_top_render_bottom.jpg"
        )
        sheet = next(
            line for line in definitions.splitlines() if line.startswith(picture)
        )
        assert re.findall(r"<Subject \d+>", sheet) == [subject]
        for section in (definitions, retention):
            lines = [line for line in section.splitlines() if line.startswith(subject)]
            assert len(lines) == 1
            assert re.findall(r"<Picture \d+>", lines[0]) == [picture]
    assert set(re.findall(r"<Picture \d+>", prompt)) == set(condition_ids)


def test_frozen_case_matches_production_input_shape():
    cases = smoke.load_cases(smoke.DEFAULT_MANIFEST)
    assert set(cases) == {CASE_ID, smoke.DEFAULT_CASE}

    case = cases[CASE_ID]
    user_content = case["messages"][0]["content"]
    target = case["messages"][1]
    source_prompt = user_content[0]
    references = user_content[1:6]

    assert source_prompt["type"] == "video"
    assert source_prompt["clip"] == {"start": 0, "end": 0}
    assert [item["id"] for item in references] == [
        f"<Picture {index}>" for index in range(1, 6)
    ]
    assert all(item["type"] == "image" for item in references)
    assert target["source"] == {
        "type": "video",
        "uri": source_prompt["uri"].replace("/videos/target/", "/videos/source/"),
    }
    assert target["content"] == [
        {"type": "video", "output": {"frames": 120, "height": 480, "width": 832}}
    ]
    assert [condition["content_id"] for condition in target["conditions"]] == [
        f"<Picture {index}>" for index in range(1, 6)
    ]
    assert len(smoke.media_uris(case)) == 7
    assert "/videos/target/" in source_prompt["uri"]
    assert "<Video 1>" not in [c["content_id"] for c in target["conditions"]]
    assert case["provenance"]["input_revision"] == 2
    assert set(smoke.expected_assets(case)) == set(smoke.media_uris(case))

    provenance = case["provenance"]
    assert provenance["source_dataset"] == (
        "h3_whitebox_multiview_v1_segment_causal_260918"
    )
    assert provenance["source_manifest_sha256"] == (
        "d24cddb259eb7e22dc9b763648934394e98e0818c80a744f79b2fc947176cbed"
    )
    assert provenance["source_manifest_line"] == 10


def test_materialize_keeps_prompt_video_separate_from_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        smoke,
        "extract_prompt_video",
        lambda src, dst: dst.write_bytes(b"one-frame-video"),
    )
    case = copy.deepcopy(smoke.load_cases(smoke.DEFAULT_MANIFEST)[CASE_ID])
    media_root = tmp_path / "bucket"
    expected = smoke.expected_assets(case)
    for uri in smoke.media_uris(case):
        path = smoke.local_source_path(uri, media_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"asset:{uri}".encode()
        path.write_bytes(payload)
        expected[uri]["bytes"] = len(payload)
        expected[uri]["sha256"] = hashlib.sha256(payload).hexdigest()

    request, receipt = smoke.materialize_case(
        case, tmp_path / "prepared", media_root=media_root
    )

    assert len(receipt["assets"]) == 7
    assert receipt["prompt_video"]["frame_index"] == 0
    assert receipt["prompt_video"]["routing"] == "prompt-only video"
    assert all(asset["bytes"] > 0 for asset in receipt["assets"])
    user_source = request["messages"][0]["content"][0]["uri"]
    target_source = request["messages"][1]["source"]["uri"]
    assert user_source != target_source
    assert user_source.endswith("training-target-frame0.mkv")
    assert request["messages"][0]["content"][0]["type"] == "video"
    assert user_source.startswith("file://")


def test_materialize_rejects_asset_drift(tmp_path):
    case = smoke.load_cases(smoke.DEFAULT_MANIFEST)[CASE_ID]
    first_uri = smoke.media_uris(case)[0]
    source = smoke.local_source_path(first_uri, tmp_path / "bucket")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"changed")

    with pytest.raises(ValueError, match="Media checksum mismatch"):
        smoke.materialize_case(
            case, tmp_path / "prepared", media_root=tmp_path / "bucket"
        )


def test_keep_remote_request_is_valid_json(tmp_path):
    case = smoke.load_cases(smoke.DEFAULT_MANIFEST)[CASE_ID]
    output = tmp_path / "request.json"
    smoke.write_json(output, case)
    restored = json.loads(output.read_text())
    assert restored["sample_id"] == CASE_ID
    assert restored["messages"][1]["source"]["uri"].startswith(
        "s3://leap-worldmodel-thailand/"
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_extract_prompt_video_preserves_only_first_rgb_frame(tmp_path):
    source, destination = tmp_path / "target.mkv", tmp_path / "frame0.mkv"
    first = bytes([255, 0, 0]) * 16 * 16
    second = bytes([0, 255, 0]) * 16 * 16
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "16x16",
            "-r",
            "24",
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
            str(source),
        ],
        input=first + second,
        check=True,
    )
    smoke.extract_prompt_video(source, destination)
    decoded = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(destination),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    assert decoded == first


def test_default_preserves_six_reference_contract():
    cases = smoke.load_cases(smoke.DEFAULT_MANIFEST)
    case = cases[smoke.DEFAULT_CASE]
    content = case["messages"][0]["content"]
    target = case["messages"][1]
    pictures = [item for item in content if item["type"] == "image"]
    assert [p["id"] for p in pictures] == [f"<Picture {i}>" for i in range(1, 7)]
    assert [c["content_id"] for c in target["conditions"]] == [
        p["id"] for p in pictures
    ]
    assert (
        smoke.expected_assets(case)[pictures[0]["uri"]]["sha256"]
        == "03475dd7769de747f16b0d13709495cce426c310a62822b93eb38db6eb92fa05"
    )
    assert target["source"]["uri"] == content[0]["uri"]
    assert case["sample_id"] != CASE_ID
    assert case["provenance"].get("input_revision") != 2
    assert len(smoke.media_uris(case)) == 7


@pytest.mark.parametrize(
    "slug,parent_id,line,event_id",
    [
        ("wasteland-snow", "zing-whitebox-video-30s-six-triview-0001", 1, 101),
        ("village-rain-night", "zing-whitebox-video-30s-six-triview-0002", 2, 102),
        ("interior-white-cyan", "zing-whitebox-video-30s-two-triview-0001", 3, 103),
    ],
)
def test_whitebox_prompt_switch_preserves_source_and_prepares_event(
    tmp_path, slug, parent_id, line, event_id
):
    parent_manifest = smoke.DEFAULT_MANIFEST.with_name(
        "whitebox_video_long_cases.jsonl"
    )
    parent = smoke.load_cases(parent_manifest)[parent_id]
    manifest = smoke.DEFAULT_MANIFEST.with_name("prompt_switch_cases.jsonl")
    case = smoke.load_cases(manifest)[f"zing-whitebox-prompt-switch-{slug}"]
    # Only the Qwen presentation becomes frame 0; full source and references stay intact.
    expected_messages = copy.deepcopy(parent["messages"])
    expected_messages[0]["content"][0]["clip"] = {"start": 0, "end": 0}
    assert case["messages"] == expected_messages
    assert all(case["provenance"][k] == v for k, v in parent["provenance"].items())
    origin = case["provenance"]["prompt_switch_derivation"]
    assert (origin["sample_id"], origin["manifest_line"]) == (parent_id, line)
    assert (
        origin["manifest_sha256"]
        == hashlib.sha256(
            parent_manifest.read_text(encoding="utf-8").encode()
        ).hexdigest()
    )
    switch = case["metadata"]["prompt_switch"]
    assert (switch["target_pixel_frame"], switch["end_pixel_frame"]) == (360, 720)
    assert (case["metadata"]["fps"], case["metadata"]["seed"]) == (24, 42)
    event = switch["event"]
    assert (event["type"], event["kind"], event["event_id"]) == (
        "event",
        "prompt",
        event_id,
    )
    user, target = case["messages"]
    assert target["source"]["type"] == "video"
    assert "clip" not in target["source"]
    assert user["content"][0]["clip"] == {"start": 0, "end": 0}
    initial_prompt = next(x["text"] for x in user["content"] if x["type"] == "text")
    # PR #303 replay keeps the original caption and changes only its events tail.
    if "\nevents:\n" in initial_prompt:
        original_body, original_events = initial_prompt.rsplit("\nevents:\n", 1)
        assert original_events.strip() == "N/A"
        updated_body, updated_events = event["payload"].rsplit("\nevents:\n", 1)
        assert updated_body == original_body
    else:
        assert event["payload"].startswith(initial_prompt + "\n\nevents:\n")
        updated_events = event["payload"][len(initial_prompt + "\n\nevents:\n") :]
    assert updated_events.strip() and updated_events.strip() != "N/A"
    assert len(re.findall(r"(?m)^events:$", event["payload"])) == 1
    assert set(re.findall(r"<(?:Picture|Video) \d+>", event["payload"])) == {
        x["id"] for x in user["content"] if "id" in x
    }
    # Synthetic media exercise URI rewriting and event preservation, not video quality.
    prepared_case = copy.deepcopy(case)
    expected = smoke.expected_assets(prepared_case)
    for uri in smoke.media_uris(prepared_case):
        path = smoke.local_source_path(uri, tmp_path / "bucket")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = uri.encode()
        path.write_bytes(payload)
        expected[uri].update(
            bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )
    request, receipt = smoke.materialize_case(
        prepared_case, tmp_path / "prepared", media_root=tmp_path / "bucket"
    )
    assert request["metadata"] == case["metadata"]
    assert request["messages"][0]["content"][0]["clip"] == {"start": 0, "end": 0}
    assert "clip" not in request["messages"][1]["source"]
    assert request["messages"][1]["conditions"] == target["conditions"]
    assert (
        request["messages"][1]["source"]["uri"]
        == request["messages"][0]["content"][0]["uri"]
    )
    assert all(uri.startswith("file://") for uri in smoke.media_uris(request))
    assert len(receipt["assets"]) == len(expected)
