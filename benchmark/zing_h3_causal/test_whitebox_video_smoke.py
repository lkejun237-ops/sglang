# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import re
import shutil
import subprocess

import pytest

import prepare_whitebox_video_smoke as smoke


CASE_ID = "zing-whitebox-video-five-triview-0001"


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
    assert set(cases) == {CASE_ID}

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
