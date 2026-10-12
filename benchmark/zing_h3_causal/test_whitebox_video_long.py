# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import re

import prepare_whitebox_video_smoke as smoke
import pytest

MANIFEST = smoke.DEFAULT_MANIFEST.with_name("whitebox_video_long_cases.jsonl")
CASE_IDS = [f"zing-whitebox-video-30s-six-triview-{i:04d}" for i in (1, 2)]


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_long_case_keeps_seven_visual_references_and_full_source(case_id):
    case = smoke.load_cases(MANIFEST)[case_id]
    user, target = case["messages"]
    pictures = [x for x in user["content"] if x["type"] == "image"]
    ids = [f"<Picture {i}>" for i in range(1, 8)]
    assert [x["id"] for x in pictures] == ids
    assert target["conditions"] == [
        {"content_id": i, "latent_modalities": ["visual"], "role": "reference"}
        for i in ids
    ]
    video = user["content"][0]
    assert video["type"] == "video" and "clip" not in video
    assert target["source"] == {"type": "video", "uri": video["uri"]}
    assert target["content"] == [
        {"type": "video", "output": {"frames": 720, "height": 480, "width": 832}}
    ]
    assert len(smoke.expected_assets(case)) == len(smoke.media_uris(case)) == 8
    assert all(
        x.startswith("s3://leap-worldmodel-thailand-single-az/")
        for x in smoke.media_uris(case)
    )
    prompt = next(x["text"] for x in user["content"] if x["type"] == "text")
    definitions = prompt.split("subject_definitions:\n", 1)[1].split("\n\n", 1)[0]
    for i in range(1, 7):
        line = next(
            x for x in definitions.splitlines() if x.startswith(f"<Subject {i}>:")
        )
        assert re.findall(r"<Picture \d+>", line) == [f"<Picture {i + 1}>"]
    assert set(re.findall(r"<Picture \d+>", prompt)) == set(ids)
    assert "cache_uri" not in case
    assert case_id not in smoke.load_cases(smoke.DEFAULT_MANIFEST)
    assert smoke.DEFAULT_CASE == "zing-whitebox-video-five-triview-0001"


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_prepare_keeps_all_seven_references_and_deduplicates_source(tmp_path, case_id):
    case = copy.deepcopy(smoke.load_cases(MANIFEST)[case_id])
    expected = smoke.expected_assets(case)
    for uri in smoke.media_uris(case):
        path = smoke.local_source_path(uri, tmp_path / "bucket")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = uri.encode()
        path.write_bytes(payload)
        expected[uri].update(
            bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )
    request, receipt = smoke.materialize_case(
        case, tmp_path / "prepared", media_root=tmp_path / "bucket"
    )
    assert len(receipt["assets"]) == 8
    assert request["messages"][1]["conditions"] == case["messages"][1]["conditions"]
    assert (
        request["messages"][0]["content"][0]["uri"]
        == request["messages"][1]["source"]["uri"]
    )
    assert all(uri.startswith("file://") for uri in smoke.media_uris(request))
