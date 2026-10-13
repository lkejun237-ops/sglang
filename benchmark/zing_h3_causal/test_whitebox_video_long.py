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


def test_wtp_case_preserves_clip_and_three_reference_mapping():
    case = smoke.load_cases(MANIFEST)["zing-whitebox-video-30s-two-triview-0001"]
    user, target = case["messages"]
    video = user["content"][0]
    assert video["type"] == "video"
    assert video["clip"] == {"start": 0, "end": 0}
    assert video["uri"].endswith(
        "19b458b45cf2baafb2352e27fc8f7da6b203f588bcbd1f5f3776afd45f1fd182.mp4"
    )
    assert target["source"] == {"type": "video", "uri": video["uri"]}
    pictures = [item for item in user["content"] if item["type"] == "image"]
    ids = [f"<Picture {i}>" for i in range(1, 4)]
    assert [item["id"] for item in pictures] == ids
    assert [item["uri"].rsplit("/", 1)[-1] for item in pictures] == [
        "6af5c6f890d0bdaeb7c1dc03e622b2ad10ada221e0649e4d59cc5d82e36a797e.png",
        "d31f804a6ee3c3ecca187d756e95c28cb21e39463a8c8484729233cdf8d7e207.png",
        "175d259b8f7310a3b3eabc0fb2cab489575b13cc01d635ecb2929f12cb5327fa.png",
    ]
    assert target["conditions"] == [
        {"content_id": i, "role": "reference", "latent_modalities": ["visual"]}
        for i in ids
    ]
    assert target["content"] == [
        {"type": "video", "output": {"frames": 720, "height": 480, "width": 832}}
    ]
    assert len(smoke.expected_assets(case)) == len(smoke.media_uris(case)) == 4
    assert all(
        uri.startswith("s3://leap-worldmodel-beijing-single-az/")
        for uri in smoke.media_uris(case)
    )


def test_wtp_prompt_and_replay_provenance_are_frozen():
    case = smoke.load_cases(MANIFEST)["zing-whitebox-video-30s-two-triview-0001"]
    prompt = next(
        item["text"]
        for item in case["messages"][0]["content"]
        if item["type"] == "text"
    )
    assert set(re.findall(r"<Picture \d+>", prompt)) == {
        "<Picture 1>",
        "<Picture 2>",
        "<Picture 3>",
    }
    assert "<Picture 2> 是主角的三视图参考" in prompt
    assert "<Picture 3> 是林姐的三视图参考" in prompt
    original = re.sub(r"<Video (\d+)>", r"@视频\1", prompt)
    original = re.sub(r"<Picture (\d+)>", r"@图片\1", original)
    assert hashlib.sha256(original.encode()).hexdigest() == (
        "4243f0ffa7ef01a35bb0697f160df0af89fb7310038634c509c8882cce9669e1"
    )
    provenance = case["provenance"]
    assert provenance["world_id"] == "world-ec2bb6fa4c7c3c6d1312f7d601bbedca"
    assert provenance["wtp_run_id"] == "run-muu2ogwl-2e6159c1dc696d4a"
    assert provenance["replay_status"] == "evidence-complete-not-reexecuted"
    assert provenance["recorder_seed"]["status"] == "unavailable"
    assert provenance["recorder_seed"]["value"] is None
    evidence = {item["role"]: item for item in provenance["evidence"]}
    assert (
        evidence["source_prompt"]["sha256"]
        == hashlib.sha256(original.encode()).hexdigest()
    )
    assert evidence["source_clip"]["uri"] == case["messages"][1]["source"]["uri"]
    assert evidence["original_recording"]["uri"] not in smoke.media_uris(case)
