import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docker_storage
from docker_storage import apply, candidates, created_epoch, fingerprint, make_plan

REPO = "registry.example/whitebox/runtime"
NOW = created_epoch("2026-11-04T16:00:00+08:00")


def image(letter="a", **updates):
    return dict(
        Id="sha256:" + letter * 64,
        Created="2026-10-28T10:00:00Z",
        Size=100,
        RepoTags=[REPO + ":old"],
        RepoDigests=[REPO + "@sha256:" + letter * 64],
        RootFS={"Layers": ["layer-" + letter]},
        **updates,
    )


def state(*images):
    return dict(
        daemon_id="one",
        root="/var/lib/docker",
        free_bytes=0,
        images=list(images or [image()]),
        containers=[],
    )


def plan(s):
    return make_plan(s, [REPO], [], age_hours=72, target_free_bytes=1000, now=NOW)


@pytest.mark.parametrize(
    "kind",
    [
        "running",
        "stopped",
        "ancestor",
        "protected",
        "fresh",
        "foreign",
        "local",
        "latest",
        "invalid_date",
    ],
)
def test_image_cannot_be_selected_without_positive_scope(kind):
    s = state()
    protected = []
    if kind in {"running", "stopped"}:
        s["containers"] = [{"image": image()["Id"], "state": kind}]
    elif kind == "ancestor":
        child = image("b")
        child["RootFS"]["Layers"] = ["layer-a", "layer-b"]
        s["images"].append(child)
        s["containers"] = [{"image": child["Id"]}]
    elif kind == "protected":
        protected = [image()["Id"]]
    elif kind == "fresh":
        s["images"][0]["Created"] = "2026-11-04T15:00:00+08:00"
    elif kind == "foreign":
        s["images"][0]["RepoTags"].append("other/service:v1")
    elif kind == "local":
        s["images"][0]["RepoDigests"] = []
    elif kind == "latest":
        s["images"][0]["RepoTags"] = [REPO + ":latest"]
    else:
        s["images"][0]["Created"] = "unknown"
    assert candidates(s, [REPO], protected, now=NOW, age_hours=72) == []


def test_timezone_and_nanosecond_created_time():
    assert created_epoch("2026-11-04T16:00:00.123456789+08:00") == created_epoch(
        "2026-11-04T08:00:00Z"
    )


@pytest.mark.parametrize("point", ["initial", "after_registry"])
def test_new_container_reference_aborts_that_deletion(tmp_path, point):
    s = state()
    p = plan(s)
    calls = []

    def recover(ref):
        s["containers"] = [{"image": image()["Id"]}]
        return True

    if point == "initial":
        s["containers"] = [{"image": image()["Id"]}]
    apply(
        p,
        fingerprint(p),
        tmp_path / "receipt",
        now=lambda: NOW,
        read=lambda: s,
        recover=recover,
        remove=calls.append,
    )
    assert not calls


def test_verified_exact_image_only_and_stop_at_free_space_target(tmp_path):
    s = state(image("a"), image("b"))
    p = plan(s)
    removed = []

    def remove(identity):
        removed.append(identity)
        s["free_bytes"] = 1000

    apply(
        p,
        fingerprint(p),
        tmp_path / "receipt",
        now=lambda: NOW,
        read=lambda: s,
        recover=lambda ref: True,
        remove=remove,
    )
    assert removed == [image()["Id"]]
    records = [json.loads(x) for x in (tmp_path / "receipt").read_text().splitlines()]
    assert records[-1]["target_met"] is True
    with pytest.raises(FileExistsError):
        apply(p, fingerprint(p), tmp_path / "receipt", now=lambda: NOW, read=lambda: s)


def test_missing_registry_proof_preserves_image(tmp_path):
    s = state()
    p = plan(s)
    removed = []
    apply(
        p,
        fingerprint(p),
        tmp_path / "receipt",
        now=lambda: NOW,
        read=lambda: s,
        recover=lambda ref: False,
        remove=removed.append,
    )
    assert not removed


@pytest.mark.parametrize("change", ["hash", "host", "stale", "age"])
def test_changed_plan_or_host_never_deletes(tmp_path, change):
    s = state()
    p = plan(s)
    expected = fingerprint(p)
    if change == "hash":
        p["images"] = []
    elif change == "host":
        s = copy.deepcopy(s)
        s["daemon_id"] = "other"
    elif change == "stale":
        p["created_at"] -= 3601
        expected = fingerprint(p)
    else:
        p["age_hours"] = -1
        expected = fingerprint(p)
    with pytest.raises(ValueError):
        apply(
            p,
            expected,
            tmp_path / "receipt",
            now=lambda: NOW,
            read=lambda: s,
            remove=lambda _: pytest.fail("Must not delete"),
        )


@pytest.mark.parametrize("reply", ["match", "wrong", "list", "invalid_json"])
def test_registry_requires_matching_immutable_descriptor(monkeypatch, reply):
    digest = "sha256:" + "a" * 64
    values = {
        "match": json.dumps({"Descriptor": {"digest": digest}}),
        "wrong": json.dumps({"Descriptor": {"digest": "sha256:" + "b" * 64}}),
        "list": json.dumps([{"Descriptor": {"digest": digest}}]),
        "invalid_json": "unavailable",
    }
    monkeypatch.setattr(docker_storage, "command", lambda *args: values[reply])
    assert docker_storage.registry_exists(REPO + "@" + digest) is (reply == "match")


def test_disk_guard_preserves_reserve_and_download_unpack_peak():
    s = state()
    s["free_bytes"] = 26 * docker_storage.GIB
    with pytest.raises(ValueError, match="headroom"):
        docker_storage.check_headroom(6, read=lambda: s)
    s["free_bytes"] = 27 * docker_storage.GIB
    assert docker_storage.check_headroom(6, read=lambda: s)["allowed"]


@pytest.mark.parametrize("size", [0, -1, float("nan"), float("inf")])
def test_invalid_disk_budget_never_calls_host(size):
    with pytest.raises(ValueError):
        docker_storage.check_headroom(size, read=lambda: pytest.fail("No host access"))
