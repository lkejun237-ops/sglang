# SPDX-License-Identifier: Apache-2.0
import json
import sys
from types import SimpleNamespace

import pytest
from resident_soak import replay_collection_decision


def test_default_soak_rejects_numerical_mismatch():
    with pytest.raises(RuntimeError, match="exact complete-case"):
        replay_collection_decision(
            valid=True, exact=False, minimum_seconds=600, collect_mismatch=False
        )


@pytest.mark.parametrize("allocated", [4, 8])
def test_dit_collection_has_four_dedicated_gpus_and_no_codec_roles(allocated):
    from run_isolated_stall_suite import role_layout
    from stall_job import instance_devices

    allocation, active, roles = instance_devices(
        dict(dit_resident_collection=True, allocated_gpu_count=allocated)
    )
    assert (allocation, active) == (allocated, 4)
    assert roles == dict(dit=[0, 1, 2, 3], encoder=[], decoder=[])
    assert role_layout(dict(sp=4, dit_only=True, role_devices=roles), 4) == roles


def test_dit_collection_rejects_shared_codec_roles():
    from stall_job import instance_devices

    with pytest.raises(ValueError, match="dedicated"):
        instance_devices(
            dict(
                dit_resident_collection=True,
                role_devices=dict(dit=[0, 1, 2, 3], encoder=[1], decoder=[2]),
            )
        )


def test_explicit_collection_preserves_numerical_failure():
    decision = replay_collection_decision(
        valid=True, exact=False, minimum_seconds=600, collect_mismatch=True
    )
    assert decision == dict(
        strict_replay_passed=False,
        numerical_status="FAIL",
        full_acceptance_passed=False,
        measurement_scope="diagnostic_resident_soak",
    )


@pytest.mark.parametrize("collect_mismatch", [False, True])
def test_invalid_warmup_never_enters_soak(collect_mismatch):
    with pytest.raises(RuntimeError, match="invalid or incomplete"):
        replay_collection_decision(
            valid=False,
            exact=False,
            minimum_seconds=600,
            collect_mismatch=collect_mismatch,
        )


@pytest.mark.parametrize("failure", ["numerical", "incomplete", "cleanup"])
@pytest.mark.parametrize("minimum", [0, 600])
def test_two_geometry_collection_preserves_failures(
    tmp_path, monkeypatch, failure, minimum
):
    import run_dit_resident_collection as collection

    captures = []
    for height in [480, 736]:
        path = tmp_path / f"capture-{height}"
        path.mkdir()
        (path / "COMPLETE.json").write_text(
            json.dumps(dict(input_frames=15921, capture_scope="dit", chunks=2))
        )
        captures.append(str(path))
    cfg = dict(
        dit_only=True,
        common_captures=captures,
        model="unused-model",
        pipeline="unused-pipeline",
    )
    (tmp_path / "runner-config.json").write_text(json.dumps(cfg))
    job = tmp_path / "job.json"
    job.write_text(json.dumps(dict(root=str(tmp_path), minimum_round_seconds=minimum)))
    started, stopped, closed = [], [], []

    def start(label, command, devices, **kwargs):
        assert ("--minimum-round-seconds" in command) is bool(minimum)
        assert ("--collect-soak-on-numerical-mismatch" in command) is bool(minimum)
        started.append(label)
        for rank in range(4):
            path = tmp_path / "results" / label / f"rank{rank}"
            path.mkdir(parents=True)
            (path / "summary.json").write_text(
                json.dumps(
                    dict(
                        collection_completed=True,
                        minimum_round_seconds=minimum,
                        strict_replay_passed=False,
                    )
                )
            )
            (path / "measured.json").write_text(
                json.dumps(
                    (
                        [
                            dict(
                                seconds=599 if failure == "incomplete" else 600,
                                sessions=[
                                    dict(
                                        fully_replayed=True,
                                        input_frames=15921,
                                        measurement=dict(blocks=[{}, {}]),
                                    )
                                ],
                            )
                        ]
                        if minimum
                        else [
                            dict(
                                seconds=1,
                                blocks=[{}] if failure == "incomplete" else [{}, {}],
                            )
                        ]
                    )
                    * 3
                )
            )
        return SimpleNamespace(wait=lambda **kwargs: 0)

    def stop(process):
        stopped.append(process)
        if failure == "cleanup":
            raise RuntimeError("Injected cleanup failure")

    def load(path, **kwargs):
        height = int(path.parent.name.split("-")[-1])
        return dict(
            params=SimpleNamespace(width=832 if height == 480 else 1280, height=height)
        )

    runner = SimpleNamespace(
        out=tmp_path / "results",
        torch=SimpleNamespace(load=load),
        bench=tmp_path,
        ports=dict(replay_master=23000),
        roles=dict(dit=[0, 1, 2, 3]),
        phase=lambda label: None,
        start=start,
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(collection, "Runner", lambda config: runner)
    monkeypatch.setattr(collection, "stop", stop)
    monkeypatch.setattr(sys, "argv", ["collection", "--config", str(job)])
    expected = {
        "numerical": "Both resident collections completed with numerical FAIL"
        if minimum
        else "Both full-context collections completed with numerical FAIL",
        "incomplete": "Incomplete resident session-work evidence"
        if minimum
        else "Incomplete full-context DiT evidence",
        "cleanup": "Injected cleanup failure",
    }
    with pytest.raises((RuntimeError, ValueError), match=expected[failure]):
        collection.main()
    assert closed == [True]
    assert len(stopped) == len(started)
    prefix = "SOAK" if minimum else "D1"
    if failure == "numerical":
        assert started == [f"{prefix}-0", f"{prefix}-1"]
        result = json.loads((runner.out / "RESIDENT_COLLECTION.json").read_text())
        assert result["collection_completed"] is True
        assert result["numerical_status"] == "FAIL"
        assert result["full_acceptance_passed"] is False
        assert result["minimum_round_seconds"] == minimum
    else:
        assert started == [f"{prefix}-0"]


@pytest.mark.parametrize("minimum", [-1, 1, True])
def test_collection_rejects_invalid_duration_before_runner(
    tmp_path, monkeypatch, minimum
):
    import run_dit_resident_collection as collection
    from stall_job import instance_devices

    with pytest.raises(ValueError, match="0 or 600"):
        instance_devices(
            dict(dit_resident_collection=True, minimum_round_seconds=minimum)
        )

    job = tmp_path / "job.json"
    job.write_text(json.dumps(dict(minimum_round_seconds=minimum)))
    monkeypatch.setattr(sys, "argv", ["collection", "--config", str(job)])
    monkeypatch.setattr(collection, "Runner", lambda _: pytest.fail("Runner started"))
    with pytest.raises(ValueError, match="0 or 600"):
        collection.main()
