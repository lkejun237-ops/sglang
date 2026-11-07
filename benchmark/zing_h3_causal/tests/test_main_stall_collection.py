# SPDX-License-Identifier: Apache-2.0
import pytest
from run_main_stall_suite import collect_cold_components, collect_cold_restarts


class FakeRunner:
    def __init__(self, tmp_path, *, failed_request=None, failed_cleanup=False):
        self.out = tmp_path
        self.switch = tmp_path / "diagnostic"
        self.failed_request = failed_request
        self.failed_cleanup = failed_cleanup
        self.events = []

    def launch_instance(self, label):
        self.events.append(("launch", label))

    def client(self, label, **kwargs):
        self.events.append(("request", label, kwargs))
        if label == self.failed_request:
            raise RuntimeError("Injected request failure")

    def shutdown_instance(self):
        self.events.append(("shutdown",))
        if self.failed_cleanup:
            raise RuntimeError("Injected cleanup failure")


def test_baseline_cold_restarts_cover_both_geometries(tmp_path):
    runner = FakeRunner(tmp_path)
    failures = []
    collect_cold_restarts(runner, failures)
    assert not failures
    assert sum(event[0] == "launch" for event in runner.events) == 6
    requests = [event for event in runner.events if event[0] == "request"]
    assert len(requests) == 24
    assert [event[2]["height"] for event in requests] == [480] * 12 + [736] * 12
    assert sum(event[0] == "shutdown" for event in runner.events) == 6


def test_failed_restart_preserves_unexecuted_requests_and_continues(tmp_path):
    runner = FakeRunner(tmp_path, failed_request="S3-832x480-restart0-warm0")
    failures = []
    collect_cold_restarts(runner, failures)
    assert len(failures) == 1
    assert failures[0]["completed"] == ["S3-832x480-restart0-cold"]
    assert failures[0]["unexecuted"] == [
        "S3-832x480-restart0-warm1",
        "S3-832x480-restart0-warm2",
    ]
    assert sum(event[0] == "launch" for event in runner.events) == 6
    assert sum(event[0] == "shutdown" for event in runner.events) == 6


def test_failed_cold_diagnostic_preserves_failure_and_collects_other_geometry(tmp_path):
    runner = FakeRunner(tmp_path, failed_request="cold-components-832x480")
    failures = []
    collect_cold_components(runner, failures)
    assert [row["stage"] for row in failures] == ["cold-components-832x480"]
    assert [event[1] for event in runner.events if event[0] == "request"] == [
        "cold-components-832x480",
        "cold-components-1280x736",
    ]
    assert not runner.switch.exists()


@pytest.mark.parametrize("collect", [collect_cold_restarts, collect_cold_components])
def test_cleanup_failure_prevents_any_next_instance(tmp_path, collect):
    runner = FakeRunner(tmp_path, failed_cleanup=True)
    with pytest.raises(RuntimeError, match="cleanup failure"):
        collect(runner, [])
    assert sum(event[0] == "launch" for event in runner.events) == 1
    assert not runner.switch.exists()
