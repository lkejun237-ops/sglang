"""Always-running tests against the SHA-pinned real relay source and exporter.

No private archive, cloud calls, credentials, inference weights or GPU required.
"""

import hashlib
import importlib.util
import json
import runpy
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import prepare_conditioning_collector as builder
import pytest
from conditioning_metric_contract import LABELS, PREFIX, validate_labels
from prepare_conditioning_collector import (
    CONTRACT,
    PINS,
    SOURCE,
    patched_sources,
    prepare,
)
from prepare_conditioning_grafana import MARKER, UID
from prepare_conditioning_grafana import prepare as prepare_dashboard

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "name,labels",
    [
        ("encodes_total", {}),
        ("encodes_total", {"prefix_outcome": "unknown", "session_id": "private"}),
        ("encodes_total", {"prefix_outcome": "arbitrary"}),
        ("tokens_total", {"kind": "total", "instance": "forged"}),
        ("results_total", {"source": "prompt-text"}),
        ("reused_tokens_bucket", {"le": "NaN"}),
        ("reused_tokens_bucket", {"le": "Inf"}),
        ("reused_tokens_bucket", {"le": "1234"}),
        ("reused_tokens_bucket", {"le": "128.00000"}),
        ("queue_duration_seconds_sum", {"le": "1"}),
        ("encode_duration_seconds_bucket", {"le": "1"}),
    ],
)
def test_reject_missing_extra_or_unbounded_labels(name, labels):
    with pytest.raises(ValueError):
        validate_labels(PREFIX + name, labels)


def test_dashboard_preserves_existing_content():
    template = json.loads(
        (ROOT / "benchmark/whitebox/grafana-dashboard.json").read_text(encoding="utf-8")
    )
    existing = {
        "dashboard": {
            "uid": UID,
            "version": 7,
            "panels": template["panels"][:6],
            "tags": ["retained"],
        },
        "meta": {"folderUid": "existing-folder"},
    }
    result = prepare_dashboard(existing, template)
    assert result["dashboard"]["panels"][:6] == existing["dashboard"]["panels"]
    assert result["dashboard"]["version"] == 7 and result["overwrite"] is False
    assert result["folderUid"] == "existing-folder"
    assert result["dashboard"]["panels"][6]["title"] == MARKER
    for panel in result["dashboard"]["panels"][7:]:
        assert panel["datasource"]["uid"] == "whitebox-sh-native"
        for target in panel["targets"]:
            assert "$cluster_id" not in target["expr"]
            assert 'pool="test-shanghai-copy"' in target["expr"]
    with pytest.raises(ValueError):
        prepare_dashboard(dict(existing, dashboard=result["dashboard"]), template)
    existing["dashboard"]["uid"] = "production"
    with pytest.raises(ValueError):
        prepare_dashboard(existing, template)


def test_reject_unreviewed_collector():
    with pytest.raises(ValueError, match="checksum"):
        patched_sources({name: b"unreviewed source" for name in PINS})


def test_reject_unreviewed_archive(tmp_path):
    baseline = tmp_path / "unreviewed.zip"
    baseline.write_bytes(b"unreviewed archive")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="archive"):
        prepare(baseline, output)
    assert not output.exists()


FIXTURE = Path(__file__).with_name("fixtures") / "conditioning-relay-pr662-r3.py.txt"
SCHEDULE_FIXTURE = FIXTURE.with_name("conditioning-schedule-pr662-r3.py.txt")


@pytest.fixture(scope="module")
def collector(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("conditioning-collector")
    sources = patched_sources({"relay.py": FIXTURE.read_bytes()})
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in (
            "conditioning_metric_contract",
            "relay",
            "conditioning_metric_source",
        ):
            path = tmp_path / (name + ".py")
            path.write_bytes(sources[name + ".py"])
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            monkeypatch.setitem(sys.modules, name, module)
            spec.loader.exec_module(module)
        yield sys.modules["relay"]


def test_archive_preserves_room_and_other_files(tmp_path, monkeypatch):
    # Exercise packaging with a tiny archive, while retaining the exact real
    # relay pin. Full release archive validation is an explicit release check.
    baseline = tmp_path / "baseline.zip"
    files = {
        "relay.py": FIXTURE.read_bytes(),
        "copy_production.py": SCHEDULE_FIXTURE.read_bytes(),
        "copy_sources.py": FIXTURE.with_name(
            "conditioning-sources-pr662-r3.py.txt"
        ).read_bytes(),
        "copy_host_snapshot.py": b"unchanged Room source",
        "asset": b"unchanged asset",
    }
    with zipfile.ZipFile(baseline, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    monkeypatch.setattr(
        builder, "BASELINE_SHA256", hashlib.sha256(baseline.read_bytes()).hexdigest()
    )
    output = tmp_path / "prepared"
    receipt = prepare(baseline, output)
    assert receipt["deployed"] is False and receipt["target_registered"] is False
    with zipfile.ZipFile(output / "collector.zip") as built:
        assert set(built.namelist()) == set(files) | {CONTRACT, SOURCE}
        assert built.read("relay.py") != files["relay.py"]
        assert built.read("copy_production.py") != files["copy_production.py"]
        assert built.read("copy_sources.py") == files["copy_sources.py"].replace(
            b"'dit': {23300, 33000}", b"'dit': {23300, 33000, 43310}"
        )
        for name in files.keys() - {
            "relay.py",
            "copy_production.py",
            "copy_sources.py",
        }:
            assert built.read(name) == files[name]
    assert "host_entrypoint" not in receipt


def test_registered_collection_rejects_unproved_targets(collector, monkeypatch):
    from types import SimpleNamespace

    source = sys.modules["conditioning_metric_source"]
    reads, writes = [], []

    async def read(url, token):
        reads.append((url, token))
        return b'world_model_conditioning_encodes_total{prefix_outcome="hit"} 2\n'

    monkeypatch.setattr(source, "_read_metrics", read)
    target = "127.0.0.1:9000"
    proof = dict(
        up=1,
        observed_at=100,
        valid_until=110,
        proof_id="current-pool",
        processes=[
            dict(
                role="encoder",
                targets=[target],
                process_id="encoder-1",
                boot_epoch="epoch-1",
            )
        ],
    )
    sink = SimpleNamespace(write=lambda batch: writes.append(batch) or True)
    result = source.collect_registered(
        proof, sink, target=target, token="read-only", clock=lambda: 102
    )
    assert result == {"status": "written", "series": 1}
    assert len(reads) == len(writes) == 1
    for invalid in [
        None,
        dict(proof, valid_until=101),
        dict(proof, processes=[]),
        dict(proof, processes=proof["processes"] * 2),
    ]:
        assert (
            source.collect_registered(
                invalid, sink, target=target, token="read-only", clock=lambda: 102
            )["status"]
            == "unknown"
        )
    assert len(reads) == len(writes) == 1
    assert (
        source.collect_registered(None, sink, target=None, token=None)["status"]
        == "disabled"
    )


def test_real_schedule_calls_conditioning_without_changing_room_flow(
    collector, monkeypatch
):
    import ast
    import copy
    import os
    from dataclasses import dataclass, replace

    source = sys.modules["conditioning_metric_source"]
    target = "127.0.0.1:9000"
    monkeypatch.setenv("CONDITIONING_METRICS_TARGET", target)
    monkeypatch.setenv("CONDITIONING_METRICS_TOKEN", "read-only")

    async def read(url, token):
        assert url == "http://" + target + "/metrics" and token == "read-only"
        return b'world_model_conditioning_encodes_total{prefix_outcome="hit"} 2\n'

    monkeypatch.setattr(source, "_read_metrics", read)

    class Sources:
        jobs = {"items": []}

        def __init__(self, *args, **kwargs):
            pass

        def capture(self):
            return self

        def observations(self, *args):
            raise ValueError("unobserved host")

        def prove_pool(self):
            return dict(
                up=1,
                http_status=200,
                observed_at=100,
                valid_until=110,
                proof_id="pool-proof",
                processes=[
                    dict(
                        role="encoder",
                        targets=[target],
                        process_id="encoder-1",
                        boot_epoch="epoch-1",
                    )
                ],
            )

        def room(self):
            return "", context("room")["proof"], 101

        def logs(self):
            return []

    class Lease:
        fence = 1

        def read_owned(self):
            return None, {}

        def checkpoint(self, changes):
            self.changes = changes

    @dataclass
    class Observation:
        valid_until: float = 110

    class Sink:
        def write(self, batch):
            return True

    tree = ast.parse(builder.patched_schedule(SCHEDULE_FIXTURE.read_bytes()))
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_round"
    )
    scope = dict(
        time=time,
        os=os,
        CopySources=Sources,
        collect_registered=source.collect_registered,
        build_remote_write=collector.build_remote_write,
        assert_sendable=collector.assert_sendable,
        log_changes=lambda *args: (
            {"error_counts": {}, "cleanup_pending": {}},
            [],
            False,
        ),
        CODES=set(),
        ENVIRONMENT="test-shanghai-copy",
        batch=lambda *args, **kwargs: Observation(),
        replace=replace,
        copy=copy,
    )
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "real-copy-schedule", "exec"
        ),
        scope,
    )
    lease = Lease()
    result = scope["run_round"](None, {}, lease, Sink(), clock=lambda: 102)
    assert result["conditioning"] == {"status": "written", "series": 1}
    assert result["room_series_written"] == 0
    assert lease.changes["coverage"]["conditioning_metrics"] is True


def context(service="sglang"):
    labels = dict(
        business_line_id="world-model-data",
        region="cn-shanghai",
        pool="test-shanghai-copy",
        deployment_id="test-shanghai-copy",
        service=service,
    )
    proof = dict(
        status="matched",
        complete=True,
        target="127.0.0.1:9000",
        labels=labels,
        proof_id="offline-test",
        observed_at=100,
        valid_until=130,
    )
    return dict(
        labels=labels, target=proof["target"], proof=proof, sampled_at=101, now=102
    )


def test_real_exporter_through_relay(collector):
    from prometheus_client import generate_latest
    from prometheus_client.parser import text_string_to_metric_families

    relay = collector
    metrics = runpy.run_path(
        str(ROOT / "python/sglang/multimodal_gen/runtime/conditioning/metrics.py")
    )["ConditioningMetrics"]()
    metrics.encoded(
        dict(hit=True, reused_tokens=512, encoded_tokens=128, total_tokens=640), 10, 2
    )
    metrics.encoded(
        dict(hit=False, reused_tokens=0, encoded_tokens=640, total_tokens=640), 20, 3
    )
    metrics.encoded(None, 30, 4)
    for source in ("encoding", "joined", "result_cache"):
        metrics.results.labels(source).inc()
    raw = generate_latest(metrics.registry)
    expected = {
        (s.name, tuple(sorted(s.labels.items()))): s.value
        for family in text_string_to_metric_families(raw.decode())
        for s in family.samples
        if s.name in LABELS
    }
    text = raw
    cfg = context()
    samples, ignored, dropped = relay._parse(
        text, cfg["labels"], cfg["target"], relay.Limits()
    )
    observed = {}
    for labels, value in samples:
        labels = dict(labels)
        name = labels.pop("__name__")
        assert labels.pop("instance") == cfg["target"]
        for key, item in cfg["labels"].items():
            assert labels.pop(key) == item
        observed[name, tuple(sorted(labels.items()))] = value
    assert observed == expected
    assert ignored > 0 and dropped == 0  # _created samples are intentionally ignored
    batch = relay.build_remote_write(text, **cfg)
    assert batch.series_count == len(expected) and batch.sampled_at_ms == 101000
    assert relay.assert_sendable(batch, proof=cfg["proof"], now=103)
    with pytest.raises(relay.RelayError):
        relay.assert_sendable(batch, proof=cfg["proof"], now=130)
    # A Room target cannot accidentally consume conditioning metrics.
    assert relay.build_remote_write(text, **context("room")).series_count == 0


@pytest.mark.parametrize(
    "sample",
    [
        "encodes_total 1",
        'encodes_total{prefix_outcome="maybe"} 1',
        'encodes_total{prefix_outcome="hit",session_id="private"} 1',
        'encodes_total{prefix_outcome="hit",service="room"} 1',
        'encodes_total{prefix_outcome="hit"} -1',
        'encodes_total{prefix_outcome="hit"} NaN',
        'encodes_total{prefix_outcome="hit"} 1 100000',
        'reused_tokens_bucket{le="1234"} 1',
    ],
)
def test_relay_rejects_invalid_known_samples(collector, sample):
    relay = collector
    raw = PREFIX + sample + "\n"
    with pytest.raises(relay.RelayError):
        relay.build_remote_write(raw, **context())


def test_collisions_empty_scope_and_room_regression(collector):
    relay = collector
    raw = PREFIX + 'encodes_total{prefix_outcome="hit"} 1\n'
    with pytest.raises(relay.RelayError):
        relay.build_remote_write(raw + raw, **context())
    assert relay.build_remote_write("", **context()).series_count == 0
    cfg = context()
    cfg["labels"]["pool"] = "prod-shanghai"
    cfg["labels"]["deployment_id"] = "zing-demo-shanghai-01"
    assert relay.build_remote_write(raw, **cfg).series_count == 0
    room = 'world_model_critical_path_stage_duration_seconds_count{stage="whitebox_prompt_switch_ttff"} 2\n'
    assert relay.build_remote_write(room, **context("room")).series_count == 1
    assert relay.build_remote_write(room, **context()).series_count == 0
    with pytest.raises(relay.RelayError):
        relay.build_remote_write(raw, **dict(context(), now=131))


@pytest.mark.parametrize(
    "status,content_type",
    [(200, "text/plain"), (401, "text/plain"), (302, "text/plain"), (200, "text/html")],
)
def test_authenticated_http_read(collector, tmp_path, status, content_type):
    source = sys.modules["conditioning_metric_source"]
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(
                (self.path, self.headers.get("Authorization") == "Bearer offline-only")
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Location", "/must-not-follow")
            self.end_headers()
            self.wfile.write(
                (PREFIX + 'encodes_total{prefix_outcome="hit"} 1\n').encode()
            )

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token_file = tmp_path / "token"
    token_file.write_text("offline-only\n", encoding="ascii")
    proof = context()["proof"]
    proof["target"] = "127.0.0.1:" + str(server.server_port)
    try:
        if status == 200 and content_type == "text/plain":
            batch = source.prepare_batch(proof, token_file, clock=lambda: 102)
            assert batch.series_count == 1
            assert batch.sampled_at_ms == 102000
        else:
            with pytest.raises(
                source.ConditioningReadError, match="^conditioning_scrape_failed$"
            ):
                source.prepare_batch(proof, token_file, clock=lambda: 102)
        assert calls == [("/metrics", True)]
        # A stale proof must fail before another HTTP request or credential read.
        with pytest.raises(collector.RelayError):
            source.prepare_batch(proof, tmp_path / "missing-token", clock=lambda: 131)
        assert len(calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("mode", ["headers", "body", "stall"])
def test_total_deadline_closes_slow_connection(collector, tmp_path, monkeypatch, mode):
    source = sys.modules["conditioning_metric_source"]
    assert source.SCRAPE_TIMEOUT_SECONDS == 3
    monkeypatch.setattr(source, "SCRAPE_TIMEOUT_SECONDS", 0.25)
    disconnected = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if mode == "stall":
                    self.connection.settimeout(2)
                    if self.connection.recv(1) == b"":
                        disconnected.set()
                    return
                if mode == "headers":
                    self.wfile.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nX-Slow: "
                    )
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"# ")
                for _ in range(50):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(
                        0.04
                    )  # Every read makes progress before a socket timeout.
            except OSError:
                disconnected.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token_file = tmp_path / "token"
    token_file.write_text("offline-only", encoding="ascii")
    proof = context()["proof"]
    proof["target"] = "127.0.0.1:" + str(server.server_port)
    try:
        started = time.monotonic()
        with pytest.raises(
            source.ConditioningReadError, match="^conditioning_scrape_timeout$"
        ):
            source.prepare_batch(proof, token_file, clock=lambda: 102)
        assert 0.15 <= time.monotonic() - started < 1.2
        assert disconnected.wait(1), "Timed-out read must release its connection"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_body_size_limit(collector, tmp_path, monkeypatch):
    source = sys.modules["conditioning_metric_source"]
    monkeypatch.setattr(source, "MAX_METRICS_BYTES", 16)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"# " + b"x" * 17)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token_file = tmp_path / "token"
    token_file.write_text("offline-only", encoding="ascii")
    proof = context()["proof"]
    proof["target"] = "127.0.0.1:" + str(server.server_port)
    try:
        with pytest.raises(
            source.ConditioningReadError, match="^conditioning_scrape_failed$"
        ):
            source.prepare_batch(proof, token_file, clock=lambda: 102)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
