"""Real boto3 wire contract against a local HTTP peer; no cloud credentials."""

import base64
import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import boto3
from botocore.config import Config

from sglang.multimodal_gen.runtime.realtime.cache_archive import S3CacheObjects
from sglang.multimodal_gen.runtime.realtime.cache_tiers import (
    CacheTiers,
    prune_persistence_states,
)


def test_oss_multi_delete_signs_serialized_body_digest():
    seen = []

    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            expected = base64.b64encode(
                hashlib.md5(body, usedforsecurity=False).digest()
            ).decode()
            seen.append((self.headers.get("Content-MD5"), expected, body))
            valid = self.headers.get("Content-MD5") == expected
            response = (
                b'<DeleteResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Deleted><Key>owned-cache-object</Key></Deleted></DeleteResult>'
                if valid
                else b"<Error><Code>MissingArgument</Code></Error>"
            )
            self.send_response(200 if valid else 400)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Peer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = boto3.client(
        "s3",
        endpoint_url=f"http://127.0.0.1:{server.server_port}",
        aws_access_key_id="test-access",
        aws_secret_access_key="test-secret",
        region_name="test",
        config=Config(
            proxies={},
            retries={"max_attempts": 0},
            request_checksum_calculation="when_required",
        ),
    )
    try:
        policy = CacheTiers.parse(
            json.loads(
                (
                    Path(__file__).parent / "cache/test-shanghai-copy.example.json"
                ).read_text()
            )["tiers"]
        )
        objects = S3CacheObjects(policy, client=client)
        result = objects.client.delete_objects(
            Bucket="test-bucket", Delete={"Objects": [{"Key": "owned-cache-object"}]}
        )
        assert result["Deleted"] == [{"Key": "owned-cache-object"}]
        assert len(seen) == 1 and seen[0][0] == seen[0][1]
        assert b"owned-cache-object" in seen[0][2]
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_status_reaping_preserves_active_and_foreign_files(tmp_path):
    expired, current, foreign = (
        tmp_path / ("a" * 64 + ".json"),
        tmp_path / ("b" * 64 + ".json"),
        tmp_path / "store-ready.json",
    )
    for path in (expired, current, foreign):
        path.write_text("{}")
    os.utime(expired, (1, 1))
    os.utime(foreign, (1, 1))
    link = tmp_path / ("c" * 64 + ".json")
    link.symlink_to(foreign)
    prune_persistence_states(tmp_path)
    assert not expired.exists()
    assert current.exists() and foreign.exists() and link.is_symlink()
