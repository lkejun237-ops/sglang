# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from fractions import Fraction

import av
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.realtime.whitebox.client_recording import (
    validate_video,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports


def video(path, times):
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=24)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "tune": "zerolatency"}
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1_000_000)
        for pts in times:
            frame = av.VideoFrame.from_ndarray(
                np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24"
            )
            frame.pts = pts
            frame.time_base = Fraction(1, 1_000_000)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


def test_upload_verifier_requires_real_model_timestamps(tmp_path):
    model = tmp_path / "model.mp4"
    actual = tmp_path / "actual.mp4"
    video(model, [0, 100000, 900000])
    video(actual, [0, 100000, 900000])

    def intent():
        return {
            "bytes": actual.stat().st_size,
            "sha256": hashlib.sha256(actual.read_bytes()).hexdigest(),
        }

    assert validate_video(actual, model, intent())["frames"] == 3
    with pytest.raises(ValueError, match="checksum"):
        validate_video(actual, model, {**intent(), "sha256": "f" * 64})
    video(actual, [0, 41667, 83333])
    with pytest.raises(ValueError, match="timestamp"):
        validate_video(actual, model, intent())
    video(actual, [0, 100000])
    with pytest.raises(ValueError, match="count"):
        validate_video(actual, model, intent())


def test_report_upload_auth_scope_and_idempotent_intent(tmp_path):
    rid = "a" * 32
    token = "session-only"
    directory = tmp_path / rid
    directory.mkdir()
    (tmp_path / ".reports").mkdir()
    (tmp_path / ".reports" / f"{rid}.json").write_text(
        json.dumps({"token_hash": hashlib.sha256(token.encode()).hexdigest()})
    )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "upload_status": "complete",
                "ui": {"mode": "browser", "files": {"main.js": "b" * 64}},
                "object_prefix": "test/world/game/streams/" + rid + "/",
            }
        )
    )
    (directory / "inputs.jsonl").write_text('{"prompt":"private"}\n')

    class Signer:
        def generate_presigned_url(self, operation, **kwargs):
            assert operation == "put_object"
            assert kwargs["Params"]["Key"].startswith(
                "test/world/game/streams/" + rid + "/client-ui/"
            )
            return "https://upload.invalid/signed"

    class Store:
        playback_client = Signer()
        bucket = "bucket"

    class Archives:
        store = Store()
        active = set()
        max_pending = 1

    report = Reports(tmp_path, Archives())
    app = FastAPI()
    report.install(app)
    http = TestClient(app)
    base = f"/v1/whitebox/recordings/{rid}"
    body = {"bytes": 100, "sha256": "b" * 64, "content_md5": "AAAAAAAAAAAAAAAAAAAAAA=="}
    assert http.post(base + "/ui-recording/intent", json=body).status_code == 404
    headers = {"Authorization": "Bearer " + token}
    assert (
        http.get(base + "/inputs", headers=headers).json()["intervals"][0]["prompt"]
        == "private"
    )
    first = http.post(base + "/ui-recording/intent", json=body, headers=headers)
    assert first.status_code == 200
    again = http.post(base + "/ui-recording/intent", json=body, headers=headers)
    assert first.json()["upload_id"] == again.json()["upload_id"]
    assert (
        http.post(
            base + "/ui-recording/intent",
            json={**body, "sha256": "c" * 64},
            headers=headers,
        ).status_code
        == 409
    )
    assert (
        http.post(
            base + "/ui-recording/intent",
            json={**body, "content_md5": "bad"},
            headers=headers,
        ).status_code
        == 422
    )
    assert (
        http.post(
            base + "/ui-recording/" + "c" * 32 + "/complete", headers=headers
        ).status_code
        == 404
    )
