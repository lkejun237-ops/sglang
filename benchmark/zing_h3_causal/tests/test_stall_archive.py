# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from archive_stall_artifacts import archive_file


@pytest.mark.parametrize("corrupt_head", [False, True])
def test_raw_file_is_removed_only_after_verified_upload_and_readback(
    tmp_path, corrupt_head
):
    objects = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_PUT(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            objects[self.path] = data
            self.send_response(200)
            self.send_header("ETag", hashlib.md5(data).hexdigest())
            self.end_headers()

        def do_HEAD(self):
            data = objects[self.path]
            self.send_response(200)
            self.send_header("Content-Length", len(data) + int(corrupt_head))
            self.send_header("ETag", hashlib.md5(data).hexdigest())
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        raw = tmp_path / "output.rgb"
        data = b"complete-raw-rgb-receipt"
        raw.write_bytes(data)
        slots = []
        for i in range(10):
            url = f"http://127.0.0.1:{server.server_port}/{i}"
            slots.append(dict(key=str(i), put_url=url, head_url=url))
        delivery = tmp_path / "delivery.private.json"
        delivery.write_text(json.dumps(dict(slots=slots)))
        if corrupt_head:
            with pytest.raises(RuntimeError, match="readback"):
                archive_file(raw, delivery, remove_verified=True, part_bytes=7)
            assert raw.read_bytes() == data
            assert not raw.with_name(raw.name + ".archive.json").exists()
        else:
            receipt = archive_file(raw, delivery, remove_verified=True, part_bytes=7)
            assert not raw.exists()
            assert b"".join(objects["/" + p["key"]] for p in receipt["parts"]) == data
            assert receipt["sha256"] == hashlib.sha256(data).hexdigest()
            assert "url" not in json.dumps(receipt)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_mounted_store_preserves_parts_and_detects_corruption(tmp_path):
    from archive_stall_artifacts import archive_to_directory

    store = tmp_path / "store"
    raw = tmp_path / "raw.rgb"
    data = b"full-raw-frames"
    raw.write_bytes(data)
    receipt = archive_to_directory(raw, store, remove_verified=True, part_bytes=5)
    assert b"".join(Path(p["path"]).read_bytes() for p in receipt["parts"]) == data
    assert receipt["verified"] and not raw.exists()
    raw = tmp_path / "retry.rgb"
    raw.write_bytes(data)
    Path(receipt["parts"][0]["path"]).write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="readback"):
        archive_to_directory(raw, store, remove_verified=True, part_bytes=5)
    assert raw.read_bytes() == data


def test_interrupted_mounted_upload_cannot_poison_a_retry(tmp_path, monkeypatch):
    from archive_stall_artifacts import archive_to_directory

    raw = tmp_path / "raw.rgb"
    data = b"full-raw-frames"
    raw.write_bytes(data)
    store = tmp_path / "store"
    original_open = Path.open

    class InterruptedWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, block):
            self.stream.write(block[:2])
            raise OSError("Injected upload interruption")

    def interrupt(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if mode == "xb" and path.suffix == ".partial":
            return InterruptedWriter(stream)
        return stream

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", interrupt)
        with pytest.raises(OSError, match="upload interruption"):
            archive_to_directory(raw, store, remove_verified=True, part_bytes=5)
    assert raw.read_bytes() == data
    assert not raw.with_name(raw.name + ".archive.json").exists()
    assert all(p.suffix == ".partial" for p in store.iterdir())
    receipt = archive_to_directory(raw, store, remove_verified=True, part_bytes=5)
    assert receipt["sha256"] == hashlib.sha256(data).hexdigest()
    assert b"".join(Path(p["path"]).read_bytes() for p in receipt["parts"]) == data
    assert not raw.exists()
