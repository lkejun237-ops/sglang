# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from compare_encoder_overlap_capture import compare
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.component_capture import (
    ComponentCapture,
)


def capture(root, packet_size=1):
    stage = SimpleNamespace(
        runtime={"steps": 4},
        config=SimpleNamespace(dit_config=SimpleNamespace(arch_config={"heads": 1})),
    )
    writer = ComponentCapture(
        root, stage, {"references": list(range(6))}, {"fps": 24}, 42, "session"
    )
    frames = torch.arange(24, dtype=torch.uint8).reshape(3, 2, 2, 2)
    for start in range(0, 2, packet_size):
        writer.append_frames("input", frames[:, start : start + packet_size], start)
        writer.append_frames("decoder", frames[:, start : start + packet_size], start)
    source = torch.ones((1, 2, 1, 1, 1))
    encoded = SimpleNamespace(start=0, end=1, source_dependency_end=2, values=source)
    writer.append_source(encoded, source)
    chunk = SimpleNamespace(
        chunk=SimpleNamespace(start=0, end=1),
        source=source,
        noise=source,
        provenance=SimpleNamespace(source_dependency_end=2),
        final=True,
    )
    writer.append(chunk, SimpleNamespace(visual=source))
    writer.finish()
    return root / "session"


def test_equal_capture_independent_of_packet_boundaries(tmp_path):
    a, b = capture(tmp_path / "a"), capture(tmp_path / "b", packet_size=2)
    assert compare(a, b)["status"] == "passed"


@pytest.mark.parametrize(
    "name",
    ["COMPLETE.json", "frames.json", "session.pt", "source-00000.pt", "chunk-00000.pt"],
)
def test_missing_evidence_cannot_pass(tmp_path, name):
    a, b = capture(tmp_path / "a"), capture(tmp_path / "b")
    (b / name).unlink()
    assert compare(a, b)["status"] == "failed"


@pytest.mark.parametrize(
    "name,key",
    [
        ("source-00000.pt", "moments"),
        ("source-00000.pt", "latent"),
        ("chunk-00000.pt", "expected_visual"),
    ],
)
def test_zero_tolerance_and_nonfinite(tmp_path, name, key):
    a, b = capture(tmp_path / "a"), capture(tmp_path / "b")
    path = b / name
    value = torch.load(path, weights_only=False)
    value[key] = value[key].clone()
    value[key].flatten()[0] += 1e-6
    torch.save(value, path)
    assert compare(a, b)["status"] == "failed"
    value[key].flatten()[0] = float("nan")
    torch.save(value, path)
    assert compare(a, b)["status"] == "failed"


def test_noise_and_watermark_are_required(tmp_path):
    a, b = capture(tmp_path / "a"), capture(tmp_path / "b")
    path = b / "chunk-00000.pt"
    row = torch.load(path, weights_only=False)
    row["chunk"].noise = row["chunk"].noise + 1
    torch.save(row, path)
    assert compare(a, b)["status"] == "failed"
    row["chunk"].noise -= 1
    row["chunk"].provenance.source_dependency_end = 3
    torch.save(row, path)
    assert compare(a, b)["status"] == "failed"


def test_raw_rgb_requires_both_and_exact_bytes(tmp_path):
    a, b = capture(tmp_path / "a"), capture(tmp_path / "b")
    x, y = tmp_path / "a.rgb", tmp_path / "b.rgb"
    x.write_bytes(b"RGB" * 8)
    y.write_bytes(b"RGB" * 8)
    assert compare(a, b, x, y)["status"] == "passed"
    assert compare(a, b, x)["status"] == "failed"
    y.write_bytes(b"RgB" * 8)
    assert compare(a, b, x, y)["status"] == "failed"


def test_only_run_identity_is_excluded(tmp_path):
    a, b = capture(tmp_path / "a"), capture(tmp_path / "b")
    for root, request in ((a, "a"), (b, "b")):
        path = root / "session.pt"
        row = torch.load(path, weights_only=False)
        row["params"]["request_id"] = request
        torch.save(row, path)
    result = compare(a, b)
    assert result["status"] == "passed"
    assert result["excluded_run_metadata"]["candidate"]["request_id"] == "b"
    path = b / "session.pt"
    row["params"]["fps"] = 25
    torch.save(row, path)
    assert compare(a, b)["status"] == "failed"


def test_equal_truncated_raw_media_cannot_pass(tmp_path):
    a, b = capture(tmp_path / "a"), capture(tmp_path / "b")
    x, y = tmp_path / "a.rgb", tmp_path / "b.rgb"
    x.write_bytes(b"RGB")
    y.write_bytes(b"RGB")
    assert compare(a, b, x, y)["status"] == "failed"


def test_capture_time_media_hashes_survive_cleanup(tmp_path):
    import json

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.component_capture import (
        media_asset_manifest,
    )

    roots = [capture(tmp_path / name) for name in ("a", "b")]
    for root in roots:
        files = [root / "media-0.png", root / "media-1.png"]
        for i, path in enumerate(files):
            path.write_bytes(bytes([i + 1]) * 10)
        value = torch.load(root / "session.pt", weights_only=False)
        value["params"].update(
            image_path=list(map(str, files)),
            zing_request=dict(
                messages=[
                    dict(
                        content=[
                            dict(type="image", id=str(i), uri=str(path))
                            for i, path in enumerate(files)
                        ]
                    )
                ]
            ),
        )
        (root / "media-assets.json").write_text(
            json.dumps(media_asset_manifest(value["params"]))
        )
        torch.save(value, root / "session.pt")
        for path in files:
            path.unlink()
    result = compare(*roots)
    assert result["status"] == "passed"
    assert len(result["normalized_media_paths"]) == 4
    manifest_path = roots[1] / "media-assets.json"
    good = manifest_path.read_text()
    value = json.loads(good)
    value["assets"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(value))
    assert compare(*roots)["status"] == "failed"
    manifest_path.write_text(good)
    session = torch.load(roots[1] / "session.pt", weights_only=False)
    session["params"]["image_path"].reverse()
    torch.save(session, roots[1] / "session.pt")
    assert "binding differs" in compare(*roots)["error"]


def test_old_media_capture_cannot_pass_without_identity(tmp_path):
    roots = [capture(tmp_path / name) for name in ("a", "b")]
    for root in roots:
        session = torch.load(root / "session.pt", weights_only=False)
        session["params"]["image_path"] = [str(root / "deleted-reference.png")]
        torch.save(session, root / "session.pt")
    result = compare(*roots)
    assert result["status"] == "failed"
    assert "recapture required" in result["error"]
