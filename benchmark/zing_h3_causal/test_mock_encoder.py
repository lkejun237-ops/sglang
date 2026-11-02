# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest
import torch
from mock_encoder import read_encoder_capture


def capture(root):
    def tensor(name, shape, dtype):
        value = (
            torch.arange(shape[0] * shape[1], dtype=torch.float32)
            .reshape(shape)
            .to(dtype)
        )
        raw = value.contiguous().view(torch.uint8).numpy().tobytes()
        (root / name).write_bytes(raw)
        return dict(
            file=name,
            bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            shape=list(shape),
            dtype=str(dtype).split(".")[-1],
        )

    context = tensor("context.bin", (3, 4), torch.bfloat16)
    roles = tensor("roles.bin", (3, 1), torch.int64)
    roles["shape"] = [3]
    source = tensor("output.moments", (2, 8), torch.float32)
    source.update(
        shape=[1, 2, 2, 2, 2],
        byte_offset=0,
        latent_start=0,
        latent_end=2,
        source_dependency_end=5,
    )
    receipt = dict(
        source_frames=5,
        latent_frames=2,
        chunks=[source],
        conditions=[
            dict(event_id=None, tensors=dict(context=context, row_role=roles)),
            dict(
                event_id=1,
                prompt_version=2,
                tensors=dict(context=context, row_role=roles),
            ),
        ],
    )
    (root / "summary.json").write_text(json.dumps(receipt))


@pytest.mark.parametrize("sink_only", [False, True])
def test_mock_reads_saved_bytes_with_explicit_prompt_order(tmp_path, sink_only):
    capture(tmp_path)
    if sink_only:
        receipt = json.loads((tmp_path / "summary.json").read_text())
        del receipt["source_frames"]
        (tmp_path / "summary.json").write_text(json.dumps(receipt))
    messages = list(
        read_encoder_capture(
            tmp_path, session_id="s", generation_id="g", prompt_at_latents=[2]
        )
    )
    assert [m["type"] for m in messages] == [
        "conditioning",
        "source.moments",
        "conditioning",
        "source.end",
    ]
    assert all(m["session_id"] == "s" and m["generation_id"] == "g" for m in messages)
    assert messages[1]["tensor"].shape == (1, 2, 2, 2, 2)
    assert messages[1]["pixel_end"] == 5
    assert messages[2]["event_id"] == 1
    assert messages[3]["latent_frames"] == 2


def test_mock_rejects_missing_prompt_order_and_corrupted_payload(tmp_path):
    capture(tmp_path)
    with pytest.raises(ValueError, match="boundaries"):
        list(read_encoder_capture(tmp_path, session_id="s", generation_id="g"))
    (tmp_path / "output.moments").write_bytes(bytes(64))
    with pytest.raises(ValueError, match="hash"):
        list(
            read_encoder_capture(
                tmp_path, session_id="s", generation_id="g", prompt_at_latents=[0]
            )
        )
