# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest
from frozen_rgb import bind_frozen_sources


def test_frozen_source_rejects_corruption_and_geometry_changes(tmp_path):
    raw = tmp_path / "input.rgb"
    raw.write_bytes(b"abc" * 4)
    entry = dict(
        path="input.rgb",
        width=2,
        height=2,
        frames=1,
        bytes=12,
        sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
    )
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({"case": entry}))
    case = dict(case_id="case", width=2, height=2, frames=1)
    assert bind_frozen_sources([case], manifest)[0]["source_rgb"] == str(raw)
    raw.write_bytes(b"xyz" * 4)
    with pytest.raises(ValueError, match="hash"):
        bind_frozen_sources([case], manifest)
    with pytest.raises(ValueError, match="geometry"):
        bind_frozen_sources([{**case, "width": 4}], manifest)
