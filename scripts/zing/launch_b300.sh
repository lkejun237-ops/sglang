#!/usr/bin/env bash
# Single-node B300 E1/Qwen + SP2 + D1 recipe; see deployment_zing_h3.mdx.
set -euo pipefail
export SGLANG_LIGHTWEIGHT_RUNTIME=1
exec "${PYTHON:-python3}" -m sglang.multimodal_gen.tools.zing_b300_launcher "$@"
