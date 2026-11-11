"""DiT entrypoint for an already allocated and partitioned E1+SP2+D1 job."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PYTHON_ROOT = str(Path(__file__).resolve().parents[2] / "python")
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)


def main():
    os.environ.setdefault("SGLANG_LIGHTWEIGHT_RUNTIME", "1")
    from sglang.multimodal_gen.tools.zing_h3_profile_launcher import fixed_roles_main

    fixed_roles_main(sys.argv[1:])


if __name__ == "__main__":
    main()
