# SPDX-License-Identifier: Apache-2.0
"""Read only file sizes and completion markers within one owned frozen root."""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if not a.root.is_absolute() or not a.root.name.startswith("pr693-frozen-"):
    p.error("Expected an absolute owned frozen experiment root")
rows = []
if a.root.is_dir():
    for path in sorted(a.root.iterdir()):
        stat = path.lstat()
        row = dict(
            name=path.name,
            bytes=stat.st_size,
            modified_ns=stat.st_mtime_ns,
            symlink=path.is_symlink(),
            directory=path.is_dir(),
        )
        if path.is_dir() and not path.is_symlink():
            row["children"] = sorted(x.name for x in path.iterdir())
        rows.append(row)
receipt = dict(root=str(a.root), exists=a.root.exists(), entries=rows)
with a.output.open("x") as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt), flush=True)
