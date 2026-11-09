#!/usr/bin/env python3
"""Project build inputs into a runtime tree without mutating the checkout.

Only multimodal_gen/test has been reviewed as removable. Dynamic plugins,
runtime resources and other test/demo-named paths remain runtime inputs.
"""

import argparse
import os
import shutil
import stat
import tempfile
from pathlib import Path

EXCLUDED = Path("multimodal_gen/test")


def project_runtime_source(source: Path, destination: Path) -> None:
    """Copy deterministic bytes/modes/links; publish only a complete safe tree."""
    source = source.resolve()
    destination = destination.absolute()
    if not source.is_dir():
        raise ValueError("Source must be an existing directory")
    if stat.S_IMODE(source.stat().st_mode) & 0o005 != 0o005:
        raise ValueError("Runtime source root is not readable by its runtime user")
    # A symlink at any destination component could redirect writes into a
    # checkout or another build. Reject it before creating any output.
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("Destination must not contain a symlink")
    destination = destination.resolve()
    if (
        destination == source
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("Source and destination must not overlap")
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ValueError("Destination must be absent or an empty directory")

    entries = []

    def collect(directory):
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(source)
            if relative == EXCLUDED:
                continue
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                try:
                    target = path.resolve(strict=True)
                except (OSError, RuntimeError) as error:
                    raise ValueError(
                        f"Symlink target cannot be resolved: {relative}"
                    ) from error
                if os.path.isabs(os.readlink(path)) or not target.is_relative_to(
                    source
                ):
                    raise ValueError(
                        f"Symlink must be relative and stay inside source: {relative}"
                    )
            elif not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError(f"Unsupported source entry: {relative}")
            if stat.S_ISDIR(mode) and stat.S_IMODE(mode) & 0o005 != 0o005:
                raise ValueError(
                    f"Runtime source directory is not readable by its runtime user: {relative}"
                )
            entries.append((relative, mode))
            if stat.S_ISDIR(mode):
                collect(path)

    collect(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Staging also lets us resolve links against the actual projected tree:
    # links into excluded content (including indirect chains) must not become
    # dangling runtime links. A rejected projection leaves the destination alone.
    with tempfile.TemporaryDirectory(
        prefix=".runtime-source-", dir=destination.parent
    ) as temporary:
        output = Path(temporary) / "sglang"
        output.mkdir()
        for relative, mode in entries:
            original, projected = source / relative, output / relative
            if stat.S_ISLNK(mode):
                projected.symlink_to(os.readlink(original))
            elif stat.S_ISDIR(mode):
                projected.mkdir()
            else:
                shutil.copyfile(original, projected)
                projected.chmod(stat.S_IMODE(mode))
        for relative, mode in entries:
            projected = output / relative
            if stat.S_ISLNK(mode):
                try:
                    target = projected.resolve(strict=True)
                except (OSError, RuntimeError) as error:
                    raise ValueError(
                        f"Symlink target is absent from runtime projection: {relative}"
                    ) from error
                if not target.is_relative_to(output):
                    raise ValueError(f"Symlink escapes runtime projection: {relative}")
        # Apply directory modes after copying children. Normalize all timestamps,
        # including directories and links, after the final child has been written.
        for relative, mode in reversed(entries):
            projected = output / relative
            if stat.S_ISDIR(mode):
                projected.chmod(stat.S_IMODE(mode))
            os.utime(projected, ns=(0, 0), follow_symlinks=False)
        os.replace(output, destination)
        # macOS requires a writable source directory for rename. Preserve a
        # read-only root's mode only after publishing the completed tree.
        destination.chmod(stat.S_IMODE(source.stat().st_mode))
        os.utime(destination, ns=(0, 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    try:
        project_runtime_source(args.source, args.destination)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
