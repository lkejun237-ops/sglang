# SPDX-License-Identifier: Apache-2.0
"""Manage a mounted shared Block0 disk cache without loading model weights."""

import argparse
import json
import sys

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
    Block0DiskCache,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="Show root usage and paginated metadata")
    listing.add_argument("--cache-key")
    listing.add_argument("--offset", type=int, default=0)
    listing.add_argument("--limit", type=int, default=100)
    for name in ("delete", "clear"):
        command = commands.add_parser(name)
        if name == "delete":
            command.add_argument("keys", nargs="+")
    args = parser.parse_args(argv)
    try:
        cache = Block0DiskCache(args.cache_dir)
        if args.command == "list":
            result = cache.list_entries(
                cache_key=args.cache_key,
                offset=args.offset,
                limit=args.limit,
            )
        else:
            result = cache.delete_entries(keys=getattr(args, "keys", None))
    except (OSError, ValueError) as exc:
        print(
            json.dumps({"code": "cache_management_failed", "message": str(exc)}),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return int(bool(result.get("failed")))


if __name__ == "__main__":
    raise SystemExit(main())
