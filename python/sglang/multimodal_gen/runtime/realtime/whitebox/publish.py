# SPDX-License-Identifier: Apache-2.0
"""Explicitly retry finalized local recording uploads; no automatic cloud job."""

import argparse
import asyncio
import os
from pathlib import Path

from .recording import S3ArchiveStore, publish_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    store = S3ArchiveStore(
        os.environ["WHITEBOX_OSS_BUCKET"],
        os.environ["WHITEBOX_OSS_ENDPOINT"],
        os.environ.get("WHITEBOX_OSS_REGION", "cn-beijing"),
    )
    asyncio.run(publish_archive(args.directory, store))


if __name__ == "__main__":
    main()
