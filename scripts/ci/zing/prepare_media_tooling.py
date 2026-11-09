"""Prepare signed Ubuntu media packages on disposable CI runners."""

import re
import subprocess
import sys
from pathlib import Path

APT_OPTIONS = [
    "-o",
    "Acquire::Retries=1",
    "-o",
    "Acquire::http::Timeout=20",
    "-o",
    "Acquire::https::Timeout=20",
]


def use_official_mirror(root):
    changed = 0
    paths = {root / "sources.list"}
    for pattern in ("*.sources", "*.list"):
        paths.update((root / "sources.list.d").glob(pattern))
    for path in sorted(paths):
        if not path.is_file():
            continue
        original = path.read_text()
        updated = re.sub(
            r"https?://azure\.archive\.ubuntu\.com/ubuntu(?=/|\s|$)",
            "https://archive.ubuntu.com/ubuntu",
            original,
        )
        # Hosted Ubuntu images can select Azure indirectly through mirror+file.
        # Replace only the two known Ubuntu mirror lists, not arbitrary local URIs.
        for suffix, host in (("", "archive"), ("-security", "security")):
            updated = re.sub(
                rf"mirror\+file:(?://)?/etc/apt/apt-mirrors{suffix}\.txt(?=\s|$)",
                f"https://{host}.ubuntu.com/ubuntu",
                updated,
            )
        if updated != original:
            path.write_text(updated)
            changed += 1
    return changed


def media_tools_ready(run):
    try:
        run(["ffprobe", "-version"], check=True, capture_output=True, timeout=20)
        result = run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return bool(re.search(r"\blibx264\b", result.stdout))
    except (OSError, subprocess.SubprocessError):
        return False


def prepare(root=Path("/etc/apt"), run=subprocess.run):
    if not media_tools_ready(run):
        # Only replace the hosted runner mirror URI. Suites, components,
        # Signed-By keyrings and APT's package/signature validation stay intact.
        use_official_mirror(root)
        run(["apt-get", *APT_OPTIONS, "update"], check=True, timeout=180)
        run(
            ["apt-get", *APT_OPTIONS, "install", "--yes", "ffmpeg"],
            check=True,
            timeout=180,
        )
        if not media_tools_ready(run):
            raise RuntimeError("FFmpeg with libx264 and ffprobe are required")
    run(["ffmpeg", "-version"], check=True, timeout=20)


if __name__ == "__main__":
    if sys.platform != "linux":
        raise SystemExit("Media package preparation is only for Linux CI runners")
    prepare()
