"""Use the fast Bookworm main mirror only when its signed Release matches Debian.

Build-only wrapper: updates/security and package authentication stay with APT.
The original command runs once; source bytes are restored even when it fails.
"""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ORIGIN_RELEASE = "https://deb.debian.org/debian/dists/bookworm/InRelease"
MIRROR_RELEASE = "https://mirrors.aliyun.com/debian/dists/bookworm/InRelease"
MIRROR_URI = "https://mirrors.aliyun.com/debian"
FETCH_SECONDS = 15
SIGNATURE_SECONDS = 5
MAX_RELEASE_BYTES = 1024 * 1024
SOURCES = Path("/etc/apt/sources.list.d/debian.sources")


def fetch_release(url):
    # A separate process gives the whole response a deadline, even if a peer
    # trickles bytes often enough to keep a socket timeout from expiring.
    result = subprocess.run(
        [sys.executable, __file__, "--fetch", url],
        capture_output=True,
        check=True,
        timeout=FETCH_SECONDS,
    )
    return result.stdout


def verify_release(data, keyring):
    gpgv = shutil.which("gpgv")
    if not gpgv or not Path(keyring).is_file():
        return False
    with tempfile.TemporaryDirectory(prefix="whitebox-release-signature-") as home:
        result = subprocess.run(
            [gpgv, "--homedir", home, "--keyring", keyring, "-"],
            input=data,
            capture_output=True,
            timeout=SIGNATURE_SECONDS,
        )
    return result.returncode == 0


def source_candidate(raw):
    blocks = raw.decode().strip().split("\n\n")
    matches = []
    for index, block in enumerate(blocks):
        fields = dict(
            line.split(": ", 1)
            for line in block.splitlines()
            if line and not line.startswith("#") and ": " in line
        )
        if (
            fields.get("URIs")
            in {"http://deb.debian.org/debian", "https://deb.debian.org/debian"}
            and "bookworm" in fields.get("Suites", "").split()
            and fields.get("Types") in {"deb", "deb deb-src"}
            and fields.get("Signed-By", "").startswith("/")
            and len(fields["Signed-By"].split()) == 1
        ):
            matches.append((index, block, fields))
    if len(matches) != 1:
        return None
    index, block, fields = matches[0]
    suites = fields["Suites"].split()
    main = block.replace("URIs: " + fields["URIs"], "URIs: " + MIRROR_URI).replace(
        "Suites: " + fields["Suites"], "Suites: bookworm"
    )
    rest = [suite for suite in suites if suite != "bookworm"]
    replacement = [main]
    if rest:
        replacement.append(
            block.replace("Suites: " + fields["Suites"], "Suites: " + " ".join(rest))
        )
    blocks[index : index + 1] = replacement
    return ("\n\n".join(blocks) + "\n").encode(), fields["Signed-By"]


def official_https(raw):
    # Only Debian's two known origin paths; leave snapshots/custom repositories
    # and every field other than the URI scheme untouched.
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.rstrip(b"\r\n") in {
            b"URIs: http://deb.debian.org/debian",
            b"URIs: http://deb.debian.org/debian-security",
        }:
            lines[index] = line.replace(b"http://", b"https://", 1)
    return b"".join(lines)


def select_sources(raw, *, fetch=fetch_release, verify=verify_release):
    started = time.monotonic()
    receipt = {
        "event": "whitebox.apt-source",
        "selected": "official",
        "reason": "unsupported-source-layout",
    }
    official = official_https(raw)
    candidate = source_candidate(official)
    if candidate is None:
        return official, receipt

    def measured(url):
        begin = time.monotonic()
        try:
            data = fetch(url)
            if not 0 < len(data) <= MAX_RELEASE_BYTES:
                raise ValueError("Release size")
            return data, {
                "seconds": round(time.monotonic() - begin, 3),
                "bytes": len(data),
            }
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            return None, {
                "seconds": round(time.monotonic() - begin, 3),
                "errorType": type(error).__name__,
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        origin, mirror = list(pool.map(measured, [ORIGIN_RELEASE, MIRROR_RELEASE]))
    receipt["fetches"] = {"official": origin[1], "mirror": mirror[1]}
    selected = official
    if origin[0] is None or mirror[0] is None:
        receipt["reason"] = "release-download-failed"
    elif origin[0] != mirror[0]:
        receipt["reason"] = "release-mismatch"
    elif b"\nCodename: bookworm\n" not in origin[0]:
        receipt["reason"] = "release-codename-mismatch"
    else:
        try:
            valid = verify(origin[0], candidate[1])
        except (OSError, subprocess.SubprocessError):
            valid = False
        if valid:
            selected = candidate[0]
            receipt.update(
                selected="aliyun-main",
                reason="identical-signed-release",
                releaseSha256=hashlib.sha256(origin[0]).hexdigest(),
            )
        else:
            receipt["reason"] = "release-signature-unavailable-or-invalid"
    receipt["seconds"] = round(time.monotonic() - started, 3)
    return selected, receipt


def execute(command):
    return subprocess.run(command, check=False).returncode


def run(
    command,
    source_path=SOURCES,
    *,
    fetch=fetch_release,
    verify=verify_release,
    execute=execute,
):
    original = source_path.read_bytes() if source_path.is_file() else None
    if original is None:
        selected, receipt = (
            None,
            {
                "event": "whitebox.apt-source",
                "selected": "official",
                "reason": "source-file-absent",
            },
        )
    else:
        selected, receipt = select_sources(original, fetch=fetch, verify=verify)
    print(json.dumps(receipt), flush=True)
    started = time.monotonic()
    changed = selected != original
    try:
        if changed:
            source_path.write_bytes(selected)
        return execute(command)
    finally:
        if changed:
            source_path.write_bytes(original)
        print(
            json.dumps(
                {
                    "event": "whitebox.apt-command-finished",
                    "seconds": round(time.monotonic() - started, 3),
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch", choices=[ORIGIN_RELEASE, MIRROR_RELEASE], help=argparse.SUPPRESS
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.fetch:
        with urllib.request.urlopen(args.fetch, timeout=FETCH_SECONDS) as response:
            data = response.read(MAX_RELEASE_BYTES + 1)
        if not 0 < len(data) <= MAX_RELEASE_BYTES:
            raise ValueError("Release exceeds byte limit")
        sys.stdout.buffer.write(data)
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("an original build command is required after --")
    result = run(command)
    return result if result >= 0 else 128 - result


if __name__ == "__main__":
    sys.exit(main())
