"""Verify and update the hash-locked prebuilt binary seedleap-kernels wheel."""

from __future__ import annotations

import argparse
import ast
import email.parser
import hashlib
import ipaddress
import json
import os
import re
import ssl
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

DEFAULT_LATEST_URL = (
    "https://zing-cdn.loopit.com.cn/releases/seedleap-kernels/latest.json"
)
DEFAULT_LOCKFILE = Path(__file__).with_name("kernels-wheel.lock.txt")
LOCKFILE_HEADER = (
    "# Managed by deploy/whitebox/bump_kernels_wheel.py "
    "(prebuilt binary CUDA wheel only)."
)

MAX_METADATA_BYTES = 1 * 1024 * 1024
MAX_WHEEL_BYTES = 512 * 1024 * 1024
MAX_UNCOMPRESSED_WHEEL_BYTES = 1024 * 1024 * 1024
STREAM_CHUNK_BYTES = 1024 * 1024

DNS_HOSTNAME_PATTERN = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+\Z"
)
VERSION_PATTERN = re.compile(
    r"^[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?\Z"
)
BUILD_TAG_PATTERN = re.compile(r"^[0-9][0-9A-Za-z_]*\Z")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}\Z")
GIT_SHA_PATTERN = re.compile(r"^(?:unknown|[0-9a-f]{7,40})\Z")
SM90_EXT_PATTERN = re.compile(
    r"^seedleap_kernels/backends/cuda/_fp8_gemm_sm90_ext[^/]*\.so\Z"
)


def validate_dns_hostname(value: Any) -> str:
    """Require a public-style multi-label DNS hostname (reject localhost/IPs)."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 253
        or any(c.isspace() or ord(c) < 32 for c in value)
    ):
        raise ValueError(f"Invalid DNS hostname: {value!r}")
    host = value.lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError(f"Refusing localhost hostname: {value!r}")
    candidate_ip = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        ipaddress.ip_address(candidate_ip)
        is_ip = True
    except ValueError:
        is_ip = False
    if is_ip:
        raise ValueError(f"Refusing literal IP address in hostname: {value!r}")
    if DNS_HOSTNAME_PATTERN.fullmatch(host) is None:
        raise ValueError(
            f"Expected a multi-label DNS hostname without trailing dot, got: {value!r}"
        )
    return host


def dns_hostname_arg(value: str) -> str:
    """Argparse type validator for --allow-download-host."""
    try:
        return validate_dns_hostname(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def validate_https_url(value: str) -> str:
    """Require a credential-free HTTPS URL on default port 443 with a DNS host."""
    if not isinstance(value, str):
        raise ValueError("Use one credential-free HTTPS URL")
    try:
        url = urlsplit(value)
        port = url.port
        valid = (
            url.scheme == "https"
            and bool(url.hostname)
            and port in (None, 443)
            and url.username is None
            and url.password is None
            and not url.query
            and not url.fragment
            and "[" not in url.netloc
            and "]" not in url.netloc
            and not any(c.isspace() or ord(c) < 32 or c in ",|" for c in value)
        )
        if valid and url.hostname is not None:
            validate_dns_hostname(url.hostname)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            "Use one credential-free HTTPS DNS URL on port 443 (no IP literals, "
            "localhost, query, or fragment)"
        )
    return value


def mirror_url(value: str) -> str:
    """Argparse type validator mirroring release.py:mirror_url with SSRF guards."""
    try:
        return validate_https_url(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


class _SafeHTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Forbid HTTP downgrade, credentials, or unapproved cross-host redirects."""

    def __init__(self, allowed_hosts: set[str]) -> None:
        super().__init__()
        self._allowed_hosts = {validate_dns_hostname(h) for h in allowed_hosts}

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        resolved = urljoin(req.full_url, newurl)
        validate_https_url(resolved)
        target_host = validate_dns_hostname(urlsplit(resolved).hostname)
        if target_host not in self._allowed_hosts:
            raise ValueError(
                f"Refusing cross-host redirect to {target_host!r}; "
                f"allowed hosts: {sorted(self._allowed_hosts)}"
            )
        return super().redirect_request(req, fp, code, msg, headers, resolved)


def _build_ssl_context() -> ssl.SSLContext:
    verify_paths = ssl.get_default_verify_paths()
    if (
        verify_paths.cafile is None
        and verify_paths.capath is None
        and Path("/etc/ssl/cert.pem").is_file()
    ):
        return ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    return ssl.create_default_context()


def _open_https_stream(url: str, *, allowed_hosts: set[str], timeout: float):
    validate_https_url(url)
    host = validate_dns_hostname(urlsplit(url).hostname)
    if host not in allowed_hosts:
        raise ValueError(
            f"Download host {host!r} is not in allowed hosts {sorted(allowed_hosts)}"
        )
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=_build_ssl_context()),
        _SafeHTTPSRedirectHandler(allowed_hosts),
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "whitebox-bump-kernels-wheel/1.0"},
    )
    return opener.open(request, timeout=timeout)


def validate_version(version: Any) -> str:
    if (
        not isinstance(version, str)
        or not version
        or "+" in version
        or any(c.isspace() or ord(c) < 32 for c in version)
        or VERSION_PATTERN.fullmatch(version) is None
    ):
        raise ValueError(
            f"Invalid seedleap-kernels public PEP 440 version (no '+local' or "
            f"control characters allowed): {version!r}"
        )
    return version


def validate_wheel_filename(
    filename: Any, *, expected_version: str | None = None
) -> str:
    if (
        not isinstance(filename, str)
        or not filename.endswith(".whl")
        or "/" in filename
        or "\\" in filename
        or ".." in filename
        or any(c.isspace() or ord(c) < 32 for c in filename)
    ):
        raise ValueError(f"Invalid wheel filename: {filename!r}")
    parts = filename[:-4].split("-")
    plat_subtags = parts[-1].split(".") if parts else []
    if (
        filename.endswith("-any.whl")
        or "-none-any.whl" in filename
        or "any" in plat_subtags
    ):
        raise ValueError(
            f"Expected a prebuilt platform binary wheel, got pure-Python or 'any' "
            f"wheel: {filename!r}"
        )
    if len(parts) not in (5, 6) or parts[0] != "seedleap_kernels":
        raise ValueError(
            f"Expected a PEP 427 platform-tagged seedleap_kernels binary wheel "
            f"filename (5 or 6 '-' components), got: {filename!r}"
        )
    validate_version(parts[1])
    if len(parts) == 6 and BUILD_TAG_PATTERN.fullmatch(parts[2]) is None:
        raise ValueError(
            f"Invalid PEP 427 build tag {parts[2]!r} in wheel filename: {filename!r}"
        )
    if expected_version is not None and parts[1] != expected_version:
        raise ValueError(
            f"Wheel filename {filename!r} does not match version {expected_version!r}"
        )
    return filename


def validate_sha256(digest: Any) -> str:
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(
            f"Invalid SHA256 digest (expected 64 lowercase hex characters): {digest!r}"
        )
    return digest


def _normalize_git_sha(raw_git_sha: Any) -> str:
    if raw_git_sha is None:
        return "unknown"
    if not isinstance(raw_git_sha, str):
        raise ValueError(f"Invalid git_sha value: {raw_git_sha!r}")
    cleaned = raw_git_sha.strip().lower()
    if not cleaned:
        return "unknown"
    if GIT_SHA_PATTERN.fullmatch(cleaned) is None:
        raise ValueError(f"Invalid git_sha value: {raw_git_sha!r}")
    return cleaned


def _extract_build_info(source_text: str) -> dict[str, Any]:
    tree = ast.parse(source_text, filename="_build_info.py")
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            raise ValueError("_build_info.py must not contain import statements.")
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "BUILD_INFO":
                    parsed = ast.literal_eval(node.value)
                    if isinstance(parsed, dict):
                        return parsed
    raise ValueError("BUILD_INFO dictionary is missing from _build_info.py.")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(STREAM_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_bounded_bytes(stream, *, max_bytes: int, label: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(min(STREAM_CHUNK_BYTES, max_bytes - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"{label} exceeds maximum size of {max_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def inspect_binary_wheel_file(
    wheel_path: Path,
    *,
    expected_version: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, str]:
    """Verify wheel size, archive safety, SHA256, binary WHEEL tag, and SM90 .so."""
    if not wheel_path.is_file():
        raise ValueError(f"Wheel file does not exist: {wheel_path}")
    file_size = wheel_path.stat().st_size
    if file_size > MAX_WHEEL_BYTES:
        raise ValueError(
            f"Wheel file {wheel_path.name} ({file_size} bytes) exceeds maximum size "
            f"of {MAX_WHEEL_BYTES} bytes"
        )
    validate_wheel_filename(wheel_path.name, expected_version=expected_version)
    filename_version = wheel_path.name[:-4].split("-")[1]
    target_version = (
        expected_version if expected_version is not None else filename_version
    )

    actual_sha256 = _sha256_file(wheel_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Downloaded wheel SHA256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )

    expected_dist_info = f"seedleap_kernels-{target_version}.dist-info"
    expected_wheel_entry = f"{expected_dist_info}/WHEEL"
    expected_meta_entry = f"{expected_dist_info}/METADATA"

    with zipfile.ZipFile(wheel_path, "r") as zf:
        total_uncompressed = 0
        for info in zf.infolist():
            arcname = info.filename
            if (
                not arcname
                or arcname.startswith("/")
                or "\\" in arcname
                or ".." in arcname.split("/")
                or any(ord(c) < 32 or ord(c) == 127 for c in arcname)
            ):
                raise ValueError(
                    f"Unsafe archive member path in {wheel_path.name}: {arcname!r}"
                )
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(
                    f"Symlink archive member is forbidden in {wheel_path.name}: "
                    f"{arcname!r}"
                )
            if info.file_size < 0:
                raise ValueError(
                    f"Invalid negative member size in {wheel_path.name}: {arcname!r}"
                )
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_UNCOMPRESSED_WHEEL_BYTES:
                raise ValueError(
                    f"Wheel {wheel_path.name} cumulative uncompressed size "
                    f"({total_uncompressed} bytes) exceeds maximum "
                    f"{MAX_UNCOMPRESSED_WHEEL_BYTES} bytes"
                )

        names = zf.namelist()
        wheel_entries = [n for n in names if n.endswith(".dist-info/WHEEL")]
        meta_entries = [n for n in names if n.endswith(".dist-info/METADATA")]
        if wheel_entries != [expected_wheel_entry] or meta_entries != [
            expected_meta_entry
        ]:
            raise ValueError(
                f"Expected exact dist-info entries "
                f"'{expected_dist_info}/{{WHEEL,METADATA}}' in {wheel_path.name}, "
                f"found WHEEL={wheel_entries}, METADATA={meta_entries}"
            )

        with zf.open(expected_wheel_entry, "r") as wheel_fp:
            wheel_text = _read_bounded_bytes(
                wheel_fp, max_bytes=MAX_METADATA_BYTES, label=expected_wheel_entry
            ).decode("utf-8")
        has_non_purelib = any(
            line.strip().lower() == "root-is-purelib: false"
            for line in wheel_text.splitlines()
        )
        if not has_non_purelib:
            raise ValueError(
                f"Wheel metadata in {wheel_path.name} must declare "
                "'Root-Is-Purelib: false' (pure-Python wheels are forbidden)."
            )

        with zf.open(expected_meta_entry, "r") as meta_fp:
            meta_text = _read_bounded_bytes(
                meta_fp, max_bytes=MAX_METADATA_BYTES, label=expected_meta_entry
            ).decode("utf-8")
        meta_msg = email.parser.Parser().parsestr(meta_text)
        pkg_name = re.sub(r"[-_.]+", "-", (meta_msg.get("Name") or "")).strip().lower()
        if pkg_name != "seedleap-kernels":
            raise ValueError(
                f"Expected wheel package Name 'seedleap-kernels', got "
                f"{meta_msg.get('Name')!r}"
            )
        meta_version = validate_version(meta_msg.get("Version") or "")
        if meta_version != target_version:
            raise ValueError(
                f"Wheel METADATA Version {meta_version!r} does not match expected "
                f"version {target_version!r}"
            )

        so_entries = [
            info.filename
            for info in zf.infolist()
            if SM90_EXT_PATTERN.fullmatch(info.filename) and info.file_size > 0
        ]
        if not so_entries:
            raise ValueError(
                f"Wheel {wheel_path.name} is missing non-empty compiled extension "
                "'seedleap_kernels/backends/cuda/_fp8_gemm_sm90_ext*.so'."
            )

        build_info_entry = "seedleap_kernels/_build_info.py"
        if build_info_entry not in names:
            raise ValueError(
                f"Wheel {wheel_path.name} is missing '{build_info_entry}'."
            )
        with zf.open(build_info_entry, "r") as bi_fp:
            build_info_text = _read_bounded_bytes(
                bi_fp, max_bytes=MAX_METADATA_BYTES, label=build_info_entry
            ).decode("utf-8")
        build_info = _extract_build_info(build_info_text)
        if build_info.get("aot_cuda") is not True:
            raise ValueError(
                f"Wheel {wheel_path.name} BUILD_INFO must declare 'aot_cuda': True."
            )
        git_sha = _normalize_git_sha(build_info.get("git_sha"))

    return {
        "version": meta_version,
        "sha256": actual_sha256,
        "git_sha": git_sha,
    }


def _stream_download_to_path(stream, destination: Path, *, max_bytes: int) -> None:
    total = 0
    with destination.open("wb") as out_fh:
        while True:
            chunk = stream.read(STREAM_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(
                    f"Downloaded wheel exceeds maximum size of {max_bytes} bytes"
                )
            out_fh.write(chunk)


def _parse_existing_lockfile(lockfile_path: Path) -> tuple[str | None, str | None]:
    if not lockfile_path.is_file():
        return None, None

    logical_lines: list[str] = []
    current = ""
    for raw_line in lockfile_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not current and (not stripped or stripped.startswith("#")):
            continue
        if stripped.startswith("#"):
            continue
        if raw_line.endswith("\\"):
            current += raw_line[:-1] + " "
        else:
            current += raw_line
            if current.strip():
                logical_lines.append(current.strip())
            current = ""
    if current.strip():
        logical_lines.append(current.strip())

    if len(logical_lines) != 1:
        raise ValueError(
            f"Existing lockfile {lockfile_path} must contain exactly 1 requirement "
            f"entry, found {len(logical_lines)}."
        )
    tokens = logical_lines[0].split()
    req_match = re.fullmatch(r"seedleap-kernels==([0-9A-Za-z._-]+)\Z", tokens[0])
    if (
        req_match is None
        or len(tokens) != 2
        or not tokens[1].startswith("--hash=sha256:")
    ):
        raise ValueError(f"Malformed existing lockfile at {lockfile_path}")
    old_version = validate_version(req_match.group(1))
    old_sha256 = validate_sha256(tokens[1].split(":", 1)[1])
    return old_version, old_sha256


def _atomic_write_text(target_path: Path, content: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = target_path.stat().st_mode & 0o777 if target_path.exists() else None
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=str(target_path.parent),
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(content)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        if existing_mode is not None:
            os.chmod(tmp_path, existing_mode)
        else:
            current_umask = os.umask(0)
            os.umask(current_umask)
            os.chmod(tmp_path, 0o666 & ~current_umask)
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _resolve_wheel_url(
    latest_url: str,
    payload: dict[str, Any],
    filename: str,
    allowed_hosts: set[str],
) -> str:
    raw_url = (
        payload.get("url") or payload.get("wheel_url") or payload.get("download_url")
    )
    if raw_url is None:
        resolved = urljoin(latest_url, f"wheels/{filename}")
    elif (
        isinstance(raw_url, str)
        and "://" not in raw_url
        and not raw_url.startswith(("/", "\\"))
        and ".." not in raw_url
    ):
        resolved = urljoin(latest_url, raw_url)
    elif isinstance(raw_url, str):
        resolved = raw_url
    else:
        raise ValueError(f"Invalid wheel URL in latest.json: {raw_url!r}")

    validate_https_url(resolved)
    parsed = urlsplit(resolved)
    wheel_host = validate_dns_hostname(parsed.hostname)
    if wheel_host not in allowed_hosts:
        raise ValueError(
            f"Wheel download host {wheel_host!r} does not match latest.json host "
            f"or allowed hosts {sorted(allowed_hosts)}"
        )
    if Path(parsed.path).name != filename:
        raise ValueError(
            f"Wheel download URL basename {Path(parsed.path).name!r} does not match "
            f"filename {filename!r}"
        )
    return resolved


def bump_kernels_wheel(
    *,
    lockfile_path: Path = DEFAULT_LOCKFILE,
    latest_url: str = DEFAULT_LATEST_URL,
    wheel_file: Path | None = None,
    allowed_download_hosts: list[str] | None = None,
    timeout: float = 60.0,
) -> dict[str, object]:
    """Validate upstream binary wheel metadata/bytes and update lockfile atomically."""
    lockfile = Path(lockfile_path)
    old_version, _ = _parse_existing_lockfile(lockfile)

    if wheel_file is not None:
        inspected = inspect_binary_wheel_file(Path(wheel_file))
        new_version = inspected["version"]
        sha256 = inspected["sha256"]
        git_sha = inspected["git_sha"]
    else:
        validate_https_url(latest_url)
        origin_host = validate_dns_hostname(urlsplit(latest_url).hostname)
        allowed_hosts = {origin_host}
        for extra_host in allowed_download_hosts or []:
            allowed_hosts.add(validate_dns_hostname(extra_host))

        with _open_https_stream(
            latest_url, allowed_hosts=allowed_hosts, timeout=timeout
        ) as meta_stream:
            raw_meta = _read_bounded_bytes(
                meta_stream, max_bytes=MAX_METADATA_BYTES, label="latest.json"
            )
        try:
            payload = json.loads(raw_meta.decode("utf-8"))
        except Exception as exc:
            raise ValueError(
                f"Failed to parse latest.json from {latest_url}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("latest.json payload must be a JSON object")

        new_version = validate_version(payload.get("version"))
        entry: dict[str, Any] = payload
        if (
            payload.get("filename") is None
            and payload.get("wheel") is None
            and isinstance(payload.get("wheels"), list)
            and len(payload["wheels"]) == 1
            and isinstance(payload["wheels"][0], dict)
        ):
            entry = payload["wheels"][0]

        raw_filename = entry.get("filename") or entry.get("wheel")
        filename = validate_wheel_filename(raw_filename, expected_version=new_version)
        sha256 = validate_sha256(entry.get("sha256"))
        raw_git_sha = payload.get("git_sha")
        if raw_git_sha is None and isinstance(payload.get("build_info"), dict):
            raw_git_sha = payload["build_info"].get("git_sha")
        git_sha = _normalize_git_sha(raw_git_sha)

        wheel_url = _resolve_wheel_url(latest_url, entry, filename, allowed_hosts)
        with tempfile.TemporaryDirectory(prefix="whitebox-kernels-whl-") as tmpdir:
            staged_whl = Path(tmpdir) / filename
            with _open_https_stream(
                wheel_url, allowed_hosts=allowed_hosts, timeout=timeout
            ) as whl_stream:
                _stream_download_to_path(
                    whl_stream, staged_whl, max_bytes=MAX_WHEEL_BYTES
                )
            inspected = inspect_binary_wheel_file(
                staged_whl,
                expected_version=new_version,
                expected_sha256=sha256,
            )
        if git_sha == "unknown" and inspected["git_sha"] != "unknown":
            git_sha = inspected["git_sha"]
        elif git_sha != "unknown" and inspected["git_sha"] not in {"unknown", git_sha}:
            raise ValueError(
                f"Wheel BUILD_INFO git_sha {inspected['git_sha']!r} does not "
                f"match latest.json git_sha {git_sha!r}"
            )

    rendered = (
        f"{LOCKFILE_HEADER}\n"
        f"seedleap-kernels=={new_version} \\\n"
        f"    --hash=sha256:{sha256}\n"
    )
    if lockfile.is_file() and lockfile.read_text(encoding="utf-8") == rendered:
        changed = False
    else:
        _atomic_write_text(lockfile, rendered)
        changed = True

    return {
        "changed": changed,
        "git_sha": git_sha,
        "new_version": new_version,
        "old_version": old_version,
        "sha256": sha256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--latest-url",
        type=mirror_url,
        default=DEFAULT_LATEST_URL,
        help="Credential-free HTTPS URL to upstream latest.json metadata.",
    )
    source_group.add_argument(
        "--wheel-file",
        type=Path,
        default=None,
        help="Local prebuilt binary .whl file path instead of remote latest.json.",
    )
    parser.add_argument(
        "--lockfile",
        type=Path,
        default=DEFAULT_LOCKFILE,
        help="Path to deploy/whitebox/kernels-wheel.lock.txt.",
    )
    parser.add_argument(
        "--allow-download-host",
        dest="allowed_download_hosts",
        action="append",
        type=dns_hostname_arg,
        default=[],
        help="Additional HTTPS DNS hostname permitted for the binary wheel download URL.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTPS request timeout in seconds.",
    )
    args = parser.parse_args(argv)
    summary = bump_kernels_wheel(
        lockfile_path=args.lockfile,
        latest_url=args.latest_url,
        wheel_file=args.wheel_file,
        allowed_download_hosts=args.allowed_download_hosts,
        timeout=args.timeout,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
