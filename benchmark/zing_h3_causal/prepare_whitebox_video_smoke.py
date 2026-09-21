# SPDX-License-Identifier: Apache-2.0
"""Prepare the production-shaped whitebox video request for Zing H3 smoke tests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlsplit


DEFAULT_CASE = "zing-whitebox-video-five-triview-0001"
DEFAULT_MANIFEST = Path(__file__).with_name("whitebox_video_smoke_cases.jsonl")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Manifest line {line_number} is invalid JSON: {error.msg}"
                ) from error
            sample_id = case.get("sample_id")
            messages = case.get("messages")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Manifest line {line_number} has no valid sample_id")
            if sample_id in cases:
                raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
            if not isinstance(messages, list) or len(messages) != 2:
                raise ValueError(f"Case {sample_id} must contain two messages")
            cases[sample_id] = case
    if not cases:
        raise ValueError(f"Manifest contains no cases: {path}")
    return cases


def media_uris(case: dict[str, Any]) -> list[str]:
    messages = case["messages"]
    descriptors = [
        item
        for item in messages[0].get("content", [])
        if item.get("type") in {"image", "video", "audio", "video_audio"}
    ]
    source = messages[1].get("source")
    if isinstance(source, dict):
        descriptors.append(source)
    uris: list[str] = []
    for item in descriptors:
        uri = item.get("uri")
        if not isinstance(uri, str) or not uri:
            raise ValueError(f"Case {case['sample_id']} has an invalid media URI")
        if uri not in uris:
            uris.append(uri)
    return uris


def expected_assets(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    assets = case.get("provenance", {}).get("assets")
    if not isinstance(assets, list):
        raise ValueError(f"Case {case['sample_id']} has no asset provenance")
    expected: dict[str, dict[str, Any]] = {}
    for asset in assets:
        if not isinstance(asset, dict) or not isinstance(asset.get("uri"), str):
            raise ValueError(f"Case {case['sample_id']} has invalid asset provenance")
        expected[asset["uri"]] = asset
    actual_uris = set(media_uris(case))
    if set(expected) != actual_uris:
        raise ValueError(
            f"Case {case['sample_id']} asset provenance does not match request media"
        )
    return expected


def local_source_path(uri: str, media_root: Path) -> Path:
    parsed = urlsplit(uri)
    if parsed.scheme == "s3":
        return media_root / parsed.path.lstrip("/")
    if parsed.scheme == "file":
        return Path(parsed.path)
    if not parsed.scheme:
        return Path(uri)
    raise ValueError(
        f"--media-root cannot resolve {parsed.scheme} URI {uri!r}; "
        "omit --media-root to use fsspec"
    )


def open_media(uri: str, media_root: Path | None) -> BinaryIO:
    if media_root is not None:
        return local_source_path(uri, media_root).open("rb")
    parsed = urlsplit(uri)
    if parsed.scheme == "file":
        return Path(parsed.path).open("rb")
    if not parsed.scheme:
        return Path(uri).open("rb")
    import fsspec

    return fsspec.open(uri, "rb").open()


def materialize_case(
    case: dict[str, Any], output_dir: Path, *, media_root: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    replacements: dict[str, str] = {}
    assets = []
    expected = expected_assets(case)
    for index, uri in enumerate(media_uris(case)):
        basename = Path(urlsplit(uri).path).name or "media.bin"
        destination = media_dir / f"{index:02d}-{basename}"
        partial = destination.with_suffix(destination.suffix + ".partial")
        try:
            with open_media(uri, media_root) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target)
            if partial.stat().st_size == 0:
                raise ValueError(f"Downloaded media is empty: {uri}")
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        bytes_copied = destination.stat().st_size
        sha256 = sha256_file(destination)
        expected_asset = expected[uri]
        if bytes_copied != expected_asset.get("bytes") or sha256 != expected_asset.get(
            "sha256"
        ):
            destination.unlink(missing_ok=True)
            raise ValueError(f"Media checksum mismatch: {uri}")
        local_uri = destination.resolve().as_uri()
        replacements[uri] = local_uri
        assets.append(
            {
                "source_uri": uri,
                "local_uri": local_uri,
                "bytes": bytes_copied,
                "sha256": sha256,
            }
        )

    request = copy.deepcopy(case)

    def rewrite(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("uri") in replacements:
                value["uri"] = replacements[value["uri"]]
            for item in value.values():
                rewrite(item)
        elif isinstance(value, list):
            for item in value:
                rewrite(item)

    rewrite(request)
    receipt = {
        "sample_id": case["sample_id"],
        "source_manifest": case.get("provenance", {}),
        "assets": assets,
    }
    return request, receipt


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--case", default=DEFAULT_CASE)
    result.add_argument("--output-dir", type=Path)
    result.add_argument(
        "--media-root",
        type=Path,
        help=(
            "Read S3 object keys below this local root instead of downloading. "
            "The root corresponds to the bucket root."
        ),
    )
    result.add_argument(
        "--keep-remote-uris",
        action="store_true",
        help="Write request.json without downloading or rewriting media URIs.",
    )
    result.add_argument("--list", action="store_true", help="List frozen cases.")
    return result


def main() -> None:
    args = parser().parse_args()
    cases = load_cases(args.manifest)
    if args.list:
        for sample_id, case in cases.items():
            target = case["messages"][1]
            output = target["content"][0]["output"]
            print(
                sample_id,
                target["content"][0]["type"],
                f"{output['width']}x{output['height']}x{output['frames']}",
                f"references={len(target.get('conditions', []))}",
            )
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --list is used")
    try:
        case = cases[args.case]
    except KeyError as error:
        raise ValueError(f"Unknown case {args.case!r}") from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_remote_uris:
        request = copy.deepcopy(case)
        expected = expected_assets(case)
        receipt = {
            "sample_id": case["sample_id"],
            "source_manifest": case.get("provenance", {}),
            "assets": [
                {
                    "source_uri": uri,
                    "local_uri": None,
                    "bytes": expected[uri]["bytes"],
                    "sha256": expected[uri]["sha256"],
                }
                for uri in media_uris(case)
            ],
        }
    else:
        request, receipt = materialize_case(
            case, args.output_dir, media_root=args.media_root
        )
    write_json(args.output_dir / "request.json", request)
    write_json(args.output_dir / "receipt.json", receipt)
    print(args.output_dir / "request.json")


if __name__ == "__main__":
    main()
