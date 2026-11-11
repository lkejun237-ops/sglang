# SPDX-License-Identifier: Apache-2.0
"""Strict CPU JSON, deployment identity and safetensors contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
import struct
from pathlib import Path
from typing import Any, Mapping

import torch

SCHEMA_VERSION = 1
TENSOR_CONTRACT = "zing-h3-context-v1"
MAX_MANIFEST_BYTES = 65536
MAX_HEADER_BYTES = 65536
_HEX = re.compile(r"^[0-9a-f]{64}$")
_PUBLIC = {
    "CONDITION_TASK_EXPIRED": "Please try preparing this request again.",
    "CONDITION_PROMPT_TOO_LONG": "The prompt is too long. Please shorten it.",
    "CONDITION_REFERENCE_LIMIT": "There are too many reference images. Please use fewer images.",
    "CONDITION_BUDGET_EXCEEDED": "The estimated input size exceeds the limit. Please shorten the prompt or use fewer reference images.",
    "CONDITION_INVALID": "We could not prepare this request. Please try again.",
    "CONDITION_UNAUTHORIZED": "We could not prepare this request. Please try again.",
    "CONDITION_PROFILE_MISMATCH": "We could not prepare this request. Please try again.",
    "CONDITION_BUSY": "The service is busy. Please try again shortly.",
    "CONDITION_TIMEOUT": "Preparing your request took too long. Please try again.",
    "CONDITION_UNAVAILABLE": "The service is unavailable. Please try again shortly.",
}


class ConditionError(ValueError):
    """Internal details must be logged, never used as browser error content."""

    def __init__(
        self,
        code: str = "CONDITION_INVALID",
        detail: str | int = "",
        status: int | None = None,
    ):
        if code not in _PUBLIC:
            raise ValueError("Unknown conditioning error code")
        if type(detail) is int:
            if status is not None:
                raise ValueError("Conditioning HTTP status was supplied twice")
            status, detail = detail, ""
        if type(detail) is not str or any(ord(c) > 127 for c in detail):
            raise ValueError("Conditioning error details must be English ASCII")
        defaults = {
            "CONDITION_TASK_EXPIRED": 410,
            "CONDITION_UNAUTHORIZED": 401,
            "CONDITION_PROFILE_MISMATCH": 409,
            "CONDITION_BUSY": 429,
            "CONDITION_TIMEOUT": 504,
            "CONDITION_UNAVAILABLE": 503,
        }
        self.status = defaults.get(code, 400) if status is None else status
        if type(self.status) is not int or not 400 <= self.status <= 599:
            raise ValueError("Invalid conditioning HTTP error status")
        self.code, self.detail = code, detail
        self.public_message = _PUBLIC[code]
        super().__init__(detail or self.public_message)


def _invalid(detail: str) -> None:
    raise ConditionError("CONDITION_INVALID", detail)


def _json_value(value: Any, *, depth: int = 0) -> None:
    if depth > 32:
        _invalid("JSON nesting exceeds the contract limit")
    if value is None or type(value) in (str, bool, int):
        if type(value) is str:
            try:
                value.encode("utf-8")
            except UnicodeError:
                _invalid("Invalid UTF-8 text")
        return
    if type(value) is float:
        if not math.isfinite(value):
            _invalid("JSON numbers must be finite")
        return
    if type(value) is list:
        for item in value:
            _json_value(item, depth=depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item, depth=depth + 1)
        return
    _invalid("Only standard JSON values are permitted")


def canonical_json(value: Any) -> bytes:
    _json_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def rgb_request_framing_bytes(max_media, max_side, max_input_bytes):
    """Bound framing for CTHW uint8 media without serializing tensor data.

    Safetensors stores a JSON entry per tensor, padded to eight bytes. Use the
    longest generated name, dimensions and offsets for every entry; ordering
    and smaller actual values can only shorten that header.
    """
    entry = canonical_json(
        {
            f"media_{max(0, max_media - 1)}": dict(
                dtype="U8",
                shape=[3, 1, max_side, max_side],
                data_offsets=[max_input_bytes, max_input_bytes],
            )
        }
    )
    header_bytes = 2 + max_media * (len(entry) - 2) + max(0, max_media - 1)
    # Four-byte RPC JSON length, eight-byte safetensors header length, padding.
    return 12 + (header_bytes + 7) // 8 * 8


def safe_json(data: bytes, *, max_bytes: int = MAX_MANIFEST_BYTES) -> Any:
    if type(data) is not bytes or len(data) > max_bytes:
        _invalid("JSON size exceeds the contract limit")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _invalid("Duplicate JSON field")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda _: _invalid("Nonfinite JSON number"),
        )
        _json_value(value)
        return value
    except ConditionError:
        raise
    except (ValueError, UnicodeError, RecursionError):
        _invalid("Invalid JSON document")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: Any, name: str) -> None:
    if type(value) is not str or not _HEX.fullmatch(value):
        _invalid(f"Invalid {name} digest")


def _text(value: Any, name: str, *, maximum: int = 256) -> None:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > maximum
        or any(ord(c) < 32 for c in value)
    ):
        _invalid(f"Invalid {name}")


def _no_paths(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {
                "uri",
                "url",
                "path",
                "filename",
                "file_path",
                "artifact_path",
            }:
                _invalid("Paths are not allowed in audit metadata")
            _no_paths(item)
    elif isinstance(value, list):
        for item in value:
            _no_paths(item)


class DeploymentProfile:
    """Immutable identity of administrator-frozen encoder and numeric config."""

    FIELDS = frozenset(
        {
            "schema_version",
            "model",
            "projection",
            "tokenizer",
            "processor",
            "runtime",
            "implementation_version",
            "preprocess_version",
            "numerical_environment",
        }
    )

    def __init__(self, document: dict):
        if (
            type(document) is not dict
            or set(document) != self.FIELDS
            or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1
        ):
            _invalid("Unsupported deployment profile schema")
        for name in (
            "model",
            "tokenizer",
            "processor",
            "runtime",
            "numerical_environment",
        ):
            if type(document[name]) is not dict or not document[name]:
                _invalid("Deployment profile component is missing")
        if (
            document["projection"] is not None
            and type(document["projection"]) is not dict
        ):
            _invalid("Projection must be explicitly null or a hash identity")
        for name in ("implementation_version", "preprocess_version"):
            _text(document[name], name)
        _no_paths(document)
        if (
            type(document["numerical_environment"].get("encoder_tp")) is not int
            or document["numerical_environment"]["encoder_tp"] != 1
        ):
            _invalid("The conditioning deployment requires encoder TP1")
        environment = document["numerical_environment"]
        if not {"encoder_tp", "gpu", "cc", "backend", "precision", "software"} <= set(
            environment
        ):
            _invalid(
                "Deployment profile must freeze the complete numerical environment"
            )
        for name in ("gpu", "cc", "backend", "precision"):
            _text(environment[name], "numerical environment " + name)
        if type(environment["software"]) is not dict or not environment["software"]:
            _invalid("Deployment profile must freeze software versions")
        self._document = copy.deepcopy(document)
        self.profile_id = sha256(canonical_json(self._document))

    @property
    def document(self) -> dict:
        return copy.deepcopy(self._document)

    @classmethod
    def load(cls, path: str | Path) -> DeploymentProfile:
        return cls(safe_json(_read_regular(Path(path), MAX_MANIFEST_BYTES)))


def _check_ancestors(path: Path) -> None:
    for parent in (path, *path.parents):
        if parent.exists() or parent.is_symlink():
            if stat.S_ISLNK(parent.lstat().st_mode):
                _invalid("Symlinks are not allowed in trusted artifact paths")


def _read_regular(path: Path, limit: int) -> bytes:
    _check_ancestors(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                _invalid("Artifact file type or size is invalid")
            data = stream.read(limit + 1)
            if len(data) != info.st_size or len(data) > limit:
                _invalid("Artifact file changed while reading")
            return data
    except ConditionError:
        raise
    except OSError:
        _invalid("Artifact file is unavailable")


def _tensor_spec(condition: Mapping[str, torch.Tensor], max_rows: int) -> dict:
    if not isinstance(condition, Mapping) or set(condition) != {"context", "row_role"}:
        _invalid("Condition must contain exactly context and row_role")
    c, r = condition["context"], condition["row_role"]
    if not isinstance(c, torch.Tensor) or not isinstance(r, torch.Tensor):
        _invalid("Condition values must be tensors")
    if (
        c.dtype != torch.bfloat16
        or c.ndim != 2
        or c.shape[1] != 5120
        or not 0 < c.shape[0] <= max_rows
        or not c.is_contiguous()
    ):
        _invalid("Invalid context dtype, shape, layout, or row count")
    if (
        r.dtype != torch.int64
        or tuple(r.shape) != (c.shape[0],)
        or not r.is_contiguous()
    ):
        _invalid("Invalid row_role dtype, shape, or layout")
    if not torch.isfinite(c).all().item() or not ((r == 0) | (r == 1)).all().item():
        _invalid("Invalid condition values")
    return {
        "context": {"dtype": "BF16", "shape": list(c.shape), "layout": "row-major"},
        "row_role": {"dtype": "I64", "shape": list(r.shape), "layout": "row-major"},
    }


def _preflight_tensor(data: bytes, max_rows: int, max_bytes: int) -> dict:
    if type(data) is not bytes or not 8 < len(data) <= max_bytes:
        _invalid("Invalid safetensors file size")
    size = struct.unpack("<Q", data[:8])[0]
    if size > MAX_HEADER_BYTES or size > len(data) - 8:
        _invalid("Invalid safetensors header size")
    header = safe_json(data[8 : 8 + size], max_bytes=MAX_HEADER_BYTES)
    if type(header) is not dict or set(header) != {"context", "row_role"}:
        _invalid("Unexpected safetensors tensor names")
    entries, spec = [], {}
    rows = None
    for name, dtype, width in (("context", "BF16", 2), ("row_role", "I64", 8)):
        item = header[name]
        if (
            type(item) is not dict
            or set(item) != {"dtype", "shape", "data_offsets"}
            or item["dtype"] != dtype
        ):
            _invalid("Invalid safetensors tensor header")
        shape, offsets = item["shape"], item["data_offsets"]
        if (
            type(shape) is not list
            or not shape
            or any(type(n) is not int or n <= 0 for n in shape)
        ):
            _invalid("Invalid safetensors tensor shape")
        if name == "context":
            if len(shape) != 2 or shape[1] != 5120 or shape[0] > max_rows:
                _invalid("Invalid safetensors context shape")
            rows = shape[0]
        elif shape != [rows]:
            _invalid("Safetensors row roles must match context rows")
        count = math.prod(shape) * width
        if (
            count > max_bytes
            or type(offsets) is not list
            or len(offsets) != 2
            or any(type(n) is not int or n < 0 for n in offsets)
        ):
            _invalid("Invalid safetensors offsets")
        start, end = offsets
        if end - start != count or end > len(data) - 8 - size:
            _invalid("Safetensors offsets exceed payload")
        entries.append((start, end))
        spec[name] = {"dtype": dtype, "shape": shape, "layout": "row-major"}
    position = 0
    for start, end in sorted(entries):
        if start != position:
            _invalid("Safetensors offsets overlap or contain gaps")
        position = end
    if position != len(data) - 8 - size:
        _invalid("Safetensors payload contains extra bytes")
    return spec
