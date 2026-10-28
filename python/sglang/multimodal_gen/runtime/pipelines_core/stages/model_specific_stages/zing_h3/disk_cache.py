# SPDX-License-Identifier: Apache-2.0
"""Tier-2 Block0 cache on CPFS or a local SSD.

The directory is intentionally simple: one immutable entry directory per
content key plus one small LRU index shared by every process.  Rank files are
written independently so sequence-parallel workers never exchange KV bytes.
"""

import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from .block0_conditions import CONDITIONS_SCHEMA, decode_conditions, encode_conditions
from .block0_identity import SCHEMA, fingerprint

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows development only
    fcntl = None

logger = init_logger(__name__)

INDEX_VERSION = 2


def _dump_json(path, value):
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_kv(path, tensor):
    """Write the logical (layer, K/V, rows, ...) tensor as raw bytes.

    Each layer/KV slice is contiguous for the pinned slot layout, so this
    avoids materializing a second full-size contiguous copy.
    """
    with open(path, "wb") as handle:
        if tensor.dim() >= 2 and tensor.shape[1] == 2:
            slices = (
                tensor[layer, part]
                for layer in range(tensor.shape[0])
                for part in range(tensor.shape[1])
            )
        else:
            slices = (tensor,)
        for value in slices:
            value = value.contiguous().reshape(-1)
            value.view(torch.uint8).numpy().tofile(handle)


def _readinto(handle, tensor):
    if tensor.is_contiguous():
        flat = tensor.reshape(-1)
        view = memoryview(flat.view(torch.uint8).numpy())
        if handle.readinto(view) != view.nbytes:
            raise EOFError("Block0 KV file ended before its declared size")
        return
    staging = torch.empty(tensor.numel() * tensor.element_size(), dtype=torch.uint8)
    view = memoryview(staging.numpy())
    if handle.readinto(view) != view.nbytes:
        raise EOFError("Block0 KV file ended before its declared size")
    tensor.copy_(staging.view(tensor.dtype).reshape(tensor.shape))


def _read_kv(path, target):
    expected = target.numel() * target.element_size()
    if path.stat().st_size != expected:
        raise ValueError("Block0 KV file size does not match the target slot")
    with open(path, "rb") as handle:
        if target.dim() >= 2 and target.shape[1] == 2:
            for layer in range(target.shape[0]):
                for part in range(target.shape[1]):
                    _readinto(handle, target[layer, part])
        else:
            _readinto(handle, target)
    return target


@dataclass(frozen=True)
class DiskEntry:
    """A complete on-disk entry whose conditions are already materialized."""

    key: str
    path: Path
    manifest: dict
    rank: int
    conditions: dict

    def load_conditions(self):
        return self.conditions

    def load_kv_into(self, target, rank=None):
        rank = self.rank if rank is None else int(rank)
        return _read_kv(self.path / f"rank-{rank}" / "kv.bin", target)


class Block0DiskCache:
    """One root-wide byte LRU; immutable complete entries contain every SP rank.

    The deployment owns and mounts the root. Only schema-v2 children enter this
    index. Old identity-free files are left untouched and can never be loaded.
    Pending entries count towards the same budget as complete entries.
    """

    def __init__(self, cache_dir, *, max_bytes, identity, rank=0, world_size=1):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Block0 disk byte budget must be positive")
        self.cache_dir = Path(cache_dir).resolve()
        if not self.cache_dir.is_dir():
            raise ValueError("Block0 disk root must already exist on mounted storage")
        self.max_bytes = max_bytes
        self.identity = identity
        self.model = fingerprint(identity["model"])
        self.layout = fingerprint(identity["layout"])
        self.local_rank, self.sp_world_size = rank, world_size
        if not 0 <= rank < world_size:
            raise ValueError("Invalid Block0 disk SP rank")
        self.tmp_root = self.cache_dir / ".tmp"
        self.tmp_root.mkdir(exist_ok=True)
        probe = self.tmp_root / uuid.uuid4().hex
        probe.write_bytes(b"")
        probe.unlink()

    @property
    def index_path(self):
        return self.cache_dir / "lru_index.json"

    def is_enabled(self):
        return True

    @staticmethod
    def _safe_key(key):
        return (
            isinstance(key, str)
            and bool(key)
            and not key.startswith(".")
            and all(char not in key for char in ("/", "\\", "\x00"))
        )

    def _entry_dir(self, key):
        return self.cache_dir / "v2" / self.model / self.layout / key[:2] / key

    def _under_root(self, path):
        target = Path(path).resolve()
        return target != self.cache_dir and target.is_relative_to(self.cache_dir)

    def new_tmp_dir(self, key):
        if not self._safe_key(key):
            return None
        try:
            path = self.tmp_root / uuid.uuid4().hex
            path.mkdir()
            return str(path)
        except OSError as exc:
            logger.warning("Block0 disk staging failed: %s", exc)
            return None

    def discard_tmp(self, tmp_dir):
        if tmp_dir is not None and self._under_root(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @contextmanager
    def _locked(self):
        with open(self.cache_dir / ".index.lock", "a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _dir_size(folder):
        return sum(p.stat().st_size for p in Path(folder).rglob("*") if p.is_file())

    @staticmethod
    def _manifest(folder):
        try:
            value = _read_json(Path(folder) / "manifest.json")
            return value if isinstance(value, dict) else None
        except (OSError, ValueError):
            return None

    def _record(self, folder, manifest):
        return {
            "dir": str(folder.relative_to(self.cache_dir)),
            "key": manifest["early_key"],
            "size_bytes": self._dir_size(folder),
            "last_accessed_at": manifest["last_accessed_at"],
            "complete": (folder / ".complete").is_file(),
        }

    @staticmethod
    def _normalize(entries):
        entries = sorted(entries, key=lambda e: (e["last_accessed_at"], e["dir"]))
        return {
            "version": INDEX_VERSION,
            "entries": entries,
            "stats": {
                "total_entries": len(entries),
                "total_bytes": sum(e["size_bytes"] for e in entries),
            },
        }

    def _write_index_locked(self, index):
        index["stats"]["max_bytes"] = self.max_bytes
        _dump_json(self.index_path, index)

    def _rebuild_locked(self):
        entries = []
        for path in (self.cache_dir / "v2").glob("*/*/*/*/manifest.json"):
            manifest = self._manifest(path.parent)
            if manifest is not None and manifest.get("schema") == SCHEMA:
                try:
                    entries.append(self._record(path.parent, manifest))
                except (KeyError, OSError, TypeError):
                    continue
        index = self._normalize(entries)
        self._write_index_locked(index)
        return index

    def _load_index_locked(self):
        try:
            raw = _read_json(self.index_path)
            if raw["version"] != INDEX_VERSION or not isinstance(raw["entries"], list):
                raise ValueError("Invalid Block0 disk index")
            entries = []
            for entry in raw["entries"]:
                folder = self.cache_dir / entry["dir"]
                if not self._under_root(folder) or not folder.resolve().is_relative_to(
                    self.cache_dir / "v2"
                ):
                    raise ValueError("Invalid Block0 index path")
                if folder.is_dir():
                    entries.append(dict(entry, size_bytes=self._dir_size(folder)))
            return self._normalize(entries)
        except (OSError, ValueError, TypeError, KeyError):
            return self._rebuild_locked()

    def _record_locked(self, entry, index):
        entries = [e for e in index["entries"] if e["dir"] != entry["dir"]] + [entry]
        entries = self._normalize(entries)["entries"]
        total = sum(e["size_bytes"] for e in entries)
        while entries and total > self.max_bytes:
            victim = entries.pop(0)
            folder = self.cache_dir / victim["dir"]
            if self._under_root(folder):
                # On failure do not pretend bytes were reclaimed.
                shutil.rmtree(folder)
            total -= victim["size_bytes"]
        return self._normalize(entries)

    def _valid_manifest(self, folder, key):
        manifest = self._manifest(folder)
        if (
            not isinstance((manifest or {}).get("generation"), str)
            or not manifest["generation"]
        ):
            return None
        if not manifest or any(
            (
                manifest.get("schema") != SCHEMA,
                manifest.get("early_key") != key,
                manifest.get("identity") != self.identity,
                manifest.get("sp_world_size") != self.sp_world_size,
            )
        ):
            return None
        ranks = manifest.get("ranks", {})
        if set(ranks) != {str(rank) for rank in range(self.sp_world_size)}:
            return None
        for rank, info in ranks.items():
            base = folder / f"rank-{rank}"
            if (
                info.get("generation") != manifest.get("generation")
                or info.get("conditions_schema") != CONDITIONS_SCHEMA
                or not (base / "conditions.pt").is_file()
                or not (base / "kv.bin").is_file()
                or (base / "kv.bin").stat().st_size != info.get("kv_bytes")
            ):
                return None
        return manifest

    def may_contain(self, key):
        """Read-only local-rank hint; lookup still validates the complete entry."""
        if not self._safe_key(key):
            return False
        try:
            folder = self._entry_dir(key)
            return (folder / ".complete").is_file() and (
                folder / f"rank-{self.local_rank}" / "kv.bin"
            ).is_file()
        except OSError:
            return False

    def lookup(self, key, rank=None):
        rank = self.local_rank if rank is None else rank
        if not self._safe_key(key) or not 0 <= rank < self.sp_world_size:
            return None
        try:
            with self._locked():
                folder = self._entry_dir(key)
                if not (folder / ".complete").is_file():
                    return None
                manifest = self._valid_manifest(folder, key)
                if manifest is None:
                    return None
                conditions = decode_conditions(
                    torch.load(
                        folder / f"rank-{rank}" / "conditions.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                manifest["last_accessed_at"] = time.time()
                _dump_json(folder / "manifest.json", manifest)
                index = self._record_locked(
                    self._record(folder, manifest), self._load_index_locked()
                )
                self._write_index_locked(index)
                if not folder.is_dir():
                    return None
                return DiskEntry(key, folder, manifest, rank, conditions)
        except Exception as exc:
            logger.warning("Block0 disk lookup skipped: %s", exc)
            return None

    def load_kv_into(self, entry, target):
        # Eviction/commit and the complete file read share the root lock. A
        # promoted L1 entry owns its bytes after this method returns.
        with self._locked():
            current = self._valid_manifest(entry.path, entry.key)
            if (
                not (entry.path / ".complete").is_file()
                or current is None
                or current["generation"] != entry.manifest["generation"]
                or current["ranks"] != entry.manifest["ranks"]
            ):
                raise ValueError("Block0 disk entry changed before promotion")
            info = current["ranks"][str(entry.rank)]
            if info["shape"] != list(target.shape) or info["dtype"] != str(
                target.dtype
            ):
                raise ValueError("Block0 disk KV shape/dtype differs from the L1 slot")
            return entry.load_kv_into(target)

    def save_rank_kv(self, key, rank, kv_tensor, tmp_dir):
        if not self._safe_key(key) or tmp_dir is None or not self._under_root(tmp_dir):
            return None
        try:
            path = Path(tmp_dir) / f"block0_kv_rank_{rank}.bin"
            _write_kv(path, kv_tensor)
            _dump_json(
                Path(tmp_dir) / f"kvmeta-{rank}.json",
                {"shape": list(kv_tensor.shape), "dtype": str(kv_tensor.dtype)},
            )
            return path
        except (OSError, RuntimeError) as exc:
            logger.warning("Block0 KV write skipped: %s", exc)
            return None

    def commit_entry(self, key, conditions_dict, manifest, rank_bins, tmp_dir):
        if not self._safe_key(key) or tmp_dir is None or not self._under_root(tmp_dir):
            return None
        tmp = Path(tmp_dir)
        try:
            # A fill cohort shares one generation across SP ranks. This prevents
            # mixing shards from competing deployments writing the same key.
            generation = manifest["generation"]
            if not isinstance(generation, str) or not generation:
                raise ValueError("Block0 snapshot generation is missing")
            if (
                manifest["identity"] != self.identity
                or manifest["sp_world_size"] != self.sp_world_size
            ):
                raise ValueError("Block0 snapshot identity/layout differs")
            prepared = {}
            for rank, source in rank_bins.items():
                if not 0 <= rank < self.sp_world_size or not self._under_root(source):
                    raise ValueError("Invalid Block0 rank staging file")
                rank_dir = tmp / f"rank-{rank}"
                rank_dir.mkdir()
                os.replace(source, rank_dir / "kv.bin")
                torch.save(
                    encode_conditions(conditions_dict), rank_dir / "conditions.pt"
                )
                prepared[str(rank)] = {
                    **_read_json(tmp / f"kvmeta-{rank}.json"),
                    "kv_bytes": (rank_dir / "kv.bin").stat().st_size,
                    "generation": generation,
                    "conditions_schema": CONDITIONS_SCHEMA,
                }
            with self._locked():
                folder = self._entry_dir(key)
                if (folder / ".complete").is_file() and self._valid_manifest(
                    folder, key
                ) is not None:
                    return self._record(folder, self._manifest(folder))
                old = self._manifest(folder)
                if old is not None and old.get("generation") != generation:
                    shutil.rmtree(folder)
                    old = None
                folder.mkdir(parents=True, exist_ok=True)
                (folder / ".complete").unlink(missing_ok=True)
                for rank in prepared:
                    destination = folder / f"rank-{rank}"
                    if destination.exists():
                        shutil.rmtree(destination)
                    os.replace(tmp / f"rank-{rank}", destination)
                now = time.time()
                value = dict(
                    manifest,
                    schema=SCHEMA,
                    early_key=key,
                    created_at=(old or {}).get("created_at", now),
                    last_accessed_at=now,
                    ranks={**(old or {}).get("ranks", {}), **prepared},
                )
                _dump_json(folder / "manifest.json", value)
                if self._valid_manifest(folder, key) is not None:
                    marker = folder / f".complete.{uuid.uuid4().hex}"
                    marker.write_bytes(b"")
                    os.replace(marker, folder / ".complete")
                entry = self._record(folder, value)
                index = self._record_locked(entry, self._load_index_locked())
                self._write_index_locked(index)
                return entry if folder.is_dir() else None
        except Exception as exc:
            logger.warning("Block0 disk commit skipped: %s", exc)
            return None
        finally:
            self.discard_tmp(tmp)
