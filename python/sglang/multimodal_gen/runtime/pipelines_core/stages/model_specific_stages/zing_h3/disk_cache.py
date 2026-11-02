# SPDX-License-Identifier: Apache-2.0
"""Shared Block0 files: atomic per-file replacement and one deployment-owned GC."""

import json
import logging
import math
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from .block0_conditions import CONDITIONS_SCHEMA, decode_conditions, encode_conditions
from .block0_identity import SCHEMA
from .cache_logging import cache_event, cache_scope
from .paged_kv import kv_slices

logger = init_logger(__name__)
DISK_FORMAT = 4


def _dump_json(path, value):
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), default=str)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            logger.warning("Block0 temporary metadata cleanup failed", exc_info=True)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_kv(path, tensor):
    """Write the logical (layer, K/V, rows, ...) tensor as raw bytes.

    Each layer/KV slice is contiguous for the pinned slot layout, so this
    avoids materializing a second full-size contiguous copy.
    """
    with open(path, "wb") as handle:
        for value in kv_slices(tensor):
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
    with open(path, "rb") as handle:
        if os.fstat(handle.fileno()).st_size != expected:
            raise ValueError("Block0 KV file size does not match the target slot")
        for value in kv_slices(target):
            _readinto(handle, value)
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
    """Same-key fills are interchangeable; only individual files are atomic.

    GC owns the approximate usage snapshot. Writers never update shared counters
    or take filesystem locks. The deployment enables GC in exactly one instance.
    Existing key directories are reused, but the previous disk format and mixed
    old/new processes are unsupported.
    """

    def __init__(
        self,
        cache_dir,
        *,
        max_bytes=None,
        identity=None,
        rank=0,
        world_size=1,
        gc_interval=60.0,
        gc_stale_seconds=600.0,
        gc_low_watermark=0.9,
    ):
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 1):
            raise ValueError("Block0 disk byte budget must be positive")
        if (
            not math.isfinite(gc_interval)
            or gc_interval <= 0
            or not math.isfinite(gc_stale_seconds)
            or gc_stale_seconds <= 0
            or not 0 < gc_low_watermark < 1
        ):
            raise ValueError("Invalid Block0 GC interval, timeout or low watermark")
        self.cache_dir = Path(cache_dir).resolve()
        if not self.cache_dir.is_dir():
            raise ValueError("Block0 disk root must already exist on mounted storage")
        self.max_bytes = max_bytes
        self.identity = identity
        self.local_rank, self.sp_world_size = rank, world_size
        if not 0 <= rank < world_size:
            raise ValueError("Invalid Block0 disk SP rank")
        self.tmp_root = self.cache_dir / ".tmp"
        self.usage_path = self.cache_dir / "cache_usage.json"
        self.gc_interval = gc_interval
        self.gc_stale_seconds = gc_stale_seconds
        self.gc_low_watermark = gc_low_watermark
        self._gc_stop = threading.Event()
        self._gc_thread = None
        # Do not probe writes here: a full quota must not prevent L1 startup.

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
        return self.cache_dir / "entries" / key

    def _under_root(self, path):
        target = Path(path).resolve()
        return target != self.cache_dir and target.is_relative_to(self.cache_dir)

    def new_tmp_dir(self, key):
        if not self._safe_key(key):
            return None
        try:
            self.tmp_root.mkdir(exist_ok=True)
            path = self.tmp_root / uuid.uuid4().hex
            path.mkdir()
            return str(path)
        except OSError as exc:
            logger.warning("Block0 disk staging failed: %s", exc)
            return None

    def discard_tmp(self, tmp_dir):
        if tmp_dir is not None and self._under_root(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _manifest(folder):
        try:
            value = _read_json(Path(folder) / "manifest.json")
            return value if isinstance(value, dict) else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _rank_info(folder, rank):
        return _read_json(folder / f"rank-{rank}" / "ready.json")

    def _valid_manifest(self, folder, key, *, check_identity=True):
        manifest = self._manifest(folder)
        if (
            not manifest
            or manifest.get("schema") != SCHEMA
            or manifest.get("disk_format") != DISK_FORMAT
            or manifest.get("early_key") != key
            or (
                check_identity
                and self.identity is not None
                and manifest.get("identity") != self.identity
            )
        ):
            return None
        if not isinstance(manifest.get("identity"), dict):
            return None
        world = manifest.get("sp_world_size")
        if (
            type(world) is not int
            or world < 1
            or (
                check_identity
                and self.identity is not None
                and world != self.sp_world_size
            )
        ):
            return None
        ranks = {}
        try:
            for rank in range(world):
                base = folder / f"rank-{rank}"
                info = self._rank_info(folder, rank)
                if (
                    info.get("conditions_schema") != CONDITIONS_SCHEMA
                    or info.get("disk_format") != DISK_FORMAT
                    or not (base / "conditions.pt").is_file()
                    or (base / "kv.bin").stat().st_size != info.get("kv_bytes")
                ):
                    return None
                ranks[str(rank)] = info
        except (OSError, ValueError, TypeError, AttributeError):
            return None
        return dict(manifest, ranks=ranks)

    def may_contain(self, key):
        if not self._safe_key(key):
            return False
        folder = self._entry_dir(key)
        return (folder / f"rank-{self.local_rank}" / "ready.json").is_file()

    def prefix_rows(self, key):
        """Metadata-only preload sizing; loading still validates actual tensors."""
        if not self._safe_key(key):
            return None
        try:
            manifest = self._valid_manifest(self._entry_dir(key), key)
            return (
                None
                if manifest is None
                else manifest.get("geometry", {}).get("prefix_rows")
            )
        except (OSError, ValueError, TypeError, AttributeError):
            return None

    def lookup(self, key, rank=None):
        rank = self.local_rank if rank is None else rank
        if not self._safe_key(key) or not 0 <= rank < self.sp_world_size:
            return None
        try:
            folder = self._entry_dir(key)
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
            return DiskEntry(key, folder, manifest, rank, conditions)
        except Exception as exc:
            cache_event(
                "l2_lookup",
                "failed",
                cache_key=key,
                rank=rank,
                reason=type(exc).__name__,
                level=logging.WARNING,
            )
            return None

    def _touch(self, entry):
        # Touch only an existing marker, never resurrect an evicted directory.
        try:
            marker = entry.path / ".access"
            if time.time() - marker.stat().st_mtime >= 60:
                os.utime(marker, None)
        except OSError:
            pass

    def load_kv_into(self, entry, target):
        info = entry.manifest["ranks"][str(entry.rank)]
        if info["shape"] != list(target.shape) or info["dtype"] != str(target.dtype):
            raise ValueError("Block0 disk KV shape/dtype differs from the L1 slot")
        # The open file owns its inode through concurrent replacement/unlink.
        result = entry.load_kv_into(target)
        self._touch(entry)
        return result

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

    def commit_entry(
        self, key, conditions_dict, manifest, rank_bins, tmp_dir, *, force=False
    ):
        if not self._safe_key(key) or tmp_dir is None or not self._under_root(tmp_dir):
            return None
        tmp = Path(tmp_dir)
        try:
            if (
                manifest["identity"] != self.identity
                or manifest["sp_world_size"] != self.sp_world_size
            ):
                raise ValueError("Block0 snapshot identity/layout differs")
            folder = self._entry_dir(key)
            if not force and self._valid_manifest(folder, key) is not None:
                return {"key": key, "dir": str(folder.relative_to(self.cache_dir))}
            for rank, source in rank_bins.items():
                if not 0 <= rank < self.sp_world_size or not self._under_root(source):
                    raise ValueError("Invalid Block0 rank staging file")
                conditions = tmp / f"conditions-{rank}.pt"
                torch.save(encode_conditions(conditions_dict), conditions)
                info = dict(
                    _read_json(tmp / f"kvmeta-{rank}.json"),
                    kv_bytes=Path(source).stat().st_size,
                    conditions_schema=CONDITIONS_SCHEMA,
                    disk_format=DISK_FORMAT,
                )
                base = folder / f"rank-{rank}"
                base.mkdir(parents=True, exist_ok=True)
                os.replace(source, base / "kv.bin")
                os.replace(conditions, base / "conditions.pt")
                _dump_json(base / "ready.json", info)
            # Shared fields are identical in meaning for the same content key;
            # there is no read/modify/write merge of rank metadata.
            value = {
                name: manifest[name]
                for name in ("identity", "sp_world_size", "geometry", "prompt_preview")
            }
            value.update(schema=SCHEMA, disk_format=DISK_FORMAT, early_key=key)
            _dump_json(folder / "manifest.json", value)
            (folder / ".access").touch()
            return {"key": key, "dir": str(folder.relative_to(self.cache_dir))}
        except Exception as exc:
            logger.warning("Block0 disk commit skipped: %s", exc)
            return None
        finally:
            self.discard_tmp(tmp)

    def _folders(self):
        for folder in (self.cache_dir / "entries").glob("*"):
            if (
                folder.is_dir()
                and self._safe_key(folder.name)
                and self._under_root(folder)
            ):
                yield folder

    def _files(self, folder):
        for path in folder.rglob("*"):
            if path.is_file() and self._under_root(path):
                yield path

    def _measure(self, folder):
        size = 0
        modified = folder.stat().st_mtime
        for path in self._files(folder):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            size += stat.st_size
            modified = max(modified, stat.st_mtime)
        return size, modified

    def _scan(self):
        entries = []
        for folder in self._folders():
            try:
                size, modified = self._measure(folder)
                manifest = self._manifest(folder) or {}
                access = folder / ".access"
                entries.append(
                    dict(
                        dir=str(folder.relative_to(self.cache_dir)),
                        key=folder.name,
                        size_bytes=size,
                        modified_at=modified,
                        last_accessed_at=access.stat().st_mtime
                        if access.exists()
                        else modified,
                        complete=self._valid_manifest(
                            folder, folder.name, check_identity=False
                        )
                        is not None,
                        prompt_preview=manifest.get("prompt_preview"),
                    )
                )
            except FileNotFoundError:
                continue
        return entries

    def _usage(self):
        try:
            value = _read_json(self.usage_path)
            if value["version"] == DISK_FORMAT and isinstance(value["entries"], list):
                return value
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return {
            "max_bytes": self.max_bytes,
            "sampled_at": None,
            "root_bytes": None,
            "root_entries": None,
            "temporary_bytes": None,
            "entries": [],
        }

    def collect(self):
        """One best-effort GC pass; only the designated collector calls this."""
        with cache_scope():
            started = time.perf_counter()
            activity = dict(
                deleted_count=0,
                stale_deleted_count=0,
                capacity_deleted_count=0,
                reclaimed_bytes=0,
            )
            try:
                usage = self._collect(activity)
            except Exception as exc:
                cache_event(
                    "gc",
                    "failed",
                    rank=self.local_rank,
                    started=started,
                    reason=type(exc).__name__,
                    level=logging.WARNING,
                    **activity,
                )
                raise
            failed = len(usage["failed"])
            changed = activity["deleted_count"] > 0
            cache_event(
                "gc",
                "partial" if failed else "completed" if changed else "idle",
                rank=self.local_rank,
                started=started,
                level=logging.WARNING
                if failed
                else logging.INFO
                if changed
                else logging.DEBUG,
                reason="delete_failed"
                if failed
                else "stale_or_capacity"
                if changed
                else None,
                root_bytes=usage["root_bytes"],
                max_bytes=self.max_bytes,
                failed_count=failed,
                **activity,
            )
            return usage

    def _collect(self, activity):
        if self.max_bytes is None:
            raise ValueError("Block0 GC requires a disk byte budget")
        now = time.time()
        entries = self._scan()
        activity["scanned_bytes"] = sum(entry["size_bytes"] for entry in entries)
        temporary_bytes = 0
        failed = []
        for folder in list(self.tmp_root.glob("*")) + list(
            self.cache_dir.glob("*.tmp")
        ):
            size = 0
            if not self._under_root(folder):
                continue
            try:
                size, modified = (
                    self._measure(folder)
                    if folder.is_dir()
                    else (folder.stat().st_size, folder.stat().st_mtime)
                )
                activity["scanned_bytes"] += size
                if now - modified >= self.gc_stale_seconds:
                    if folder.is_dir():
                        shutil.rmtree(folder)
                    else:
                        folder.unlink()
                    activity["deleted_count"] += 1
                    activity["stale_deleted_count"] += 1
                    activity["reclaimed_bytes"] += size
                else:
                    temporary_bytes += size
            except FileNotFoundError:
                continue
            except OSError:
                temporary_bytes += size
                failed.append(str(folder.relative_to(self.cache_dir)))
        remaining = []
        for entry in entries:
            folder = self.cache_dir / entry["dir"]
            try:
                # Abandoned JSON temp files can also live alongside valid data.
                for path in folder.rglob("*.tmp"):
                    if self._under_root(path):
                        stat = path.stat()
                        if now - stat.st_mtime >= self.gc_stale_seconds:
                            path.unlink()
                            activity["deleted_count"] += 1
                            activity["stale_deleted_count"] += 1
                            activity["reclaimed_bytes"] += stat.st_size
                            entry["size_bytes"] = max(
                                0, entry["size_bytes"] - stat.st_size
                            )
                if (
                    not entry["complete"]
                    and now - entry["modified_at"] >= self.gc_stale_seconds
                ):
                    shutil.rmtree(folder)
                    activity["deleted_count"] += 1
                    activity["stale_deleted_count"] += 1
                    activity["reclaimed_bytes"] += entry["size_bytes"]
                    continue
            except FileNotFoundError:
                pass
            except OSError:
                failed.append(entry["key"])
            remaining.append(entry)
        total = temporary_bytes + sum(e["size_bytes"] for e in remaining)
        if total > self.max_bytes:
            for entry in sorted(
                remaining.copy(), key=lambda e: (e["last_accessed_at"], e["dir"])
            ):
                if total <= self.max_bytes * self.gc_low_watermark:
                    break
                # Fresh incomplete writes are left for the timeout rule.
                if not entry["complete"]:
                    continue
                try:
                    shutil.rmtree(self.cache_dir / entry["dir"])
                except FileNotFoundError:
                    pass
                except OSError:
                    failed.append(entry["key"])
                    continue
                total -= entry["size_bytes"]
                remaining.remove(entry)
                activity["deleted_count"] += 1
                activity["capacity_deleted_count"] += 1
                activity["reclaimed_bytes"] += entry["size_bytes"]
        usage = dict(
            version=DISK_FORMAT,
            max_bytes=self.max_bytes,
            sampled_at=now,
            root_bytes=total,
            root_entries=len(remaining),
            temporary_bytes=temporary_bytes,
            entries=remaining,
            failed=failed,
        )
        _dump_json(self.usage_path, usage)
        return usage

    def start_gc(self):
        if self.local_rank != 0 or self._gc_thread is not None:
            return
        if self.max_bytes is None:
            raise ValueError("Block0 GC requires a disk byte budget")

        def run():
            while not self._gc_stop.is_set():
                try:
                    self.collect()
                except Exception:
                    logger.warning("Block0 disk GC failed", exc_info=True)
                if self._gc_stop.wait(self.gc_interval):
                    break

        self._gc_thread = threading.Thread(
            target=run, name="zing-block0-gc", daemon=True
        )
        self._gc_thread.start()

    def close(self):
        self._gc_stop.set()
        if self._gc_thread is not None:
            self._gc_thread.join()

    def list_entries(self, *, cache_key=None, offset=0, limit=100):
        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= 1000
        ):
            raise ValueError("Offset must be nonnegative and limit must be 1..1000")
        if cache_key is not None and not self._safe_key(cache_key):
            raise ValueError("Invalid cache key")
        cache_event(
            "list",
            "completed",
            rank=self.local_rank,
            level=logging.DEBUG,
            cache_key=cache_key,
        )
        usage = self._usage()
        selected = sorted(
            (e for e in usage["entries"] if cache_key is None or e["key"] == cache_key),
            key=lambda e: e["dir"],
        )
        return dict(
            max_bytes=usage["max_bytes"],
            root_entries=usage["root_entries"],
            root_bytes=usage["root_bytes"],
            sampled_at=usage["sampled_at"],
            temporary_bytes=usage["temporary_bytes"],
            total_entries=len(selected) if usage["sampled_at"] is not None else None,
            total_bytes=sum(e["size_bytes"] for e in selected)
            if usage["sampled_at"] is not None
            else None,
            complete_entries=sum(e["complete"] for e in selected)
            if usage["sampled_at"] is not None
            else None,
            offset=offset,
            entries=selected[offset : offset + limit],
            next_offset=offset + limit if offset + limit < len(selected) else None,
        )

    def delete_entries(self, *, keys=None):
        """Live scoped deletion, independent of GC's possibly stale snapshot."""
        with cache_scope():
            started = time.perf_counter()
            event = "clear" if keys is None else "delete"
            cache_event(
                event,
                "started",
                rank=self.local_rank,
                cache_key=keys[0] if keys and len(keys) == 1 else None,
            )
            try:
                value = self._delete_entries(keys=keys)
            except Exception as exc:
                cache_event(
                    event,
                    "failed",
                    rank=self.local_rank,
                    started=started,
                    reason=type(exc).__name__,
                    level=logging.WARNING,
                )
                raise
            failed = len(value["failed"])
            cache_event(
                event,
                "partial" if failed else "completed",
                rank=self.local_rank,
                cache_key=keys[0] if keys and len(keys) == 1 else None,
                started=started,
                reason="delete_failed" if failed else None,
                level=logging.WARNING if failed else logging.INFO,
                deleted_count=len(value["deleted"]),
                missing_count=len(value["missing"]),
                failed_count=failed,
                reclaimed_bytes=value["reclaimed_bytes"],
            )
            return value

    def _delete_entries(self, *, keys=None):
        if keys is not None and (
            not isinstance(keys, (list, tuple))
            or not all(self._safe_key(k) for k in keys)
        ):
            raise ValueError("Invalid cache keys")
        deleted, failed, reclaimed = [], [], 0
        found = set()
        for folder in self._folders():
            key = folder.name
            if keys is not None and key not in keys:
                continue
            found.add(key)
            try:
                size, _ = self._measure(folder)
                shutil.rmtree(folder)
                reclaimed += size
                deleted.append(key)
                cache_event(
                    "l2_delete",
                    "completed",
                    cache_key=key,
                    rank=self.local_rank,
                    reason="clear" if keys is None else "delete",
                    reclaimed_bytes=size,
                )
            except FileNotFoundError:
                deleted.append(key)
                cache_event("l2_delete", "missing", cache_key=key, rank=self.local_rank)
            except OSError as exc:
                cache_event(
                    "l2_delete",
                    "failed",
                    cache_key=key,
                    rank=self.local_rank,
                    reason=type(exc).__name__,
                    level=logging.WARNING,
                )
                failed.append({"key": key, "code": "delete_failed"})
        usage = self._usage()
        return dict(
            deleted=deleted,
            missing=sorted(set(keys or ()) - found),
            failed=failed,
            reclaimed_bytes=reclaimed,
            remaining_bytes=usage["root_bytes"],
            sampled_at=usage["sampled_at"],
        )
