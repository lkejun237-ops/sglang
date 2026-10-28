# SPDX-License-Identifier: Apache-2.0
"""Tier-2 Block0 cache on CPFS or a local SSD.

The directory is intentionally simple: one immutable entry directory per
content key plus one small LRU index shared by every process.  Rank files are
written independently so sequence-parallel workers never exchange KV bytes.
"""

import json
import os
import pickle
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows development only
    fcntl = None

logger = init_logger(__name__)

ROOT_ENV = "SGLANG_ZING_DISK_CACHE_DIR"
MAX_ENTRIES_ENV = "SGLANG_ZING_DISK_CACHE_MAX_ENTRIES"
MAX_GB_ENV = "SGLANG_ZING_DISK_CACHE_MAX_GB"
INDEX_VERSION = 1


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


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
        return _read_kv(self.path / f"block0_kv_rank_{rank}.bin", target)


class Block0DiskCache:
    """Bounded, self-healing, process-safe LRU over Block0 KV directories."""

    def __init__(self, cache_dir=None):
        raw = os.environ.get(ROOT_ENV) if cache_dir is None else cache_dir
        self.cache_dir = Path(raw).expanduser() if raw else None
        self.max_entries = _env_int(MAX_ENTRIES_ENV, 50)
        self.max_gb = _env_float(MAX_GB_ENV, 1024.0)
        self.max_bytes = max(0, int(self.max_gb * (1 << 30)))
        self._writable = None
        self._rank = None
        self._world_size = None

    @property
    def local_rank(self):
        if self._rank is None:
            self._rank = self._sp_rank()
        return self._rank

    @property
    def sp_world_size(self):
        if self._world_size is None:
            self._world_size = self._sp_world_size()
        return self._world_size

    @staticmethod
    def _sp_rank():
        try:
            from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                get_sp_parallel_rank,
                model_parallel_is_initialized,
            )

            if model_parallel_is_initialized():
                return int(get_sp_parallel_rank())
        except Exception:
            pass
        return 0

    @staticmethod
    def _sp_world_size():
        try:
            from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                get_sp_world_size,
                model_parallel_is_initialized,
            )

            if model_parallel_is_initialized():
                return int(get_sp_world_size())
        except Exception:
            pass
        return 1

    @property
    def index_path(self):
        return self.cache_dir / "lru_index.json"

    @property
    def tmp_root(self):
        return self.cache_dir / ".tmp"

    def is_enabled(self):
        if self.cache_dir is None:
            return False
        if self._writable is not None:
            return self._writable
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.tmp_root.mkdir(exist_ok=True)
            probe = self.cache_dir / f".probe-{os.getpid()}-{uuid.uuid4().hex}"
            probe.write_bytes(b"")
            probe.unlink()
        except (OSError, ValueError):
            self._writable = False
        else:
            self._writable = True
        return self._writable

    @staticmethod
    def _safe_key(key):
        return (
            isinstance(key, str)
            and bool(key)
            and not key.startswith(".")
            and all(char not in key for char in ("/", "\\", "\x00"))
            and key not in {".", ".."}
        )

    def _entry_dir(self, key):
        return self.cache_dir / key[:2] / key

    def _under_root(self, path):
        try:
            target = Path(path).resolve()
            target.relative_to(self.cache_dir.resolve())
        except (OSError, ValueError):
            return False
        return target != self.cache_dir.resolve()

    def new_tmp_dir(self, key):
        if not self.is_enabled() or not self._safe_key(key):
            return None
        try:
            path = self.tmp_root / f"{key}_{uuid.uuid4().hex}"
            path.mkdir(parents=True, exist_ok=False)
            return str(path)
        except OSError as exc:
            logger.warning("Zing Block0 disk tmp dir failed: %s", exc)
            return None

    def discard_tmp(self, tmp_dir):
        if tmp_dir is None or not self.is_enabled():
            return
        if self._under_root(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @contextmanager
    def _locked(self):
        if not self.is_enabled():
            raise OSError("Block0 disk cache is disabled")
        lock_path = self.cache_dir / ".index.lock"
        handle = open(lock_path, "a+b")
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    @staticmethod
    def _dir_size(folder):
        try:
            return sum(
                item.stat().st_size for item in Path(folder).iterdir() if item.is_file()
            )
        except OSError:
            return 0

    def _manifest(self, folder):
        try:
            value = _read_json(Path(folder) / "manifest.json")
        except (OSError, ValueError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    def _write_manifest(self, folder, manifest):
        """Write manifest.json until its own encoded size field is stable."""
        size = 0
        for _ in range(4):
            manifest["size_bytes"] = size
            _dump_json(Path(folder) / "manifest.json", manifest)
            actual = self._dir_size(folder)
            if actual == size:
                break
            size = actual
        return manifest

    def _entry_from_manifest(self, key, folder, manifest):
        folder = Path(folder)
        try:
            mtime = folder.stat().st_mtime
        except OSError:
            mtime = time.time()
        return {
            "key": key,
            "dir": str(folder),
            "created_at": float(manifest.get("created_at", mtime)),
            "last_accessed_at": float(manifest.get("last_accessed_at", mtime)),
            "size_bytes": self._dir_size(folder),
            "sp_world_size": manifest.get("sp_world_size"),
            "geometry": manifest.get("geometry"),
            "prompt_preview": manifest.get("prompt_preview"),
        }

    @staticmethod
    def _normalize(entries):
        entries = sorted(
            entries, key=lambda item: (item.get("last_accessed_at", 0.0), item["key"])
        )
        return {
            "version": INDEX_VERSION,
            "stats": {
                "total_entries": len(entries),
                "total_bytes": sum(int(item.get("size_bytes", 0)) for item in entries),
            },
            "entries": entries,
        }

    def _write_index_locked(self, index):
        stats = index.setdefault("stats", {})
        stats["max_entries"] = self.max_entries
        stats["max_bytes"] = self.max_bytes
        _dump_json(self.index_path, index)

    def _rebuild_locked(self):
        entries = []
        try:
            prefixes = [
                item
                for item in self.cache_dir.iterdir()
                if item.is_dir() and not item.name.startswith(".")
            ]
        except OSError:
            prefixes = []
        for prefix in prefixes:
            try:
                folders = [item for item in prefix.iterdir() if item.is_dir()]
            except OSError:
                continue
            for folder in folders:
                if not (folder / ".complete").is_file():
                    continue
                manifest = self._manifest(folder)
                if manifest is None:
                    continue
                entries.append(self._entry_from_manifest(folder.name, folder, manifest))
        index = self._normalize(entries)
        self._write_index_locked(index)
        return index

    def _load_index_locked(self):
        try:
            raw = _read_json(self.index_path)
            if raw.get("version") != INDEX_VERSION:
                raise ValueError("unsupported index version")
            stored = raw.get("entries")
            if not isinstance(stored, list):
                raise ValueError("invalid index entries")
        except (OSError, ValueError, TypeError, AttributeError):
            return self._rebuild_locked()

        entries, changed = [], False
        for entry in stored:
            if not isinstance(entry, dict) or not self._safe_key(entry.get("key")):
                changed = True
                continue
            folder = self._entry_dir(entry["key"])
            if not (folder / ".complete").is_file():
                changed = True
                continue
            size = self._dir_size(folder)
            if entry.get("size_bytes") != size:
                entry["size_bytes"] = size
                changed = True
            entry["dir"] = str(folder)
            entries.append(entry)
        index = self._normalize(entries)
        stats = index["stats"]
        old = raw.get("stats") or {}
        if (
            changed
            or old.get("total_entries") != stats["total_entries"]
            or old.get("total_bytes") != stats["total_bytes"]
            or old.get("max_entries") != self.max_entries
            or old.get("max_bytes") != self.max_bytes
        ):
            self._write_index_locked(index)
        return index

    def _remove_entry(self, entry):
        path = Path(entry.get("dir") or self._entry_dir(entry["key"]))
        if not self._under_root(path):
            logger.warning("Refusing to remove Block0 entry outside cache: %s", path)
            return
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Zing Block0 disk eviction failed for %s: %s", path, exc)

    def _record_locked(self, entry, index):
        entries = [item for item in index["entries"] if item.get("key") != entry["key"]]
        entries.append(entry)
        entries = self._normalize(entries)["entries"]
        total = sum(int(item.get("size_bytes", 0)) for item in entries)
        while entries and (len(entries) > self.max_entries or total > self.max_bytes):
            victim = entries.pop(0)
            total -= int(victim.get("size_bytes", 0))
            self._remove_entry(victim)
        return self._normalize(entries)["entries"]

    def may_contain(self, key):
        """Read-only local-rank hint; do not load conditions/KV or update LRU."""
        if self.cache_dir is None or self._writable is False or not self._safe_key(key):
            return False
        try:
            folder = self._entry_dir(key)
            return (folder / ".complete").is_file() and (
                folder / f"block0_kv_rank_{self.local_rank}.bin"
            ).is_file()
        except OSError:
            return False

    def lookup(self, key, rank=0):
        if not self.is_enabled() or not self._safe_key(key):
            return None
        try:
            with self._locked():
                index = self._load_index_locked()
                entry = next(
                    (item for item in index["entries"] if item.get("key") == key),
                    None,
                )
                if entry is None:
                    # An entry directory may exist after a crash between the
                    # rename and the index update. Recover it once.
                    folder = self._entry_dir(key)
                    if (folder / ".complete").is_file():
                        index = self._rebuild_locked()
                        entry = next(
                            (
                                item
                                for item in index["entries"]
                                if item.get("key") == key
                            ),
                            None,
                        )
                if entry is None:
                    return None
                folder = Path(entry.get("dir") or self._entry_dir(key))
                if not (folder / ".complete").is_file():
                    del index["entries"][index["entries"].index(entry)]
                    self._write_index_locked(self._normalize(index["entries"]))
                    return None
                rank = int(rank)
                if not (folder / f"block0_kv_rank_{rank}.bin").is_file():
                    return None
                entry = dict(entry, last_accessed_at=time.time())
                index["entries"] = [
                    entry if item.get("key") == key else item
                    for item in index["entries"]
                ]
                index = self._normalize(index["entries"])
                self._write_index_locked(index)
                manifest = self._manifest(folder) or {}
                try:
                    conditions = torch.load(
                        folder / "conditions.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                except (
                    OSError,
                    RuntimeError,
                    ValueError,
                    TypeError,
                    EOFError,
                    pickle.UnpicklingError,
                ) as exc:
                    logger.warning("Zing Block0 conditions load failed: %s", exc)
                    conditions = None
                if not isinstance(conditions, dict):
                    return None
                return DiskEntry(key, folder, manifest, rank, conditions)
        except OSError as exc:
            logger.debug("Zing Block0 disk lookup skipped: %s", exc)
            return None

    def save_rank_kv(self, key, rank, kv_tensor, tmp_dir):
        if (
            not self.is_enabled()
            or not self._safe_key(key)
            or tmp_dir is None
            or not self._under_root(tmp_dir)
        ):
            return None
        try:
            tmp = Path(tmp_dir)
            tmp.mkdir(parents=True, exist_ok=True)
            path = tmp / f"block0_kv_rank_{int(rank)}.bin"
            _write_kv(path, kv_tensor)
            return path
        except (OSError, RuntimeError) as exc:
            logger.warning("Zing Block0 rank KV write failed: %s", exc)
            return None

    def commit_entry(self, key, conditions_dict, manifest, rank_bins, tmp_dir):
        if (
            not self.is_enabled()
            or not self._safe_key(key)
            or tmp_dir is None
            or not self._under_root(tmp_dir)
        ):
            return None
        tmp = Path(tmp_dir)
        try:
            tmp.mkdir(parents=True, exist_ok=True)
            torch.save(conditions_dict, tmp / "conditions.pt")
            manifest = dict(manifest or {})
            now = time.time()
            manifest.update(
                early_key=key,
                created_at=float(manifest.get("created_at", now)),
                last_accessed_at=float(manifest.get("last_accessed_at", now)),
                sp_world_size=int(manifest.get("sp_world_size", self.sp_world_size)),
            )
            manifest.setdefault("geometry", {})
            manifest.setdefault("prompt_preview", None)

            if isinstance(rank_bins, dict):
                bins = list(rank_bins.items())
            else:
                bins = [(None, path) for path in (rank_bins or ())]
            ranks = set()
            for rank, path in bins:
                if path is None:
                    continue
                source = Path(path)
                if not source.is_file():
                    continue
                if rank is None:
                    try:
                        rank = int(source.stem.rsplit("_", 1)[-1])
                    except ValueError:
                        continue
                ranks.add(int(rank))
            _dump_json(tmp / "manifest.json", manifest)

            final = self._entry_dir(key)
            final.parent.mkdir(parents=True, exist_ok=True)
            moved = False
            if not final.exists():
                try:
                    os.replace(tmp, final)
                    moved = True
                except OSError:
                    # Another rank created the destination first.
                    pass
            if not moved:
                final.mkdir(parents=True, exist_ok=True)
                for item in tmp.iterdir():
                    os.replace(item, final / item.name)

            # Serialize the shared manifest and availability marker with the
            # index so concurrent ranks cannot lose one another's rank file.
            with self._locked():
                existing = self._manifest(final) or {}
                manifest["rank_files"] = sorted(
                    set(existing.get("rank_files", ()))
                    | {f"block0_kv_rank_{rank}.bin" for rank in ranks}
                )
                manifest = self._write_manifest(final, manifest)
                marker = final / f".complete.{uuid.uuid4().hex}"
                marker.write_bytes(b"")
                os.replace(marker, final / ".complete")
                entry = self._entry_from_manifest(key, final, manifest)
                index = self._load_index_locked()
                index["entries"] = self._record_locked(entry, index)
                index = self._normalize(index["entries"])
                self._write_index_locked(index)
            return entry
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Zing Block0 disk commit failed: %s", exc)
            return None
        finally:
            if self._under_root(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
