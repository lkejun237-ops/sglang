# SPDX-License-Identifier: Apache-2.0
"""Cross-process disk admission and entry pins for one local cache domain."""

import contextlib
import fcntl
import hashlib
import json
import os
import weakref
from pathlib import Path

from sglang.multimodal_gen.runtime.utils.shared_capacity import (
    CapacityUnavailable,
    SharedCapacity,
    _open,
)


def tree_bytes(root, *, exclude=None):
    """Logical bytes, conservatively counting legacy data and sparse files."""
    total = 0
    for parent, directories, files in os.walk(root, followlinks=False):
        directories[:] = [
            n
            for n in directories
            if not (Path(parent) / n).is_symlink()
            and Path(parent) / n != exclude
            and not os.path.ismount(Path(parent) / n)
        ]
        for name in files:
            path = Path(parent) / name
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                pass
    return total


class StorageBudget:
    @classmethod
    def existing(cls, namespace):
        """Attach metadata-only API/CLI deletion to its registered live domain."""
        namespace = Path(namespace).resolve()
        name = hashlib.sha256(str(namespace).encode()).hexdigest() + ".root"
        matches = [
            root
            for root in (namespace, *namespace.parents)
            if (root / ".capacity" / name).exists()
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise CapacityUnavailable("Cache namespace has multiple capacity domains")
        root = matches[0]
        path = root / ".capacity" / (hashlib.sha256(b"disk").hexdigest() + ".policy")
        return cls(root, namespace, **json.loads(path.read_text()))

    def __init__(
        self,
        root,
        namespace,
        *,
        limit,
        reserve,
        inflight,
        writers,
        collection=(1.0, 0.9, 600.0),
    ):
        self.root, self.namespace = Path(root).resolve(), Path(namespace).resolve()
        if self.root == Path("/") or not self.namespace.is_dir():
            raise ValueError("A dedicated existing cache directory is required")
        if not self.namespace.is_relative_to(self.root):
            raise ValueError("Cache directory must be inside its shared disk root")
        if self.namespace.stat().st_dev != self.root.stat().st_dev:
            raise ValueError("Cache namespace must use the budget root filesystem")
        if any(
            type(n) is not int or n < 0 for n in (limit, reserve, inflight, writers)
        ):
            raise ValueError("Storage limits must be nonnegative integers")
        if not limit or not inflight or not writers:
            raise ValueError("Storage capacity must be positive")
        self.limit, self.reserve, self.inflight, self.writers = (
            limit,
            reserve,
            inflight,
            writers,
        )
        self.state = SharedCapacity(self.root / ".capacity")
        self.policy = dict(
            limit=limit,
            reserve=reserve,
            inflight=inflight,
            writers=writers,
            collection=list(collection),
        )
        # OverlayFS without persistent xino may recycle a directory's synthetic
        # inode after its last reference disappears. Keep the actual directory
        # pinned while this budget is live; never refresh a saved inode to hide
        # a replacement. A genuine rename/replacement still fails roots().
        fd = os.open(self.namespace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            with self.state.locked():
                self.state.policy("disk", self.policy)
                for registered in self.roots():
                    if registered != self.namespace and (
                        registered.is_relative_to(self.namespace)
                        or self.namespace.is_relative_to(registered)
                    ):
                        raise CapacityUnavailable("Cache namespaces cannot overlap")
                path = self.state.root / (
                    hashlib.sha256(str(self.namespace).encode()).hexdigest() + ".root"
                )
                if path.is_symlink():
                    raise ValueError("Cache registration cannot be a symlink")
                stat = os.fstat(fd)
                current = self.namespace.stat()
                if (current.st_dev, current.st_ino) != (stat.st_dev, stat.st_ino):
                    raise CapacityUnavailable(
                        "Cache namespace changed during registration"
                    )
                if not path.exists():
                    temporary = path.with_suffix(".registering")
                    try:
                        temporary.write_text(
                            json.dumps(
                                dict(
                                    path=str(self.namespace),
                                    device=stat.st_dev,
                                    inode=stat.st_ino,
                                )
                            )
                        )
                        os.replace(temporary, path)
                    finally:
                        temporary.unlink(missing_ok=True)
        except BaseException:
            os.close(fd)
            raise
        self._namespace_finalizer = weakref.finalize(self, os.close, fd)

    def roots(self):
        result = []
        for path in self.state.root.glob("*.root"):
            registered = json.loads(path.read_text())
            root = Path(registered["path"])
            if (
                root.is_symlink()
                or root.resolve() != root
                or not root.is_relative_to(self.root)
            ):
                raise CapacityUnavailable("Registered cache directory changed")
            if not root.exists():
                continue
            stat = root.stat()
            if (stat.st_dev, stat.st_ino) != (
                registered["device"],
                registered["inode"],
            ):
                raise CapacityUnavailable("Registered cache directory was replaced")
            if not root.is_dir() or stat.st_dev != self.root.stat().st_dev:
                raise CapacityUnavailable("Registered cache filesystem changed")
            result.append(root)
        return result

    def _remaining(self, row):
        staging = Path(row["staging"])
        if not staging.is_relative_to(self.root) or staging.is_symlink():
            raise CapacityUnavailable("Storage owner path changed")
        return max(0, row["bytes"] - tree_bytes(staging))

    def reserve_write(self, key, staging, amount):
        staging = Path(staging).resolve()
        if not staging.is_relative_to(self.namespace / ".tmp"):
            raise ValueError("Write staging must belong to this cache")
        if type(amount) is not int or amount <= 0:
            raise ValueError("Storage write size must be positive")
        with self.state.locked():
            self.roots()
            rows = [r for r in self.state.owners() if r["resource"] == "disk"]
            remaining = sum(self._remaining(r) for r in rows)
            groups = {(r["namespace"], r["key"]) for r in rows}
            used = tree_bytes(self.root, exclude=self.state.root)
            st = os.statvfs(self.root)
            free = st.f_bavail * st.f_frsize
            if (
                used + remaining + amount > self.limit
                or sum(r["bytes"] for r in rows) + amount > self.inflight
                or len(groups | {(str(self.namespace), key)}) > self.writers
                or free - remaining - amount < self.reserve
            ):
                raise CapacityUnavailable(
                    "Disk write budget or free-space reserve exhausted"
                )
            return self.state.create(
                dict(
                    resource="disk",
                    bytes=amount,
                    namespace=str(self.namespace),
                    key=key,
                    staging=str(staging),
                )
            )

    def require_free_space(self, next_bytes):
        st = os.statvfs(self.root)
        if st.f_bavail * st.f_frsize - next_bytes < self.reserve:
            raise CapacityUnavailable("Filesystem reserve reached during cache write")

    @contextlib.contextmanager
    def metadata_write(self, amount):
        """Small index writes must not bypass the same disk/free-space budget."""
        with self.state.locked():
            self.roots()
            remaining = sum(
                self._remaining(r)
                for r in self.state.owners()
                if r["resource"] == "disk"
            )
            if (
                tree_bytes(self.root, exclude=self.state.root) + remaining + amount
                > self.limit
            ):
                raise CapacityUnavailable("Disk metadata budget exhausted")
            self.require_free_space(remaining + amount)
            yield

    def pin(self, key):
        with self.state.locked():
            self.roots()
            return self.state.create(
                dict(resource="read", bytes=0, namespace=str(self.namespace), key=key)
            )

    def protected(self):
        return {
            (r["namespace"], r["key"])
            for r in self.state.owners()
            if r["resource"] in {"disk", "read"}
        }

    @contextlib.contextmanager
    def collector(self):
        """Kernel election; a second instance can take over after owner exit."""
        fd = _open(self.state.root / "collector.lock")
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            yield True
        finally:
            os.close(fd)
