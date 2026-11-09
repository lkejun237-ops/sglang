# SPDX-License-Identifier: Apache-2.0
"""Local-machine capacity leases, held by kernel locks rather than timeouts.

All cooperating processes/containers must mount the same local directory.
No model or DMA work runs under the admission lock. Disk collection uses the
same admission gate while removing entries; contenders fail closed after a
bounded wait. An exited process leaves an unlocked record which the next
admission can reap. A slow/live owner never loses its allocation to a TTL.
"""

import contextlib
import fcntl
import hashlib
import json
import os
import time
import uuid
import weakref
from pathlib import Path

_PROCESS_LEASES = []


def retain_until_exit(lease):
    # PyTorch's pinned host allocator can keep physical pages after a Tensor is
    # freed. Do not lend that memory to another process based on Python GC.
    _PROCESS_LEASES.append(lease)


class CapacityUnavailable(RuntimeError):
    pass


def _open(path):
    return os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)


def _read(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    raw = os.read(fd, 131073)
    if not raw or len(raw) > 131072:
        raise CapacityUnavailable("Invalid capacity owner record")
    return json.loads(raw)


def _write(fd, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    if len(raw) > 131072:
        raise ValueError("Capacity owner record exceeds its bound")
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    view = memoryview(raw)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("Capacity record write made no progress")
        view = view[count:]


class CapacityLease:
    def __init__(self, path, fd, value):
        self.path, self.fd, self.value = path, fd, value
        self._finalizer = weakref.finalize(self, os.close, fd)

    def close(self):
        if self.fd is not None:
            # Leave the record for locked reconciliation. Unlinking here could
            # race an inventory scan; unlocking is sufficient proof of release.
            self._finalizer()
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SharedCapacity:
    def __init__(self, root, *, lock_timeout=0.2):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_timeout = lock_timeout

    @contextlib.contextmanager
    def locked(self):
        fd = _open(self.root / "admission.lock")
        deadline = time.monotonic() + self.lock_timeout
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise CapacityUnavailable("Capacity admission is busy")
                    time.sleep(0.005)
            yield
        finally:
            os.close(fd)

    def owners(self):
        """Read/reap while holding locked(); invalid live records fail closed."""
        result = []
        for path in self.root.glob("*.lease"):
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    result.append(_read(fd))
                else:
                    path.unlink()
            finally:
                os.close(fd)
        return result

    def create(self, value):
        """Create a locked record under locked(), before allocation starts."""
        path = self.root / (uuid.uuid4().hex + ".lease")
        fd = _open(path)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _write(fd, value)
            return CapacityLease(path, fd, value)
        except BaseException:
            os.close(fd)
            path.unlink(missing_ok=True)
            raise

    def policy(self, name, value):
        """A shared domain cannot silently accept conflicting process limits."""
        path = self.root / (hashlib.sha256(name.encode()).hexdigest() + ".policy")
        fd = _open(path)
        try:
            if os.fstat(fd).st_size:
                if _read(fd) != value:
                    raise CapacityUnavailable("Shared capacity policy differs")
            else:
                _write(fd, value)
                os.fsync(fd)
        finally:
            os.close(fd)

    def reserve(self, resource, amount, limit, *, owner=None):
        with self.locked():
            return self.reserve_locked(resource, amount, limit, owner=owner)

    def reserve_locked(self, resource, amount, limit, *, owner=None):
        """Reserve as part of an admission transaction already holding locked()."""
        if (
            not isinstance(resource, str)
            or not resource
            or type(amount) is not int
            or type(limit) is not int
            or not 0 < amount <= limit
        ):
            raise ValueError("Capacity request must fit a positive integer budget")
        self.policy(resource, {"limit": limit})
        used = sum(row["bytes"] for row in self.owners() if row["resource"] == resource)
        if used + amount > limit:
            raise CapacityUnavailable("Shared capacity budget exhausted")
        return self.create(
            {"resource": resource, "bytes": amount, "owner": owner or str(os.getpid())}
        )

    def snapshot(self):
        with self.locked():
            return self.owners()
