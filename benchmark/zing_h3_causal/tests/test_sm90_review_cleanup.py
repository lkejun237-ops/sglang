"""An exited, unreaped worker must not block verified experiment cleanup."""

import os
import sys
import time
from pathlib import Path

import pytest
from run_sm90_product_review import live_group_members


@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux /proc")
def test_zombie_group_does_not_count_as_live_worker():
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read)
        os.setpgid(0, 0)
        os.write(write, b"ready")
        os.close(write)
        os._exit(0)
    os.close(write)
    try:
        assert os.read(read, 5) == b"ready"
        deadline = time.monotonic() + 5
        path = Path(f"/proc/{pid}/stat")
        while ") Z " not in path.read_text() and time.monotonic() < deadline:
            time.sleep(0.01)
        # The old check succeeds even though no executable worker exists.
        os.killpg(pid, 0)
        assert live_group_members(pid) == []
    finally:
        os.close(read)
        os.waitpid(pid, 0)
