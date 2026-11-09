"""CPU-only process tree used through the Room image's real entrypoint."""

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path


async def direct_children():
    for code in (0, 17, 31):
        child = await asyncio.create_subprocess_exec(
            sys.executable, "-c", f"raise SystemExit({code})"
        )
        assert await child.wait() == code, "Direct child exit status was lost"


def reap_orphaned_descendants():
    asyncio.run(direct_children())
    descendants = []
    for _ in range(6):
        read_fd, write_fd = os.pipe()
        child = os.fork()
        if child == 0:
            os.close(read_fd)
            descendant = os.fork()
            if descendant == 0:
                os.close(write_fd)
                time.sleep(0.15)
                os._exit(0)
            os.write(write_fd, str(descendant).encode())
            os.close(write_fd)
            os._exit(0)
        os.close(write_fd)
        descendants.append(int(os.read(read_fd, 32)))
        os.close(read_fd)
        assert os.waitpid(child, 0) == (child, 0)
    deadline = time.monotonic() + 5
    remaining = descendants
    while remaining and time.monotonic() < deadline:
        remaining = [pid for pid in remaining if Path(f"/proc/{pid}").exists()]
        time.sleep(0.02)
    print(json.dumps({"descendants": descendants, "remaining": remaining}), flush=True)
    assert not remaining, "Room entrypoint did not reap orphaned descendants"


def main():
    mode = sys.argv[1]
    if mode == "reap":
        reap_orphaned_descendants()
    elif mode == "exit":
        raise SystemExit(42)
    elif mode == "signal":

        def stopped(_signal, _frame):
            print("room-init-signal-received", flush=True)
            raise SystemExit(23)

        signal.signal(signal.SIGTERM, stopped)
        print("room-init-signal-ready", flush=True)
        while True:
            signal.pause()
    else:
        raise ValueError("Unknown process probe")


if __name__ == "__main__":
    main()
