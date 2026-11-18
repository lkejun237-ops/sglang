"""Supervise this deployment's optional native Mooncake store process."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def owned_store(config, source):
    if not config.get("manage_cache_store", False):
        yield lambda: None
        return
    path = Path(config["dit_options"]["zing_block0_mooncake_config"])
    settings = json.loads(path.read_text())
    ready = Path(settings["tiers"]["state_directory"]) / "store-ready.json"
    env = dict(os.environ, **config.get("environment", {}))
    env["PYTHONPATH"] = str(source / "python") + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.Popen(
        [
            sys.executable,
            str(source / "deploy/whitebox/cache_store.py"),
            "--config",
            str(path),
        ],
        cwd=source,
        env=env,
        start_new_session=True,
    )

    def check():
        if process.poll() is not None:
            raise RuntimeError("Owned cache Store exited")

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, interrupted)
    try:
        deadline = time.monotonic() + config.get("startup_timeout", 900)
        while time.monotonic() < deadline:
            check()
            try:
                receipt = json.loads(ready.read_text())
                if receipt.get("pid") == process.pid:
                    if receipt.get("memory_bytes") != settings["store_memory_bytes"]:
                        raise RuntimeError("Owned cache Store memory identity differs")
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError("Owned cache Store readiness deadline exceeded")
        yield check
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        # This process group was created here; never discover/stop shared stores.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
