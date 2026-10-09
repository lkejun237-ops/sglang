# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only monitoring budgets for resident cells containing a full trace.

Trace serialization can take longer than production session/heartbeat limits.
This launcher changes no model, request, profiler coverage or production default.
"""

import functools
import multiprocessing as mp
import os
import sys
from types import SimpleNamespace

PROFILE_MONITOR_SECONDS = 3600


def configure_monitoring(launch):
    # Replace only this launch module's reference, not the shared uvicorn module.
    launch.uvicorn = SimpleNamespace(
        run=functools.partial(
            launch.uvicorn.run, ws_ping_timeout=PROFILE_MONITOR_SECONDS
        )
    )


def main():
    from sglang.multimodal_gen.runtime import launch_server as launch

    mp.set_start_method("spawn", force=True)
    args = launch.prepare_server_args(sys.argv[1:])
    configure_monitoring(launch)
    try:
        launch.dispatch_launch(args)
    finally:
        launch.kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
