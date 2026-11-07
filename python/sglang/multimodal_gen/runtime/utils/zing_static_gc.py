# SPDX-License-Identifier: Apache-2.0
"""Keep Zing's process-lifetime heap out of hot gen2 collections.

Call once after model loading and before admitting any requests. New request
objects remain subject to normal automatic GC. Unfreeze during worker shutdown
before its existing final collection; never freeze an active session.
"""

import gc
import json
import logging
import os
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def defer_cyclic_gc():
    """Defer automatic finalizers until CUDA capture and stream cleanup finish.

    Compiled kernel finalizers can unload CUDA modules and invalidate an active
    capture. This does not suppress explicit collect() or reference counting.
    Preserve an already disabled policy, including through nested/error paths.
    """
    enabled = gc.isenabled()
    if enabled:
        gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


class StaticGCHandle:
    def __init__(self, receipt):
        self.receipt = receipt
        self.closed = False

    def close(self):
        if not self.closed:
            gc.unfreeze()
            self.closed = True


def prepare_static_gc(context):
    value = os.environ.get("SGLANG_ZING_FREEZE_STATIC_GC", "1")
    if value not in {"0", "1"}:
        raise ValueError("SGLANG_ZING_FREEZE_STATIC_GC must be 0 or 1")
    if value == "0":
        return None
    started = time.perf_counter()
    previous = gc.get_freeze_count()
    gc.collect()
    gc.freeze()
    receipt = dict(
        enabled=True,
        context=context,
        pid=os.getpid(),
        automatic_gc_enabled=gc.isenabled(),
        frozen_before=previous,
        frozen_after=gc.get_freeze_count(),
        preparation_seconds=time.perf_counter() - started,
        completed_unix_ns=time.time_ns(),
        scope="Static process heap before request admission; new request objects remain collectible",
    )
    logger.info("ZING_STATIC_GC_READY_JSON %s", json.dumps(receipt))
    return StaticGCHandle(receipt)
