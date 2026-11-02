# SPDX-License-Identifier: Apache-2.0
"""Single-consumer, event-loop-local work queue with configurable polling.

The processor owns consumption (including batching) and must be async. Admission
limits remain with the caller; this queue never silently drops input. Existing
pipelines can split wait_ready/process to prepare their request between the two.
Cancellation and processor errors propagate to the owning lifecycle.
"""

import asyncio
import math
from collections import deque


class PollingTaskQueue(deque):
    def __init__(self, processor, *, ready=None, poll_interval_s=0.005, arrange=None):
        super().__init__()
        if (
            isinstance(poll_interval_s, bool)
            or not isinstance(poll_interval_s, (int, float))
            or not math.isfinite(poll_interval_s)
            or poll_interval_s <= 0
        ):
            raise ValueError("Polling interval must be a positive finite number")
        if not callable(processor) or (ready is not None and not callable(ready)):
            raise TypeError("Processor and readiness predicate must be callable")
        if arrange is not None and not callable(arrange):
            raise TypeError("Input arranger must be callable")
        self.processor = processor
        self.arrange = arrange
        self.ready = ready if ready is not None else bool
        self.poll_interval_s = poll_interval_s

    def append(self, item):
        """Arrange one input into zero or more runnable tasks, in order.

        The default admits the original item unchanged. A stateful arranger
        owns its staging buffer and must be cleared by its session owner.
        """
        if self.arrange is None:
            super().append(item)
        else:
            super().extend(self.arrange(item))

    def extend(self, items):
        for item in items:
            self.append(item)

    async def wait_ready(self):
        while not self.ready(self):
            await asyncio.sleep(self.poll_interval_s)

    async def process(self, *args, **kwargs):
        """Process work selected after wait_ready, without rechecking its queue.

        Request preparation may already have removed the selected input. The
        caller must serialize this step and retain its existing cleanup rules.
        """
        return await self.processor(self, *args, **kwargs)

    async def run_once(self, *args, **kwargs):
        await self.wait_ready()
        return await self.process(*args, **kwargs)
