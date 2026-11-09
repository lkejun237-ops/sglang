# SPDX-License-Identifier: Apache-2.0
"""An RTP presentation clock; capture/UI/archive timestamps are never rewritten."""

from collections import deque
from statistics import median_high


class ModelPresentationClock:
    """Smooth selection holes with 17 source intervals, never buffered pixels.

    Every interval contributes once to the moving average, including partial
    selection groups. Unlike integrating a sparse throughput estimate, this
    preserves elapsed time without accumulating estimation bias. One instance
    belongs to one model epoch on one RTC attachment.
    """

    def __init__(self):
        self.last_source = None
        self.presentation = 0.0
        self.count = 0
        self.intervals = deque(maxlen=17)
        self.smoothing = False

    def observe(self, source_pts):
        if self.last_source is None:
            self.presentation = float(source_pts)
        else:
            delta = source_pts - self.last_source
            if delta <= 0:
                raise ValueError("Source timestamps must increase")
            if delta > 2_000_000:
                self.intervals.clear()
                self.smoothing = False
                self.presentation = max(source_pts, self.presentation + 1)
            else:
                if not self.intervals:
                    self.intervals.extend([delta] * 17)
                else:
                    self.intervals.append(delta)
                if delta > max(80_000, 2.5 * median_high(self.intervals)):
                    self.smoothing = True
                self.presentation += (
                    sum(self.intervals) / len(self.intervals)
                    if self.smoothing
                    else delta
                )
                # A real large hole or a sudden cadence change may exceed one
                # filter window. Rejoin source time instead of accruing debt.
                self.presentation = max(
                    source_pts - 1_000_000,
                    min(source_pts + 1_000_000, self.presentation),
                )
        self.last_source = source_pts
        self.count += 1
        return round(self.presentation)
