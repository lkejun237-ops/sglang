# SPDX-License-Identifier: Apache-2.0
"""Bounded cache receipts joined to the existing recording identities."""

import time

PATH = "cache-events.jsonl"


def record_cache_event(recording, event):
    if recording is None:
        return False
    metadata = recording.manifest.setdefault(
        "cache_events", {"path": PATH, "accepted": 0, "dropped": 0}
    )
    offer = getattr(recording, "offer", None)
    accepted = bool(
        offer
        and offer(
            "cache_event",
            {"schema_version": 1, "observed_epoch_ms": time.time() * 1000, **event},
            optional=True,
        )
    )
    metadata["accepted" if accepted else "dropped"] += 1
    return accepted
