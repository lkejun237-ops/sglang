# SPDX-License-Identifier: Apache-2.0
"""Immutable player-close identities shared by Room and Coordinator.

Receipts are issued only by the authenticated resource owner, never the browser.
The close ID is stable across retries and independent of renewable lease times.
"""

import hashlib
import json


def player_close_id(assignment):
    identity = [assignment.token, assignment.session_id, assignment.generation_id]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def sdk_cleanup_receipt(assignment, *, state, source_id, owner_epoch, view_id=None):
    if state not in {"retired", "not_created"}:
        raise ValueError("SDK cleanup has no positive terminal evidence")
    return dict(
        close_id=player_close_id(assignment),
        session_id=assignment.session_id,
        generation_id=assignment.generation_id,
        state=state,
        source_id=source_id,
        owner_epoch=owner_epoch,
        view_id=view_id,
    )


def validate_sdk_cleanup(assignment, receipt):
    if not isinstance(receipt, dict):
        raise ValueError("SDK cleanup receipt required")
    if (
        receipt.get("close_id") != player_close_id(assignment)
        or receipt.get("session_id") != assignment.session_id
        or receipt.get("generation_id") != assignment.generation_id
        or receipt.get("state") not in {"retired", "not_created"}
        or any(
            not isinstance(receipt.get(field), str)
            or not 1 <= len(receipt[field]) <= 256
            for field in ("source_id", "owner_epoch")
        )
        or (receipt["state"] == "retired" and not receipt.get("view_id"))
    ):
        raise ValueError("SDK cleanup identity or terminal evidence mismatch")
    # Persist a bounded, canonical receipt instead of arbitrary remote fields.
    result = {
        key: receipt[key]
        for key in (
            "close_id",
            "session_id",
            "generation_id",
            "state",
            "source_id",
            "owner_epoch",
        )
    }
    view_id = receipt.get("view_id")
    if view_id is not None and (
        not isinstance(view_id, str) or not 1 <= len(view_id) <= 256
    ):
        raise ValueError("Invalid retired view identity")
    result["view_id"] = view_id
    return result
