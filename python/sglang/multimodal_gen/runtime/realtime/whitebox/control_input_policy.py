# SPDX-License-Identifier: Apache-2.0
"""Gameplay transport policy shared by both existing SDK control relays."""

import json

CONTROL_GAMEPLAY_MAX_AGE_SECONDS = 0.2
CONTROL_MAX_QUEUED_MESSAGES = 64
GAMEPLAY_TYPES = frozenset(("input.state", "input.action", "input.semantic.batch"))


def is_gameplay_message(payload):
    try:
        message = json.loads(payload)
        return isinstance(message, dict) and message.get("type") in GAMEPLAY_TYPES
    except (ValueError, TypeError, RecursionError):
        return False  # The SDK remains the packet/schema authority.
