# SPDX-License-Identifier: Apache-2.0
"""Keep SDK blocking diagnostics collectible without exporting raw CLI stderr."""

import json
import logging
import os
import re
from contextlib import contextmanager

COLLECTOR_FD_ENV = "WORLDKIT_PRODUCTION_LOG_FD"
_IDENTITY = re.compile(r"[a-zA-Z0-9_.:-]{1,256}")
_CODE = re.compile(r"[A-Z][A-Z0-9_]{1,95}")
_IDENTITIES = (
    "projectId",
    "sessionId",
    "requestId",
    "operationId",
    "tool",
    "actionName",
    "sourceCommit",
    "sourceHash",
    "runtimeHash",
    "worldBuildHash",
    "roomId",
    "playerId",
    "playerSessionId",
    "viewportId",
    "diagnosticHash",
    "consumerId",
    "consumerEpoch",
    "segmentId",
    "conditionId",
    "occurrenceId",
    "configHash",
    "payloadHash",
    "gameBuildHash",
    "worldId",
    "sourceEpoch",
)
_NUMBERS = (
    "roomEpoch",
    "epoch",
    "connectionEpoch",
    "viewportBindingRevision",
    "toolkitGeneration",
    "controlSequence",
    "contextRevision",
    "viewRevision",
    "promptVersion",
    "durationMilliseconds",
    "executionCount",
    "omittedExecutionCount",
)
_LEVELS = {
    "failed": logging.ERROR,
    "cancelled": logging.INFO,
    "not-completed": logging.WARNING,
}


@contextmanager
def cli_log_sink(environment):
    """Give long-lived SDK children their own reference to the container stderr sink."""
    child_environment = dict(environment)
    child_environment.pop(COLLECTOR_FD_ENV, None)
    try:
        descriptor = os.dup(2)
    except OSError:
        # A missing diagnostic sink must not prevent a healthy source from starting.
        yield child_environment, ()
        return
    try:
        child_environment[COLLECTOR_FD_ENV] = str(descriptor)
        yield child_environment, (descriptor,)
    finally:
        os.close(descriptor)


def _failure(value, depth=0):
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("code"), str)
        or not _CODE.fullmatch(value["code"])
    ):
        return None
    result = {"code": value["code"]}
    fingerprint = value.get("messageHash")
    if isinstance(fingerprint, str) and re.fullmatch(r"[a-f0-9]{64}", fingerprint):
        result["messageHash"] = fingerprint
    if depth < 2:
        cause = _failure(value.get("cause"), depth + 1)
        if cause is not None:
            result["cause"] = cause
        causes = value.get("causes")
        if isinstance(causes, list):
            result["causes"] = [
                item
                for entry in causes[:3]
                if (item := _failure(entry, depth + 1)) is not None
            ]
    return result


def forward_blocked_log(value, trace_id):
    """Forward only the SDK's bounded public event fields, never arbitrary text or messages."""
    if (
        not isinstance(value, dict)
        or value.get("event") != "worldkit.production.blocked"
    ):
        return False
    if type(value.get("schemaVersion")) is not int or value["schemaVersion"] != 1:
        return False
    stage, outcome = value.get("stage"), value.get("outcome")
    if (
        not isinstance(stage, str)
        or not re.fullmatch(r"[a-z][a-z0-9.-]{0,63}", stage)
        or outcome not in _LEVELS
    ):
        return False
    failure = _failure(value.get("error"))
    if failure is None:
        return False
    row = dict(
        schemaVersion=1,
        event=value["event"],
        stage=stage,
        outcome=outcome,
        level=logging.getLevelName(_LEVELS[outcome]).lower(),
        error=failure,
    )
    for key in _IDENTITIES:
        field = value.get(key)
        if isinstance(field, str) and _IDENTITY.fullmatch(field):
            row[key] = field
    for key in _NUMBERS:
        field = value.get(key)
        if type(field) is int and 0 <= field <= 9007199254740991:
            row[key] = field
    if isinstance(trace_id, str) and _IDENTITY.fullmatch(trace_id):
        row["trace_id"] = trace_id
    try:
        logging.getLogger(__name__).log(
            _LEVELS[outcome], "%s", json.dumps(row, separators=(",", ":"))
        )
    except Exception:
        # Collection is auxiliary; a broken logging handler cannot change the CLI outcome.
        pass
    return True
