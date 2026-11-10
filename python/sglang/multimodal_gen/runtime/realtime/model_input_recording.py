# SPDX-License-Identifier: Apache-2.0
"""Content identities for accepted model inputs; no storage or GPU dependencies."""

import hashlib

PROTOCOL = "model-input-v1"


def asset(data, format):
    return dict(format=format, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def input_snapshot(kind, value, *, event_id=None, request_id=None):
    """Describe validated API values, preserving reference order and omission.

    An omitted update reference list means inherit the worker's active media;
    an explicit empty list means clear. Acceptance is not execution evidence.
    """
    result = dict(schema_version=1, protocol=PROTOCOL, kind=kind)
    if event_id is not None:
        result["event_id"] = event_id
    if request_id is not None:
        result["request_id"] = request_id
    if kind == "source.first":
        result.update(
            source_frame_index=value["source_frame_index"],
            source_frame=dict(
                asset(value["source_frame"], "rgb24"),
                width=value["width"],
                height=value["height"],
            ),
        )
        return result
    prompt = value["prompt"]
    result.update(
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        references_mode=(
            "inherit"
            if kind == "prompt.update" and "references" not in value
            else "replace"
        ),
        references=[
            dict(asset(r["payload"], r["format"]), id=r["id"])
            for r in value.get("references", []) or []
        ],
    )
    if kind in {"init", "prefill"}:
        result.update(
            seed=value.get("seed", 42),
            size=value.get("size", "832x480"),
            source_stream=dict(value["source_stream"])
            if value.get("source_stream") is not None
            else None,
        )
    if kind == "prompt.update":
        result["prompt_version"] = value.get("version")
    elif kind == "init":
        result["prompt_version"] = 1
    if kind == "prefill":
        stream = value["source_stream"]
        size = value.get("source_frame_size") or stream
        result["source_frame"] = dict(
            asset(value["source_frame"], "rgb24"),
            width=size["width"],
            height=size["height"],
        )
    return result


def accepted_input(session, snapshot):
    import time

    return dict(
        snapshot,
        capture_stage="model_api_accepted",
        model_session_id=session.id,
        generation_id=session.generation_id,
        accepted_at_unix_ns=time.time_ns(),
    )
