# SPDX-License-Identifier: Apache-2.0
"""Match model-side receipts to bounded original bytes before archive admission."""

import copy
import hashlib
import json
import time

from ..model_input_recording import PROTOCOL, input_snapshot

PATH = "model-inputs/events.jsonl"


class ModelInputCapture:
    def __init__(self, recording, session_id, generation_id):
        self.recording = recording
        self.identity = dict(model_session_id=session_id, generation_id=generation_id)
        self.pending = {}
        self.closed = False
        if not hasattr(recording, "model_input_captures"):
            recording.model_input_captures = []
        recording.model_input_captures.append(self)
        self.pending_bytes = 0
        self.stream = None
        # Metadata only: original image bytes are released after archive enqueue.
        self.confirmed_inputs = {}
        self.confirmed_bytes = 0
        self.reference_assets = {}
        self.output_observations = {}
        self.last_output_key = object()
        self.summary = recording.manifest.setdefault("model_inputs", {})
        for key, value in dict(
            protocol=PROTOCOL,
            path=PATH,
            status="pending",
            accepted=0,
            applied=0,
            unconfirmed=0,
            unresolved_applied=0,
            failed=0,
        ).items():
            self.summary.setdefault(key, value)

    def failed(self, code):
        self.summary.update(status="incomplete", last_error=code)
        self.summary["failed"] += 1

    def sent(self, message):
        try:
            kind = message.get("type")
            event = message.get("event_id")
            if kind in {"init", "prefill"}:
                self.stream = message.get("source_stream")
                value = message
                key = (kind, message.get("request_id"))
            elif kind == "event" and message.get("kind") in {"prompt", "prompt.update"}:
                kind = "prompt.update"
                value = message["payload"]
                if isinstance(value, str):
                    value = dict(
                        prompt=value,
                        **(
                            {"references": message["references"]}
                            if "references" in message
                            else {}
                        ),
                    )
                else:
                    value = dict(value, prompt=value["text"])
                key = (kind, event)
            elif (
                kind == "event"
                and message.get("kind") == "source.append"
                and message["payload"]["source_frame_start"] == 0
            ):
                kind = "source.first"
                size = self.stream
                value = dict(
                    width=size["width"],
                    height=size["height"],
                    source_frame_index=0,
                    source_frame=message["payload"]["data"][
                        : size["width"] * size["height"] * 3
                    ],
                )
                key = (kind, event)
            else:
                return
            snapshot = input_snapshot(
                kind, value, event_id=event, request_id=message.get("request_id")
            )
            assets = {}
            for r, descriptor in zip(
                value.get("references", []) or [], snapshot.get("references", [])
            ):
                assets[self.asset_path(descriptor)] = r["payload"]
            if "source_frame" in snapshot:
                assets[self.asset_path(snapshot["source_frame"])] = value[
                    "source_frame"
                ]
            size = sum(len(b) for b in assets.values())
            if (
                key in self.pending
                or len(self.pending) >= 4
                or self.pending_bytes + size > 64 << 20
            ):
                raise ValueError("MODEL_INPUT_CAPTURE_CAPACITY")
            self.pending[key] = snapshot, assets, size
            self.pending_bytes += size
        except (KeyError, TypeError, ValueError):
            self.failed("MODEL_INPUT_CAPTURE_INVALID")

    @staticmethod
    def asset_path(descriptor):
        return (
            "model-inputs/assets/" + descriptor["sha256"] + "." + descriptor["format"]
        )

    def received(self, message):
        receipt = message.get("model_input")
        if receipt is None:
            # Rejections are not accepted model input. Retain only the fact that
            # a submitted input was not confirmed; never manufacture a receipt.
            if message.get("state") == "rejected" or message.get("type") == "error":
                key = ("prompt.update", message.get("event_id"))
                pending = self.pending.pop(key, None)
                if pending:
                    self.pending_bytes -= pending[2]
                    self.summary["unconfirmed"] += 1
                    self.summary["status"] = "incomplete"
                rejected = self.confirmed_inputs.pop(message.get("event_id"), None)
                if rejected is not None:
                    self.confirmed_bytes -= rejected[1]
            return
        try:
            kind = receipt["kind"]
            key = (
                kind,
                receipt.get("event_id")
                if kind in {"prompt.update", "source.first"}
                else receipt.get("request_id"),
            )
            expected, assets, size = self.pending.pop(key)
            self.pending_bytes -= size
            if any(
                receipt.get(k) != v for k, v in {**expected, **self.identity}.items()
            ):
                raise ValueError("MODEL_INPUT_RECEIPT_MISMATCH")
            if receipt.get("capture_stage") != "model_api_accepted":
                raise ValueError("MODEL_INPUT_RECEIPT_STAGE")
            if type(receipt.get("accepted_at_unix_ns")) is not int:
                raise ValueError("MODEL_INPUT_RECEIPT_TIME")
            # Accept only the defined receipt fields; never persist arbitrary
            # server fields or caller-provided archive paths.
            event = dict(
                expected,
                **self.identity,
                capture_stage=receipt["capture_stage"],
                accepted_at_unix_ns=receipt["accepted_at_unix_ns"],
            )
            event = copy.deepcopy(event)
            for descriptor in event.get("references", []):
                descriptor["path"] = self.asset_path(descriptor)
            if "source_frame" in event:
                event["source_frame"]["path"] = self.asset_path(event["source_frame"])
            offer = getattr(self.recording, "offer", None)
            if offer is None or not offer("model_input", (event, assets)):
                raise ValueError("MODEL_INPUT_ARCHIVE_UNAVAILABLE")
            self.summary["accepted"] += 1
            if kind in {"init", "prompt.update"}:
                size = len(json.dumps(event, ensure_ascii=False).encode())
                if (
                    len(self.confirmed_inputs) >= 64
                    or self.confirmed_bytes + size > 2 << 20
                ):
                    raise ValueError("MODEL_INPUT_HISTORY_CAPACITY")
                for descriptor in event["references"]:
                    asset_key = (
                        descriptor["id"],
                        descriptor["sha256"],
                        descriptor["format"],
                    )
                    if (
                        asset_key not in self.reference_assets
                        and len(self.reference_assets) >= 192
                    ):
                        raise ValueError("MODEL_INPUT_ASSET_CAPACITY")
                    self.reference_assets[asset_key] = descriptor
                self.confirmed_inputs[event.get("event_id")] = (event, size)
                self.confirmed_bytes += size
                self._resolve_output(event.get("event_id"))
        except (KeyError, TypeError, ValueError):
            self.failed("MODEL_INPUT_RECEIPT_INVALID")

    def applied(self, message, prompt, references, prompt_version):
        """Called after the real model output and its source range are validated.

        Room's output-condition owner supplies the effective reference snapshot.
        API acceptance alone (including another Prefill) never changes it.
        """
        event_id = message.get("event_id")
        key = (event_id, prompt_version)
        if key == self.last_output_key:
            return
        self.last_output_key = key
        try:
            if len(self.output_observations) >= 4:
                raise ValueError("MODEL_OUTPUT_CAPTURE_CAPACITY")
            if (
                type(message.get("output_frame_start")) is not int
                or type(message.get("num_frames")) is not int
                or message["num_frames"] < 1
            ):
                raise ValueError("MODEL_OUTPUT_CAPTURE_RANGE")
            self.output_observations[event_id] = dict(
                prompt=prompt,
                references=copy.deepcopy(references),
                prompt_version=prompt_version,
                first_output_frame=message["output_frame_start"],
                num_frames=message["num_frames"],
                observed_at_unix_ns=time.time_ns(),
            )
            # Input and decoder control channels may race. Wait for the same
            # accepted event, not for a guessed previous input or Prefill.
            self._resolve_output(event_id)
        except (KeyError, TypeError, ValueError):
            self.failed("MODEL_OUTPUT_INPUT_UNRESOLVED")

    def _resolve_output(self, event_id):
        if (
            event_id not in self.output_observations
            or event_id not in self.confirmed_inputs
        ):
            return
        observed = self.output_observations.pop(event_id)
        accepted, size = self.confirmed_inputs.pop(event_id)
        self.confirmed_bytes -= size
        if accepted["prompt"] != observed["prompt"] or (
            accepted.get("prompt_version") is not None
            and accepted["prompt_version"] != observed["prompt_version"]
        ):
            raise ValueError("MODEL_OUTPUT_PROMPT_MISMATCH")
        resolved = [
            copy.deepcopy(
                self.reference_assets[
                    (r["id"], r["sha256"], r["path"].rsplit(".", 1)[-1])
                ]
            )
            for r in observed["references"]
        ]
        if (
            accepted["references_mode"] == "replace"
            and resolved != accepted["references"]
        ):
            raise ValueError("MODEL_OUTPUT_REFERENCES_MISMATCH")
        event = dict(
            accepted,
            capture_stage="model_output_observed",
            input_references_mode=accepted["references_mode"],
            references_mode="resolved",
            references=resolved,
            prompt_version=observed["prompt_version"],
            first_output_frame=observed["first_output_frame"],
            num_frames=observed["num_frames"],
            observed_at_unix_ns=observed["observed_at_unix_ns"],
        )
        if not self.recording.offer("model_input", (event, {})):
            raise ValueError("MODEL_OUTPUT_ARCHIVE_UNAVAILABLE")
        self.summary["applied"] += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.recording.model_input_captures.remove(self)
        self.summary["unconfirmed"] += len(self.pending)
        self.summary["unresolved_applied"] += len(self.output_observations)
        self.pending.clear()
        self.output_observations.clear()
        self.confirmed_inputs.clear()
        self.reference_assets.clear()
        self.confirmed_bytes = 0
        self.pending_bytes = 0
        self.summary["status"] = (
            "incomplete"
            if self.summary["failed"]
            or self.summary["unconfirmed"]
            or self.summary["unresolved_applied"]
            else "confirmed"
        )


def archive_model_input(event, assets, blob, append, known_assets):
    """Worker-only byte verification and upload; called after fail-open enqueue."""
    descriptors = list(event.get("references", []))
    if "source_frame" in event:
        descriptors.append(event["source_frame"])
    expected = {ModelInputCapture.asset_path(d) for d in descriptors}
    resolved = event.get("capture_stage") == "model_output_observed"
    if resolved:
        if assets or event.get("references_mode") != "resolved":
            raise ValueError("Resolved input must reuse verified assets")
        for d in descriptors:
            if d["path"] != ModelInputCapture.asset_path(d) or known_assets.get(
                d["path"]
            ) != (d["sha256"], d["bytes"]):
                raise ValueError("Resolved model input asset is unavailable")
        append(event)
        return
    if expected != set(assets):
        raise ValueError("Model input assets do not match the receipt")
    for d in descriptors:
        data = assets[d["path"]]
        if len(data) != d["bytes"] or hashlib.sha256(data).hexdigest() != d["sha256"]:
            raise ValueError("Model input asset identity mismatch")
    for path, data in assets.items():
        blob(path, data)
    for d in descriptors:
        known_assets[d["path"]] = (d["sha256"], d["bytes"])
    append(event)
