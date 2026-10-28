# SPDX-License-Identifier: Apache-2.0
"""Data-only persistence for Block0 conditioning snapshots."""

from dataclasses import fields

import torch

from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import ZingKVRequest

from .packing import H3InferenceInputs

CONDITIONS_SCHEMA = "zing-block0-conditions-v1"
_RECORDS = {"inputs": H3InferenceInputs, "request": ZingKVRequest}


def _check_data(value):
    """Keep custom objects, tensor subclasses and device storage out of the file."""
    kind = type(value)
    if value is None or kind in (str, bytes, bool, int, float):
        return
    if kind is torch.Tensor and value.device.type == "cpu":
        return
    if kind in (list, tuple):
        for item in value:
            _check_data(item)
        return
    if kind is dict and all(type(key) is str for key in value):
        for item in value.values():
            _check_data(item)
        return
    raise ValueError("Block0 conditions must contain only CPU tensors and plain data")


def encode_conditions(conditions):
    """Expand the two known records without deepcopying their tensor storage."""
    if (
        type(conditions) is not dict
        or type(conditions.get("inputs")) is not H3InferenceInputs
    ):
        raise ValueError("Block0 conditions require prepared inference inputs")
    data, records = dict(conditions), {}
    for name, record in _RECORDS.items():
        value = data.get(name)
        if type(value) is record:
            data[name] = {
                field.name: getattr(value, field.name) for field in fields(record)
            }
            records[name] = record.__name__
    payload = {"schema": CONDITIONS_SCHEMA, "records": records, "conditions": data}
    _check_data(payload)
    return payload


def decode_conditions(payload):
    """Only caller-owned constructors may rebuild records after restricted loading."""
    _check_data(payload)
    if (
        type(payload) is not dict
        or set(payload) != {"schema", "records", "conditions"}
        or payload["schema"] != CONDITIONS_SCHEMA
        or type(payload["records"]) is not dict
        or payload["records"].get("inputs") != H3InferenceInputs.__name__
        or type(payload["conditions"]) is not dict
    ):
        raise ValueError("Unsupported Block0 conditions schema")
    data = dict(payload["conditions"])
    for name, tag in payload["records"].items():
        record = _RECORDS.get(name)
        value = data.get(name)
        if (
            record is None
            or tag != record.__name__
            or type(value) is not dict
            or set(value) != {field.name for field in fields(record)}
        ):
            raise ValueError("Invalid Block0 conditions record")
        if record is H3InferenceInputs:
            for tensors in (value["positive"], value["visual"], value["audio"]):
                if type(tensors) is not dict or any(
                    type(tensor) is not torch.Tensor for tensor in tensors.values()
                ):
                    raise ValueError("Invalid Block0 conditioning tensors")
            for shape_name in ("visual_shape", "audio_shape"):
                shape = value[shape_name]
                if shape is None and shape_name == "audio_shape":
                    continue
                if (
                    type(shape) is not tuple
                    or len(shape) != 3
                    or any(type(size) is not int or size < 1 for size in shape)
                ):
                    raise ValueError("Invalid Block0 conditioning shape")
        data[name] = record(**value)
    return data
