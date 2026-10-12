# SPDX-License-Identifier: Apache-2.0
"""Byte-accounted CPU tensor LRU and synchronous singleflight for local use."""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import Future

import torch


def tensor_bytes(condition):
    return sum(t.numel() * t.element_size() for t in condition.values())


def clone_condition(condition):
    if not isinstance(condition, dict) or any(
        not isinstance(t, torch.Tensor) or t.device.type != "cpu"
        for t in condition.values()
    ):
        raise ValueError("Conditioning cache accepts CPU tensors only")
    return {
        name: tensor.detach().contiguous().clone() for name, tensor in condition.items()
    }


class ConditioningCache:
    def __init__(self, max_bytes, *, per_tenant_bytes=None):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("Invalid conditioning cache capacity")
        self.max_bytes = max_bytes
        self.per_tenant_bytes = (
            max_bytes if per_tenant_bytes is None else per_tenant_bytes
        )
        if (
            type(self.per_tenant_bytes) is not int
            or not 0 <= self.per_tenant_bytes <= max_bytes
        ):
            raise ValueError("Invalid tenant cache capacity")
        self.current_bytes = 0
        self._entries = OrderedDict()
        self._flights = {}
        self._lock = threading.RLock()

    @staticmethod
    def _tenant(key):
        return key[0] if isinstance(key, tuple) else None

    def entry_size(self, key):
        """Inspect a charge before allocating the caller's isolated copy."""
        with self._lock:
            entry = self._entries.get(key)
            return None if entry is None else entry[1]

    def get(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return clone_condition(entry[0])

    def _remove(self, key):
        self.current_bytes -= self._entries.pop(key)[1]

    def put(self, key, condition):
        if self.max_bytes == 0:
            return
        condition = clone_condition(condition)
        size = tensor_bytes(condition)
        if size > min(self.max_bytes, self.per_tenant_bytes):
            return
        tenant = self._tenant(key)
        with self._lock:
            if key in self._entries:
                self._remove(key)
            while self.current_bytes + size > self.max_bytes:
                self._remove(next(iter(self._entries)))
            while (
                sum(
                    entry[1]
                    for k, entry in self._entries.items()
                    if self._tenant(k) == tenant
                )
                + size
                > self.per_tenant_bytes
            ):
                self._remove(
                    next(k for k in self._entries if self._tenant(k) == tenant)
                )
            self._entries[key] = (condition, size)
            self.current_bytes += size

    def get_or_encode(self, key, encode):
        if self.max_bytes == 0:
            return clone_condition(encode())
        with self._lock:
            cached = self.get(key)
            if cached is not None:
                return cached
            flight = self._flights.get(key)
            owner = flight is None
            if owner:
                flight = self._flights[key] = Future()
        if owner:
            try:
                condition = clone_condition(encode())
                self.put(key, condition)
                flight.set_result(condition)
            except BaseException as error:
                flight.set_exception(error)
            finally:
                with self._lock:
                    self._flights.pop(key, None)
        return clone_condition(flight.result())
