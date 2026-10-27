# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU command rendezvous for the dedicated encoder worker."""

from sglang.multimodal_gen.runtime.disaggregation.transport.worker_control import (
    RoleCommandControl,
    RoleControlError,
)

EncoderControlError = RoleControlError
EncoderCommandControl = RoleCommandControl
