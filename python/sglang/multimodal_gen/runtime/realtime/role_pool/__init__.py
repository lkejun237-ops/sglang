# SPDX-License-Identifier: Apache-2.0
"""Managed, service-granular Encoder/DiT/Decoder allocation.

Ownership split for the control-plane extraction:
- WORKER_AGENT_MODULES remain in SGLang with the GPU model workers.
- CONTROL_PLANE_MODULES move out with Gateway, Coordinator, and Room.
"""

WORKER_AGENT_MODULES = (
    "contracts",
    "worker",
    "reporting",
    "native_http",
)
CONTROL_PLANE_MODULES = (
    "admission",
    "store",
    "scheduler",
    "api",
    "admin",
    "transport",
)
