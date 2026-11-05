# SPDX-License-Identifier: Apache-2.0
"""NCCL environment policy shared by the standalone Zing workers."""

import logging
import os

logger = logging.getLogger(__name__)


def configure_zing_nccl():
    """Apply before any NCCL communicator is initialized, on every worker rank.

    A channel cap intended for an external bridge also caps DiT's SP group.
    Keep a common minimum across roles without imposing a maximum. Clear the
    legacy MAX alias and CTA overrides so they cannot restore inherited limits.
    Network/interface settings are left to the deployment.
    """
    cleared = (
        "NCCL_MAX_NRINGS",
        "NCCL_MAX_NCHANNELS",
        "NCCL_MIN_CTAS",
        "NCCL_MAX_CTAS",
    )
    previous = {
        key: os.environ[key]
        for key in ("NCCL_MIN_NCHANNELS", *cleared)
        if key in os.environ
    }
    os.environ["NCCL_MIN_NCHANNELS"] = "2"
    for key in cleared:
        os.environ.pop(key, None)
    logger.warning(
        "Zing NCCL startup policy: NCCL_MIN_NCHANNELS=2; channel/CTA upper "
        "bounds and other channel/CTA overrides unset; previous values=%s. "
        "NCCL config files must not reintroduce these overrides.",
        previous,
    )
