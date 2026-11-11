# SPDX-License-Identifier: Apache-2.0
"""Single-node Zing B300 recipe: thin forwarder to zing_h3_profile_launcher."""

from __future__ import annotations

from sglang.multimodal_gen.tools.zing_h3_profile_launcher import (
    B300_ENVIRONMENT as ENVIRONMENT,
)
from sglang.multimodal_gen.tools.zing_h3_profile_launcher import B300_MODULE as MODULE
from sglang.multimodal_gen.tools.zing_h3_profile_launcher import (
    COMMON_ENV,
)
from sglang.multimodal_gen.tools.zing_h3_profile_launcher import (
    b300_build_plan as build_plan,
)
from sglang.multimodal_gen.tools.zing_h3_profile_launcher import (
    b300_check_hardware as check_hardware,
)
from sglang.multimodal_gen.tools.zing_h3_profile_launcher import (
    b300_check_ports as check_ports,
)
from sglang.multimodal_gen.tools.zing_h3_profile_launcher import b300_main as main
from sglang.multimodal_gen.tools.zing_h3_profile_launcher import (
    b300_materialize as materialize,
)
from sglang.multimodal_gen.tools.zing_h3_profile_launcher import b300_parser as parser
from sglang.multimodal_gen.tools.zing_h3_profile_launcher import (
    b300_serve as _b300_serve,
)
from sglang.multimodal_gen.tools.zing_h3_profile_launcher import (
    b300_server_main as server_main,
)
from sglang.multimodal_gen.tools.zing_h3_profile_launcher import (
    b300_stop_processes as stop_processes,
)
from sglang.multimodal_gen.tools.zing_h3_profile_launcher import (
    b300_wait_conditioning as wait_conditioning,
)
from sglang.multimodal_gen.tools.zing_h3_profile_launcher import (
    policy_arguments,
    server_policy,
)


def serve(args, plan):
    return _b300_serve(
        args,
        plan,
        check_ports_fn=check_ports,
        check_hardware_fn=check_hardware,
        materialize_fn=materialize,
        wait_conditioning_fn=wait_conditioning,
        stop_processes_fn=stop_processes,
    )


__all__ = [
    "COMMON_ENV",
    "ENVIRONMENT",
    "MODULE",
    "build_plan",
    "check_hardware",
    "check_ports",
    "main",
    "materialize",
    "parser",
    "policy_arguments",
    "serve",
    "server_main",
    "server_policy",
    "stop_processes",
    "wait_conditioning",
]


if __name__ == "__main__":
    main()
