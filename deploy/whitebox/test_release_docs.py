"""Exercise the operator template against the actual standalone DiT parser."""

import re
import shlex
from pathlib import Path


def test_standalone_dit_manual_uses_supported_parameters():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_worker import (
        argument_parser,
    )

    root = Path(__file__).resolve().parents[2]
    manual = (
        root / "docs/docs/sglang-diffusion/zing/plans/whitebox-manual-releases.mdx"
    ).read_text()
    command = next(
        block
        for block in re.findall(r"```bash\n(.*?)```", manual, re.S)
        if "zing_h3.dit_worker" in block
    )
    tokens = shlex.split(command.replace("\\\n", " "))
    start = (
        next(
            i for i, token in enumerate(tokens) if token.endswith("zing_h3.dit_worker")
        )
        + 1
    )
    values = {
        "SP": "2",
        "HTTP_PORT": "33000",
        "TRANSPORT_PORT": "34000",
        "RENDEZVOUS_PORT": "35000",
        "MEM_FRACTION": "0.8",
        "ADVERTISE_HOST": "10.0.0.2",
        "MODEL_VIEW": "/private/model",
        "PIPELINE": "/private/pipeline.json",
        "CACHE_DIR": "/private/cache/model",
        "CACHE_BUDGET_ROOT": "/private/cache",
        "CAPACITY_DIR": "/private/capacity",
    }
    args = [
        values[token[1:]] if token.startswith("$") else token
        for token in tokens[start:]
    ]
    parsed = argument_parser().parse_args(args)
    assert parsed.service_count == 1 and parsed.sp_degree == 2
    assert parsed.zing_block0_cache_gc is True
    assert parsed.zing_initial_block_count == 3
    assert parsed.zing_block0_cache_storage == "local"


def test_observation_template_covers_every_verified_role():
    import json

    from release import RELEASE_ROLES

    root = Path(__file__).resolve().parents[2]
    text = (
        root / "docs/docs/sglang-diffusion/zing/plans/whitebox-manual-releases.mdx"
    ).read_text()
    observations = next(
        json.loads(block)
        for block in re.findall(r"```json\n(.*?)```", text, re.S)
        if '"platform_image"' in block
    )
    assert set(observations) == set(RELEASE_ROLES)
