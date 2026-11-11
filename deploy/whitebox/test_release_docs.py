"""Check deployment examples against native worker parsers without loading models."""

import argparse
import importlib
import json
import re
import shlex
import sys
from pathlib import Path

import pytest


def manual_worker_arguments(module_name):
    root = Path(__file__).resolve().parents[2]
    manual = (
        root / "docs/docs/sglang-diffusion/zing/plans/whitebox-manual-releases.mdx"
    ).read_text()
    for block in re.findall(r"```bash\n(.*?)```", manual, re.S):
        tokens = shlex.split(block.replace("\\\n", " "))
        if module_name in tokens:
            return tokens[tokens.index(module_name) + 1 :]
    raise AssertionError(f"Missing worker command: {module_name}")


def test_standalone_dit_manual_uses_supported_parameters():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_worker import (
        argument_parser,
    )

    arguments = manual_worker_arguments(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_worker"
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
        values[token[1:]] if token.startswith("$") else token for token in arguments
    ]
    parsed = argument_parser().parse_args(args)
    assert parsed.service_count == 1 and parsed.sp_degree == 2
    assert parsed.zing_block0_cache_gc is True
    assert parsed.zing_initial_block_count == 0
    assert parsed.zing_block0_cache_storage == "local"


def test_observation_template_covers_every_verified_role():
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


@pytest.mark.parametrize(
    "role,module_name",
    [
        (
            "encoder",
            "sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server",
        ),
        (
            "decoder",
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker",
        ),
    ],
)
def test_independent_worker_manual_uses_native_parser(monkeypatch, role, module_name):
    arguments = manual_worker_arguments(module_name)
    values = {
        "MODEL_VIEW": "/private/model",
        "ADVERTISE_HOST": "10.0.0.2",
        "HTTP_PORT": "31000",
        "TRANSPORT_PORT": "30000",
        "RENDEZVOUS_PORT": "32000",
        "OUTPUT_PORT": "38000",
        "DECODER_CONFIG_JSON": json.dumps(
            {
                "type": "lynn_streaming",
                "checkpoint_path": "/private/lynn/decoder.pt",
                "model_dir": "/private/lynn",
                "precision": "bf16",
                "weights": "ema",
            }
        ),
    }
    arguments = [
        values[token[1:]] if token.startswith("$") else token for token in arguments
    ]
    entrypoint = importlib.import_module(module_name)
    parse_args = argparse.ArgumentParser.parse_args

    class ArgumentsParsed(Exception):
        pass

    def capture_arguments(parser, *args, **kwargs):
        # Use the production parser, then stop before model loading or registration.
        raise ArgumentsParsed(parse_args(parser, *args, **kwargs))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_arguments)
    monkeypatch.setattr(sys, "argv", [module_name, *arguments])
    with pytest.raises(ArgumentsParsed) as result:
        entrypoint.main()
    parsed = result.value.args[0]
    assert parsed.service_count == 1
    assert parsed.advertise_host == values["ADVERTISE_HOST"]
    if role == "encoder":
        assert parsed.pipeline_config_path is None
        assert parsed.session_max_lifetime_s == 1800
    else:
        assert json.loads(parsed.decoder_config)["type"] == "lynn_streaming"
        assert parsed.static_connection is False
