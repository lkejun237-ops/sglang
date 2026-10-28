# SPDX-License-Identifier: Apache-2.0
"""Deployment CLI parsing without loading model weights or creating CUDA groups."""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import FlexibleArgumentParser


def _from_dict_without_model_resolution(kwargs):
    with patch.object(
        PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
    ):
        return ServerArgs.from_dict(kwargs)


class TestZingPipelineCli(unittest.TestCase):
    def parse(self, argv):
        parser = FlexibleArgumentParser()
        # The serve subcommand owns --config, before ServerArgs adds its flags.
        parser.add_argument("--config", type=str, default="")
        ServerArgs.add_cli_args(parser)
        argv = ["--model-path", "/fake", *argv]
        with (
            patch.object(sys, "argv", ["sglang", *argv]),
            patch.object(
                PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
            ),
        ):
            args, unknown = parser.parse_known_args(argv)
            return ServerArgs.from_cli_args(args, unknown)

    def test_independent_flags_parse_to_server_args(self):
        args = self.parse(
            [
                "--zing-conditioning-mode",
                "remote",
                "--conditioning-service-url",
                "http://127.0.0.1:8093",
                "--conditioning-config",
                "client.json",
                "--conditioning-expected-profile",
                "profile.json",
                "--realtime-max-sessions",
                "1",
                "--realtime-max-sessions-per-worker",
                "1",
                "--zing-conditioning-prefetch",
                "--conditioning-result-transport",
                "nccl",
                "--zing-source-decoder-overlap",
                "--zing-source-encoder-mode",
                "overlap",
                "--zing-source-encoder-prefetch",
                "--zing-source-async-output",
                "--zing-source-work-quantum",
                "1",
                "--zing-source-joint-forward",
                "cross_chunk",
                "--zing-source-defer-final",
                "false",
                "--zing-source-prompt-joint",
                "false",
            ]
        )
        self.assertTrue(args.zing_conditioning_prefetch)
        self.assertEqual(args.conditioning_result_transport, "nccl")
        self.assertTrue(args.zing_source_decoder_overlap)
        self.assertTrue(args.zing_source_encoder_prefetch)
        self.assertTrue(args.zing_source_async_output)
        self.assertEqual(args.zing_source_encoder_mode, "overlap")
        self.assertEqual(args.zing_source_work_quantum, 1)
        self.assertEqual(args.zing_source_joint_forward, "cross_chunk")
        self.assertFalse(args.zing_source_defer_final)
        self.assertFalse(args.zing_source_prompt_joint)

    def test_explicit_false_overrides_server_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "server.json")
            with open(path, "w") as f:
                json.dump(
                    {
                        "zing_conditioning_prefetch": True,
                        "zing_source_decoder_overlap": True,
                        "zing_source_async_output": True,
                        "zing_source_encoder_prefetch": True,
                        "zing_source_work_quantum": 1,
                        "conditioning_result_transport": "nccl",
                    },
                    f,
                )
            args = self.parse(
                [
                    "--config",
                    path,
                    "--zing-conditioning-prefetch",
                    "false",
                    "--zing-source-decoder-overlap",
                    "false",
                    "--zing-source-async-output",
                    "false",
                    "--zing-source-encoder-prefetch",
                    "false",
                    "--zing-source-work-quantum",
                    "0",
                    "--conditioning-result-transport",
                    "http",
                ]
            )
        self.assertFalse(args.zing_conditioning_prefetch)
        self.assertFalse(args.zing_source_decoder_overlap)
        self.assertFalse(args.zing_source_async_output)
        self.assertFalse(args.zing_source_encoder_prefetch)
        self.assertEqual(args.zing_source_work_quantum, 0)
        self.assertEqual(args.conditioning_result_transport, "http")

    def test_omitted_overrides_preserve_legacy_inputs(self):
        args = self.parse([])
        for name in (
            "zing_source_encoder_mode",
            "zing_source_joint_forward",
            "zing_source_encoder_prefetch",
            "zing_source_async_output",
            "zing_source_work_quantum",
            "conditioning_result_transport",
            "zing_source_defer_final",
            "zing_source_prompt_joint",
        ):
            self.assertIsNone(getattr(args, name))

    def test_programmatic_invalid_quantum_is_rejected(self):
        for value in (True, "1", 2):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "quantum"),
            ):
                _from_dict_without_model_resolution(
                    {"model_path": "/fake", "zing_source_work_quantum": value}
                )
