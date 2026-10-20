import pathlib
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.nn as nn
from safetensors.torch import save_file as safetensors_save_file

from sglang.multimodal_gen.runtime.loader.component_loaders import vae_loader
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import (
    _assign_direct_gpu_vae_state,
)
from sglang.multimodal_gen.runtime.loader.utils import load_model_state_dict


class TestH3EncoderOnlyLoading(unittest.TestCase):
    def test_decoder_is_not_constructed_and_encoder_state_is_preserved(self):
        from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
            MiniMaxH3VideoVAEConfig,
        )
        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import (
            MiniMaxH3VideoVAE,
        )
        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import klvae

        with torch.device("meta"):
            with patch.object(klvae, "ViT3DDecoder", return_value=nn.Linear(1, 1)):
                full = MiniMaxH3VideoVAE(MiniMaxH3VideoVAEConfig())
            with patch.object(
                klvae,
                "ViT3DDecoder",
                side_effect=AssertionError("Decoder must not be constructed"),
            ):
                encoder = MiniMaxH3VideoVAE(MiniMaxH3VideoVAEConfig(load_decoder=False))
        expected = {
            k: v.shape
            for k, v in full.state_dict().items()
            if not k.startswith(("decoder.", "post_quant_conv."))
        }
        self.assertEqual(
            {k: v.shape for k, v in encoder.state_dict().items()}, expected
        )
        self.assertFalse(hasattr(encoder, "decoder"))
        self.assertFalse(hasattr(encoder, "post_quant_conv"))
        self.assertEqual(encoder.prepare_decoder_autocast_weights(torch.float16), 0)

    def test_filtered_checkpoint_load_is_exact_and_still_strict(self):
        from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
            MiniMaxH3VideoVAEConfig,
        )

        config = MiniMaxH3VideoVAEConfig(load_decoder=False)
        model = nn.Module()
        model.encoder = nn.Linear(2, 2)
        model.quant_conv = nn.Linear(2, 2)
        expected = {k: torch.full_like(v, 3) for k, v in model.state_dict().items()}
        with TemporaryDirectory() as folder:
            path = str(pathlib.Path(folder) / "model.safetensors")
            safetensors_save_file(
                {
                    **expected,
                    "decoder.weight": torch.ones(3),
                    "post_quant_conv.weight": torch.ones(2),
                },
                path,
            )
            filtered = dict(
                vae_loader.safetensors_weights_iterator(
                    [path], key_filter=vae_loader._vae_checkpoint_key_filter(config)
                )
            )
            load_model_state_dict(model, filtered, strict=True)
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, expected[k], rtol=0, atol=0)
            _assign_direct_gpu_vae_state(
                model,
                [path],
                component_name="video_vae",
                device=torch.device("cpu"),
                vae_config=config,
            )
            safetensors_save_file({**expected, "encoder.typo": torch.ones(1)}, path)
            with self.assertRaises(ComponentCheckpointUnsupportedError):
                _assign_direct_gpu_vae_state(
                    model,
                    [path],
                    component_name="video_vae",
                    device=torch.device("cpu"),
                    vae_config=config,
                )
            safetensors_save_file({"encoder.weight": expected["encoder.weight"]}, path)
            with self.assertRaises(ComponentCheckpointUnsupportedError):
                _assign_direct_gpu_vae_state(
                    model,
                    [path],
                    component_name="video_vae",
                    device=torch.device("cpu"),
                    vae_config=config,
                )
        self.assertIsNone(
            vae_loader._vae_checkpoint_key_filter(MiniMaxH3VideoVAEConfig())
        )

    def test_native_backend_rejects_missing_decoder(self):
        from types import SimpleNamespace

        from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
            ZingH3CausalPipelineConfig,
        )

        config = ZingH3CausalPipelineConfig(
            video_decoder_backend="native", zing_adaln_mode="resident"
        )
        config.vae_config.load_decoder = False
        with self.assertRaisesRegex(ValueError, "requires the Lynn"):
            config.validate_server_args(SimpleNamespace())

    def test_lynn_selection_omits_native_decoder_without_extra_config(self):
        from types import SimpleNamespace

        from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
            ZingH3CausalPipelineConfig,
        )

        config = ZingH3CausalPipelineConfig(zing_adaln_mode="resident")
        config.update_pipeline_config(
            {"decoder_config": {"checkpoint_path": "/test/lynn.pt", "model_dir": "/m"}}
        )
        config.validate_server_args(
            SimpleNamespace(
                attention_backend=None,
                resolve_component_attention_backend=lambda _: (None, None),
            )
        )
        self.assertEqual(config.decoder_type, "lynn_streaming")
        self.assertFalse(config.vae_config.load_decoder)
        self.assertTrue(config.vae_config.load_encoder)

    def test_native_selection_keeps_its_decoder(self):
        from types import SimpleNamespace

        from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
            ZingH3CausalPipelineConfig,
        )

        config = ZingH3CausalPipelineConfig(
            video_decoder_backend="native", zing_adaln_mode="resident"
        )
        config.validate_server_args(
            SimpleNamespace(
                attention_backend=None,
                resolve_component_attention_backend=lambda _: (None, None),
            )
        )
        self.assertTrue(config.vae_config.load_decoder)
