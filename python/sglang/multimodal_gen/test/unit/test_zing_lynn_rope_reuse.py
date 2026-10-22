# SPDX-License-Identifier: Apache-2.0
"""RoPE geometry is local to a tile batch and ages on each decoder step."""

import unittest
from unittest.mock import patch

import torch
from msgspec.structs import replace
from test_zing_lynn_stream import ARCHITECTURE, CONFIG

from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.model import (
    StreamingDecoder,
)


class TestLynnRopeReuse(unittest.TestCase):
    def make_model(self, window=3, tile_batch_size=0):
        architecture = {
            **ARCHITECTURE,
            "decoder_num_layers": 3,
            "decoder_num_register_tokens": 4,
        }
        model = StreamingDecoder(
            replace(
                CONFIG,
                temporal_window=window,
                tile_overlap=0,
                tile_batch_size=tile_batch_size,
            ),
            architecture,
        ).eval()
        with torch.no_grad():
            # Zero residual scales would hide attention history from outputs.
            for parameter in model.parameters():
                if not parameter.any():
                    parameter.normal_(std=0.5)
        model.compute_dtype = torch.float32
        model.attention = lambda q, k, v: (
            torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)
        )
        return model

    @torch.no_grad()
    def test_startup_aging_special_tokens_and_new_stream(self):
        torch.manual_seed(352)
        model = self.make_model()
        state = model.init_stream()
        z = torch.randn(1, 24, 1, 2, 2)
        for step in range(7):
            with patch.object(
                model.decoder.rope, "forward", wraps=model.decoder.rope.forward
            ) as rope:
                pixels, state = model.decode_step(z, state)
            self.assertEqual(rope.call_count, 1 if step == 0 else 2)
            self.assertEqual(pixels.shape[2], model.protocol.new_frames(step))
            query = rope.call_args_list[0].args[0]
            self.assertEqual(query.shape, (1, 9, 3))
            self.assertTrue(torch.equal(query[:, -5:], torch.zeros(1, 5, 3)))
            if step:
                key = rope.call_args_list[1].args[0]
                history = min(step, 2)
                expected_times = torch.tensor(
                    [
                        2 * -d / model.protocol.teacher_window
                        for d in range(history, 0, -1)
                    ]
                ).repeat_interleave(4)
                torch.testing.assert_close(key[0, : history * 4, 0], expected_times)
                torch.testing.assert_close(key[:, history * 4 :], query)
            for layers in state.tiles.values():
                for k, v in layers:
                    self.assertEqual(k.shape[1], min(step + 1, 2) * 4)
                    self.assertEqual(k.shape, v.shape)
                    for cached in (k, v):
                        self.assertEqual(
                            cached.untyped_storage().nbytes(),
                            cached.numel() * cached.element_size(),
                        )
        # Geometry memoization must not survive a call or hold the prior session.
        with patch.object(
            model.decoder.rope, "forward", wraps=model.decoder.rope.forward
        ) as rope:
            model.decode_step(z, model.init_stream())
        self.assertEqual(rope.call_count, 1)

    @torch.no_grad()
    def test_no_history_window(self):
        model = self.make_model(window=1)
        state = model.init_stream()
        for _ in range(3):
            with patch.object(
                model.decoder.rope, "forward", wraps=model.decoder.rope.forward
            ) as rope:
                _, state = model.decode_step(torch.zeros(1, 24, 1, 2, 2), state)
            self.assertEqual(rope.call_count, 1)
            self.assertTrue(
                all(
                    k.shape[1] == v.shape[1] == 0
                    for layers in state.tiles.values()
                    for k, v in layers
                )
            )

    @torch.no_grad()
    def test_geometry_reuse_respects_partial_tile_batches(self):
        model = self.make_model(tile_batch_size=2)
        state = model.init_stream()
        attend = model.attention
        batches = []

        def checked_attention(q, k, v):
            self.assertEqual(q.shape[0], k.shape[0])
            self.assertEqual(q.shape[0], v.shape[0])
            batches.append(q.shape[0])
            return attend(q, k, v)

        model.attention = checked_attention
        # Three spatial tiles, two videos: batches of four and then two.
        for _ in range(3):
            _, state = model.decode_step(torch.randn(2, 24, 1, 5, 2), state)
        self.assertEqual(batches, [4, 4, 4, 2, 2, 2] * 3)


if __name__ == "__main__":
    unittest.main()
