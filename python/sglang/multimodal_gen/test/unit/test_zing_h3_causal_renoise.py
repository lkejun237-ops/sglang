# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_zing_h3_causal_renoise import (
    ZingH3CausalRenoiseSchedule,
)


def _schedule(steps=(1000, 750, 500, 250), shift=5.0):
    return ZingH3CausalRenoiseSchedule(steps, num_train_timesteps=1000, shift=shift)


def test_video_and_audio_use_separate_shifted_noise_grids():
    video, audio = _schedule(), _schedule(shift=3.0)
    torch.testing.assert_close(video.sigmas, torch.tensor([1.0, 15 / 16, 5 / 6, 5 / 8]))
    torch.testing.assert_close(audio.sigmas, torch.tensor([1.0, 9 / 10, 3 / 4, 1 / 2]))
    assert torch.equal(video.timesteps, 1 - video.sigmas)
    assert torch.equal(audio.timesteps, 1 - audio.sigmas)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_each_nonterminal_step_consumes_one_global_noise_draw(dtype):
    schedule = _schedule()
    sample = torch.linspace(-1, 1, 35).reshape(5, 7).to(dtype)
    velocity = torch.full_like(sample, 0.37)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        for index in range(len(schedule.sigmas)):
            before = torch.get_rng_state()
            out, master = schedule.step(velocity, sample, step_index=index)
            after = torch.get_rng_state()
            expected_master = sample.float() + schedule.sigmas[index] * velocity.float()
            assert master.dtype == torch.float32
            assert torch.equal(master, expected_master)
            torch.set_rng_state(before)
            if index == len(schedule.sigmas) - 1:
                expected = expected_master.to(dtype)
            else:
                epsilon = torch.randn_like(sample)
                sigma = schedule.sigmas[index + 1]
                # Rounding x0 before mixing is part of the checkpoint contract.
                expected = (
                    (1 - sigma) * expected_master.to(dtype).float()
                    + sigma * epsilon.float()
                ).to(dtype)
            assert torch.equal(out, expected)
            assert torch.equal(after, torch.get_rng_state())
            sample = out


def test_single_step_returns_clean_latent_without_consuming_rng():
    with torch.random.fork_rng(devices=[]):
        before = torch.get_rng_state()
        out, master = _schedule((1000,)).step(
            torch.tensor([2.0]), torch.tensor([3.0]), step_index=0
        )
        assert out.item() == master.item() == 5.0
        assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize(
    "steps", [[], [999], [1000, 0], [1000, 500, 750], [1000, 500, 500], [1000, 0.5]]
)
def test_rejects_invalid_checkpoint_grid(steps):
    with pytest.raises(ValueError):
        _schedule(steps)


@pytest.mark.parametrize("shift", [0, -1, float("inf"), float("nan")])
def test_rejects_invalid_shift(shift):
    with pytest.raises(ValueError):
        _schedule(shift=shift)


@pytest.mark.parametrize("index", [-1, 4])
def test_rejects_invalid_step_without_consuming_rng(index):
    before = torch.get_rng_state()
    with pytest.raises(IndexError):
        _schedule().step(torch.ones(2), torch.ones(2), step_index=index)
    assert torch.equal(before, torch.get_rng_state())
