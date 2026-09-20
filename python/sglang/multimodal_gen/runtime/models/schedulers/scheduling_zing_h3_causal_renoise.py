# SPDX-License-Identifier: Apache-2.0
"""Zing1 H3 dataward CD sampling; independent of the FastH3 Euler path."""

import math
from collections.abc import Sequence

import torch
from sglang.multimodal_gen.runtime.utils.zing_diagnostics import current_diagnostic


class ZingH3CausalRenoiseSchedule:
    """Reconstruct x0 in FP32, round to model dtype, then re-noise.

    Pass the checkpoint's unshifted integer noise grid and modality-specific
    shift. A K-entry grid performs K forwards and consumes K-1 global RNG
    draws, matching Zing/minWM FewStepRenoiseScheduler(dataward=True). The caller
    owns step_index and the order of video/audio calls. This class does not
    register a pipeline or make a bidirectional checkpoint causal.
    """

    def __init__(
        self,
        denoising_step_list: Sequence[int],
        *,
        num_train_timesteps: int,
        shift: float,
    ) -> None:
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("shift must be finite and positive")
        steps = torch.as_tensor(denoising_step_list, dtype=torch.float32)
        if steps.ndim != 1 or steps.numel() == 0:
            raise ValueError("denoising_step_list must be a non-empty list")
        if not torch.equal(steps, steps.round()):
            raise ValueError("denoising_step_list must contain integer noise levels")
        if steps[0] != num_train_timesteps:
            raise ValueError("denoising_step_list must start at num_train_timesteps")
        if not bool(((steps > 0) & (steps <= num_train_timesteps)).all()):
            raise ValueError("denoising_step_list entries must be in (0, N]")
        if steps.numel() > 1 and not bool((steps[:-1] > steps[1:]).all()):
            raise ValueError("denoising_step_list must be strictly descending")
        base_sigmas = steps / float(num_train_timesteps)
        self.sigmas = shift * base_sigmas / (1.0 + (shift - 1.0) * base_sigmas)
        self.timesteps = 1.0 - self.sigmas

    def to(self, device: torch.device | str) -> "ZingH3CausalRenoiseSchedule":
        self.sigmas = self.sigmas.to(device)
        self.timesteps = self.timesteps.to(device)
        return self

    def step(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        *,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next model input and the unrounded clean master latent."""
        if not 0 <= step_index < len(self.sigmas):
            raise IndexError("step_index is outside the denoising grid")
        if model_output.shape != sample.shape:
            raise ValueError("model_output and sample must have the same shape")
        # Keep the reference's FP64 promotion for diagnostics as well as its
        # FP32 master for BF16/FP16 inference. Do not fuse away the BF16 round.
        compute_dtype = torch.promote_types(torch.float32, sample.dtype)
        compute_dtype = torch.promote_types(compute_dtype, model_output.dtype)
        sigma = self.sigmas[step_index].to(compute_dtype)
        x0 = sample.to(compute_dtype) + sigma * model_output.to(compute_dtype)
        x0_model = x0.to(sample.dtype)
        # Opt-in parity capture only: record the actual RNG draw below without
        # drawing again. Disabled outside a diagnostic context; tensor copies
        # in that context are intentionally excluded from performance runs.
        diagnostic = current_diagnostic()
        if step_index + 1 == len(self.sigmas):
            if diagnostic:
                diagnostic.record(
                    "scheduler." + getattr(self, "diagnostic_modality", "unknown"),
                    step=step_index,
                    sample=sample,
                    prediction=model_output,
                    x0=x0,
                    x0_model=x0_model,
                    sigma=sigma,
                    next_sigma=None,
                    noise=None,
                    next_sample=x0_model,
                )
            return x0_model, x0
        noise = torch.randn_like(x0_model)
        noise_dtype = torch.promote_types(torch.float32, sample.dtype)
        next_sigma = self.sigmas[step_index + 1].to(noise_dtype)
        next_sample = (1 - next_sigma) * x0_model.to(noise_dtype)
        next_sample = next_sample + next_sigma * noise.to(noise_dtype)
        if diagnostic:
            diagnostic.record(
                "scheduler." + getattr(self, "diagnostic_modality", "unknown"),
                step=step_index,
                sample=sample,
                prediction=model_output,
                x0=x0,
                x0_model=x0_model,
                sigma=sigma,
                next_sigma=next_sigma,
                noise=noise,
                next_sample=next_sample.to(sample.dtype),
            )
        return next_sample.to(sample.dtype), x0
