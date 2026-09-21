# SPDX-License-Identifier: Apache-2.0
"""Same-latent native/TAEH3 metrics; subjective quality always remains pending."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def ssim(x, y):
    """Wang SSIM: 11x11 Gaussian sigma=1.5, valid window, RGB mean, range=1."""
    axis = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    kernel = torch.exp(-axis.square() / (2 * 1.5**2))
    kernel /= kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).expand(3, 1, 11, 11)

    def smooth(z):
        return F.conv2d(z, window, groups=3)

    mx, my = smooth(x), smooth(y)
    vx, vy = smooth(x * x) - mx * mx, smooth(y * y) - my * my
    covariance = smooth(x * y) - mx * my
    return (
        ((2 * mx * my + 0.01**2) * (2 * covariance + 0.03**2))
        / ((mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2))
    ).mean()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", required=True)
    p.add_argument("--taeh3", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    native = torch.load(args.native, map_location="cpu", weights_only=True)
    tae = torch.load(args.taeh3, map_location="cpu", weights_only=True)
    if not torch.equal(native["visual_latents"], tae["visual_latents"]):
        raise ValueError("Quality comparison requires identical DiT latents")
    if (native.get("audio") is None) != (tae.get("audio") is None):
        raise ValueError("Decoder switch changed audio presence")
    if native.get("audio") is not None and not torch.equal(
        native["audio"], tae["audio"]
    ):
        raise ValueError("Decoder switch changed audio samples")
    x, y = native["frames"], tae["frames"]
    if x.shape != y.shape or x.ndim != 5:
        raise ValueError("Quality comparison requires matching NCTHW frames")
    report = {
        "latent_exact": True,
        "audio_exact": True,
        "shape": list(x.shape),
        "subjective_quality": "pending user review",
        "lpips": {
            "status": "not measured; optional dependency and pretrained weights not provisioned"
        },
        "ssim_definition": "11x11 Gaussian sigma=1.5 valid-window RGB mean, range=1; no resizing",
        "frames": [],
    }
    perceptual = None
    try:
        import lpips
        from torchvision.models import AlexNet_Weights

        checkpoint = (
            Path(torch.hub.get_dir())
            / "checkpoints"
            / Path(AlexNet_Weights.IMAGENET1K_V1.url).name
        )
        if checkpoint.is_file():
            perceptual = lpips.LPIPS(net="alex").eval().to(args.device)
            report["lpips"] = {
                "status": "measured",
                "network": "alex",
                "input_range": [-1, 1],
                "resize": False,
            }
        else:
            report["lpips"] = {
                "status": "unavailable",
                "reason": "AlexNet weights are not cached; no implicit network download during evaluation",
            }
    except ImportError as error:
        report["lpips"] = {"status": "unavailable", "reason": str(error)}
    previous = None
    with torch.inference_mode():
        for index in range(x.shape[2]):
            a, b = (
                x[:, :, index].float().to(args.device),
                y[:, :, index].float().to(args.device),
            )
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("Non-finite decoded pixels")
            if min(a.min(), b.min()) < 0 or max(a.max(), b.max()) > 1:
                raise ValueError("Decoded pixels are outside [0, 1]")
            mse = (a - b).square().mean()
            row = {
                "frame": index,
                "mse": mse.item(),
                "psnr": (-10 * torch.log10(mse)).item(),
                "ssim": ssim(a, b).item(),
            }
            if perceptual is not None:
                row["lpips"] = perceptual(a * 2 - 1, b * 2 - 1).mean().item()
            if previous is not None:
                pa, pb = previous
                row["temporal_delta_mae"] = ((a - pa) - (b - pb)).abs().mean().item()
                row["native_delta_mae"] = (a - pa).abs().mean().item()
                row["taeh3_delta_mae"] = (b - pb).abs().mean().item()
            previous = a, b
            report["frames"].append(row)
    rows = report["frames"]
    report["psnr_global"] = (
        -10 * torch.tensor(sum(row["mse"] for row in rows) / len(rows)).log10().item()
    )
    report["ssim_mean"] = sum(row["ssim"] for row in rows) / len(rows)
    if perceptual is not None:
        report["lpips"]["mean"] = sum(row["lpips"] for row in rows) / len(rows)
    if len(rows) > 1:
        report["temporal_delta_mae_mean"] = sum(
            row["temporal_delta_mae"] for row in rows[1:]
        ) / (len(rows) - 1)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}))


if __name__ == "__main__":
    main()
