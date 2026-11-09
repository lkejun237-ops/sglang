# SPDX-License-Identifier: Apache-2.0
"""Input geometry shared by H3 media processing and CPU reference admission."""


def validate_h3_media_geometry(height: int, width: int, *, multiple: int) -> None:
    """Check fixed input constraints without resizing or running model inference.

    The processor still selects the bucket under its runtime area/edge budgets.
    CPU admission can reject impossible geometry before authorizing a model commit.
    """
    if min(height, width, multiple) < 1:
        raise ValueError("H3 media geometry must be positive")
    if not 0.25 <= width / height <= 4.0:
        raise ValueError(
            f"H3 supports image/video aspect ratios in [1:4,4:1], got {width}:{height}"
        )
    if min(height, width) < multiple:
        raise ValueError(
            f"H3 media {height}x{width} has no non-upscaling bucket for multiple={multiple}"
        )
