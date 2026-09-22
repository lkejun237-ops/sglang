# SPDX-License-Identifier: Apache-2.0
import pytest
from PIL import Image

from continuous_source_media import reference_payload


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP"])
@pytest.mark.parametrize("filename", ["03475dd7769de747", "misleading.jpg"])
def test_reference_format_comes_from_bytes_and_preserves_payload(
    tmp_path, image_format, filename
):
    path = tmp_path / filename
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(path, format=image_format)
    data = path.read_bytes()
    assert reference_payload(path, "<Picture 1>") == dict(
        id="<Picture 1>", format=image_format.lower(), payload=data
    )


def test_unsupported_reference_format_is_not_mislabeled(tmp_path):
    path = tmp_path / "image.webp"
    Image.new("RGB", (8, 8)).save(path, format="BMP")
    with pytest.raises(ValueError, match="JPEG, PNG, or WebP"):
        reference_payload(path, "<Picture 1>")
