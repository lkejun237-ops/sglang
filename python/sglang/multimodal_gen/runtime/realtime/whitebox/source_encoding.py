# SPDX-License-Identifier: Apache-2.0
"""Strict software source-codec settings shared by RTC sessions and local probes."""

H264_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
)


def default_preset(codec):
    return {"h264": "veryfast", "vp9": "6", "av1": "10"}.get(codec)


def default_quality(codec):
    return {"h264": 18, "vp9": 56, "av1": 56}.get(codec)


def encoder_implementation(codec):
    return {"h264": "libx264", "vp9": "libvpx-vp9", "av1": "libsvtav1"}[codec]


def encoding_capabilities():
    return {
        codec: dict(
            available=False,
            encoderImplementation=encoder_implementation(codec),
            presets=list(
                H264_PRESETS
                if codec == "h264"
                else map(str, range(9 if codec == "vp9" else 13))
            ),
            # Baseline H.264 cannot encode x264's CRF=0 lossless mode.
            qualityRange=[0 if codec == "vp9" else 1, 51 if codec == "h264" else 63],
            qualitySemantics="crf" if codec == "h264" else "cbr-max-qp",
            minQpRange=[0, 63 if codec != "h264" else 0],
            defaultMinQp=0,
            bitrateRange=[100000, 50000000],
            gopRange=[1, 24],
            vbvMsRange=[50, 5000],
            intraRefresh=codec == "h264",
            hardware=False,
            defaultPreset=default_preset(codec),
            defaultQuality=default_quality(codec),
        )
        for codec in ("h264", "vp9", "av1")
    }


def validate_source_encoding(value):
    required = {
        "codec",
        "bitrate",
        "gop",
        "vbvMs",
        "preset",
        "quality",
        "intraRefresh",
    }
    if (
        not isinstance(value, dict)
        or not required <= set(value)
        or set(value) - required - {"minQp"}
    ):
        raise ValueError(
            "The source encoder settings are incomplete or contain unknown fields"
        )
    codec = value["codec"]
    if not isinstance(codec, str) or codec not in encoding_capabilities():
        raise ValueError("Unsupported source codec")
    limits = encoding_capabilities()[codec]
    for key, bounds in [
        ("bitrate", limits["bitrateRange"]),
        ("gop", limits["gopRange"]),
        ("vbvMs", limits["vbvMsRange"]),
        ("quality", limits["qualityRange"]),
    ]:
        if type(value[key]) is not int or not bounds[0] <= value[key] <= bounds[1]:
            raise ValueError("Invalid source encoder " + key)
    if not isinstance(value["preset"], str) or value["preset"] not in limits["presets"]:
        raise ValueError("Invalid source encoder preset")
    if (
        type(value["intraRefresh"]) is not bool
        or value["intraRefresh"]
        and codec != "h264"
    ):
        raise ValueError("Intra refresh is only supported by H.264")
    min_qp = value.get("minQp", 0)
    if (
        type(min_qp) is not int
        or not 0 <= min_qp <= value["quality"]
        or (codec == "h264" and min_qp != 0)
    ):
        raise ValueError("Invalid source encoder minQp")
    result = dict(value)
    if codec == "h264":
        result.pop("minQp", None)
    else:
        result["minQp"] = min_qp
    return result
