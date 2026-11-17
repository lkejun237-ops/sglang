# SPDX-License-Identifier: Apache-2.0
"""Frozen Golden Wire Vectors locking cross-repo wire compatibility.

`seedleap/sglang` and `seedleap/world-realtime` share zero Python packages and
interact strictly over network wire protocols. Identical frozen byte/JSON/HMAC
vectors are asserted independently in both repositories' CI suites without any
cross-repository source checkout or `sys.path` coupling.
"""

from __future__ import annotations

import struct
import zlib

import pytest

from sglang.multimodal_gen.configs import zing_h3_media_geometry
from sglang.multimodal_gen.runtime.realtime import (
    async_vae_protocol,
    chunk_telemetry,
    source_prepare,
    zing_contracts,
)
from sglang.multimodal_gen.runtime.realtime.role_pool import (
    contracts as role_pool_contracts,
)

EXPECTED_ASYNC_VAE_WIRE_HEX = (
    "85a776657273696f6e02a474797065ac6c6174656e745f6368756e6ba373657109"
    "a66865616465728eaa73657373696f6e5f6964a6736573732d31ad67656e657261"
    "74696f6e5f6964a567656e2d31aa726571756573745f6964a57265712d31ab6368"
    "756e6b5f696e64657803a56474797065a862666c6f61743136a573686170659501"
    "10043c68ab627974655f6c656e67746840a8636865636b73756dd9406665393064"
    "313364653766393264623362643765613439653462613532336365666431333634"
    "6432623339613236326466353536616430303430376230663139a86576656e745f"
    "696407ae616374696f6e5f76657273696f6e02ae70726f6d70745f76657273696f"
    "6e01b1646561646c696e655f65706f63685f6d73cf00000199c82cc07bad686173"
    "5f7265666572656e6365c3ae69735f66696e616c5f6368756e6bc2a77061796c6f"
    "6164c4400102030401020304010203040102030401020304010203040102030401"
    "020304010203040102030401020304010203040102030401020304010203040102"
    "0304"
)
EXPECTED_ASYNC_VAE_CHECKSUM = (
    "fe90d13de7f92db3bd7ea49e4ba523cefd1364d2b39a262df556ad00407b0f19"
)
EXPECTED_ROLE_POOL_HMAC_TOKEN = (
    "eyJhY3Rpb24iOiJjb25uZWN0IiwiYm9keSI6ImM3MmU1MTMzMzA3YTQ1N2NhZmIxMz"
    "U3Mzg5NWQ4MTQ4Njc3N2UwZmU1NTVhOTIxMTFmMTA5YjVhNmQxZDhhZDMiLCJlcG9j"
    "aCI6ImVwb2NoLTQyIiwiZXhwIjoxNzYwMDAwMDYwLjB9."
    "9c643279ddbfe9792da409e2bc450045090156cd4a1faa7ef6f8c5649d80ea96"
)
EXPECTED_FP_V1 = "fe6a8eed1c2768045aa2b9f37058cc60b30311596ac26374a8a9a9301c293eb7"
EXPECTED_FP_V2 = "34dd7ab74eaa0ef3c9be041dfcbc32001ab0ce0e6736895a40df6bbd089190e2"


def _deterministic_png_rgb(
    width: int = 64,
    height: int = 32,
    rgb_pixel: tuple[int, int, int] = (12, 134, 210),
) -> tuple[bytes, bytes]:
    rgb = bytes(rgb_pixel) * (width * height)
    row = b"\x00" + bytes(rgb_pixel) * width
    compressed = zlib.compress(row * height, level=0)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", crc)

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
    return png, rgb


def test_async_vae_protocol_matches_frozen_golden_vector():
    assert async_vae_protocol.PROTOCOL_VERSION == 2
    assert async_vae_protocol.DEFAULT_MAX_MESSAGE_BYTES == 67_108_864

    payload = b"\x01\x02\x03\x04" * 16
    checksum = async_vae_protocol.checksum_payload(payload)
    assert checksum == EXPECTED_ASYNC_VAE_CHECKSUM

    header = async_vae_protocol.LatentChunkHeader(
        session_id="sess-1",
        generation_id="gen-1",
        request_id="req-1",
        chunk_index=3,
        dtype="bfloat16",
        shape=(1, 16, 4, 60, 104),
        byte_length=len(payload),
        checksum=checksum,
        event_id=7,
        action_version=2,
        prompt_version=1,
        deadline_epoch_ms=1760000000123,
        has_reference=True,
        is_final_chunk=False,
    )
    wire = async_vae_protocol.encode_message(
        "latent_chunk", header=header, payload=payload, seq=9
    )
    assert wire == bytes.fromhex(EXPECTED_ASYNC_VAE_WIRE_HEX)

    decoded = async_vae_protocol.decode_message(
        bytes.fromhex(EXPECTED_ASYNC_VAE_WIRE_HEX)
    )
    parsed_header = async_vae_protocol.latent_header_from_message(decoded)
    async_vae_protocol.validate_payload(parsed_header, decoded["payload"])
    assert parsed_header == header


def test_role_pool_hmac_matches_frozen_golden_vector():
    secret = "golden-wire-parity-secret-32-chars-minimum!!"
    now = 1_760_000_000.0
    creds = role_pool_contracts.Credentials(secret, "epoch-42", clock=lambda: now)
    body = {
        "allocation_id": "alloc-99",
        "admission_id": "adm-99",
        "peers": {"encoder": "tcp://10.0.0.1:31002", "dit": "tcp://10.0.0.2:34002"},
    }
    token = creds.sign("connect", body, expires_at=now + 60)
    assert token == EXPECTED_ROLE_POOL_HMAC_TOKEN
    verified = creds.verify(EXPECTED_ROLE_POOL_HMAC_TOKEN, "connect", body)
    assert verified["epoch"] == "epoch-42"

    png_bytes, _ = _deterministic_png_rgb(64, 32)
    limits = role_pool_contracts.profile_limits(
        {
            "source_sizes": [[832, 480]],
            "target_sizes": [[832, 480], [1280, 736]],
            "max_fps": 24,
            "max_references": 12,
            "max_reference_bytes": 16_777_216,
            "max_reference_pixels": 33_554_432,
            "max_prompt_chars": 16_384,
            "max_preview_pixels": 399_360,
        }
    )
    role_pool_contracts.validate_business_message(
        {
            "type": "prefill",
            "size": "1280x736",
            "source_stream": {"width": 832, "height": 480, "fps": 24},
            "source_frame_size": {"width": 832, "height": 480},
            "prompt": "cinematic neon street",
            "references": [{"id": "ref-0", "payload": png_bytes}],
        },
        limits,
    )
    with pytest.raises(role_pool_contracts.PoolError) as exc_info:
        role_pool_contracts.validate_business_message(
            {
                "type": "init",
                "size": "1920x1080",
                "source_stream": {"width": 832, "height": 480, "fps": 24},
                "prompt": "oversized target",
            },
            limits,
        )
    assert exc_info.value.code == "PROFILE_TARGET_EXCEEDED"


def test_source_prepare_fingerprints_match_frozen_golden_vectors():
    assert source_prepare.PROTOCOL == "whitebox-startup-prepare-v1"
    assert source_prepare.V2_PROTOCOL == "whitebox-startup-prepare-v2"
    assert tuple(source_prepare.IDENTITY_FIELDS) == (
        "preparation_id",
        "source_session_id",
        "source_epoch",
        "worker_epoch",
        "condition_fingerprint",
    )

    png_bytes, rgb_bytes = _deterministic_png_rgb(64, 32)
    frame = {
        "format": "png",
        "width": 64,
        "height": 32,
        "encoded_sha256": source_prepare.rgb_digest(png_bytes),
        "rgb_sha256": source_prepare.rgb_digest(rgb_bytes),
        "payload": png_bytes,
    }
    assert source_prepare.snapshot_rgb(frame, 64, 32) == rgb_bytes

    common = dict(
        rgb=rgb_bytes,
        prompt="cinematic neon street in rain",
        references=[{"id": "ref-0", "format": "png", "payload": png_bytes}],
        seed=12345,
        source_stream={"width": 832, "height": 480, "fps": 24},
        profile_sha256="a" * 64,
    )
    fp_v1 = source_prepare.condition_fingerprint(
        **common,
        protocol=source_prepare.PROTOCOL,
        preview_width=None,
        preview_height=None,
    )
    fp_v2 = source_prepare.condition_fingerprint(
        **common,
        protocol=source_prepare.V2_PROTOCOL,
        preview_width=64,
        preview_height=32,
    )
    assert fp_v1 == EXPECTED_FP_V1
    assert fp_v2 == EXPECTED_FP_V2


def test_zing_contracts_and_telemetry_match_frozen_golden_vectors():
    assert zing_contracts.MAX_PROMPT_CHARS == 16_384
    assert zing_contracts.MAX_PROMPT_BYTES == 65_536
    assert zing_contracts.validate_prompt_text("  hello world  ") == "  hello world  "

    prov = zing_contracts.Provenance(
        zing_contracts.PixelRange(0, 12),
        (zing_contracts.SourceSpan(1, 0, 0, 12),),
        12,
    )
    assert prov.metadata() == {
        "output_frame_start": 0,
        "num_frames": 12,
        "source_alignment": [
            {
                "source_id": 1,
                "source_frame_start": 0,
                "output_frame_start": 0,
                "num_frames": 12,
            }
        ],
        "source_dependency_end": 12,
    }

    raw_receipt = {
        "cache_key": "b" * 64,
        "prompt_version": 2,
        "cache_result": "l2_hit",
        "miss_reason": None,
        "internal_secret_field": "dropped",
        "timings": {
            "elapsed_host_ms": 18.4567,
            "stages": {"qwen_encode": {"host_ms": 12.3456, "status": "skipped"}},
        },
    }
    assert zing_contracts.public_cache_receipt(raw_receipt) == {
        "cache_key": "b" * 64,
        "prompt_version": 2,
        "cache_result": "l2_hit",
        "miss_reason": None,
        "timings": {
            "elapsed_host_ms": 18.457,
            "stages": {"qwen_encode": {"host_ms": 12.346, "status": "skipped"}},
            "cuda_ms": {},
        },
    }

    assert tuple(chunk_telemetry.HOST_FIELDS) == (
        "prepare",
        "qwen_init",
        "prompt",
        "input_copy",
        "encoder",
        "bridge",
        "dit",
        "decoder",
    )
    assert set(chunk_telemetry.CHUNK_FIELDS) == {
        "request_prepare_ms",
        "scheduler_forward_ms",
        "chunk_total_ms",
        "output_pace_ms",
        "transport_encode_ms",
        "transport_write_ms",
        "model_denoise_ms",
        "model_vae_encode_ms",
        "model_vae_decode_ms",
        "model_post_decode_ms",
        "vae_queue_wait_ms",
        "vae_decode_ms",
        "raw_frame_async_enqueue_ms",
        "encoder_wait_ms",
        "encoder_process_ms",
        "encoder_return_ms",
        "dit_wait_ms",
        "dit_process_ms",
        "dit_return_ms",
        "decoder_wait_ms",
        "decoder_process_ms",
        "decoder_return_ms",
        "h3_prepare_host_ms",
        "h3_qwen_init_host_ms",
        "h3_prompt_host_ms",
        "h3_input_copy_host_ms",
        "h3_encoder_host_ms",
        "h3_bridge_host_ms",
        "h3_dit_host_ms",
        "h3_decoder_host_ms",
    }
    assert chunk_telemetry.h3_host_durations(
        {
            "zing_source_stage_host_ms": {
                "prepare_host_ms": 1.2345,
                "dit_host_ms": 42.6789,
                "secret_prompt": "ignored",
            }
        }
    ) == {"h3_prepare_host_ms": 1.234, "h3_dit_host_ms": 42.679}
    assert chunk_telemetry.sanitize_frame_stage_breakdown(
        [
            {
                "output_frame": 4,
                "encoder": {"wait_ms": 0.5123, "process_ms": 3.4567, "extra": 99},
                "dit": {"process_ms": 18.9001},
                "decoder": {"return_ms": 1.0004},
            }
        ],
        start_frame=4,
        num_frames=1,
    ) == [
        {
            "output_frame": 4,
            "encoder": {"wait_ms": 0.512, "process_ms": 3.457},
            "dit": {"process_ms": 18.9},
            "decoder": {"return_ms": 1.0},
        }
    ]

    zing_h3_media_geometry.validate_h3_media_geometry(480, 832, multiple=32)
    zing_h3_media_geometry.validate_h3_media_geometry(736, 1280, multiple=32)
    with pytest.raises(ValueError):
        zing_h3_media_geometry.validate_h3_media_geometry(100, 800, multiple=32)


# Keep these receipt vectors identical in both repositories.
@pytest.mark.parametrize(
    "cache_result", ["l1_hit", "l2_hit", "l3_hit", "mixed_hit", "miss", "bypass"]
)
@pytest.mark.parametrize("nested_lookup", [True, False])
def test_cache_receipt_provenance_matches_frozen_golden_vector(
    cache_result, nested_lookup
):
    provenance = {
        "artifact_id": "a" * 32,
        "source_layer": "l3",
        "baseline_s0_sha256": "c" * 64,
        "online_s0_sha256": "d" * 64,
    }
    raw = {
        "cache_key": "b" * 64,
        "prompt_version": 2,
        "cache_result": cache_result,
        "internal_secret_field": "dropped",
        "timings": {
            "elapsed_host_ms": 18.4567,
            "stages": {"qwen_encode": {"host_ms": 12.3456, "status": "skipped"}},
            "cuda_ms": {"qwen_encode": 1.23456, "invalid/name": 99},
        },
    }
    if nested_lookup:
        raw["lookup"] = dict(provenance, internal_path="/private/cache")
    else:
        raw.update(provenance)
    expected = {
        "cache_key": "b" * 64,
        "prompt_version": 2,
        "cache_result": cache_result,
        "miss_reason": None,
        "timings": {
            "elapsed_host_ms": 18.457,
            "stages": {"qwen_encode": {"host_ms": 12.346, "status": "skipped"}},
            "cuda_ms": {"qwen_encode": 1.235},
        },
        **provenance,
    }
    receipt = zing_contracts.public_cache_receipt(raw)
    assert receipt == expected
    # Adapter -> Gateway -> Room -> archive each sanitize the public receipt.
    for _ in range(3):
        receipt = zing_contracts.public_cache_receipt(receipt)
        assert receipt == expected


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("artifact_id", "a" * 64),
        ("source_layer", "l4"),
        ("baseline_s0_sha256", "not-a-hash"),
        ("online_s0_sha256", "D" * 64),
    ],
)
def test_cache_receipt_drops_invalid_provenance(field, invalid):
    receipt = zing_contracts.public_cache_receipt(
        {
            "cache_key": "b" * 64,
            "prompt_version": 2,
            "cache_result": "l3_hit",
            "lookup": {field: invalid},
        }
    )
    assert receipt == {
        "cache_key": "b" * 64,
        "prompt_version": 2,
        "cache_result": "l3_hit",
        "miss_reason": None,
        "timings": {"elapsed_host_ms": None, "stages": {}, "cuda_ms": {}},
    }
