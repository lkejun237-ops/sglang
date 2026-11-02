# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy

import pytest
from encoder_test_master import MomentsSink


def sink():
    result = MomentsSink()
    result.negotiate(
        dict(
            output_kind="source_moments",
            session_id="s",
            generation_id="g",
            encoded_components=["source_vae", "qwen"],
        )
    )
    return result


def condition(event=None):
    return dict(
        type="conditioning",
        session_id="s",
        generation_id="g",
        event_id=event,
        prompt_version=event,
        tensors=dict(
            context=dict(dtype="bfloat16", shape=[2, 3], payload=bytes(12)),
            row_role=dict(dtype="int64", shape=[2], payload=bytes(16)),
        ),
    )


def test_conditions_saved_separately_without_advancing_source_watermarks(tmp_path):
    receiver = sink()
    with (tmp_path / receiver.filename).open("wb") as file:
        for event in [None, 4, 7]:
            receipt = asyncio.run(receiver.receive_conditioning(condition(event), file))
            for name, item in receipt["tensors"].items():
                assert (tmp_path / item["file"]).read_bytes() == condition()["tensors"][
                    name
                ]["payload"]
    assert receiver.frames == receiver.latents == receiver.offset == 0
    assert (tmp_path / receiver.filename).read_bytes() == b""
    assert [c["event_id"] for c in receiver.summary()["conditions"]] == [None, 4, 7]


@pytest.mark.parametrize(
    "corruption", ["session", "generation", "bytes", "rows", "manifest", "dtype"]
)
def test_invalid_condition_not_persisted(tmp_path, corruption):
    message = copy.deepcopy(condition())
    if corruption == "session":
        message["session_id"] = "other"
    elif corruption == "generation":
        message["generation_id"] = "old"
    elif corruption == "bytes":
        message["tensors"]["context"]["payload"] = bytes(11)
    elif corruption == "rows":
        message["tensors"]["row_role"].update(shape=[1], payload=bytes(8))
    elif corruption == "manifest":
        del message["tensors"]["row_role"]
    else:
        message["tensors"]["row_role"]["dtype"] = "float32"
    receiver = sink()
    with (tmp_path / receiver.filename).open("wb") as file:
        with pytest.raises(ValueError):
            asyncio.run(receiver.receive_conditioning(message, file))
    assert not receiver.conditions
    assert not list(tmp_path.glob("condition-*.bin"))


@pytest.mark.parametrize("event", [None, 2, 3])
def test_duplicate_or_stale_condition_rejected(tmp_path, event):
    receiver = sink()
    with (tmp_path / receiver.filename).open("wb") as file:
        asyncio.run(receiver.receive_conditioning(condition(3), file))
        with pytest.raises(ValueError):
            asyncio.run(receiver.receive_conditioning(condition(event), file))
    assert len(receiver.conditions) == 1


def test_qwen_negotiation_requires_condition_before_completion():
    receiver = sink()
    state = dict(
        ended=True, completed=False, submitted=0, accepted=0, consumed=0, received=0
    )
    message = dict(
        output_kind="source_moments",
        session_id="s",
        generation_id="g",
        source_frames=0,
        latent_frames=0,
    )
    with pytest.raises(ValueError, match="incomplete"):
        receiver.complete(message, state)


def test_dit_latents_record_actual_prompt_epoch_and_output_coverage(tmp_path):
    from encoder_test_master import DiTLatentSink

    receiver = DiTLatentSink()
    receiver.negotiate(
        dict(
            output_kind="dit_latents",
            session_id="s",
            generation_id="g",
            encoded_components=["source_vae", "qwen"],
        )
    )
    with (tmp_path / receiver.filename).open("wb") as file:
        for i, event in enumerate([None, None, 3]):
            message = dict(
                type="dit.latents",
                format="dit_latents",
                session_id="s",
                generation_id="g",
                dtype="bfloat16",
                shape=[1, 2, 1, 1, 1],
                payload=bytes(4),
                latent_start=i,
                latent_end=i + 1,
                output_frame_start=i,
                num_frames=1,
                source_dependency_end=3,
                event_id=event,
            )
            assert (
                asyncio.run(receiver.receive(message, file, accepted_frames=3)) == i + 1
            )
        with pytest.raises(ValueError, match="coverage or prompt"):
            asyncio.run(
                receiver.receive(
                    dict(
                        message,
                        output_frame_start=3,
                        latent_start=3,
                        latent_end=4,
                        event_id=2,
                    ),
                    file,
                    accepted_frames=4,
                )
            )
    receiver.complete(
        dict(
            output_kind="dit_latents",
            session_id="s",
            generation_id="g",
            source_frames=3,
            latent_frames=3,
        ),
        dict(
            ended=True, completed=False, submitted=3, accepted=3, consumed=3, received=3
        ),
    )
    assert [item["event_id"] for item in receiver.summary()["chunks"]] == [
        None,
        None,
        3,
    ]
    assert (tmp_path / receiver.filename).stat().st_size == 12
