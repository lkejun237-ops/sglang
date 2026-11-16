# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import RenderError
from sglang.multimodal_gen.runtime.realtime.whitebox.scene_transition import (
    SceneTransition,
    read_scene_transitions,
)


@pytest.mark.parametrize("missing_entries", [True, False])
def test_entry_only_inventory_still_requires_a_declared_initial_entry(missing_entries):
    initial = {"sceneId": "yard", "entryId": "start"}
    with pytest.raises(ValueError, match="Invalid scene transition"):
        read_scene_transitions(
            dict(
                schemaVersion=1,
                initial=initial
                if missing_entries
                else {"sceneId": "unknown", "entryId": "start"},
                entries=[] if missing_entries else [initial],
                transitions=[],
                videos=[],
            )
        )


@pytest.mark.parametrize(
    "story,with_video,prepare_first",
    [
        (False, True, False),
        (False, False, False),
        (True, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
@pytest.mark.parametrize(
    "model,commit_fails", [(False, False), (True, False), (True, True)]
)
def test_cutscene_waits_for_playback_preparation_and_target_presentation(
    story, with_video, prepare_first, model, commit_fails
):
    async def run():
        source_calls = []
        snapshots = asyncio.Queue()
        ready = asyncio.Event()
        prefill_ready = asyncio.Event()
        old_send_finished = asyncio.Event()
        model_ready = asyncio.Event()
        frozen_input = {}
        fail_once = model and story
        snapshot = {"condition": {"prompt": "Forest"}, "rgb_sha256": "a" * 64}
        target = {"sceneId": "forest", "entryId": "start"}
        initial = {"sceneId": "yard", "entryId": "start"}
        video = {
            "path": "media/chapter.mp4",
            "byteLength": 20,
            "durationSeconds": 3,
            "sha256": "a" * 64,
        }
        frame = {
            "protocolVersion": 1,
            "sessionId": "sdk",
            "epoch": 2,
            "mediaGeneration": 3,
            "outputFrameId": 4,
            "outputPtsUs": 500,
            "source": {
                "presentationId": "p",
                "sdkEpoch": 0,
                "sourceFrameId": 4,
                "sourceTimeUs": 500,
                "simulationTick": 12,
                "worldRevision": 1,
            },
            "uiRevision": 1,
            "uiCompleteThroughUs": 500,
            "widthPixels": 160,
            "heightPixels": 90,
        }

        async def source(command):
            action = command["action"]
            source_calls.append(action)
            if action == "prepare":
                await ready.wait()
            if action == "frame":
                return {"frame": frame}
            if action == "snapshot":
                return snapshot
            if action == "commit" and commit_fails:
                raise RuntimeError("fixture commit failure")
            return {}

        async def fence():
            source_calls.append("fence")
            await old_send_finished.wait()

        async def prepare_model(value):
            assert value is snapshot
            source_calls.append("freeze")
            return frozen_input

        async def prefill_model(value, deadline, *, observe):
            nonlocal fail_once
            assert value is frozen_input
            assert deadline > asyncio.get_running_loop().time()
            source_calls.append("prefill")
            await prefill_ready.wait()
            if fail_once:
                fail_once = False
                raise RenderError("PREFILL_FAILED")
            await observe(dict(type="prefill.memory_ready", request_id="prefill"))
            await observe(dict(type="prefill.persisted", request_id="prefill"))
            return {"cache_key": "c" * 64}

        async def generate(value, first_frame, deadline, *, transition_id):
            assert transition_id == transition.current["transition_id"]
            assert value is frozen_input
            assert first_frame == frame
            assert deadline > asyncio.get_running_loop().time()
            source_calls.append("generate")
            return frozen_input

        async def start_model(candidate):
            assert candidate is frozen_input
            source_calls.append("start")
            await model_ready.wait()
            return "forest-model"

        def verify_model(stream_id, presented):
            assert stream_id == "forest-model"
            assert presented == frame
            source_calls.append("model proof")

        async def discard():
            source_calls.append("discard")

        transition = SceneTransition(
            dict(
                schemaVersion=1,
                initial=initial,
                entries=[initial, target],
                videos=[video] if with_video else [],
                transitions=(
                    []
                    if not story and not with_video
                    else [
                        dict(
                            id="chapter",
                            target=target,
                            trigger={"eventId": "enter.forest"},
                            **({"video": video["path"]} if with_video else {}),
                        )
                    ]
                ),
            ),
            source,
            snapshots.put,
            lambda: 60,
            model_lifecycle=SimpleNamespace(
                fence=fence,
                prepare=prepare_model,
                prefill=prefill_model,
                generate=generate,
                start=start_model,
                discard=discard,
            ),
            verify_model_frame=verify_model,
        )

        async def send(op, request, generation=1, *, with_model=False, **fields):
            await transition.command(
                dict(
                    type="command",
                    version=2,
                    op="scene.transition." + op,
                    request_id=request,
                    source_session_id="room",
                    source_epoch=1,
                    connection_generation=generation,
                    **fields,
                ),
                model=with_model,
            )

        async def next_state(phase):
            state = await asyncio.wait_for(snapshots.get(), 1)
            assert state["phase"] == phase
            return state

        try:
            await transition.attach(1)
            occurrence = dict(
                eventId="enter.forest", occurrenceId="original-1", simulationTick=12
            )
            if story:
                await transition.trigger({**occurrence, "eventId": "unknown"})
                assert snapshots.empty()
                await transition.trigger(occurrence, model=model)
                await transition.trigger(occurrence, model=model)
            else:
                await send(
                    "request",
                    "request",
                    expected_revision=0,
                    **({"transition": "chapter"} if with_video else {"target": target}),
                    with_model=model,
                )
            await next_state("entering")
            if model:
                await asyncio.sleep(0)
                assert source_calls == ["fence"]
                assert snapshots.empty()
                old_send_finished.set()
            playing = await next_state("playing")
            assert playing.get("occurrence") == (occurrence if story else None)
            assert playing["active_entry"] == initial
            expected_calls = (["fence"] if model else []) + ["hold", "prepare"]
            assert source_calls == expected_calls
            transition_id = playing["transition_id"]
            assert playing["receipt"]["has_video"] is with_video
            playback_attempt = 3 if with_video else 1
            if with_video:
                await send(
                    "media",
                    "blocked",
                    transition_id=transition_id,
                    playback_attempt=1,
                    media_state="failed",
                )
                assert (await next_state("playing"))["media_state"] == "failed"
                await send(
                    "retry", "retry", transition_id=transition_id, playback_attempt=1
                )
                assert (await next_state("playing"))["playback_attempt"] == 2
                await transition.attach(2)
                assert (await next_state("playing"))["playback_attempt"] == 3
                await transition.attach(1)
                assert transition.generation == 2
                assert transition.snapshot()["playback_attempt"] == 3
                assert snapshots.empty()
                with pytest.raises(RenderError):
                    await send(
                        "media",
                        "stale",
                        transition_id=transition_id,
                        playback_attempt=2,
                        media_state="ended",
                    )
                await send(
                    "media",
                    "play",
                    generation=2,
                    transition_id=transition_id,
                    playback_attempt=3,
                    media_state="playing",
                )
                await next_state("playing")
                if not prepare_first:
                    await send(
                        "media",
                        "end",
                        generation=2,
                        transition_id=transition_id,
                        playback_attempt=3,
                        media_state="ended",
                    )
                    assert (await next_state("playing"))["media_ended"]
            else:
                assert playing["video"] is None and playing["media_ended"]
                with pytest.raises(RenderError):
                    await send(
                        "media",
                        "unexpected-video",
                        transition_id=transition_id,
                        playback_attempt=1,
                        media_state="ended",
                    )
            deadline = transition.deadline
            await transition.attach(3)
            recovered = transition.snapshot()
            if prepare_first:
                playback_attempt += 1
                assert (await next_state("playing"))["media_state"] == "waiting"
                await send(
                    "media",
                    "resumed-play",
                    generation=3,
                    transition_id=transition_id,
                    playback_attempt=playback_attempt,
                    media_state="playing",
                )
                assert not (await next_state("playing"))["media_ended"]
            assert recovered["media_state"] == ("waiting" if prepare_first else "ended")
            assert recovered["playback_attempt"] == playback_attempt
            assert transition.deadline == deadline
            assert snapshots.empty()
            assert source_calls == expected_calls
            ready.set()
            if model:
                prefilling = await next_state("playing")
                assert prefilling["model_phase"] == "prefilling"
                expected_calls += ["snapshot", "freeze", "prefill"]
                assert source_calls == expected_calls
                assert snapshots.empty()
                assert transition.model_input is frozen_input
                prefill_ready.set()
                if story:
                    failed = await next_state("playing")
                    assert failed["receipt"]["prefill_retryable"]
                    assert failed["media_ended"] is (not prepare_first)
                    assert not failed["scene_prepared"]
                    assert transition.deadline == deadline
                    await send(
                        "retry_prefill",
                        "retry-prefill",
                        generation=3,
                        transition_id=transition_id,
                    )
                    retrying = await next_state("playing")
                    assert "prefill_retryable" not in retrying["receipt"]
                    assert retrying["playback_attempt"] == playback_attempt
                    expected_calls += ["prefill"]
                memory = (await next_state("playing"))["receipt"]
                assert memory["condition_rgb_sha256"] == "a" * 64
                assert memory["prefill"] == dict(
                    request_id="prefill", memory_ready=True
                )
                persisted = (await next_state("playing"))["receipt"]
                assert persisted["prefill"]["persistence"] == "persisted"
            if prepare_first:
                prepared = await next_state("playing")
                assert prepared["scene_prepared"] and not prepared["media_ended"]
                assert prepared["media_state"] == "playing"
                assert source_calls == expected_calls
                assert "generate" not in source_calls and "commit" not in source_calls
                assert snapshots.empty()
                await send(
                    "media",
                    "end-after-preparation",
                    generation=3,
                    transition_id=transition_id,
                    playback_attempt=playback_attempt,
                    media_state="ended",
                )
            await next_state("starting")
            expected_calls += ["publish", "frame"]
            if model:
                starting = await next_state("starting")
                assert starting["model_phase"] == "starting"
                if commit_fails:
                    await next_state("failed")
                    assert source_calls == expected_calls + [
                        "generate",
                        "commit",
                        "discard",
                    ]
                    assert transition.model_input is None
                    return
                expected_calls += ["generate", "commit", "start"]
                assert source_calls == expected_calls
                assert snapshots.empty()
                model_ready.set()
            assert (await next_state("revealing"))["first_frame"] == frame
            with pytest.raises(RenderError):
                await send(
                    "presented",
                    "old",
                    generation=3,
                    transition_id=transition_id,
                    frame={**frame, "mediaGeneration": 2},
                )
            await send(
                "presented",
                "presented",
                generation=3,
                transition_id=transition_id,
                frame=frame,
            )
            await next_state("activating")
            running = await next_state("running")
            assert running["active_entry"] == target
            assert running["receipt"]["presented"] == frame
            if model:
                await transition.record_generation_cache(
                    transition_id,
                    dict(cache_key="b" * 64, cache_result="l1_hit", miss_reason=None),
                )
                receipt = (await next_state("running"))["receipt"]
                assert receipt["generation_cache"]["cache_result"] == "l1_hit"
                assert receipt["prefill"]["persistence"] == "persisted"
            else:
                assert "prefill" not in running["receipt"]
            if story:
                await transition.trigger(occurrence)
                assert snapshots.empty()
            await send(
                "presented",
                "presented",
                generation=3,
                transition_id=transition_id,
                frame=frame,
            )
            assert source_calls == expected_calls + (
                ["model proof"] if model else []
            ) + ["activate"]
            assert transition.model_input is None
        finally:
            await transition.close()

    asyncio.run(run())
