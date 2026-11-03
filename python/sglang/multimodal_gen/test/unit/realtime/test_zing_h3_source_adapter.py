# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec.msgpack
import pytest

from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters import (
    zing_h3_source_adapter as source,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RawRGBRealtimeOutputAdapter,
)
from sglang.multimodal_gen.runtime.realtime import zing_input_budget as budget
from sglang.multimodal_gen.runtime.realtime.source_transport import (
    DEFAULT_MESSAGE_BYTES,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    PixelRange,
    Provenance,
    SourceSpan,
)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    raw = json.dumps(
        {
            "causal": True,
            "h3_media_config": {
                "fps": 24,
                "max_short_edge": 2048,
                "max_area_config": {"source": 1032192},
                "max_source_target_aspect_ratio_factor": 1.02,
            },
        }
    ).encode()
    (tmp_path / "zing_config.json").write_bytes(raw)
    (tmp_path / "zing_realtime_profile.json").write_text(
        json.dumps(
            dict(
                model_config_sha256=hashlib.sha256(raw).hexdigest(),
                solver="few_step",
                denoising_step_list=[1000, 750, 500, 250],
                initial_block_count=0,
                guidance_scale=1,
                noise_version="indexed-cpu-v1",
            )
        )
    )
    args = SimpleNamespace(
        model_path=str(tmp_path),
        disagg_encoder_port=None,
        disagg_decoder_port=None,
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(
                    vae_clip_length=17, spatial_compression_ratio=16
                )
            ),
            dit_config=SimpleNamespace(
                arch_config=SimpleNamespace(patch_size=(1, 2, 2))
            ),
        ),
    )
    monkeypatch.setattr(source, "get_global_server_args", lambda: args)
    # These ingress tests do not ship a model export; real tokenizer loading and
    # non-truncated counting are covered separately below in this module.
    monkeypatch.setattr(
        source, "count_prompt_tokens", lambda root, text: len(text.split())
    )
    return args


def initialize(**overrides):
    request = RealtimeVideoGenerationsRequest(
        **{
            "type": "init",
            "prompt": "Render snow",
            "zing_realtime_api_version": 1,
            "source_stream": {"format": "rgb24", "width": 32, "height": 32, "fps": 24},
            "size": "32x32",
            **overrides,
        }
    )
    adapter, session = source.ZingH3SourceStreamAdapter(), GenerateSession()
    session.set_adapter(adapter)
    asyncio.run(adapter.on_init(session, request))
    session.set_request(request)
    return adapter, session


def _enable_reference_updates(args, *, split=True):
    from pathlib import Path

    root = Path(args.model_path)
    runtime = json.loads((root / "zing_config.json").read_text())
    runtime.update(num_frame_per_block=[3, 2], split_first_latent=split)
    raw = json.dumps(runtime).encode()
    (root / "zing_config.json").write_bytes(raw)
    profile = json.loads((root / "zing_realtime_profile.json").read_text())
    profile["model_config_sha256"] = hashlib.sha256(raw).hexdigest()
    (root / "zing_realtime_profile.json").write_text(json.dumps(profile))
    args.pipeline_config.prompt_switch_prefix_latent_frames = 5


def _reference(name="<Picture 1>", color="red"):
    import io

    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(output, format="PNG")
    return dict(id=name, format="png", payload=output.getvalue())


class PromptSocket:
    def __init__(self):
        self.messages = []

    async def send_bytes(self, raw):
        self.messages.append(msgspec.msgpack.decode(raw))


@pytest.mark.parametrize("split", [False, True])
def test_reference_capability_requires_supported_deployment(configured, split):
    adapter, _ = initialize()
    assert adapter.ready["prompt_reference_updates"] is False
    _enable_reference_updates(configured, split=split)
    adapter, _ = initialize()
    assert adapter.ready["prompt_reference_updates"] is True


@pytest.mark.parametrize("dispatched", [0, 1])
def test_text_only_update_checks_reference_budget_before_and_after_dispatch(
    configured, dispatched
):
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.conditioning import (
        release_api_media,
    )

    configured.pipeline_config.zing_condition_max_tokens = 2202
    adapter, session = initialize(prompt="old", references=[_reference()])
    adapter.dispatched_source_frames = dispatched
    ws = PromptSocket()

    async def run():
        try:
            await adapter.ingest_wire_event(
                ws,
                session,
                dict(type="event", kind="prompt", event_id=1, payload="one two three"),
                100,
            )
            assert ws.messages[-1]["code"] == "CONDITION_BUDGET_EXCEEDED"
            assert ws.messages[-1]["fatal"] is False
            assert adapter._pending_prompt_command["rejected"]
            assert session.request.prompt == "old"
        finally:
            await adapter.close_conditioning(session)
            release_api_media(session)

    asyncio.run(run())


@pytest.mark.parametrize(
    "references",
    [
        None,
        {},
        [_reference(str(i)) for i in range(22)],
        [_reference(), _reference()],
        [_reference("<Video 1>")],
        [dict(id="bad", format="png", payload=b"not an image")],
    ],
)
def test_invalid_reference_event_is_nonfatal_and_supersedes_pending(
    configured, references
):
    _enable_reference_updates(configured)
    adapter, session = initialize()
    ws = PromptSocket()

    async def run():
        await adapter.ingest_wire_event(
            ws,
            session,
            dict(
                type="event",
                kind="prompt",
                event_id=1,
                payload="same",
                references=[_reference()],
            ),
            100,
        )
        upload = adapter._pending_reference_upload
        leases = list(upload.reservations)
        await adapter.ingest_wire_event(
            ws,
            session,
            dict(
                type="event",
                kind="prompt",
                event_id=2,
                payload="same",
                references=references,
            ),
            100,
        )
        assert upload.items == [] and all(lease.released for lease in leases)
        assert adapter._pending_prompt_command == dict(
            prompt="same", event_id=2, rejected=True
        )
        assert ws.messages[-1]["fatal"] is False
        assert ws.messages[-1]["event_id"] == 2
        assert session.request.prompt == "Render snow"
        await adapter.ingest_wire_event(
            ws,
            session,
            dict(type="event", kind="prompt", event_id=3, payload="valid"),
            100,
        )
        assert ws.messages[-1]["type"] == "control_ack"
        assert "references" not in adapter._pending_prompt_command
        await adapter.close_conditioning(session)

    asyncio.run(run())


@pytest.mark.parametrize("count", [1, 21])
def test_reference_upload_lifetime_and_empty_snapshot(configured, count):
    _enable_reference_updates(configured)
    adapter, session = initialize()
    ws = PromptSocket()

    async def run():
        await adapter.ingest_wire_event(
            ws,
            session,
            dict(
                type="event",
                kind="prompt",
                event_id=1,
                payload="same",
                references=[_reference(f"<Picture {i + 1}>") for i in range(count)],
            ),
            100,
        )
        upload = adapter._pending_reference_upload
        assert [item["id"] for item in upload.items] == [
            f"<Picture {i + 1}>" for i in range(count)
        ]
        adapter.sample_chunk_inputs(session, configured, None, 17)
        assert adapter._selected_prompt_command["references"] == upload.items
        await adapter.ingest_wire_event(
            ws,
            session,
            dict(
                type="event", kind="prompt", event_id=2, payload="same", references=[]
            ),
            100,
        )
        assert upload.items and all(not lease.released for lease in upload.reservations)
        assert adapter._pending_prompt_command["references"] == []
        await adapter.close_conditioning(session)
        assert upload.items == [] and upload.reservations == []
        assert session._api_media_reservations == []

    asyncio.run(run())


def test_reference_memory_cleanup_releases_initial_pending_and_inflight(
    configured, monkeypatch
):
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
        realtime_video_api,
    )

    _enable_reference_updates(configured)
    # Failing all media writes proves neither init nor update spools images.
    monkeypatch.setattr(
        Path, "write_bytes", lambda *a, **k: pytest.fail("Media must stay in memory")
    )
    adapter, session = initialize(references=[_reference("initial")])
    ws = PromptSocket()

    async def run():
        for event_id in (1, 2):
            await adapter.ingest_wire_event(
                ws,
                session,
                dict(
                    type="event",
                    kind="prompt",
                    event_id=event_id,
                    payload="same",
                    references=[_reference()],
                ),
                100,
            )
            if event_id == 1:
                adapter.sample_chunk_inputs(session, configured, None, 17)
        leases = list(session._api_media_reservations)
        assert len(leases) == 3 and session.input_temp_dir is None

        async def release(_):
            return True

        monkeypatch.setattr(
            realtime_video_api, "_release_scheduler_realtime_session", release
        )
        await realtime_video_api._cleanup_realtime_session(session, None, None)
        assert all(lease.released for lease in leases)
        assert session._api_media_reservations == []
        assert adapter._pending_reference_upload is None
        assert adapter._inflight_reference_uploads == []
        assert session.conditioning_resources.ledger.snapshot()["global"]["ram"] == 0

    asyncio.run(run())


def test_cancelled_reference_verification_returns_ram_credit(configured, monkeypatch):
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters import (
        zing_reference_upload as uploads,
    )

    adapter, session = initialize(references=[_reference("initial")])
    initial = list(session._api_media_reservations)

    async def run():
        entered = asyncio.Event()

        async def blocked(*_):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(uploads, "verify_presentation_upload", blocked)
        task = asyncio.create_task(
            uploads.prepare_reference_upload(
                session,
                [_reference()],
            )
        )
        await entered.wait()
        candidate = session._api_media_reservations[-1]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert candidate.released and session._api_media_reservations == initial
        assert not initial[0].released
        from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.conditioning import (
            release_api_media,
        )

        release_api_media(session)
        await adapter.close_conditioning(session)

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["image", "clear", "invalid", "text"])
def test_early_reference_event_dispatch_preserves_initial_epoch(
    configured, monkeypatch, mode
):
    _enable_reference_updates(configured)
    adapter, session = initialize()
    ws = PromptSocket()

    def prepare(self, session, args, chunk):
        inputs = self.sample_chunk_inputs(session, args, chunk, 17)
        return SimpleNamespace(
            block_idx=0,
            extra={},
            realtime_event_id=self.prompt_event,
            sampling_params=SimpleNamespace(
                prompt=inputs.prompt,
                zing_request={
                    "messages": [{"content": [{"type": "text", "text": inputs.prompt}]}]
                },
            ),
        )

    monkeypatch.setattr(
        source.BaseRealtimeModelAdapter, "prepare_next_request", prepare
    )
    monkeypatch.setattr(
        source.BaseRealtimeModelAdapter, "on_chunk_complete", lambda *args: None
    )

    async def run():
        fields = (
            {}
            if mode == "text"
            else {
                "references": None
                if mode == "invalid"
                else []
                if mode == "clear"
                else [_reference()]
            }
        )
        await adapter.ingest_wire_event(
            ws,
            session,
            dict(type="event", kind="prompt", event_id=1, payload="new", **fields),
            100,
        )
        upload = adapter._pending_reference_upload
        batch = adapter.prepare_next_request(session, configured, None)
        expected = "new" if mode == "text" else "Render snow"
        assert batch.sampling_params.prompt == expected
        assert (
            batch.sampling_params.zing_request["messages"][0]["content"][-1]["text"]
            == expected
        )
        assert batch.realtime_event_id == (1 if mode == "text" else None)
        assert batch.extra["zing_prompt_update"]["event_id"] == 1
        assert batch.extra["zing_prompt_update"]["prompt"] == "new"
        if upload is not None:
            assert all(not lease.released for lease in upload.reservations)
        adapter.on_chunk_complete(session, SimpleNamespace(usage={}))
        if upload is not None:
            assert upload.items == [] and upload.reservations == []
        await adapter.close_conditioning(session)

    asyncio.run(run())


@pytest.mark.parametrize(
    "profile_steps,profile_count,cli_steps,count",
    [
        (4, 0, None, 0),
        (2, 0, None, 0),
        (3, 0, None, 0),
        (1, 0, None, 0),
        (5, 0, None, 0),
        (2, 0, 5, 3),
        (4, 0, 2, 0),
        (2, 0, 4, 0),
        (4, 0, 2, 3),
        (2, 0, 2, 3),
        (2, 3, None, 0),
        (2, 3, 2, 0),
        (2, 3, 4, 0),
    ],
)
@pytest.mark.parametrize("explicit_steps", [False, True])
def test_effective_sampling_is_reported_and_dispatched(
    configured,
    monkeypatch,
    profile_steps,
    profile_count,
    cli_steps,
    count,
    explicit_steps,
):
    path = Path(configured.model_path) / "zing_realtime_profile.json"
    profile = json.loads(path.read_text())
    profile["denoising_step_list"] = [1000, 750, 500, 250, 125][:profile_steps]
    if profile_count:
        profile.update(
            initial_block_count=profile_count,
            initial_denoising_step_list=[1000, 750, 500, 250],
        )
    path.write_text(json.dumps(profile))
    configured.zing_solver = "few_step" if cli_steps else None
    configured.zing_denoising_step_list = (
        [1000, 750, 500, 250, 125][:cli_steps] if cli_steps else None
    )
    configured.zing_initial_block_count = count
    configured.zing_initial_denoising_step_list = (
        [1000, 750, 500, 250] if count else None
    )
    expected_steps = cli_steps or profile_steps
    expected_count = count if cli_steps is not None else profile_count
    adapter, session = initialize(
        num_inference_steps=expected_steps if explicit_steps else None
    )
    ready = adapter.session_ready_message(session)
    expected = dict(
        solver="few_step",
        denoising_step_list=[1000, 750, 500, 250, 125][:expected_steps],
        initial_block_count=expected_count,
        initial_denoising_step_list=[1000, 750, 500, 250] if expected_count else None,
    )
    assert (
        ready["num_inference_steps"]
        == session.request.num_inference_steps
        == expected_steps
    )
    assert ready["sampling_schedule"] == expected
    # Isolate the generic API factory; exercise the source adapter's real assembly.
    monkeypatch.setattr(
        source, "build_realtime_sampling_params", lambda *a, **kw: SimpleNamespace(**kw)
    )
    params = adapter.build_sampling_params(
        session,
        configured,
        SimpleNamespace(request_id="test"),
        SimpleNamespace(prompt="snow"),
        17,
    )
    assert params.num_inference_steps == expected_steps
    assert params.zing_solver == expected["solver"]
    assert params.zing_denoising_step_list == expected["denoising_step_list"]
    monkeypatch.setattr(
        source.BaseRealtimeModelAdapter,
        "prepare_next_request",
        lambda *a: SimpleNamespace(extra={}, block_idx=0),
    )
    batch = adapter.prepare_next_request(session, configured, None)
    wire = msgspec.msgpack.decode(msgspec.msgpack.encode(batch.extra))
    assert wire["zing_source_command"]["sampling_schedule"] == expected
    params.zing_denoising_step_list.append(125)
    assert ready["sampling_schedule"] == expected
    assert adapter.profile == profile
    adapter.dispose(session)
    with pytest.raises(ValueError, match="requested step count"):
        initialize(num_inference_steps=4 if expected_steps == 2 else 2)


@pytest.mark.parametrize(
    "grid", [[], [1000, 1000], [750, 500], [1000, 0], [1000, True]]
)
def test_invalid_profile_grid_is_not_hidden_by_cli(configured, grid):
    path = Path(configured.model_path) / "zing_realtime_profile.json"
    profile = json.loads(path.read_text())
    profile["denoising_step_list"] = grid
    path.write_text(json.dumps(profile))
    configured.zing_solver = "few_step"
    configured.zing_denoising_step_list = [1000, 500]
    with pytest.raises(ValueError):
        initialize()


def test_cli_euler_rejected_before_source_session(configured):
    configured.zing_solver = "euler"
    configured.zing_denoising_step_list = None
    with pytest.raises(ValueError, match="deployed model"):
        initialize()


def test_explicit_zero_disables_profile_initial_schedule(configured):
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    path = Path(configured.model_path) / "zing_realtime_profile.json"
    profile = json.loads(path.read_text())
    profile.update(
        denoising_step_list=[1000, 500],
        initial_block_count=3,
        initial_denoising_step_list=[1000, 750, 500, 250],
    )
    path.write_text(json.dumps(profile))
    configured._explicit_arg_names = {"zing_initial_block_count"}
    configured.is_arg_explicitly_set = lambda name: ServerArgs.is_arg_explicitly_set(
        configured, name
    )
    configured.zing_initial_block_count = 0
    adapter, session = initialize()
    assert adapter.ready["sampling_schedule"] == dict(
        solver="few_step",
        denoising_step_list=[1000, 500],
        initial_block_count=0,
        initial_denoising_step_list=None,
    )
    adapter.dispose(session)


@pytest.mark.parametrize("profile_count", [None, 0, 3])
@pytest.mark.parametrize("cli_count", [None, 0, 3])
def test_default_mixed_schedule_and_explicit_overrides(
    configured, profile_count, cli_count
):
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    path = Path(configured.model_path) / "zing_realtime_profile.json"
    profile = json.loads(path.read_text())
    profile.pop("initial_block_count")
    if profile_count is not None:
        profile["initial_block_count"] = profile_count
    if profile_count == 3:
        profile.update(
            denoising_step_list=[1000, 500],
            initial_denoising_step_list=[1000, 750, 500, 250],
        )
    path.write_text(json.dumps(profile))
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    argv = ["--model-path", str(configured.model_path)]
    if cli_count is not None:
        argv += ["--zing-initial-block-count", str(cli_count)]
    parsed = parser.parse_args(argv)
    ServerArgs._validate_zing_sampling(parsed)
    for name in (
        "zing_solver",
        "zing_denoising_step_list",
        "zing_initial_block_count",
        "zing_initial_denoising_step_list",
    ):
        setattr(configured, name, getattr(parsed, name))
    configured.is_arg_explicitly_set = lambda name: (
        cli_count is not None and name == "zing_initial_block_count"
    )
    adapter, session = initialize()
    count = (
        cli_count
        if cli_count is not None
        else (profile_count if profile_count is not None else 3)
    )
    grid = [1000, 500] if count else profile["denoising_step_list"]
    assert adapter.ready["num_inference_steps"] == len(grid)
    assert adapter.ready["sampling_schedule"] == dict(
        solver="few_step",
        denoising_step_list=grid,
        initial_block_count=count,
        initial_denoising_step_list=[1000, 750, 500, 250] if count else None,
    )
    adapter.dispose(session)


@pytest.mark.parametrize(
    "fields",
    [
        {"initial_block_count": 3},
        {"initial_block_count": True},
        {"initial_block_count": 1},
        {
            "initial_block_count": 0,
            "initial_denoising_step_list": [1000, 750, 500, 250],
        },
        {"initial_block_count": 3, "initial_denoising_step_list": [1000, 0]},
        {
            "initial_block_count": 3,
            "initial_denoising_step_list": [1000, 750, 500, 250],
            "denoising_step_list": [1000, 750, 800],
        },
    ],
)
def test_bad_profile_initial_schedule_is_not_hidden_by_cli(configured, fields):
    path = Path(configured.model_path) / "zing_realtime_profile.json"
    profile = json.loads(path.read_text())
    profile.update(denoising_step_list=[1000, 500])
    profile.update(fields)
    path.write_text(json.dumps(profile))
    configured.zing_solver = "few_step"
    configured.zing_denoising_step_list = [1000, 500]
    with pytest.raises(ValueError):
        initialize()


@pytest.mark.parametrize(
    "overrides",
    [
        {"zing_realtime_api_version": 2},
        {"zing_request": {}},
        {"num_inference_steps": 50},
        {"zing_solver": "euler"},
        {"zing_initial_block_count": 3},
        {"zing_initial_denoising_step_list": [1000, 750, 500, 250]},
        {"max_chunks": 3},
        {"zing_window_chunks": 100},
        {"zing_rope_position_mode": "absolute"},
        {"guidance_scale": 2},
        {"realtime_preview_max_width": 16},
        {"condition_inputs": {}},
    ],
)
def test_reject_mixed_protocol_or_profile(configured, overrides):
    with pytest.raises(ValueError):
        initialize(**overrides)


def test_source_over_capacity_rejects_with_transport_error(configured):
    from sglang.multimodal_gen.runtime.realtime.source_transport import (
        SourceTransportError,
    )

    with pytest.raises(SourceTransportError) as error:
        initialize(
            size="32x32",
            source_stream={"format": "rgb24", "width": 4096, "height": 4096, "fps": 24},
        )
    assert error.value.code == "source_message_too_large"


def test_default_encoder_dispatches_first_frame(configured):
    adapter, session = initialize()
    assert adapter.source_encoder["mode"] == "progressive"
    assert adapter.source_encoder["factory"].endswith(
        "progressive_encoder:create_session"
    )
    adapter.pending_frames = 1
    asyncio.run(asyncio.wait_for(adapter.wait_for_next_chunk(session), 1))


@pytest.mark.parametrize("window", [None, {"window_chunks": 6, "sink_chunks": 3}])
def test_window_is_frozen_from_deployment_only(configured, monkeypatch, window):
    from pathlib import Path

    path = Path(configured.model_path) / "zing_realtime_profile.json"
    profile = json.loads(path.read_text())
    if window is not None:
        profile["source_attention_window"] = window
        path.write_text(json.dumps(profile))
    adapter, session = initialize()
    monkeypatch.setattr(
        source, "build_realtime_sampling_params", lambda *a, **kw: SimpleNamespace()
    )
    params = adapter.build_sampling_params(
        session,
        None,
        SimpleNamespace(request_id="test"),
        SimpleNamespace(prompt="original prompt"),
        17,
    )
    expected = window or {"window_chunks": 21, "sink_chunks": 3}
    assert params.zing_window_chunks == expected["window_chunks"]
    assert params.zing_sink_chunks == expected["sink_chunks"]
    with pytest.raises(ValueError):
        initialize(zing_window_chunks=6)


@pytest.mark.parametrize(
    "window",
    [
        None,
        {},
        {"window_chunks": True, "sink_chunks": 0},
        {"window_chunks": 22, "sink_chunks": 3},
        {"window_chunks": 3, "sink_chunks": 3},
        {"window_chunks": 6, "sink_chunks": -1},
    ],
)
def test_invalid_deployment_window_fails_before_session(configured, window):
    from pathlib import Path

    path = Path(configured.model_path) / "zing_realtime_profile.json"
    profile = json.loads(path.read_text())
    profile["source_attention_window"] = window
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="deployed model"):
        initialize()


def test_source_presentation_dispatches_one_frame_even_with_clip17_encoder(configured):
    from pathlib import Path

    path = Path(configured.model_path) / "zing_realtime_profile.json"
    profile = json.loads(path.read_text())
    profile["source_encoder"] = {"mode": "clip17"}
    path.write_text(json.dumps(profile))
    adapter, session = initialize()
    assert adapter.source_encoder == {"mode": "clip17"}

    async def run():
        adapter.pending_frames = 1
        waiter = asyncio.create_task(adapter.wait_for_next_chunk(session))
        await asyncio.sleep(0)
        assert waiter.done()
        await asyncio.wait_for(waiter, 1)

    asyncio.run(run())


@pytest.mark.parametrize(
    "mode,frames", [("clip17", 17)] + [("progressive", n) for n in [1, 2, 8, 17]]
)
def test_ready_preserves_unknown_total_and_accepted_precedes_dispatch(
    configured, mode, frames
):
    adapter, session = initialize()
    adapter.source_encoder = {"mode": mode}
    assert session.request.num_frames is None and session.request.max_chunks is None
    assert adapter.session_ready_message(session)["realtime_output_format"] == "raw"

    async def run():
        sent = []

        class Socket:
            async def send_bytes(self, payload):
                assert not adapter.pending
                sent.append(msgspec.msgpack.decode(payload))
                await asyncio.sleep(0)

        waiter = asyncio.create_task(adapter.wait_for_next_chunk(session))
        await asyncio.sleep(0)
        assert not waiter.done()
        payload = dict(
            client_seq=0,
            source_frame_start=0,
            num_frames=frames,
            pts_us=0,
            data=bytes(frames * 32 * 32 * 3),
        )
        await adapter.ingest_wire_event(
            Socket(),
            session,
            dict(type="event", kind="source.append", event_id=1, payload=payload),
            len(payload["data"]) + 200,
        )
        await asyncio.wait_for(waiter, 1)
        assert len(adapter.pending) == 1 and sent[0]["type"] == "source.accepted"
        assert adapter.ledger.consumed_frames == 0

    asyncio.run(run())
    adapter.dispose(session)
    assert not adapter.pending and adapter.ledger.closed


def test_admission_receiver_can_start_after_first_dispatch(configured, tmp_path):
    adapter, session = initialize()
    adapter.source_encoder = {"mode": "progressive"}

    async def run():
        context = source.zmq.asyncio.Context()
        adapter.admission_context = context
        adapter.admission_socket = context.socket(source.zmq.PUSH)
        adapter.admission_socket.setsockopt(source.zmq.LINGER, 0)
        endpoint = "ipc://" + str(tmp_path / "bootstrap.sock")
        adapter.admission_socket.bind(endpoint)

        class Socket:
            async def send_bytes(self, wire):
                assert not adapter.pending
                assert msgspec.msgpack.decode(wire)["type"] == "source.accepted"

        async def start_receiver():
            await adapter.wait_for_next_chunk(session)
            receiver = context.socket(source.zmq.PULL)
            receiver.setsockopt(source.zmq.LINGER, 0)
            receiver.connect(endpoint)
            try:
                header, pixels = await receiver.recv_multipart()
                assert json.loads(header)["source_frame_start"] == 0
                assert pixels == bytes(32 * 32 * 3)
            finally:
                receiver.close()

        receiver = asyncio.create_task(start_receiver())
        try:
            await asyncio.wait_for(
                adapter.ingest_wire_event(
                    Socket(),
                    session,
                    dict(
                        type="event",
                        kind="source.append",
                        event_id=1,
                        payload=dict(
                            client_seq=0,
                            source_frame_start=0,
                            num_frames=1,
                            pts_us=0,
                            data=bytes(32 * 32 * 3),
                        ),
                    ),
                    32 * 32 * 3 + 200,
                ),
                2,
            )
            await asyncio.wait_for(receiver, 2)
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            adapter.clear_state(session)

    asyncio.run(run())


@pytest.mark.parametrize("output_format", ["raw", "jpeg", "webp"])
def test_source_provenance_survives_transport_splitting(output_format):
    p = Provenance(
        PixelRange(0, 17), (SourceSpan(1, 0, 0, 10), SourceSpan(2, 10, 10, 7)), 34
    )
    sent = []

    class Socket:
        async def send_bytes(self, payload):
            sent.append(payload)

    async def run():
        return await RawRGBRealtimeOutputAdapter()._send_frame_batches(
            Socket(),
            [[bytes(32 * 32 * 3)] * 17],
            content_type="application/x-raw-rgb",
            chunk_index_start=0,
            request_id="dispatch",
            output_format=output_format,
            preview_max_width=32,
            frame_metadata={
                "format": "rgb24",
                "width": 32,
                "height": 32,
                "channels": 3,
                "bytes_per_frame": 32 * 32 * 3,
            },
            source_blocks=[
                dict(block_idx=7, request_id="origin", event_id=91, provenance=p)
            ],
            source_fps=24,
        )

    stats = asyncio.run(run())
    cursor, wires = 0, iter(sent)
    for wire in wires:
        header = msgspec.msgpack.decode(wire)
        payload = (
            next(wires) if header["type"] == "frame_batch_header" else header["payload"]
        )
        assert len(payload) == header["total_size"]
        assert header["output_frame_start"] == cursor
        assert header["pts_us"] == cursor * 1_000_000 // 24
        assert header["event_id"] == 91 and header["chunk_index"] == 7
        assert (
            header["request_id"] == "origin" and header["source_dependency_end"] == 34
        )
        end = cursor + header["num_frames"]
        assert (
            header["source_alignment"]
            == p.slice(PixelRange(cursor, end)).metadata()["source_alignment"]
        )
        cursor = end
    assert cursor == stats["num_frames"] == 17


def test_one_decoded_batch_is_split_across_its_source_blocks():
    """A dedicated decoder returns one RGB batch per dispatch; every source block
    must still reach the client as its own ordered frame batch."""
    whole = Provenance(
        PixelRange(0, 17), (SourceSpan(1, 0, 0, 10), SourceSpan(2, 10, 10, 7)), 34
    )
    blocks = [
        dict(
            block_idx=7 + index,
            request_id="origin",
            event_id=91,
            provenance=whole.slice(PixelRange(start, end)),
        )
        for index, (start, end) in enumerate(((0, 10), (10, 17)))
    ]
    sent = []

    class Socket:
        async def send_bytes(self, payload):
            sent.append(payload)

    frames = [bytes([index]) * (2 * 2 * 3) for index in range(17)]

    stats = asyncio.run(
        RawRGBRealtimeOutputAdapter()._send_frame_batches(
            Socket(),
            [frames],
            content_type="application/x-raw-rgb",
            chunk_index_start=0,
            request_id="dispatch",
            output_format="raw",
            frame_metadata={
                "format": "rgb24",
                "width": 2,
                "height": 2,
                "channels": 3,
                "bytes_per_frame": 2 * 2 * 3,
            },
            source_blocks=blocks,
            source_fps=24,
        )
    )

    headers = []
    wires = iter(sent)
    for wire in wires:
        header = msgspec.msgpack.decode(wire)
        payload = (
            next(wires) if header["type"] == "frame_batch_header" else header["payload"]
        )
        headers.append((header, payload))
    assert [header["chunk_index"] for header, _ in headers] == [7, 8]
    assert [header["output_frame_start"] for header, _ in headers] == [0, 10]
    assert [header["num_frames"] for header, _ in headers] == [10, 7]
    assert b"".join(payload for _, payload in headers) == b"".join(frames)
    assert stats["num_frames"] == 17


def test_reference_only_init_preserves_ids_and_dispatches_bytes_once(
    configured, monkeypatch
):
    refs = [_reference(f"<Picture {i + 1}>") for i in range(21)]
    adapter, session = initialize(references=refs)
    assert adapter.ready["max_references"] == 21
    assert not any("user_media" in key for key in adapter.ready)
    monkeypatch.setattr(
        source, "build_realtime_sampling_params", lambda *a, **kw: SimpleNamespace()
    )

    def build(prompt):
        return adapter.build_sampling_params(
            session,
            None,
            SimpleNamespace(request_id="test"),
            SimpleNamespace(prompt=prompt),
            17,
        ).zing_request["messages"]

    user, target = build("original prompt")
    assert [c["id"] for c in user["content"][:-1]] == [r["id"] for r in refs]
    assert all(c["type"] == "image" for c in user["content"][:-1])
    assert [c["payload"] for c in user["content"][:-1]] == [r["payload"] for r in refs]
    assert user["content"][-1]["text"] == "original prompt"
    assert [c["content_id"] for c in target["conditions"]] == [r["id"] for r in refs]
    assert "source" not in target
    session.generate_chunk_cnt = 1
    user, target = build("next prompt")
    assert user["content"] == [dict(type="text", text="next prompt")]
    assert target["conditions"] == []
    adapter.clear_state(session)
    assert not adapter.reference_items and adapter.source_preview_sha256 is None


def test_reference_bytes_remain_bounded(configured):
    with pytest.raises(ValueError, match="too large"):
        initialize(
            references=[
                dict(id="large", format="jpeg", payload=b"x" * ((64 << 20) + 1))
            ]
        )


@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_reference_snapshot_byte_budget_at_init_and_update(configured, extra_bytes):
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_reference_upload import (
        prepare_reference_upload,
    )

    ref = _reference()
    size = (64 << 20) + extra_bytes
    ref["payload"] += bytes(size - len(ref["payload"]))
    if extra_bytes:
        with pytest.raises(ValueError, match="too large"):
            initialize(references=[ref])
        adapter, session = initialize()
    else:
        adapter, session = initialize(references=[ref])

    async def run():
        try:
            if extra_bytes:
                with pytest.raises(ValueError, match="too large"):
                    await prepare_reference_upload(session, [ref])
            else:
                upload = await prepare_reference_upload(session, [ref])
                assert len(upload.items[0]["payload"]) == size
                upload.release(session)
        finally:
            from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.conditioning import (
                release_api_media,
            )

            await adapter.close_conditioning(session)
            release_api_media(session)

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["local", "remote"])
@pytest.mark.parametrize("value", [None, [], [{"type": "image"}], [{"type": "video"}]])
def test_removed_user_media_is_rejected_before_image_verification(
    configured, monkeypatch, mode, value
):
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import conditioning

    configured.zing_conditioning_mode = mode
    monkeypatch.setattr(
        conditioning, "verify_presentation_upload", lambda *a: pytest.fail("verify")
    )
    with pytest.raises(ConditionError, match="source_stream, references and prompt"):
        initialize(user_media=value)


@pytest.mark.parametrize(
    "extra",
    [
        {"type": "video"},
        {"reference": False},
        {"uri": "file:///tmp/image.png"},
        {"clip": {"start": 0, "end": 0}},
        {"fps": 24},
    ],
)
def test_reference_entries_cannot_add_presentation_fields(configured, extra):
    with pytest.raises(ValueError, match="reference images"):
        initialize(references=[dict(_reference(), **extra)])


def test_reference_ids_must_be_unique_and_not_claim_source(configured):
    with pytest.raises(ValueError, match="uniquely named"):
        initialize(references=[_reference("same"), _reference("same")])
    with pytest.raises(ValueError, match="different name"):
        initialize(references=[_reference("<Video 1>")])


def test_21_references_are_preserved_and_22_rejected(configured):
    import io

    from PIL import Image

    refs = []
    for i in range(22):
        data = io.BytesIO()
        fmt = "PNG" if i == 0 else "JPEG"
        Image.new("RGB", (32, 32), color=(i * 10, 0, 0)).save(data, format=fmt)
        refs.append(
            dict(id=f"picture-{21 - i}", format=fmt.lower(), payload=data.getvalue())
        )
    adapter, session = initialize(references=refs[:21])
    assert session.input_temp_dir is None
    assert adapter.session_ready_message(session)["max_references"] == 21
    assert len(adapter.reference_items) == 21
    assert adapter.reference_ids == [ref["id"] for ref in refs[:21]]
    assert [item["payload"] for item in adapter.reference_items] == [
        ref["payload"] for ref in refs[:21]
    ]
    adapter.dispose(session)
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.conditioning import (
        release_api_media,
    )

    release_api_media(session)
    with pytest.raises(ValueError, match="at most 21"):
        initialize(references=refs)


@pytest.mark.parametrize("parts", [(1, 3), (3, 1), (2, 2)])
def test_source_preview_uses_s0_once_without_consuming_credit(configured, parts):
    adapter, session = initialize()
    assert "source_preview" not in adapter.ready
    assert adapter.source_preview_sha256 is None
    assert "prompt_video_frame_format" not in adapter.ready

    class Socket:
        async def send_bytes(self, payload):
            pass

    async def run():
        start = 0
        preview = None
        for seq, count in enumerate(parts):
            payload = dict(
                client_seq=seq,
                source_frame_start=start,
                num_frames=count,
                pts_us=start * 1_000_000 // 24,
                data=b"".join(
                    bytes([i + 10]) * (32 * 32 * 3) for i in range(start, start + count)
                ),
            )
            await adapter.ingest_wire_event(
                Socket(),
                session,
                dict(
                    type="event",
                    kind="source.append",
                    event_id=seq + 1,
                    payload=payload,
                ),
                len(payload["data"]) + 200,
            )
            await asyncio.wait_for(adapter.wait_for_next_chunk(session), 1)
            if preview is None:
                preview = adapter.source_preview_sha256
            assert adapter.source_preview_sha256 == preview
            assert adapter.pending[-1]["payload"] is payload
            start += count
            await asyncio.sleep(0)  # paused producer resumes with the same S0
        assert adapter.ledger.accepted_frames == sum(parts)
        assert adapter.ledger.consumed_frames == 0
        assert (
            adapter.source_preview_sha256
            == hashlib.sha256(bytes([10]) * 3072).hexdigest()
        )
        assert session.input_temp_dir is None  # S0 stays in the original append.
        assert adapter.reference_items == []

    asyncio.run(run())
    adapter.dispose(session)
    assert adapter.source_preview_sha256 is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_preview": "ar-first-frame"},
        {"source_preview": "guess"},
        {"prompt_video_frame": {}},
        {"user_media": [{"type": "video", "id": "v", "uri": "/etc/passwd"}]},
    ],
)
def test_removed_preview_options_and_unsafe_media_are_rejected(configured, kwargs):
    with pytest.raises(ValueError):
        initialize(**kwargs)


def test_source_preview_reserves_video_id_for_automatic_media(configured):
    with pytest.raises(ValueError, match="different name"):
        initialize(references=[dict(id="<Video 1>", format="png", payload=b"")])


def test_invalid_first_packet_and_empty_eos_do_not_create_preview(configured):
    adapter, session = initialize()

    class Socket:
        async def send_bytes(self, payload):
            raise AssertionError("Invalid input must not be acknowledged")

    async def run():
        for kind, payload in [
            ("source.end", {"source_frame_end": 0}),
            (
                "source.append",
                dict(
                    client_seq=1,
                    source_frame_start=0,
                    num_frames=1,
                    pts_us=0,
                    data=bytes(3072),
                ),
            ),
        ]:
            with pytest.raises(ValueError):
                await adapter.ingest_wire_event(
                    Socket(),
                    session,
                    dict(type="event", kind=kind, event_id=1, payload=payload),
                    3300,
                )
            assert adapter.source_preview_sha256 is None
            assert adapter.ledger.accepted_frames == 0

    asyncio.run(run())
    adapter.dispose(session)


def test_dispose_before_s0_wakes_waiter(configured):
    adapter, session = initialize()

    async def run():
        waiter = asyncio.create_task(adapter.wait_for_next_chunk(session))
        await asyncio.sleep(0)
        adapter.dispose(session)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, 1)

    asyncio.run(run())
    replacement, new_session = initialize()
    assert replacement.source_preview_sha256 is None
    replacement.dispose(new_session)


@pytest.mark.parametrize("width,height", [(832, 480), (896, 512), (1280, 736)])
def test_high_resolution_ready(configured, width, height):
    adapter, session = initialize(
        size=f"{width}x{height}",
        source_stream=dict(format="rgb24", width=width, height=height, fps=24),
        source_message_budget_bytes=DEFAULT_MESSAGE_BYTES,
    )
    ready = adapter.session_ready_message(session)
    assert adapter.clip == 17
    assert ready["max_source_batch_frames"] == 17
    assert adapter.ledger.limits.pending_frames == 4 * 17
    assert ready["max_message_bytes"] == DEFAULT_MESSAGE_BYTES
    assert ready["max_source_message_bytes"] == 17 * width * height * 3 + 4096
    assert ready["max_pending_source_bytes"] == 4 * 17 * width * height * 3


@pytest.mark.parametrize("width,height", [(1280, 736), (1280, 720), (832, 480)])
@pytest.mark.parametrize("target_width,target_height", [(832, 480), (1280, 736)])
def test_resize_negotiates_encoded_size_but_accounts_original_bytes(
    configured, monkeypatch, width, height, target_width, target_height
):
    path = Path(configured.model_path)
    runtime = json.loads((path / "zing_config.json").read_text())
    runtime["h3_media_config"].update(
        max_short_edge=1024,
        max_area_config={"source": 1280 * 1024},
        max_source_target_aspect_ratio_factor=2.0,
    )
    raw = json.dumps(runtime).encode()
    (path / "zing_config.json").write_bytes(raw)
    profile = json.loads((path / "zing_realtime_profile.json").read_text())
    profile["model_config_sha256"] = hashlib.sha256(raw).hexdigest()
    (path / "zing_realtime_profile.json").write_text(json.dumps(profile))
    adapter, session = initialize(
        size=f"{target_width}x{target_height}",
        source_stream=dict(format="rgb24", width=width, height=height, fps=24),
    )
    ready = adapter.session_ready_message(session)
    assert ready["source_stream"]["width"] == width
    assert ready["source_encoding"]["width"] == 832
    assert ready["source_encoding"]["height"] == 480
    assert ready["source_encoding"]["resize_applied"] == (width == 1280)
    assert session.request.size == f"{target_width}x{target_height}"
    assert session.request.realtime_preview_max_width == target_width
    assert adapter.ledger.frame_bytes == width * height * 3
    assert ready["max_pending_source_bytes"] == 4 * 17 * width * height * 3
    payload = dict(client_seq=0, source_frame_start=0, num_frames=1, pts_us=0, data=b"")
    with pytest.raises(ValueError, match="dimensions"):
        adapter.ledger.accept(payload, event_id=1, message_bytes=100)
    assert adapter.ledger.accepted_frames == 0
    monkeypatch.setattr(
        source, "build_realtime_sampling_params", lambda *a, **kw: SimpleNamespace(**kw)
    )
    params = adapter.build_sampling_params(
        session,
        configured,
        SimpleNamespace(request_id="geometry"),
        SimpleNamespace(prompt="Edit"),
        1,
    )
    output = params.zing_request["messages"][1]["content"][0]["output"]
    assert (output["width"], output["height"]) == (target_width, target_height)


@pytest.mark.parametrize(
    "width,height,budget,code",
    [
        (1280, 720, DEFAULT_MESSAGE_BYTES, "source_dimensions_unaligned"),
        (1920, 1088, DEFAULT_MESSAGE_BYTES, "source_message_too_large"),
        (1280, 736, 2 * DEFAULT_MESSAGE_BYTES, "source_transport_mismatch"),
    ],
)
def test_transport_rejected_before_media_allocation(
    configured, monkeypatch, width, height, budget, code
):
    from sglang.multimodal_gen.runtime.realtime.source_transport import (
        SourceTransportError,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Media allocation happened before transport validation")

    monkeypatch.setattr(source.tempfile, "mkdtemp", forbidden)
    monkeypatch.setattr(source.Image, "open", forbidden)
    with pytest.raises(SourceTransportError) as error:
        initialize(
            size=f"{width}x{height}",
            source_stream=dict(format="rgb24", width=width, height=height, fps=24),
            source_message_budget_bytes=budget,
            references=[dict(id="image", format="png", payload=b"image")],
        )
    assert error.value.code == code


@pytest.mark.parametrize("oversized_init", [False, True])
def test_init_errors_have_stable_wire_codes(configured, monkeypatch, oversized_init):
    from fastapi import WebSocketDisconnect

    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
        realtime_video_api as api,
    )

    initial = dict(
        type="init",
        prompt="Render snow",
        size="1280x720",
        zing_realtime_api_version=1,
        source_stream=dict(format="rgb24", width=1280, height=720, fps=24),
    )
    if oversized_init:
        configured.pipeline_config.zing_reference_max_images = 2
        initial.update(
            size="32x32",
            source_stream=dict(format="rgb24", width=32, height=32, fps=24),
            source_message_budget_bytes=65536,
            references=[dict(id="ref", format="png", payload=bytes(80 << 10))],
        )
        monkeypatch.setattr(
            source.Image,
            "open",
            lambda *a, **kw: pytest.fail(
                "An oversized init must be rejected before image verification"
            ),
        )
    adapter = source.ZingH3SourceStreamAdapter()
    monkeypatch.setattr(api, "get_global_server_args", lambda: configured)
    monkeypatch.setattr(
        api,
        "get_realtime_model_adapter",
        lambda args: SimpleNamespace(for_request=lambda req: adapter),
    )

    class Socket:
        def __init__(self):
            self.received = False
            self.sent = []

        async def receive_bytes(self):
            if self.received:
                raise WebSocketDisconnect()
            self.received = True
            return msgspec.msgpack.encode(initial)

        async def send_bytes(self, data):
            self.sent.append(msgspec.msgpack.decode(data))

    ws = Socket()
    session = GenerateSession()
    with pytest.raises(WebSocketDisconnect):
        asyncio.run(api._listen_generate_request(ws, session))
    assert len(ws.sent) == 1
    assert ws.sent[0]["code"] == (
        "source_init_too_large" if oversized_init else "source_dimensions_unaligned"
    )
    if not oversized_init:
        assert "divisible by 32" in ws.sent[0]["content"]
    assert session.input_temp_dir is None
    assert session.request is None


def test_startup_rejects_insufficient_concurrent_transport_memory():
    from sglang.multimodal_gen.runtime.realtime.source_transport import (
        SourceTransportBudget,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    args = SimpleNamespace(
        pipeline_config=None,  # This test covers the generic transport budget.
        batching_mode="dynamic",
        batching_max_size=1,
        batching_delay_ms=0,
        realtime_max_sessions=8,
        realtime_max_sessions_per_worker=8,
        realtime_max_message_bytes=SourceTransportBudget().message_bytes,
        realtime_transport_host_bytes=8 * SourceTransportBudget().session_bytes - 1,
    )
    with pytest.raises(ValueError, match="host budget"):
        ServerArgs._validate_batching(args)


@pytest.mark.parametrize("reason", ["ready", "timeout", "prompt", "eos", "cancelled"])
def test_joint_wait_is_bounded_and_interruptible(configured, reason):
    adapter, session = initialize()
    adapter.profile.update(joint_forward="cross_chunk", joint_wait_ms=30)
    adapter.pending_frames = 1
    adapter.source_encoder = {"mode": "progressive"}

    async def run():
        task = asyncio.create_task(adapter.wait_for_next_chunk(session))
        await asyncio.sleep(0.002)
        if reason == "ready":
            adapter.pending_frames = adapter.clip
        elif reason == "prompt":
            adapter._pending_prompt_command = dict(prompt="New prompt", event_id=2)
        elif reason == "eos":
            adapter.end = {"frame_end": 1}
        elif reason == "cancelled":
            task.cancel()
        adapter.available.set()
        if reason == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await asyncio.wait_for(task, 0.5)

    asyncio.run(run())
    assert adapter.joint_wait["reason"] == reason
    assert 0 <= adapter.joint_wait["actual_ms"] < 500
    if reason == "timeout":
        assert adapter.joint_wait["actual_ms"] >= 25


@pytest.mark.parametrize("value", [0, -1, 1001, True, "20", float("inf"), float("nan")])
def test_joint_wait_rejects_invalid_deployment_budget(configured, value):
    path = (
        __import__("pathlib").Path(configured.model_path) / "zing_realtime_profile.json"
    )
    profile = json.loads(path.read_text())
    profile.update(joint_forward="cross_chunk", joint_wait_ms=value)
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="streaming settings"):
        initialize()


@pytest.mark.parametrize(
    "lengths,split,dispatched,pending,target,missing",
    [
        ([1, 2, 2], False, 0, 1, 1, 0),
        ([1, 2, 2], False, 1, 1, 16, 15),
        ([3, 2], True, 0, 1, 1, 0),
        ([3, 2], True, 9, 1, 17, 16),
        ([1, 2, 2], False, 0, 9, 1, 0),
        ([3, 2], False, 0, 1, 9, 8),
        ([3, 2], False, 0, 9, 9, 0),
        ([5], False, 0, 1, 17, 16),
    ],
)
def test_auto_wait_uses_checkpoint_chunks_and_codec_watermarks(
    configured, lengths, split, dispatched, pending, target, missing
):
    adapter, _ = initialize()
    adapter.source_encoder = {"mode": "progressive"}
    adapter.joint_plan = source.ChunkPlan(lengths, split)
    adapter.dispatched_source_frames = dispatched
    adapter.pending_frames = pending
    frames, budget = adapter._joint_wait_target()
    assert frames == target
    assert budget == pytest.approx(missing * 1000 / 24)


@pytest.mark.parametrize("lengths,split", [([1, 2, 2], False), ([3, 2], True)])
@pytest.mark.parametrize("budget", ["auto", 30])
def test_first_latent_dispatch_does_not_wait_for_following_chunk(
    configured, monkeypatch, lengths, split, budget
):
    adapter, session = initialize()
    adapter.profile.update(joint_forward="cross_chunk", joint_wait_ms=budget)
    adapter.source_encoder = {"mode": "progressive"}
    adapter.joint_plan = source.ChunkPlan(lengths, split)
    adapter.pending_frames = 1

    async def unexpected_wait():
        pytest.fail("The first latent is ready without another source frame")

    monkeypatch.setattr(adapter.available, "wait", unexpected_wait)
    asyncio.run(adapter.wait_for_next_chunk(session))
    assert adapter.joint_wait["target_frames"] == 1
    assert adapter.joint_wait["reason"] == "ready"
    if budget == "auto":
        assert adapter.joint_wait["budget_ms"] == 0


def test_unsplit_first_chunk_waits_for_its_source_dependency(configured):
    adapter, session = initialize()
    adapter.profile.update(joint_forward="cross_chunk", joint_wait_ms="auto")
    adapter.source_encoder = {"mode": "progressive"}
    adapter.joint_plan = source.ChunkPlan([3, 2], False)
    adapter.pending_frames = 1

    async def run():
        task = asyncio.create_task(adapter.wait_for_next_chunk(session))
        await asyncio.sleep(0)
        assert not task.done()
        adapter.pending_frames = 9
        adapter.available.set()
        await asyncio.wait_for(task, 0.5)

    asyncio.run(run())
    assert adapter.joint_wait["target_frames"] == 9
    assert adapter.joint_wait["reason"] == "ready"


def test_joint_defaults_need_no_user_tuning(configured):
    from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams

    adapter, _ = initialize()
    assert adapter.profile.get("joint_forward", "cross_chunk") == "cross_chunk"
    assert adapter.profile.get("joint_wait_ms", "auto") == "auto"
    assert ZingH3SamplingParams().zing_joint_forward == "cross_chunk"


@pytest.mark.parametrize("limit", [None, 6143, 6144, 6145])
def test_initial_reference_decoded_budget_is_aggregate_and_header_only(
    configured, monkeypatch, limit
):
    from PIL import Image

    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.conditioning import (
        release_api_media,
    )

    configured.pipeline_config.zing_reference_max_decoded_bytes = limit
    refs = [_reference("one"), _reference("two")]
    # A reference always decodes to RGB, independently of its compressed size.
    assert sum(len(ref["payload"]) for ref in refs) < 6143
    monkeypatch.setattr(
        Image.Image,
        "convert",
        lambda *a, **k: pytest.fail("Admission must precede RGB decode"),
    )
    request = RealtimeVideoGenerationsRequest(
        type="init",
        prompt="Render snow",
        zing_realtime_api_version=1,
        source_stream=dict(format="rgb24", width=32, height=32, fps=24),
        size="32x32",
        references=refs,
    )
    adapter, session = source.ZingH3SourceStreamAdapter(), GenerateSession()
    session.set_adapter(adapter)

    async def run():
        try:
            if limit == 6143:
                with pytest.raises(ValueError, match="configured decoded size limit"):
                    await adapter.on_init(session, request)
            else:
                await adapter.on_init(session, request)
                assert len(adapter.reference_items) == 2
        finally:
            await adapter.close_conditioning(session)
            release_api_media(session)
        assert (
            session.conditioning_resources.ledger.snapshot()["tenants"]
            .get(session.id, {})
            .get("ram", 0)
            == 0
        )

    asyncio.run(run())


@pytest.mark.parametrize("limit", [6143, 6144, 6145, None])
def test_reference_event_decoded_budget_rejects_without_decode_and_recovers(
    configured, monkeypatch, limit
):
    from PIL import Image

    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.conditioning import (
        release_api_media,
    )

    _enable_reference_updates(configured)
    configured.pipeline_config.zing_reference_max_decoded_bytes = limit
    adapter, session = initialize(references=[_reference("old")])
    active = list(adapter.reference_items)
    initial_ram = (
        session.conditioning_resources.ledger.snapshot()["tenants"]
        .get(session.id, {})
        .get("ram", 0)
    )
    ws = PromptSocket()
    refs = [_reference("new-1"), _reference("new-2")]
    monkeypatch.setattr(
        Image.Image,
        "convert",
        lambda *a, **k: pytest.fail("Admission must precede RGB decode"),
    )

    async def run():
        try:
            await adapter.ingest_wire_event(
                ws,
                session,
                dict(
                    type="event",
                    kind="prompt",
                    event_id=1,
                    payload="new",
                    references=refs,
                ),
                100,
            )
            if limit == 6143:
                assert ws.messages[-1]["type"] == "error"
                assert ws.messages[-1]["fatal"] is False
                assert adapter._pending_prompt_command["rejected"] is True
                assert adapter._pending_reference_upload is None
                assert (
                    session.conditioning_resources.ledger.snapshot()["tenants"]
                    .get(session.id, {})
                    .get("ram", 0)
                    == initial_ram
                )
            else:
                assert ws.messages[-1]["type"] == "control_ack"
                assert len(adapter._pending_reference_upload.items) == 2
            assert adapter.reference_items == active
            assert session.request.prompt == "Render snow"
            # Both a smaller valid set and clearing the set remain possible.
            for event_id, references in [(2, refs[:1]), (3, [])]:
                await adapter.ingest_wire_event(
                    ws,
                    session,
                    dict(
                        type="event",
                        kind="prompt",
                        event_id=event_id,
                        payload="valid",
                        references=references,
                    ),
                    100,
                )
                assert ws.messages[-1]["type"] == "control_ack"
                assert len(adapter._pending_reference_upload.items) == len(references)
        finally:
            await adapter.close_conditioning(session)
            release_api_media(session)
        assert (
            session.conditioning_resources.ledger.snapshot()["tenants"]
            .get(session.id, {})
            .get("ram", 0)
            == 0
        )

    asyncio.run(run())


@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.5, "1024"])
def test_reference_decoded_budget_requires_positive_integer_or_null(invalid):
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
        ZingH3CausalPipelineConfig,
    )

    config = ZingH3CausalPipelineConfig(zing_reference_max_decoded_bytes=invalid)
    with pytest.raises(ValueError, match="zing_reference_max_decoded_bytes"):
        config.validate_server_args(SimpleNamespace())


@pytest.mark.parametrize("limit", [None, 1, 1073741824])
def test_reference_decoded_budget_pipeline_override(limit):
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
        ZingH3CausalPipelineConfig,
    )

    config = ZingH3CausalPipelineConfig()
    assert config.zing_reference_max_decoded_bytes is None
    config.update_pipeline_config(dict(zing_reference_max_decoded_bytes=limit))
    assert config.zing_reference_max_decoded_bytes == limit


@pytest.mark.parametrize("limit", [0, 2, 32])
def test_configured_reference_count_init_update_and_ready(configured, limit):
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.conditioning import (
        release_api_media,
    )

    configured.pipeline_config.zing_reference_max_images = limit
    configured.pipeline_config.zing_condition_max_tokens = max(
        50000, 1700 * limit + 1000
    )
    _enable_reference_updates(configured)
    refs = [_reference(f"picture-{i}") for i in range(limit + 1)]
    adapter, session = initialize(references=refs[:limit])
    assert adapter.ready["max_references"] == limit
    assert not any("user_media" in key for key in adapter.ready)
    assert adapter.ready["max_reference_bytes"] == limit * (64 << 20)
    assert adapter.ready["max_reference_decoded_bytes"] == limit * (48 << 20)
    assert len(adapter.reference_items) == limit
    ws = PromptSocket()

    async def run():
        try:
            await adapter.ingest_wire_event(
                ws,
                session,
                dict(
                    type="event",
                    kind="prompt",
                    event_id=1,
                    payload="new",
                    references=refs[:limit],
                ),
                100,
            )
            assert ws.messages[-1]["type"] == "control_ack"
            assert len(adapter._pending_reference_upload.items) == limit
            await adapter.ingest_wire_event(
                ws,
                session,
                dict(
                    type="event",
                    kind="prompt",
                    event_id=2,
                    payload="invalid",
                    references=refs,
                ),
                100,
            )
            assert ws.messages[-1]["type"] == "error" and not ws.messages[-1]["fatal"]
            assert session.request.prompt == "Render snow"
            assert len(adapter.reference_items) == limit
            await adapter.ingest_wire_event(
                ws,
                session,
                dict(
                    type="event",
                    kind="prompt",
                    event_id=3,
                    payload="recover",
                    references=[],
                ),
                100,
            )
            assert ws.messages[-1]["type"] == "control_ack"
        finally:
            await adapter.close_conditioning(session)
            release_api_media(session)

    asyncio.run(run())
    with pytest.raises(ValueError, match=f"at most {limit}"):
        initialize(references=refs)


@pytest.mark.parametrize("invalid", [-1, True, False, 1.5, "32", None])
def test_reference_count_requires_nonnegative_integer(invalid):
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
        ZingH3CausalPipelineConfig,
    )

    config = ZingH3CausalPipelineConfig(zing_reference_max_images=invalid)
    with pytest.raises(ValueError, match="zing_reference_max_images"):
        config.validate_server_args(SimpleNamespace())


@pytest.mark.parametrize("limit", [0, 2, 32])
def test_reference_count_pipeline_override(limit):
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
        ZingH3CausalPipelineConfig,
    )

    config = ZingH3CausalPipelineConfig(decoder_config=dict(type="official_h3"))
    assert config.zing_reference_max_images == 21
    config.update_pipeline_config(dict(zing_reference_max_images=limit))
    assert config.zing_reference_max_images == limit


@pytest.mark.parametrize("count", [0, 9, 21, 32])
def test_reference_capacity_derives_worst_case_bytes_and_transport(count):
    from sglang.multimodal_gen.runtime.realtime.source_transport import (
        SourceTransportBudget,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_contracts import ReferenceLimits

    limits = ReferenceLimits(count)
    assert limits.reference_bytes == count * (64 << 20)
    assert limits.decoded_reference_bytes == count * (48 << 20)
    args = SimpleNamespace(
        pipeline_config=SimpleNamespace(zing_reference_max_images=count),
        realtime_max_sessions=1,
        realtime_max_sessions_per_worker=1,
    )
    transport = SourceTransportBudget.from_args(args)
    assert transport.message_bytes == (count + 1) * (64 << 20) + 4096
    assert transport.reference_bytes == (count + 1) * (64 << 20)
    assert (
        transport.session_bytes
        == 14 * transport.message_bytes + transport.reference_bytes + (1 << 20)
    )
    transport.validate_sessions(1)
    # Explicitly smaller resource budgets must not silently grow.
    args.realtime_transport_host_bytes = 1 << 20
    with pytest.raises(ValueError, match="host budget is too small"):
        SourceTransportBudget.from_args(args).validate_sessions(1)


def test_client_can_negotiate_smaller_message_budget(configured):
    configured.pipeline_config.zing_reference_max_images = 32
    adapter, session = initialize(source_message_budget_bytes=DEFAULT_MESSAGE_BYTES)
    assert adapter.ready["max_message_bytes"] == DEFAULT_MESSAGE_BYTES
    assert adapter.ready["max_references"] == 32
    assert adapter.ready["max_reference_decoded_bytes"] == 32 * (48 << 20)
    ws = PromptSocket()
    from sglang.multimodal_gen.runtime.realtime.source_transport import (
        SourceTransportError,
    )

    with pytest.raises(SourceTransportError, match="exceeds"):
        asyncio.run(
            adapter.ingest_wire_event(
                ws,
                session,
                dict(type="event", kind="heartbeat"),
                DEFAULT_MESSAGE_BYTES + 1,
            )
        )


def test_worst_case_budget_allows_over_legacy_64m_on_real_websocket(configured):
    import websockets

    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.conditioning import (
        release_api_media,
    )
    from sglang.multimodal_gen.runtime.realtime.source_transport import (
        SourceTransportBudget,
    )

    configured.pipeline_config.zing_reference_max_images = 2
    _enable_reference_updates(configured)
    ref = _reference("picture-1")
    ref["payload"] += bytes((32 << 20) + 8192 - len(ref["payload"]))
    refs = [ref, dict(ref, id="picture-2")]
    initial = dict(
        type="init",
        prompt="old",
        zing_realtime_api_version=1,
        source_stream=dict(format="rgb24", width=32, height=32, fps=24),
        size="32x32",
        references=refs,
    )
    wire = msgspec.msgpack.encode(initial)
    assert len(wire) > DEFAULT_MESSAGE_BYTES
    budget = SourceTransportBudget.from_args(configured)
    assert len(wire) < budget.message_bytes
    receipts = []

    async def exercise():
        async def peer(ws):
            session = GenerateSession()
            adapter = source.ZingH3SourceStreamAdapter()
            session.set_adapter(adapter)
            try:
                raw = await ws.recv()
                request = RealtimeVideoGenerationsRequest.model_validate(
                    msgspec.msgpack.decode(raw)
                )
                await adapter.on_init(session, request)
                session.set_request(request)
                await ws.send(msgspec.msgpack.encode(adapter.ready))
                raw = await ws.recv()

                class Writer:
                    async def send_bytes(self, value):
                        await ws.send(value)

                await adapter.ingest_wire_event(
                    Writer(), session, msgspec.msgpack.decode(raw), len(raw)
                )
                receipts.append(len(adapter._pending_reference_upload.items))
            finally:
                await adapter.close_conditioning(session)
                release_api_media(session)

        async with websockets.serve(
            peer, "127.0.0.1", 0, max_size=budget.message_bytes, compression=None
        ) as server:
            port = server.sockets[0].getsockname()[1]
            async with websockets.connect(
                f"ws://127.0.0.1:{port}",
                max_size=budget.message_bytes,
                compression=None,
            ) as ws:
                await ws.send(wire)
                ready = msgspec.msgpack.decode(await ws.recv())
                assert ready["max_references"] == 2
                assert ready["max_message_bytes"] == budget.message_bytes
                await ws.send(
                    msgspec.msgpack.encode(
                        dict(
                            type="event",
                            kind="prompt",
                            event_id=1,
                            payload="new",
                            references=refs,
                        )
                    )
                )
                ack = msgspec.msgpack.decode(await ws.recv())
                assert ack["type"] == "control_ack"
        assert receipts == [2]

    asyncio.run(exercise())


def test_reference_capacity_rejects_unaddressable_transport():
    from sglang.multimodal_gen.runtime.realtime.source_transport import (
        SourceTransportBudget,
    )

    args = SimpleNamespace(
        pipeline_config=SimpleNamespace(zing_reference_max_images=2**64)
    )
    with pytest.raises(ValueError, match="addressable"):
        SourceTransportBudget.from_args(args)


@pytest.mark.parametrize(
    "json_count,cli_count,expected",
    [(None, None, 21), (2, None, 2), (None, 32, 32), (2, 32, 32), (32, 0, 0)],
)
def test_reference_count_cli_overrides_json_and_drives_transport(
    tmp_path, monkeypatch, json_count, cli_count, expected
):
    import sys

    import sglang.multimodal_gen.registry as registry
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
        ZingH3CausalPipelineConfig,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_contracts import ReferenceLimits
    from sglang.multimodal_gen.runtime.server_args.server_args import (
        ServerArgs,
        prepare_server_args,
    )

    # Resolve a CPU-only config without loading model files or unrelated GPU setup.
    monkeypatch.setattr(
        registry,
        "get_model_info",
        lambda *a, **k: SimpleNamespace(pipeline_config_cls=ZingH3CausalPipelineConfig),
        raising=False,
    )
    monkeypatch.setattr(
        ServerArgs, "__post_init__", lambda self: self._validate_batching()
    )
    argv = ["--model-path", str(tmp_path)]
    if json_count is not None:
        path = tmp_path / "pipeline.json"
        path.write_text(json.dumps(dict(zing_reference_max_images=json_count)))
        argv += ["--pipeline-config-path", str(path)]
    if cli_count is not None:
        if json_count is None:
            argv += [f"--zing-reference-max-images={cli_count}"]
        else:
            argv += ["--zing-reference-max-images", str(cli_count)]
    monkeypatch.setattr(sys, "argv", ["sglang", *argv])
    args = prepare_server_args(argv)
    assert args.pipeline_config.zing_reference_max_images == expected
    assert (
        args.realtime_max_message_bytes == ReferenceLimits(expected).media_bytes + 4096
    )
    assert args.is_arg_explicitly_set("zing_reference_max_images") == (
        cli_count is not None
    )


@pytest.mark.parametrize("value", ["1.5", "abc", None])
def test_reference_count_cli_rejects_non_integer_or_missing_value(value):
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    argv = ["--model-path", "unused", "--zing-reference-max-images"]
    if value is not None:
        argv.append(value)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(argv)
    assert error.value.code == 2


def test_reference_count_cli_negative_rejected_by_startup(monkeypatch, tmp_path):
    import sys

    import sglang.multimodal_gen.registry as registry
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
        ZingH3CausalPipelineConfig,
    )
    from sglang.multimodal_gen.runtime.server_args.server_args import (
        ServerArgs,
        prepare_server_args,
    )

    monkeypatch.setattr(
        registry,
        "get_model_info",
        lambda *a, **k: SimpleNamespace(pipeline_config_cls=ZingH3CausalPipelineConfig),
        raising=False,
    )
    monkeypatch.setattr(
        ServerArgs, "__post_init__", lambda self: self._validate_batching()
    )
    argv = ["--model-path", str(tmp_path), "--zing-reference-max-images", "-1"]
    monkeypatch.setattr(sys, "argv", ["sglang", *argv])
    with pytest.raises(ValueError, match="nonnegative integer"):
        prepare_server_args(argv)


def test_reference_count_cli_does_not_add_setting_to_other_models():
    from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig

    parser = argparse.ArgumentParser()
    PipelineConfig.add_cli_args(parser)
    parsed = parser.parse_args(["--zing-reference-max-images", "32"])
    config = PipelineConfig()
    config.update_config_from_dict(vars(parsed))
    assert not hasattr(config, "zing_reference_max_images")


@pytest.mark.parametrize("enabled", [None, True, False])
def test_prompt_prefetch_publishes_before_dispatch_and_closes(
    configured, monkeypatch, enabled
):
    from sglang.multimodal_gen.runtime.conditioning import prefetch

    class Publisher:
        def __init__(self, endpoint, **kwargs):
            self.endpoint, self.messages, self.closed = endpoint, [], False

        def submit(self, command):
            self.messages.append(
                msgspec.msgpack.decode(msgspec.msgpack.encode(command))
            )

        def close(self):
            self.closed = True

    monkeypatch.setattr(prefetch, "PromptPrefetchPublisher", Publisher)
    configured.zing_conditioning_prefetch = enabled
    configured.zing_conditioning_mode = "remote"
    _enable_reference_updates(configured)
    adapter, session = initialize()
    publisher = adapter.prompt_prefetch
    if enabled is False:
        assert publisher is None
        adapter.clear_state(session)
        return
    ws = PromptSocket()

    async def run():
        await adapter.ingest_wire_event(
            ws,
            session,
            dict(
                type="event",
                kind="prompt",
                event_id=1,
                payload="new",
                references=[_reference()],
            ),
            100,
        )
        assert publisher.messages[0]["prompt"] == "new"
        assert (
            publisher.messages[0]["references"][0]["payload"] == _reference()["payload"]
        )
        assert adapter._selected_prompt_command is None
        assert adapter._pending_prompt_command["event_id"] == 1
        assert ws.messages[-1]["type"] == "control_ack"
        adapter.clear_state(session)
        assert publisher.closed

    asyncio.run(run())


@pytest.mark.parametrize("remote,quantum", [(False, "1"), (True, "0"), (True, "1")])
def test_source_decoder_overlap_requires_dedicated_single_block_path(
    configured, monkeypatch, remote, quantum
):
    configured.zing_source_decoder_overlap = True
    configured.disagg_decoder_port = 8098 if remote else None
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM", quantum)
    if not remote or quantum != "1":
        with pytest.raises(ValueError, match="dedicated decoder and source quantum 1"):
            initialize()
    else:
        adapter, session = initialize()
        assert adapter.source_decoder_overlap is True
        assert adapter.profile["source_decoder"] == {"remote": {"port": 8098}}
        adapter.clear_state(session)


def _queue_source_packets(adapter, sizes):
    for seq, size in enumerate(sizes):
        start = adapter.ledger.accepted_frames
        payload = dict(
            client_seq=seq,
            source_frame_start=start,
            num_frames=size,
            pts_us=start * 1_000_000 // 24,
            data=bytes(size * adapter.ledger.frame_bytes),
        )
        item = dict(
            payload=payload, event_id=seq + 1, message_bytes=len(payload["data"]) + 100
        )
        adapter.ledger.accept(**item)
        adapter.pending.append(item)
        adapter.pending_frames += size


@pytest.mark.parametrize(
    "overlap,capacity,expected",
    [(False, 17, 0), (True, 17, 16), (True, 8, 8), (True, 7, 0)],
)
def test_continuation_can_append_whole_packets_within_capacity(
    configured, monkeypatch, overlap, capacity, expected
):
    adapter, session = initialize()
    adapter.more_work = True
    adapter.source_append_overlap = overlap
    adapter.append_capacity_frames = capacity
    _queue_source_packets(adapter, [8, 8])
    monkeypatch.setattr(
        source.BaseRealtimeModelAdapter,
        "prepare_next_request",
        lambda *a: SimpleNamespace(block_idx=1, extra={}),
    )
    batch = adapter.prepare_next_request(session, configured, None)
    assert (
        sum(
            item["payload"]["num_frames"]
            for item in batch.extra["zing_source_command"]["appends"]
        )
        == expected
    )
    assert adapter.dispatched_source_frames == expected
    assert adapter.pending_frames == 16 - expected
    adapter.clear_state(session)


def test_no_capacity_and_no_progress_is_rejected_instead_of_spinning(
    configured, monkeypatch
):
    adapter, session = initialize()
    adapter.source_append_overlap = True
    adapter.append_capacity_frames = 7
    _queue_source_packets(adapter, [8])
    monkeypatch.setattr(
        source.BaseRealtimeModelAdapter,
        "prepare_next_request",
        lambda *a: SimpleNamespace(block_idx=1, extra={}),
    )
    with pytest.raises(ValueError, match="cannot progress"):
        adapter.prepare_next_request(session, configured, None)
    assert len(adapter.pending) == 1
    adapter.clear_state(session)


def test_credit_coalesces_progress_and_preserves_old_output_provenance(configured):
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_video_api import (
        _LockedRealtimeWebSocket,
    )

    adapter, session = initialize()
    adapter.source_decoder_overlap = True
    adapter.source_work_quantum = 1
    _queue_source_packets(adapter, [8, 8])
    adapter.dispatched_source_frames = 16
    messages = []

    class Socket:
        async def send_bytes(self, payload):
            messages.append(msgspec.msgpack.decode(payload))

    def progress(end):
        session.new_chunk()
        adapter.on_chunk_complete(
            session,
            SimpleNamespace(
                usage={"zing_source_consumed_end": end, "zing_source_more_work": True}
            ),
        )

    async def run():
        ws = _LockedRealtimeWebSocket(Socket())
        progress(1)  # A codec slice may end inside a transport packet.
        await adapter.flush_source_credit(ws)
        assert messages == [] and adapter.ledger.consumed_frames == 0
        progress(8)
        progress(9)
        async with ws.send_group():  # Existing RGB group calls this reentrantly.
            await adapter.flush_source_credit(ws)
        assert [m["source_frame_end"] for m in messages] == [8]
        assert adapter.ledger.provenance_for(PixelRange(0, 8), 8).pixels == PixelRange(
            0, 8
        )
        adapter.ledger.delivered(PixelRange(0, 8))
        progress(16)
        await adapter.flush_source_credit(ws)
        await adapter.flush_source_credit(ws)
        assert [m["source_frame_end"] for m in messages] == [8, 16]
        assert len(adapter.ledger.records) == 1
        for invalid in (15, 17, True):
            with pytest.raises(ValueError, match="ownership watermark"):
                progress(invalid)
        assert adapter.pending_source_consumed_end == 16

    asyncio.run(run())
    adapter.clear_state(session)


def test_output_completion_flushes_progress_before_waiting_for_credit_input(configured):
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
        realtime_video_api as api,
    )

    adapter, session = initialize()
    adapter.source_decoder_overlap = True
    _queue_source_packets(adapter, [8, 8])

    async def run():
        input_ready = asyncio.Event()
        output_flushed = asyncio.Event()
        output_release = asyncio.Event()
        messages = []

        class Socket:
            async def send_bytes(self, payload):
                message = msgspec.msgpack.decode(payload)
                messages.append(message)
                if message.get("source_frame_end") == 16:
                    input_ready.set()

        ws = api._LockedRealtimeWebSocket(Socket())

        async def wait_input(_session):
            await input_ready.wait()

        adapter.wait_for_next_chunk = wait_input

        async def old_output():
            adapter.pending_source_consumed_end = 8
            await adapter.flush_source_credit(ws)
            output_flushed.set()
            await output_release.wait()

        output = asyncio.create_task(old_output())
        await output_flushed.wait()
        adapter.pending_source_consumed_end = 16
        waiter = asyncio.create_task(
            api._wait_for_source_input(adapter, session, output, ws)
        )
        await asyncio.sleep(0)
        assert not waiter.done()
        output_release.set()
        await asyncio.wait_for(waiter, 2)
        assert [m["source_frame_end"] for m in messages] == [8, 16]

    asyncio.run(run())
    adapter.clear_state(session)


def test_adapter_init_applies_disagg_role_policies(configured):
    configured.disagg_encoder_port = 31005
    configured.disagg_decoder_port = 31007
    adapter, session = initialize()
    assert (
        getattr(adapter, "profile", {})
        .get("source_decoder", {})
        .get("remote", {})
        .get("port")
        == 31007
    ), "profile source_decoder remote port must be injected from disagg_decoder_port"
    assert (
        getattr(adapter, "profile", {})
        .get("source_encoder", {})
        .get("remote", {})
        .get("port")
        == 31005
    ), "profile source_encoder remote port must be injected from disagg_encoder_port"
    asyncio.run(adapter.close_conditioning(session))


@pytest.mark.parametrize("quantum", [0, 1])
def test_cli_source_policy_reaches_worker_command(configured, monkeypatch, quantum):
    configured.disagg_encoder_port = 31005
    configured.disagg_decoder_port = 31007
    configured.zing_source_encoder_mode = "overlap"
    configured.zing_source_encoder_prefetch = True
    configured.zing_source_async_output = True
    configured.zing_source_work_quantum = quantum
    configured.zing_source_defer_final = False
    configured.zing_source_prompt_joint = False
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM", "invalid")
    monkeypatch.setenv("SGLANG_ZING_SOURCE_DEFER_FINAL", "1")
    monkeypatch.setenv("SGLANG_ZING_SOURCE_PROMPT_JOINT", "1")
    profile_path = Path(configured.model_path) / "zing_realtime_profile.json"
    original = profile_path.read_bytes()
    # Keep the generic request factory isolated; test the real source command assembly.
    monkeypatch.setattr(
        source.BaseRealtimeModelAdapter,
        "prepare_next_request",
        lambda *a: SimpleNamespace(extra={}, block_idx=0),
    )
    adapter, session = initialize()
    try:
        command = adapter.prepare_next_request(session, configured, None).extra[
            "zing_source_command"
        ]
        command = msgspec.msgpack.decode(msgspec.msgpack.encode(command))
        assert command["work_quantum"] == quantum
        assert command["defer_final"] is False and command["prompt_joint"] is False
        assert command["async_output"] is True
        assert command["encoder"]["remote"]["mode"] == "overlap"
        assert command["encoder"]["remote"]["cross_dispatch_prefetch"] is True
        assert command["encoder"]["remote"]["admission_endpoint"].startswith("ipc:///")
        assert command["decoder"]["remote"]["port"] == 31007
        assert profile_path.read_bytes() == original
    finally:
        adapter.clear_state(session)


@pytest.mark.parametrize("remote_codecs", [False, True])
def test_server_defaults_reach_source_command(configured, monkeypatch, remote_codecs):
    from unittest.mock import patch

    from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
        ZingH3CausalPipelineConfig,
    )
    from sglang.multimodal_gen.runtime.platforms import current_platform
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    # Validate real auto defaults without probing a CUDA device in CPU tests.
    monkeypatch.setattr(
        current_platform, "get_available_gpu_memory", lambda **kwargs: 80.0
    )
    # The adapter consumes validated Zing values, not public class defaults.
    with patch.object(
        PipelineConfig,
        "from_kwargs",
        return_value=ZingH3CausalPipelineConfig(decoder_config={"type": "official_h3"}),
    ):
        validated = ServerArgs.from_dict({"model_path": str(configured.model_path)})

    for name in (
        "zing_source_encoder_mode",
        "zing_source_encoder_prefetch",
        "zing_source_async_output",
        "zing_source_joint_forward",
        "zing_source_work_quantum",
        "zing_source_defer_final",
        "zing_source_prompt_joint",
        "zing_source_decoder_overlap",
        "zing_conditioning_prefetch",
        "realtime_max_sessions",
        "realtime_max_sessions_per_worker",
    ):
        setattr(configured, name, getattr(validated, name))
    if remote_codecs:
        configured.disagg_encoder_port = 31005
        configured.disagg_decoder_port = 31007
    monkeypatch.setattr(
        source.BaseRealtimeModelAdapter,
        "prepare_next_request",
        lambda *a: SimpleNamespace(extra={}, block_idx=0),
    )
    adapter, session = initialize()
    try:
        command = adapter.prepare_next_request(session, configured, None).extra[
            "zing_source_command"
        ]
        command = msgspec.msgpack.decode(msgspec.msgpack.encode(command))
        assert command["work_quantum"] == 1
        assert command["defer_final"] is True and command["prompt_joint"] is True
        assert command["async_output"] is True
        assert adapter.profile["joint_forward"] == "cross_chunk"
        assert adapter.source_decoder_overlap is remote_codecs
        if remote_codecs:
            assert command["encoder"]["remote"]["mode"] == "overlap"
            assert command["encoder"]["remote"]["cross_dispatch_prefetch"] is True
        else:
            assert "remote" not in command["encoder"]
        assert adapter.prompt_prefetch is None  # Local Qwen needs no publisher.
    finally:
        adapter.clear_state(session)


def test_cli_false_disables_profile_policies(configured):
    profile_path = Path(configured.model_path) / "zing_realtime_profile.json"
    profile = json.loads(profile_path.read_text())
    profile.update(
        async_source_output=True,
        source_encoder={
            "mode": "progressive",
            "remote": {
                "port": 31005,
                "mode": "overlap",
                "cross_dispatch_prefetch": True,
            },
        },
    )
    profile_path.write_text(json.dumps(profile))
    configured.zing_source_encoder_prefetch = False
    configured.zing_source_encoder_mode = "serial"
    configured.zing_source_async_output = False
    adapter, session = initialize()
    try:
        assert adapter.profile["async_source_output"] is False
        assert adapter.source_encoder["remote"] == {
            "port": 31005,
            "mode": "serial",
            "cross_dispatch_prefetch": False,
        }
        assert adapter.admission_socket is None
        assert json.loads(profile_path.read_text()) == profile
    finally:
        adapter.clear_state(session)


def test_cli_prefetch_requires_remote_overlap(configured):
    configured.zing_source_encoder_prefetch = True
    with pytest.raises(ValueError, match="requires a remote Encoder"):
        initialize()
    configured.disagg_encoder_port = 31005
    configured.zing_source_encoder_mode = "serial"
    with pytest.raises(ValueError, match="overlap deployment"):
        initialize()


def test_cli_decoder_overlap_uses_explicit_quantum(configured, monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM", "0")
    configured.zing_source_work_quantum = 1
    configured.zing_source_decoder_overlap = True
    configured.disagg_decoder_port = 31007
    adapter, session = initialize()
    try:
        assert adapter.source_decoder_overlap and adapter.source_work_quantum == 1
    finally:
        adapter.clear_state(session)


@pytest.mark.parametrize("mode", ["off", "source_first", "cross_chunk"])
def test_cli_joint_forward_overrides_profile(configured, monkeypatch, mode):
    profile_path = Path(configured.model_path) / "zing_realtime_profile.json"
    profile = json.loads(profile_path.read_text())
    profile["joint_forward"] = "off" if mode == "cross_chunk" else "cross_chunk"
    profile_path.write_text(json.dumps(profile))
    configured.zing_source_joint_forward = mode
    adapter, session = initialize()
    try:
        monkeypatch.setattr(
            source,
            "build_realtime_sampling_params",
            lambda *a, **kw: SimpleNamespace(**kw),
        )
        params = adapter.build_sampling_params(
            session,
            configured,
            SimpleNamespace(request_id="joint-cli"),
            SimpleNamespace(prompt="snow"),
            17,
        )
        assert params.zing_joint_forward == mode
        assert json.loads(profile_path.read_text()) == profile
    finally:
        adapter.clear_state(session)


@pytest.mark.parametrize(
    "tokens,images,error",
    [
        (40000, 5, None),
        (40001, 0, "CONDITION_PROMPT_TOO_LONG"),
        (7000, 25, None),  # Exactly 50000 estimated condition tokens.
        (7001, 25, "CONDITION_BUDGET_EXCEEDED"),
        (39999, 25, "CONDITION_BUDGET_EXCEEDED"),
        (1, 26, "CONDITION_REFERENCE_LIMIT"),
        (30000, 0, None),  # Performance recommendation is not a rejection gate.
    ],
)
def test_individual_and_combined_boundaries(tokens, images, error):
    policy = budget.ZingInputBudget(max_references=25)
    if error:
        with pytest.raises(ConditionError) as caught:
            policy.validate(tokens, images)
        assert caught.value.code == error
        assert caught.value.public_message.isascii()
    else:
        policy.validate(tokens, images)


@pytest.mark.parametrize(
    "name", ["zing_prompt_max_tokens", "zing_condition_max_tokens"]
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_invalid_deployment_budget(name, value):
    with pytest.raises(ValueError, match=name):
        budget.ZingInputBudget.from_config(SimpleNamespace(**{name: value}))


def test_real_export_tokenizer_counts_without_truncation(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    native = Tokenizer(
        WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "你好": 3}, unk_token="[UNK]")
    )
    native.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=native, unk_token="[UNK]", model_max_length=2
    )
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    (tmp_path / "model_index.json").write_text(
        json.dumps({"tokenizer": ["transformers", "PreTrainedTokenizerFast"]})
    )
    budget._prompt_tokenizer.cache_clear()
    try:
        for prompt in ("hello world 你好", "你好 unknown world hello", ""):
            assert budget.count_prompt_tokens(tmp_path, prompt) == len(
                tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
            )
        assert budget.count_prompt_tokens(tmp_path, "hello world 你好") == 3
        assert budget._prompt_tokenizer.cache_info().misses == 1
    finally:
        budget._prompt_tokenizer.cache_clear()


def test_missing_tokenizer_fails_closed(tmp_path):
    with pytest.raises(ConditionError) as caught:
        budget.count_prompt_tokens(tmp_path, "hello")
    assert caught.value.code == "CONDITION_UNAVAILABLE"


@pytest.mark.parametrize("reference_count", [0, 2])
@pytest.mark.parametrize("outcome", ["applied", "failed", "waiting"])
def test_api_prompt_budget_tracks_dispatched_reference_updates(
    configured, monkeypatch, reference_count, outcome
):
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.conditioning import (
        release_api_media,
    )

    _enable_reference_updates(configured)
    configured.pipeline_config.zing_condition_max_tokens = 5000
    initial = [_reference(str(i)) for i in range(2 if reference_count == 0 else 0)]
    adapter, session = initialize(prompt="old", references=initial)
    ws = PromptSocket()

    def prepare(self, session, args, chunk):
        inputs = self.sample_chunk_inputs(session, args, chunk, 17)
        return SimpleNamespace(
            block_idx=1,
            extra={},
            realtime_event_id=self.prompt_event,
            sampling_params=SimpleNamespace(prompt=inputs.prompt),
        )

    monkeypatch.setattr(
        source.BaseRealtimeModelAdapter, "prepare_next_request", prepare
    )
    monkeypatch.setattr(
        source.BaseRealtimeModelAdapter, "on_chunk_complete", lambda *a: None
    )

    async def run():
        try:
            await adapter.ingest_wire_event(
                ws,
                session,
                dict(
                    type="event",
                    kind="prompt",
                    event_id=1,
                    payload="new",
                    references=[_reference(str(i)) for i in range(reference_count)],
                ),
                100,
            )
            batch = adapter.prepare_next_request(session, configured, None)
            assert "prompt_tokens" not in batch.extra["zing_prompt_update"]
            adapter.on_chunk_complete(
                session,
                SimpleNamespace(
                    usage={
                        "zing_realtime_blocks": [{"event_id": 1}]
                        if outcome == "applied"
                        else []
                    },
                    conditioning_notice={"event_id": 1, "code": "CONDITION_INVALID"}
                    if outcome == "failed"
                    else None,
                ),
            )
            await adapter.ingest_wire_event(
                ws,
                session,
                dict(type="event", kind="prompt", event_id=2, payload="word " * 3000),
                100,
            )
            if reference_count == 0 and outcome == "applied":
                assert ws.messages[-1]["type"] == "control_ack"
            else:
                assert ws.messages[-1]["code"] == "CONDITION_BUDGET_EXCEEDED"
        finally:
            await adapter.close_conditioning(session)
            release_api_media(session)

    asyncio.run(run())


def test_init_rejects_before_reference_decode(configured, monkeypatch):
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import conditioning

    decode = Mock(side_effect=AssertionError("Image decoding must not start"))
    monkeypatch.setattr(conditioning, "verify_presentation_upload", decode)
    configured.pipeline_config.zing_condition_max_tokens = 2200
    with pytest.raises(ConditionError) as caught:
        initialize(prompt="hello", references=[_reference()])
    assert caught.value.code == "CONDITION_BUDGET_EXCEEDED"
    decode.assert_not_called()


def test_api_update_rejection_supersedes_and_recovers(configured):
    configured.pipeline_config.zing_condition_max_tokens = 2202
    configured.pipeline_config.zing_prompt_max_tokens = 3
    _enable_reference_updates(configured)
    adapter, session = initialize(prompt="old")
    ws = PromptSocket()

    async def update(event_id, prompt, **fields):
        await adapter.ingest_wire_event(
            ws,
            session,
            dict(
                type="event", kind="prompt", event_id=event_id, payload=prompt, **fields
            ),
            100,
        )

    async def run():
        try:
            await update(1, "two words", references=[_reference()])
            previous = adapter._pending_reference_upload
            assert ws.messages[-1]["type"] == "control_ack"
            await update(2, "one two three four", prompt_tokens=0)
            assert previous.items == []
            assert adapter._pending_reference_upload is None
            assert adapter._pending_prompt_command["rejected"]
            assert ws.messages[-1]["code"] == "CONDITION_PROMPT_TOO_LONG"
            assert ws.messages[-1]["fatal"] is False
            assert ws.messages[-1]["event_id"] == 2
            await update(3, "one two three", references=[_reference()])
            assert ws.messages[-1]["code"] == "CONDITION_BUDGET_EXCEEDED"
            await update(4, "one two three", references=[])
            assert ws.messages[-1]["type"] == "control_ack"
            assert adapter._pending_prompt_command["references"] == []
            assert "prompt_tokens" not in adapter._pending_prompt_command
            assert session.request.prompt == "old"
            assert adapter.ready["input_budget"]["max_prompt_tokens"] == 3
        finally:
            await adapter.close_conditioning(session)

    asyncio.run(run())


@pytest.mark.parametrize("cli_override", [False, True])
def test_budget_cli_overrides_pipeline_json(tmp_path, monkeypatch, cli_override):
    import sys

    import sglang.multimodal_gen.registry as registry
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
        ZingH3CausalPipelineConfig,
    )
    from sglang.multimodal_gen.runtime.server_args.server_args import (
        ServerArgs,
        prepare_server_args,
    )

    monkeypatch.setattr(
        registry,
        "get_model_info",
        lambda *a, **k: SimpleNamespace(pipeline_config_cls=ZingH3CausalPipelineConfig),
        raising=False,
    )
    monkeypatch.setattr(
        ServerArgs, "__post_init__", lambda self: self._validate_batching()
    )
    path = tmp_path / "pipeline.json"
    path.write_text(
        json.dumps(dict(zing_prompt_max_tokens=2000, zing_condition_max_tokens=8000))
    )
    argv = ["--model-path", str(tmp_path), "--pipeline-config-path", str(path)]
    if cli_override:
        argv += [
            "--zing-prompt-max-tokens",
            "40000",
            "--zing-condition-max-tokens=50000",
        ]
    monkeypatch.setattr(sys, "argv", ["sglang", *argv])
    args = prepare_server_args(argv)
    assert args.pipeline_config.zing_prompt_max_tokens == (
        40000 if cli_override else 2000
    )
    assert args.pipeline_config.zing_condition_max_tokens == (
        50000 if cli_override else 8000
    )
    assert args.pipeline_config.zing_reference_max_images == 21
