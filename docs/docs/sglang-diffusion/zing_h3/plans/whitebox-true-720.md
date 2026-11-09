# True 720p Whitebox implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement these owner-scoped tasks. The user approved the model-connection padding/cropping design in the implementation request.

**Goal:** Capture, display and record actual 1280×720 source/output video while H3 consumes an explicitly identified 1280×736 private canvas.

**Architecture:** Keep immutable source geometry on each session and Room. Replicate the bottom 16 rows at the model connection, then crop each decoded model frame before composition, playback or recording. Existing H3 codecs, ledgers and GPU workers remain unchanged and see aligned dimensions.

**Tech stack:** Python, existing Whitebox Room/CLI and MessagePack paths, pytest CPU protocol peers.

**Spec:** [Owning resolution contract](../specs/whitebox-render-segments.md#public-source-and-private-model-resolution), plus the user-confirmed implementation request.

## Constraints and review focus

- Default 480p is 832×480; an omitted selector preserves the configured default. Explicit selectors are `whitebox.resolution="480p"|"720p"`.
- Preserve source clocks, CLI SHA/proofs, native UI coordinates and archived source frames. Never resize video or label 736 as 720.
- Existing games pin their original geometry. Unsupported selectors/paths reject before CLI/model allocation; shared runtime defaults never change.
- V1 startup fingerprints use padded model RGB and private source geometry, while original CLI hashes remain public. V2 native condition previews remain independent.
- Check packed frame boundaries, padded input credit bytes, source-sized recordings, cancellation, default compatibility and concurrent Rooms with different geometry.

## Tasks

- [x] Add and verify immutable source/model geometry and strict resolution selection.
- [x] Freeze source spec per Room/session; validate joins and classic-path rejection before allocation.
- [x] Adapt private InteractiveSession init/source batches/credit/output and PlayStartup prepare identities, leaving public source data intact.
- [x] Register and run CPU regressions for 480, true 720 pixels, recording, independent Rooms and cleanup; update owning documentation.
- [x] Run affected checks and prepare the exact source candidate. GPU correctness/performance and online publication remain unverified and separately authorized.

## Local verification (2026-10-30)

Base: shared `white_box` commit `dd6554d5deb9d2c69cafb52e11075c3af82340bb`.
Tests ran on macOS CPU with `SGLANG_LIGHTWEIGHT_RUNTIME=1`, `PYTHONPATH=python`
and pytest `--confcutdir=python/sglang/multimodal_gen/test/unit/realtime`.

- Geometry/resolution owners: **38 passed**, including actual 480p/720p H.264 decode and MP4 recording dimensions, source/model RGB boundaries and private startup fingerprints.
- Geometry/resolution + startup + error origins + Gateway server: **175 passed**.
- Existing Whitebox, interactive, Room pipeline/rooms/control/member media, RTC lifecycle/pipeline, render contract/lifecycle and Room cleanup: **243 passed, 1 deselected**. The unmodified `test_worker_prompt_boundary_includes_admitted_prefetch` requires Torch, unavailable in this environment; it was separately attempted and its import failure recorded rather than claimed as passing. No GPU worker code changed.
- Existing classic Gateway close regression initially exposed a test synchronization race: coordinator release precedes socket close. The test now waits for the actual close completion too; product close behavior/timeouts were not changed. The normal classic path and general Gateway suite then passed **79 tests**.
- Follow-up: that close barrier does not make its late-metric assumption deterministic. With `drain_seconds=0.2`, source completion/encoding/ASGI scheduling can consume the existing tail deadline before the browser sends its final metric; cancellation then need not observe that metric. The reported `[False]` case remains unresolved as a test assumption. One identical-consumer comparison against exact `dd6554d5deb9d2c69cafb52e11075c3af82340bb` Gateway source and the candidate passed both, which is not a stability claim or a fix. Product deadlines were left unchanged. A separate controlled completed-delivery barrier should replace this timing assumption.
- Ruff, formatting, isort, diff checks, new AREA owner registration and owning-document links passed. Dependency deprecation warnings remain unrelated to this change.

These are local source/CPU results. They do not establish deployed CLI/Canvas
dimensions, live GPU acceptance, quality, FPS, latency or memory equivalence.

## Native RTC codec follow-up (2026-10-30)

- [x] Validate selectors before replacing an authenticated RTC resume/source.bind attachment. Conflicts preserve the old peer, subscription, Room and preparation state; no admission occurs.
- [x] Check the actual browser offer before choosing a codec policy. System Chrome **156.0.8078.48**, headless, with the product's two recvonly video transceivers and H.264/RTX preferences, advertised only H.264 Level 3.1 (`42001f`, `42e01f` and other profiles ending `1f`), without `max-recv-level`.
- [x] Use the actual source rate for larger capture-clock canvases, preserving the original 480p hint. Product PyAV SPS is `avc1.42c01f` for 720p at 24/30 FPS, and `avc1.42c020` at 60 FPS. Real decode and MP4 remux preserve all eight test frames and bitrate rollover preserves the hint. Model, live source and paired archive consumers share the policy.
- [x] Keep the existing native RTC Level 3.1 envelope. 720p above 30 effective CLI FPS is rejected with `RTC_CODEC_UNSUPPORTED`, `stage=validation`, before reservation/allocation or replacing a current attachment; WS keeps rates through 60. This does not negotiate Level 3.2 or falsify an SPS/SDP value.
- [x] Existing-Room RTC preflight reads its already validated CLI `driver.descriptor.fps` before adapter metadata or launch defaults. Four reverse-environment cases reproduce the original wrong decision and verify that 24/30 FPS Rooms remain accepted while 60 FPS Rooms are rejected before reservation, even when the current environment says otherwise.
- [x] CPU follow-up: **62 passed** across resolution, RTC lifecycle and capture-clock owners; **88 passed, 1 skipped** across render lifecycle, Room pipeline, interactive and native-RTC owners. The skip requires the optional native capture encoder library, not built on this CPU host. Ruff, isort, formatting and diff checks passed. New tests use the already registered resolution owner; no new AREA manifest entry is needed.
- [ ] Verify actual deployed Chrome/native RTC media, startup/input, end/archive/release and the next session at 720p ≤30 FPS. Offer inspection and CPU SPS/decode checks do not establish this end-to-end result.

## WTP consumer and publication handoff

The parent task owns the separate `seedleap/world-test-platform` change targeting
`main`: `web/src/WhiteboxPlaytest.tsx`, `web/src/whitebox-client.mjs` and its declared
types, `web/src/whitebox-playtest-settings.mjs` and its declared types, and
`docs/whitebox-playtest.md`. The UI selects the canonical `480p`/`720p` wire value,
shows actual dimensions, disables overrides for an existing game, and rejects a
`session.ready` whose public dimensions do not match the requested preset. WTP
does not pad video or relabel 736p. This plan is the cross-repository owner record;
the WTP implementation need not duplicate it.

- [ ] Parent verifies and merges the WTP consumer separately from the SGLang source candidate.
- [ ] Before publishing, identify the current Room base and every retained overlay. Port these owning source changes together; the online historical base plus local overlay is not equivalent to the shared `white_box` tree. Per-room `spec` must exist before new frame/render paths run, so do not hot-patch only the frame readers into existing live Room objects.
- [ ] With separately authorized GPU/deployment work, verify real CLI/Canvas/H.264/recording dimensions and original source SHA in both WS and native RTC, prepare/attach and first append, UI alignment, input, end/drain/archive, slot release and the next session.
- [ ] Run same-hardware baseline/candidate 480 and 736 model benchmarks with full Encoder/DiT/Decoder timing/memory and the public-720 padding/crop path. Source merge, CPU proof, deployment, and GPU/browser acceptance remain distinct.
