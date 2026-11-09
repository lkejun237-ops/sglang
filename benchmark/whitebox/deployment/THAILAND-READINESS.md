# Historical Thailand readiness — 2026-10-12

This is a historical preparation/validation record, **not current deployment
instructions or evidence of current GPU acceptance**. Use
[the release entry](../../../deploy/whitebox/README.md) for the supported workflow.
The [private archive index](../history/20261109-evidence-index.json) preserves the
original document, exact site requests and volume manifests with SHA256/format.

The old plan kept Beijing catalog reads and Thailand recording writes separate.
Its H3 settings were a single player, SP2, two denoising steps and a six-chunk
window. Later public policy supersedes these dated configuration details.

At this checkpoint ACK/PAI private connectivity was unproven: the read-only probe
retained the quota VPC despite the requested VPC. Bidirectional model/Coordinator
reachability, complete asset/image compatibility, Thai CI/CD wiring, real H3,
cloud recordings and 24fps browser behavior were not established. A VPC peering
dry run returned DryRunOperation; it created no connection or route.

Historical rollback covered only the newly created inference/lane applications,
retaining recordings and existing jobs. It is not authorization to stop today's
services. Source cleanup removes no cloud resources, model assets or runtime data.

## Verification so far

- `white_box` merged SGLang main `0cfc3d6f79b707fd25e396d02882fd101a1cff57`
  (merge commit `15f6d13206`), including source synchronization and cancellation
  cleanup improvements. The 151 CPU control-chain tests pass after this merge.
- All 22 offline regional configuration and GPU-guard tests pass, covering region separation,
  budget limits, exclusive sessions, unhealthy/busy nodes and forbidden job
  expansion or preemption. These do not exercise a live GPU allocation.
- All 77 source-adapter CPU tests pass with the repository's CPU import harness,
  including trusted two/four-step profiles, mismatch rejection and window binding.
  Gateway/room/whitebox regressions: 96 passed, including both step counts through
  the three-video pipeline. GPU execution is still untested for this deployment.
- All 43 platform regression tests pass. The isolated frontend supports
  `WORLD_TEST_WHITEBOX_MODEL_MODE=h3` and the Thailand gateway URL while preserving
  its default mock mode and Beijing catalog.
- Earlier real CLI/headless UI local tests remain useful integration evidence,
  but used a mock model and local recordings. They do not prove real H3, Thai
  networking, OSS delivery or realtime 24 fps.

One authorized **zero-GPU**, two-CPU asset inspection job was created:
`dlc1rqfvnvd57o6t` (`whitebox-th-assets-readonly-20261012`). It used a read-only
CPFS mount, finished successfully at 21:24 CST and did not modify model assets.
It identified the existing model at
`/cpfs/world-model/sglang/evals/issue253-client-5s-20261010-r1/model` and TAEH3 at
`/cpfs/world-model/sglang/evals/zing176-20260928-r3/taeh3.pth`.
The observed config SHA256 is
`6b343cc671048731eb7219d4d1c3958414d1c2457f097cababcfb7ae40a19228`.
Full weights/image compatibility and the new deployment profile still need
verification. No GPU allocation, Redis/OSS write or backend activation has occurred.

## Latest CLI integration and authorized capture repair

Upstream `codex/wtp-stream-cli` was merged at
`7e589b2e4d4278d0999bb2dd18a3fe23497233a5`. SDK `white_box` now pins
`26d08ff08af0e303dd7c3d0f50cd575df3f28fcc`, containing the user-authorized capture
initialization repair. Chromium can return a permanently null Capture Handle
when capture starts before configuration propagation. Startup now waits and
retries only that missing-handle case, stopping every unverified stream; wrong
identity, origin, surface, DPR or dimensions still fail closed. No Beijing
platform or OSS loading behavior changed. All 10 capture tests, affected
TypeScript checking, workspace boundaries, test census and runtime prebuild pass.

The SGLang Room adapter uses the official public `stream-compositor` API and
validates source frame, UI revision, output PTS, generation and final EOF. Local
model/compositor transport uses VP8; public playback and all three archived
tracks remain H.264. Native codec teardown is serialized with cancelled worker
threads so contexts cannot be dropped during encoding/decoding.

The uninstrumented Linux AMD64 room image passed the real two-game CLI/UI
integration in 55.82 seconds (final rebuild after codec teardown repair), including three H.264 tracks and UI drain timing.
The same build passed 160 control-chain and 22 regional tests.
Its model and archive dependencies are local fixtures. This does not demonstrate
real H3, cloud OSS delivery or 24 fps wall-clock performance. No diagnostic
preload is present in the image.

A second authorized zero-GPU, read-only probe `dlc1h72byjpn0ryd` also completed.
It verified model metadata and runtime packages, but retained the PAI VPC despite
the requested ACK VPC. Both probe jobs are terminal. No training job was changed.

