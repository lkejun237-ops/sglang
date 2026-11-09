# Historical CLI lane review — 2026-10-12

Historical preparation record, **not applied at this checkpoint**. It does not
represent current production topology. In particular the path `/whitebox-test/`
now denotes Thailand production; do not reuse this old test-lane routing plan.
Current operations start at [the release entry](../../../deploy/whitebox/README.md).

The then-reviewed scope was isolated CPU UI/Room and mock H3, separate from the
existing platform and Zing services. Planned limits were two model slots, one
view per game, startup 90s and play 60s; none is current policy. Historical
rollback was limited to the newly added lane/images/config, retaining PVCs, OSS
and existing services. Source was white_box at the time; current SG uses main.

Full original review, exact request body and retained-volume manifest are in the
[private archive index](../history/20261109-evidence-index.json), with member hashes,
format and authenticated access requirements. They are evidence, not deploy inputs.
No cloud resource or data has been removed by this source cleanup.

## Validation limits

Validation on 2026-10-12, before any new cloud writes:

- Linux AMD64 Room image built successfully as a non-root runtime. Its build
  ran 151 CPU control-plane tests and the real CLI/UI two-game integration test
  (61.46 seconds), including slot admission/release, timer and all three matching
  H.264 frame/PTS sequences. Model and archive destination are local fixtures.
- The isolated frontend AMD64 image passed 42 tests, TypeScript and Vite build.
  Its running container served health, the prefixed tasks page and CLI/mock
  configuration; write requests were rejected with HTTP 405.
- SDK headless/UI and real CLI contract tests, type checks, workspace boundaries
  and the test census passed. SDK is pinned to
  `920be54ab127b1dc2838015540965361bbbf6375`; frontend is
  `b910397d98b4e944b2e1d52d2bba10ed276b42e4`.
- Deployment payload validation, client-side Secret/PVC dry-runs and Jenkins
  pipeline syntax validation passed. No new cloud resources were created.

Software Chromium UI capture measured about 82ms/frame after warmup on the local
machine, so **realtime 24fps is not established**. The configured source/video
clock remains 24fps. Cloud CPU throughput, OSS and actual-world browser use
must be measured after release; do not reuse old browser-mock receipts as proof.
