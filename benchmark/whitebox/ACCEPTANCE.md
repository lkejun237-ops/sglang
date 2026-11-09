# Whitebox non-model acceptance — 2026-10-12

## What is implemented

- Actual `generate` gateway ingress, HTTP coordinator, Redis state, worker
  reservation/epoch fencing, lease renewal, capacity limits and bounded deadlines.
- A deployable mock worker implementing the H3 continuous-source contract without
  loading a model. One worker replica advertises exactly two slots.
- An explicit browser-engine plugin. Only the playtest's own same-origin iframe
  runs the existing delivered world. Controls pass through the gateway before
  being applied. The gateway requests one capture at a time and records the
  applied input sequence, rather than assuming arrival equals application.
- The browser source uses JPEG transport only in mock mode. Internal H3 input is
  RGB24. Live gateway output remains Annex-B H.264, decoded using WebCodecs.
- Backend mock HUD rendering, concurrent H.264 encoders, asynchronous uploads,
  final manifest publication and automatic retry of finalized spools on restart.
- Capability-scoped recording status and signed playback URLs. Capability hashes
  live under `.reports/` outside the uploaded dataset. Signed URLs last 15 minutes.
- Per-session trace table: count, mean, min/max, error count, recent-512 p95.
  Shared Prometheus histograms and structured trace logs remain available.
- All mock recordings have `dataset_status=excluded_mock`. Model video is source
  passthrough; UI video adds an explicitly labelled server-rendered test HUD.

## Reproducible local acceptance

Requirements: the CPU dependencies in `requirements.txt`, pytest, and redis-server.
This starts its own Redis on localhost:16389 with persistence disabled. It refuses
to continue if that Redis cannot start. It never uses production Redis or OSS.

```sh
PYTHONPATH=python SGLANG_LIGHTWEIGHT_RUNTIME=1 python benchmark/whitebox/acceptance_server.py
# Separate terminal:
PYTHONPATH=python SGLANG_LIGHTWEIGHT_RUNTIME=1 python benchmark/whitebox/acceptance_check.py
```

The test platform's `local-validation/whitebox-preview.mjs` exercises its actual
TaskDetailView and WhiteboxPlaytest. Its local world is a small keyboard-driven
canvas fixture. Production configuration uses the world's existing `embedUrl`;
no shared world playback bridge, runtime pin, task scheduler or runner is modified.

Observed local results:

| Check | Result |
|---|---|
| Concurrent sessions | 2 active, 0 free; third rejected with CAPACITY_EXHAUSTED |
| Renewal | Both remain active after the 15-second test lease TTL |
| Deadline | 60.017 s and 60.032 s from first media to completion |
| Recordings | Each session: 1,441 frames in all three H.264 MP4s |
| Alignment | All three decoded PTS sequences identical |
| Release | 2 free slots, then a new session succeeds |
| Browser controls | D key changed source pixels; actual applied sequence recorded |
| Prompt | Update owned by output from frame 24 in the short browser run |
| Playback | Whitebox/model/UI recordings all play; HTTP Range returns 206 |
| Diagnostics | 25 measured stages in short browser run; no stage errors |
| Access | Missing/wrong recording capability rejected; persisted capability survives restart |
| Recovery | Finalized failed archive republished automatically, manifest last |

These are local integration results, not online deployment or real OSS acceptance.

## Deployment boundaries and current network finding

The actual test platform is `seedleap/world-test-platform`, service
`s-18d2f876a3529a1cac14e233ad404f5e`, namespace `world-test-platform`, on loopit-ops
cluster `c716251ae51e8420d93e67621f0357d2c`. Only this API/frontend role needs a
new image and two whitebox configuration values. Preserve all existing runtime,
storage, authentication, sealed SDK inputs, task and publication settings.

The requested existing Redis is `r-2ze358d0dce684b4`, Beijing,
`r-2ze358d0dce684b4.redis.rds.aliyuncs.com:6379`, in a different VPC from loopit-ops.
A read-only TCP/PING probe from a loopit-ops node timed out. The user approved
placing the backend in the existing Beijing inference environment. The selected
target is the platform-managed `world-model-cn` ACK cluster
`cd2b7cca02b684de09537a079ba8ce9ea`, platform ID
`c-18ce327cb18a47fd4f6f3afe809113b5`, business line `world-model`.
Its existing coordinator runs with Redis; a read-only AUTH/PING succeeded and
EXISTS returned zero for both new keys. Pod IPs are already whitelisted; no
Redis whitelist, VPC or DNS change is needed. The separate historical k3s control
host is not selected (its coordinator is memory-backed and its IP is not allowed).

New service Redis scope is exactly:

```text
DB 0
{whitebox-playtest-v1}:coordinator:state
{whitebox-playtest-v1}:coordinator:lock
```

No existing Zing keys should be read, overwritten or deleted. Stop the new
services to roll back; their state/lock expire (24 hours / 2 seconds). Slot TTL
is 30 seconds, heartbeat TTL 15 seconds, gateway renewal every 10 seconds.

The existing platform OSS credential passed a read-only prefix listing:
`oss://world-model-test-platform/whitebox-playtest/mock-v1/`, HTTP 200, empty prefix.
Only new UUID recording directories belong to this rollout. Existing platform
object prefixes must not be changed. Real uploads still require the concrete
data-store-write approval required by the workspace AGENTS.md.

## Deployment and monitoring configuration

- Build `docker/whitebox.Dockerfile` through the deployment system; its release
  stage depends on 145 passing CPU contract tests. Use
  `deployment.env.example` to configure three independent roles. No WebUI is
  included in the control-plane image, and no GPU is required for this acceptance.
- Keep one mock worker replica with capacity 2. Use no-overlap worker rollout
  (`maxSurge=0,maxUnavailable=1`) so two epochs cannot advertise one identity
  simultaneously. Do not change the existing platform rollout strategy.
- Gateway needs a persistent writable spool owned by UID 10001. Keep a single
  gateway replica while status is served from that spool. Finished uploads are
  retained locally; monitor disk usage and set the retention/cleanup policy
  before increasing sustained test traffic. Process-killed open MP4s are not
  falsely published as complete; only finalized manifests are retried.
- Expose only generate WS, `/v1/slots`, and `/v1/whitebox/recordings/{id}` to the
  platform. Keep coordinator, worker reservation endpoints, and `/metrics` private.
  Configure exact `WHITEBOX_BROWSER_ORIGIN` for cross-origin playback status.
- Scrape gateway :18080/metrics and coordinator :18081/metrics. The accompanying
  `grafana-dashboard.json` uses cluster/service ownership labels and stage p95,
  throughput/error rate, worker selection/reservation latency and process memory.
  On Loopit-managed services, configure metrics with the business-monitoring API,
  preserving unrelated monitoring settings. Logs are collected by that platform.
- Session API: `GET /v1/whitebox/recordings/{recording_id}` with
  `Authorization: Bearer <report_token>` returned in session.ready. Response has
  timing rows, recording/upload/dataset status, timeline and the three signed URLs.
  It never returns input/prompt/reference bytes or opaque states.
- Enable actual H3/CLI later by replacing configured factories and worker routing
  revision. Disable `WORLD_TEST_WHITEBOX_SOURCE_MODE=browser_mock` then. The
  gateway/coordinator/recording protocols and admission path remain the same.

The concrete proposed platform requests, three narrowly scoped HTTPS routes and
50 GiB retained recording PVC are historical inputs preserved in [the private evidence index](history/20261109-evidence-index.json). The dedicated `Jenkinsfile` is CI-only, derives
the existing Beijing registry/callback workflow and allows only the three new
whitebox service names. Do not edit the old Zing CI jobs or workloads. New
services are registered with zero replicas, built, checked, then activated in
dependency order after approval. The gateway volume init container uses that
same verified image digest and only fixes the new volume root's UID/permissions.

## Online acceptance after approved deployment

1. Read health and verify capacity 2 before testing.
2. Open actual ready worlds in two independent tabs; keep a third waiting client.
   Verify controls visibly affect frames, the third cannot exceed capacity, and
   all slots release after a timed or manual end. Test browser disconnect too.
3. Keep a session alive through at least two 30-second lease windows. Verify
   renewals, the 60-second play deadline and subsequent admission.
4. Wait for `upload_status=complete`. HEAD/GET every manifest-listed OSS object,
   compare size/SHA256, decode all three MP4s, compare counts and PTS, then use
   each signed URL in the platform video player (including Range requests).
5. Check the trace table, manifest trace summary, private Prometheus scrape and
   dashboards. Mock inference latency must never be described as real H3 latency.
6. Smoke-test task list/detail, existing Enter World and an existing artifact;
   do not resubmit jobs, migrate data or rebuild scheduler/runner/account services.
