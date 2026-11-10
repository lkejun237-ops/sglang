# Managed Encoder / DiT / Decoder role pools

This is the opt-in production-control adapter for the standalone services described in [the disaggregated serving guide](/docs/sglang-diffusion/zing/DISAGGREGATED_SERVING_GUIDE). It adds process registration, service-level scheduling, shared process budgets, player deadlines, and verified group release. It does not create cloud instances or restart workers.

Implementation baseline: `white_box@5d7e71a87ab81cf61bcea60e03c606a2121ea3a0` integrated with `main@0a4623e1299efc386a81473ed03d6a79ed234eaa`. Local CPU/Redis/HTTP/WebSocket tests are available below. GPU/NCCL, heterogeneous hardware performance, and production browser acceptance remain unverified for this integration.

## Ownership and capacity

An Encoder process with three services registers one process record and three separate service records. Each service can belong to one player group; all services share the process budget. Three free service ports do not imply enough throughput for three players.

An assignment reserves one Encoder service, one DiT service, and one Decoder service. Whitebox-only play retains all three. Prefill and Generate children run sequentially within the same logical parent binding. The DiT service is exclusive to one player; SP2 ranks together form one service and only rank zero registers it. Encoder/Decoder model processes may host multiple independently owned services.

Admission uses per-record Redis transactions, per-profile ready indices, and a shared cross-pool admission home. It checks both service availability and the remaining process budget. Heartbeats cannot erase allocation or cleanup ownership. Missing heartbeats remove candidates without declaring their resources released.

The first implementation balances declared committed budget and candidates. Budget profiles must be measured for the intended hardware and workload. It does not infer safe concurrency from GPU name, service count, current low utilization, or historical results from another configuration. Automatic performance-based budget tuning and active-session migration are not implemented.

## Prerequisites and startup order

An explicitly authorized isolated development test can build with
`WHITEBOX_SOURCE_BRANCH=codex/<branch>` and start with
`WHITEBOX_ALLOW_DEVELOPMENT_RELEASE=1`. The version endpoint preserves the actual
branch/SHA and marks `release_scope=isolated_development_test`. Production defaults
and the production release builder require an exact `main`-reachable SGLang SHA
and a reviewed `dev`-reachable SDK pin; this test option
does not authorize deployment or changes to a shared lane.

These files extend an existing Whitebox deployment; they are not a complete deployment manifest. Retain the current internal tokens, `WHITEBOX_ROOM_OWNERS`, `WHITEBOX_ROOM_OWNER_ID`, `WHITEBOX_ROOM_DRIVER`, `WHITEBOX_ROOM_COORDINATOR_URL`, `WHITEBOX_ROUTING_REVISION`, pinned CLI/SDK, media/ICE, storage and publication configuration. Use the existing [Whitebox deployment entry](/docs/sglang-diffusion/zing/deployment_zing_h3) and the repository `benchmark/whitebox/README.md` for those prerequisites.

The original Coordinator entrypoint still defaults to DynamoDB. For a Redis-backed Whitebox lane, select its backend explicitly in addition to the role-pool configuration:

```sh
SGLANG_ROLE_POOL_COORDINATOR_CONFIG=/run/config/coordinator.json python -m sglang.multimodal_gen.runtime.entrypoints.realtime_coordinator_server --backend redis --redis-host redis.internal --redis-port 6379 --redis-db 0 --port 18081
```

The CLI backend above continues to own the original Room directory/legacy state; role-pool and shared-admission stores use their explicit JSON settings and key namespaces. Start/configure Redis, verify its incarnation and initialize only a new empty pool, then start Coordinator, standalone roles, Room and Gateway/WTP. Wait for registrations before enabling player admission. This integration uses the render-v2 bound Room path; classic direct-generate does not use the new pool.

## Coordinator configuration

Set `SGLANG_ROLE_POOL_COORDINATOR_CONFIG` to a deployment-owned JSON file. Example structure:

```json
{
  "pool_id": "shanghai-zing-a",
  "control_epoch": "deployment-issued-unique-control-epoch",
  "redis_url": "redis://redis.internal:6379/0",
  "redis_run_id": "verified-primary-process-run-id",
  "admission_redis_url": "redis://admission-redis.internal:6379/0",
  "admission_redis_run_id": "verified-admission-primary-run-id",
  "signing_secret_env": "ROLE_POOL_SIGNING_KEY",
  "role_token_env": "ROLE_POOL_REGISTRATION_TOKEN",
  "room_token_env": "ROLE_POOL_ROOM_TOKEN",
  "operator_token_env": "ROLE_POOL_OPERATOR_TOKEN",
  "coordinator_url": "http://coordinator.internal:18081",
  "capacity_routes": {
    "whitebox-zing-revision": {
      "profile": "480",
      "compatibility": "validated-model-and-protocol-bundle",
      "network_domain": "same-zone-network-a"
    }
  },
  "default_capacity_route": "whitebox-zing-revision"
}
```

Use distinct registration, Room, operator, and signing credentials, each at least 32 characters, supplied through the existing secret system. Do not commit secret values. All pool shards which enforce a common user/session limit use the same admission store. Cross-pool retries retain the original admission identity; they cannot acquire a second group elsewhere while the first is unresolved.

`redis_run_id` is the actual Redis process incarnation, not its address or persistent cluster node ID. Writes verify both run ID and primary role on the same watched connection as the transaction. A restart/promotion fails closed. The trusted configuration and `control_epoch` must not be recovered from the Redis data being repaired. Redis 6.2 or later is required for the bounded ready-index sampling command. This version uses explicitly pinned primary endpoints, not an automatically routed Redis Cluster/Sentinel client. Use a pinned primary for each pool store and a shared pinned primary for the admission home; online resharding or transparent primary changes are not supported. Hash tags keep each pool transaction colocated for future cluster integration.

Authority is never initialized by API startup. For a genuinely new, empty pool, an operator may inspect and then explicitly initialize it:

```sh
python -m sglang.multimodal_gen.runtime.realtime.role_pool.admin inspect-store --config /run/config/coordinator.json
python -m sglang.multimodal_gen.runtime.realtime.role_pool.admin bootstrap --config /run/config/coordinator.json --new-pool
```

`inspect-store` checks `redis_url` only. If the admission store is a separate instance, independently inspect and verify its process incarnation before setting `admission_redis_run_id`.

These are management operations, not part of ordinary player admission. `bootstrap` refuses a pool containing state and is not a disaster-recovery shortcut. After data loss/promotion, keep admission closed, fence the old control paths, reconcile native group/admission ownership and deadlines, and obtain operator authorization before installing a new authority. Do not simply repin the Redis run ID or delete an unresolved ledger.

## Worker configuration and registration

Each role process receives its own `SGLANG_ROLE_POOL_WORKER_CONFIG` file. The following **illustrates the schema and arithmetic only**; these capacity numbers are not hardware performance measurements:

```json
{
  "role": "encoder",
  "pool_id": "shanghai-zing-a",
  "control_epoch": "deployment-issued-unique-control-epoch",
  "coordinator_url": "http://coordinator.internal:18081",
  "signing_secret_env": "ROLE_POOL_SIGNING_KEY",
  "role_token_env": "ROLE_POOL_REGISTRATION_TOKEN",
  "compatibility": "validated-model-and-protocol-bundle",
  "network_domain": "same-zone-network-a",
  "hardware_ids": ["node-a/GPU-verified-uuid"],
  "service_count": 3,
  "capacity": {"compute_units": 2, "dynamic_memory_units": 2},
  "profiles": {
    "480": {"compute_units": 1, "dynamic_memory_units": 1}
  },
  "profile_limits": {
    "480": {
      "source_sizes": [[832, 480]],
      "target_sizes": [[832, 480]],
      "max_fps": 24,
      "max_references": 3,
      "max_reference_bytes": 8388608,
      "max_reference_pixels": 16777216,
      "max_prompt_chars": 16384,
      "max_preview_pixels": 16777216
    }
  }
}
```

Give DiT/Decoder their own measured process capacities and per-player costs; the same profile's request limits and compatibility identity must agree across the selected roles. Include static weights, cache reservations, communication overhead and operating headroom when determining the remaining dynamic budget. A profile outside the validated envelope is rejected before shared GPU work. New target geometry/reference limits require a suitable profile, not silent overcommit.

Run the existing standalone role entrypoints with matching `--service-count` and reachable `--advertise-host`. DiT remains `--service-count 1`, with SP1/SP2 as supported by the native entrypoint. SP2 hardware identity covers the participating GPUs; followers do not register additional service/boot records.

Use role-specific pipeline inputs: the standalone Encoder selects its native
source codec and must not receive the output Decoder's `decoder_config`. Keep
the original output decoder configuration on DiT/Decoder. DiT reserves its
management/rendezvous endpoints before loading weights and broadcasts either the
descriptor or the startup error to every SP peer. This prevents a leader setup
failure from leaving a follower waiting for a descriptor during teardown.
The Encoder business ASGI app installs managed authorization independently of
its management HTTP app. Both generation and Prefill require the signed child
identity, including requests rejected before model execution. Set Room's
`WHITEBOX_INFERENCE_STEPS` to the model's deployed normal step count (for example
`2` for `[1000, 500]`); the model keeps its configured initial-step schedule.
Keep `WHITEBOX_MODEL_SUBCLIP_ADMISSION=0` unless the deployed standalone worker
actually advertises `source_output_release_protocol=valid-source-frames-v1`.
The conservative input window remains supported without that optional capability.

A single reporter batches the process manifest and all service observations. Registration is idempotent for the same immutable manifest and boot epoch. A changed manifest/epoch cannot overwrite an unresolved process. Model load and native readiness precede READY admission. No endpoint addresses are inferred by Room; native service descriptors provide the actual input/output and transport endpoints.

## Managed control protocol

With managed mode enabled, unauthenticated legacy group operations are rejected. Existing standalone mode remains available when the worker configuration is absent.

Coordinator-to-service requests use signed `POST /managed/{action}` calls. The signature binds action, complete body, control epoch, and a bounded executor lease. Important actions are `observe`, `reserve`, `build`, `renew`, `destroy`, `finalize`, `child_status`, `persistence`, and process `drain`.

- `observe` is read-only. Unlike native `get_connection_status`, it never takes a 10-second reservation.
- `reserve` records the complete allocation/admission identity and charges the local process budget, then takes the original native reservation.
- `build` reuses the original native topology/token protocol. Communicator setup is serialized within a shared role process, after reservation acceptance. Control/lease traffic has separate connection capacity from long builds, cleanup and persistence reads.
- Three matching native confirmations are required before BOUND. Partial/unknown results retain all potentially affected resources in the original operation ledger.
- `destroy` for a managed exit cancels this group's work using native abort/rollback cleanup. It does not wait for a disconnected viewer to acknowledge graceful EOS, and it still requires execution/communication owners to stop.
- CLEANED does not open the service for another owner. `finalize` returns an exact epoch/group/allocation release receipt; only then can the Coordinator return the service and process budget.
- Delayed reserve/build/release requests cannot replace a different owner. Uncertain cleanup preserves occupancy. A process-level restart-required failure locally fences sibling admission and is reported as failed; no restart is requested.

Business WebSockets require a connection-scoped signed claim. Each service allows one attached child; replayed/overlapping child attachments are rejected. A cleanup receipt is published only after native compute cleanup **and** local admission release. Child receipts are bounded while preserving the credential replay window.

The Coordinator retains unresolved ownership without TTL deletion. Pool-local fully released assignments and released cross-pool user-owner records retain 24 hours. Cancellation-before-admission tombstones retain 600 seconds, longer than the 480-second new-admission window. Active, closing, pending-SDK and quarantined ownership never expires merely because it is old. A new user claim clears the previous terminal retention TTL. Initial admission requests have a fixed creation timestamp and bounded admission window, so an old request cannot become a fresh lease after its pool terminal record expires.

## Room adapter and browser behavior

Configure Room with:

```text
WHITEBOX_MODEL_BACKEND=role_pool
WHITEBOX_BOUND_COMPUTE=1
WHITEBOX_ROOM_PIPELINE=1
WHITEBOX_RENDER_SEGMENTS=1
WHITEBOX_MODEL_STARTUP_PREPARE=0
WHITEBOX_ROLE_POOL_CONFIG=/run/config/room-role-pool.json
```

Gateway must also set `WHITEBOX_ROOM_PIPELINE=1`, advertise the same render-v2
capability and disable its `WHITEBOX_MODEL_STARTUP_PREPARE` option for this lane.
For the standalone WTP lane, set `WORLD_TEST_WHITEBOX_RENDER_SEGMENTS=1` and
`WORLD_TEST_WHITEBOX_RENDER_PLAY_SECONDS` to the configured player limit.
Classic single-WS direct-generate routes stay on the legacy backend.

Room configuration:

```json
{
  "coordinator_url": "http://coordinator.internal:18081",
  "room_token_env": "ROLE_POOL_ROOM_TOKEN",
  "player_ttl_seconds": 1800,
  "startup_seconds": 480,
  "routes": {
    "whitebox-zing-revision:832x480": {
      "profile": "480",
      "compatibility": "validated-model-and-protocol-bundle",
      "network_domain": "same-zone-network-a"
    }
  }
}
```

Add separately validated routes for other target sizes/revisions. A route key uses the padded model canvas, e.g. the existing 720p source canvas maps to model `1280x736`; its source and target limits must reflect the actual model request. Region-specific TTL and resource settings are configuration differences, not code forks.

The existing Room `ModelBinding` remains the logical parent. Its local adapter sends init/source/updates to Encoder and receives RGB plus source consumption from Decoder. Encoder input ACK and Decoder output can arrive in either order: bounded holding ensures source ACK precedes dependent frames/credits and a frame header stays adjacent to its payload. Completion requires both channels and matching frame/credit counts.

Explicit Prefill uses the original Encoder/DiT cache path. It reports memory-ready after verified child cleanup; persistence is separately inspected on DiT, off the model command thread, and never extends the player lease. No Encoder-local disk is mistaken for DiT L2. Normal generation remains available without Prefill.

The adapter retains `ordered-boundary-v2` updates and uses output event IDs to identify effective conditions. The older `clip-barrier-v1` fast condition-switch extension is not advertised by this backend. The legacy `source.prepare` early-start extension is not implemented in standalone workers; enabling `WHITEBOX_MODEL_STARTUP_PREPARE` with this backend fails configuration validation. Existing non-pool serving keeps those extensions. These are explicit capability boundaries, not claims of full legacy-extension parity.

Each valid player keeps its E/T/D group during whitebox-only play, Prefill, and Segment transitions. Reconnection attaches to the existing valid player; it does not move an active model session or reset its deadline. CLI `room resume` remains an extension point. GameRoom has no platform-wide maximum lifetime.

The existing public capacity endpoint reports complete player groups for configured pool revisions, including shared-process budget constraints. Gateway must not divide this logical count by the three physical roles. Capacity snapshots are advisory; admission performs the authoritative checks.

## Scaling down and manual recovery

Validated SDK retirement receipts are persisted before waiting for GPU cleanup.
A caller timeout does not discard that positive evidence; reconciliation finishes
the player only after all native roles have separately confirmed release.
Duplicate receipts must match the original retirement identity.

Operator-only commands:

```sh
python -m sglang.multimodal_gen.runtime.realtime.role_pool.admin drain --config /run/config/coordinator.json --process-id PROCESS_ID --boot-epoch BOOT_EPOCH
python -m sglang.multimodal_gen.runtime.realtime.role_pool.admin process-status --config /run/config/coordinator.json --process-id PROCESS_ID --boot-epoch BOOT_EPOCH
```

Drain stops new reservations while existing players retain their deadlines. `safe_to_stop` requires a complete native observation, no service allocations, no process-budget use, and native drain acknowledgement. The deployment manager must stop that exact process and verify its exit before submitting `retire-process --stop-receipt RECEIPT_REFERENCE`. These commands never kill a process themselves.

For a failed process whose normal cleanup cannot return, `confirm-stopped` requires operator credentials, the exact old boot epoch, an independently verified stop receipt and `--operator-id`. It releases only that process's roles with an explicit **operator process-exit evidence** category, not a fabricated native ACK. Other roles and SDK cleanup still require their own confirmation. A lost heartbeat alone never authorizes this operation.

SDK member/view cleanup and GPU-role cleanup proceed independently. Unconfirmed SDK cleanup can remain pending while positively released GPU roles serve other players. A PlayerSession is not reported fully released until its SDK receipt is validated. Storage recovery must reconcile cross-pool admission ownership as well as service occupancy. Authenticated `native-identity` and paged `native-recovery` commands provide active ownership and retained terminal owner/receipt mappings, including original admission, topology, profile and deadline. They perform no mutation. Native terminal history is bounded (up to 4096 groups per service, with at least the 180-second replay window protected when compacting); it is not an unlimited audit log. Beyond that window, use retained Coordinator/WTP evidence. If any original ownership cannot be established, keep admission closed; this release does not provide an automatic disaster-recovery importer.

Read-only recovery examples (using the signing key from the trusted configuration, not the operator bearer token):

```sh
python -m sglang.multimodal_gen.runtime.realtime.role_pool.admin native-identity --config /run/config/coordinator.json --service-http http://role.internal:31000
python -m sglang.multimodal_gen.runtime.realtime.role_pool.admin native-recovery --config /run/config/coordinator.json --service-http http://role.internal:31000 --boot-epoch OBSERVED_BOOT_EPOCH --offset 0 --limit 32
```

Continue with the returned `next_offset` until it is null. A complete page traversal is still only the retained native window, not proof that all historical admissions have been recovered.

## Local validation and review route

```sh
python -m pip install -r scripts/ci/zing/requirements-role-pool.txt
python scripts/ci/zing/run_role_pool_tests.py -q
```

Install a local `redis-server` to run the real Redis cases. Tests start their own ephemeral loopback server; they do not use an existing database or cloud endpoint. The test harness skips unrelated package-level GPU registries, not the tested implementation. Native callbacks use CPU doubles; Redis transactions, HTTP control, process reporting and split WebSocket transport are real local components.

Read `contracts.py` and `store.py` first, then `scheduler.py` and `worker.py`, then `native_http.py` / `reporting.py`, and finally `whitebox/disaggregated_binding.py` plus `transport.py`. Tests cover registration granularity, budget contention, shared admission fencing, partial build/cleanup, shutdown authority, process drain/replacement, signed controls, replay protection, and split-channel ordering.

The wider merged conditioning/Prefill/decoder/player-lifecycle suites remain required. Local CPU tests do not establish CUDA correctness, heterogeneous NCCL interoperability, per-player FPS/latency, or production readiness. Future release still requires an exact `main`-reachable SGLang SHA and reviewed `dev`-reachable SDK pin, no hot patches, five current browser cases of at least 180 seconds per formal player, multiplayer interfaces, cache/recording coverage, and explicit reporting of every observed regression. This change does not authorize deployment.
