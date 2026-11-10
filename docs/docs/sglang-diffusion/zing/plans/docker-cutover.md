# Immutable Docker Room/Gateway cutover

`deploy/whitebox/docker_cutover.py` prepares and switches an explicitly authorized
Room and optional Gateway. It does not build/pull images, change models,
Coordinator, executor pools, prune storage, remove containers or create cloud
resources. It requires Linux, Python 3.12, Docker CLI/API and root access to the
selected Docker socket, Nginx configuration and `/proc`.

The existing image builder and its source ancestry, native-image and layer-history
gates remain prerequisites. Source SHA, SDK SHA and image digest are independent
identities. Resolve each full SHA from its owning reviewed release; do not reuse a
moving tag or change a task/runtime pin to match this operator.

## Private configuration

Copy [the example](../../../../../deploy/whitebox/docker-cutover.example.json)
outside Git and replace every placeholder from fresh observations. The example is
not a deployment request. Keep settings, stage receipts and mounted bundle copies
private; they contain inherited container environment and bindings. The CLI emits
only mode/stage/completion, and does not print raw exceptions or Docker output.

- `docker_socket`: exact absolute socket of the intended engine. Default and test
  engines can live on different hosts or sockets; never infer the target from region.
- `authorized_roles`: exactly the roles listed in `containers`, restricted to one
  `room` and optional `gateway`. Full old IDs, exact old names and distinct candidate
  names bind the request. No other container is stopped.
- `image`: locally present `repository@sha256:...`, checked against RepoDigests.
  `old_source`/`old_sdk` bind the actual old `/app/release.json`;
  `source`/`sdk`/`branch` bind the candidate's baked file. A no-network image probe
  also checks `sdk_files`, mapping reviewed relative SDK files to SHA256 hashes.
  Supply hashes measured from the exact reviewed SDK checkout, including a relevant
  CLI/Stream Host source file. Room requires at least one. This is independent of
  the manifest's SDK claim and does not replace the full image-build gates.
- `remove_mounts`: exact `{source,target}` pairs for obsolete read-only runtime
  overlays, restricted to `/app/python` and `/app/release.json`. Unexpected source
  overlays fail closed, including nested overlays and SDK mounts. An empty list
  means none may exist. Config/HostConfig, commands, environment, limits and other
  bind mounts are inherited. Only existing host networking and explicit bind mounts
  are supported; anonymous/named volumes and VolumesFrom require separate work.
- `idle.room_status`: original authenticated Room status POST. `role_pools` must
  include every current admitting route/pool, including any legacy/default route
  that remains reachable. Supply each original body and token **environment name**;
  tokens are read from the authorized old container, never printed. Check all model
  group active/queued/waiting counters and any returned role lease counters.
  `metrics` must expose all four recording/pending-work gauges; missing evidence
  rejects the switch. These endpoints must describe the actual owners being switched.
- `nginx`: exact absolute config path, original raw-text SHA256, unique route marker,
  and complete list of HTTP/WS admission paths, including prefixed/unprefixed and
  managed control entry points when exposed. The operator inserts exact 503
  locations, validates with `nginx -t` and reloads. `pid_file` accepts only
  `/run/nginx.pid` or `/var/run/nginx.pid`. If the deployment uses another PID file,
  extend the reviewed whitelist before execution. The timeout is 1–300 seconds.
  Inventory must establish that all public admission uses this gated Nginx; direct
  alternative ingress is outside this operator's supported scope.

`WHITEBOX_RELEASE_BUNDLE` must be one read-only bind. Preparation copies its JSON
into the private stage, updates only policy source/SDK/branch and the resulting
configuration digest, and points the candidate at that copy. Original files are
untouched. Other root fields, GPU source, model/media policy, role pool, FPS,
authentication and routing remain inherited. The candidate runs with the old
Config.User; a probe resolves its actual UID/GID for the private bundle ownership.
Image revision labels are updated while service ownership labels remain.

## Commands and evidence

Use the exact pushed operator commit and reviewed image inputs:

```sh
/path/to/python3.12 deploy/whitebox/docker_cutover.py prepare \
  /private/cutover.json /private/new-stage
# Inspect private prepared.json, per-role bundles and created candidate IDs.
/path/to/python3.12 deploy/whitebox/docker_cutover.py switch \
  /private/cutover.json /private/new-stage
```

Preparation probes images and **creates** stopped candidates; it does not start
service candidates or stop originals. `prepare-intent.json` and per-role created
receipts preserve partial results. A failed/uncertain request requires inspection
of those exact names/IDs before any retry; do not delete intent receipts and retry
blindly. Candidate preparation does not constitute deployment authorization.

Switch rechecks original Config/HostConfig/image/start identity, original bundle
bytes, candidate Config/HostConfig/image and candidate bundle bytes. It checks
Room/role leases/recording idle, installs the ingress gate, and waits for all
pre-gate Nginx worker PID/start-time identities to disappear naturally. This drains
already accepted WebSockets and in-flight requests that a reload alone would leave
alive. It never signals or kills workers. An unrelated long connection can block
the whole Nginx worker: timeout restores the old ingress and stops no containers.
`nginx-workers.private.json` records original and remaining worker identities.
Only after draining does the second idle check run, followed by exact old stop,
candidate start and source/SDK/config/health verification per authorized role.

Old containers remain stopped for rollback. A failure restores affected original
containers in reverse order, verifies their original source/SDK/config/health,
then restores the exact original Nginx text. If rollback readiness is unconfirmed,
the gate remains closed even if receipt writing also fails. If someone changes
Nginx during the switch, the operator refuses to overwrite their bytes: reconcile
explicitly. SIGTERM and keyboard interruption enter rollback; a host crash/SIGKILL
requires manual reconciliation from the private intent. No crash-recovery claim
is made. `deployed.json` means identity and health only; real 720p/VP9 playback,
input, high-FPS and exit/resource-release acceptance remain separate.

## Implementation and verification

The design uses one owner module and fake-Docker lifecycle tests; no general
orchestration framework or model deployment path is introduced. Tests cover exact
source/SDK and mount protection, unchanged runtime settings, two-role reverse
rollback, config drift, incomplete idle evidence, old WebSocket drain, PID reuse,
PID whitelist and receipt-failure behavior. They make no external requests and
start no actual service.

```sh
python3.12 -m pytest deploy/whitebox/test_docker_cutover.py -q
```

Local fake tests and existing release/image-storage tests do not prove a real
Docker/Nginx cutover or browser playback. Record those gates independently.
