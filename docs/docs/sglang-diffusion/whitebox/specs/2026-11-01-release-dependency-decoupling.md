# Whitebox release dependency decoupling

Status: proposed design and execution plan only. No runtime implementation,
image build, Jenkins installation or deployment is included in this change.
Source inspected: `509dc194ca4253b481fb724cbec2b433425aec9b` on `white_box`.
The user authorized progressing dependency decoupling without repeated questions;
the choices below are the agent's recommendations, not user-specified technology.

## Intent and boundaries

Separate runtime payload from test-only inputs, and let actual Jenkins builds
reuse trusted layers across source commits. Preserve the existing checks as
release dependencies and preserve every artifact's real producer identity.
These are two independently implementable tasks; neither depends on changing
model code, WTP, the SDK, service configuration or the other task.

- Integration target remains `white_box`.
- SDK pin remains `e99e03be52410a37732e2007969de862ecfb75b5` and Room-only.
- Preserve x264 bridge, Go/Pion RTC sender, Chromium, ffmpeg/ffprobe, tini,
  native SDK checks, dynamic plugins and existing runtime assets.
- Preserve single-active-model-session policy, model/profile settings, media and
  public admission protocols. No new publication, service restart or cloud edit.
- Preserve registry/account/authentication/TLS boundaries, regional push routing,
  immutable final-image handling and success-callback digest verification.
- A source merge, a candidate build, a hosted-recipe update and deployed
  acceptance are four separate facts. This document proves none of the latter.

## Current source evidence

`docker/whitebox.Dockerfile` copies all `python/sglang` separately into `runtime`
and `room-runtime`. `checks` inherits runtime and adds test dependencies;
`release` copies its success marker. `room-runtime` also depends on that marker;
`room-checks` runs actual SDK/native codec checks before `room` copies its markers.
Test files currently arrive through the broad runtime copy. The Dockerfile-specific
ignore file admits the complete Python subtree. Go tests run in `rtc-builder`.

`benchmark/whitebox/run_service.py` dispatches coordinator, gateway, mock-worker
and room roles. `session.load_plugin` loads module strings dynamically;
`whitebox_room_server.compose_room_driver` also selects `worldkit_room` lazily.
Import reachability cannot be established by static imports alone. Names such as
mock, demo, testset or examples do not establish that content is disposable.

`benchmark/whitebox/Jenkinsfile` has `configureSharedBuildCache` and
`prepareSharedBuildCache` helpers, but its pipeline stages call neither.
`prepareWorldModelEcr` currently discovers `${prefix}-${GIT_SHA}` tags across
known prefixes; the actual BuildKit command imports those references and exports
inline cache. Existing same-tag final-image reuse remains a different operation.
`Platform Dockerfile` downloads managed bytes, so the repository Dockerfile is
not sufficient evidence of the recipe used by a live job. Installed inline
Jenkins jobs also require separate reconciliation; see
[registry routing](../../../../../benchmark/whitebox/CI-REGISTRIES.md).

## Task 1: runtime source projection

Two viable approaches are a positive runtime package allowlist, or a conservative
source projection excluding only reviewed test-only paths. Recommend the latter:
a positive allowlist is smaller but risks losing dynamic factories and packaged
assets unrelated to this task. Do not upgrade the Dockerfile frontend just to
obtain newer COPY exclusion syntax.

Introduce a build-only source projection stage using a read-only bind of
`python/sglang` and a small standard-library helper. It writes a separate output
tree; runtime and room-runtime copy that output. Initially exclude exactly
`multimodal_gen/test/` after validating runtime consumers. Preserve all other
paths, including `sglang/test` until separately proven test-only. Expand the
exclusion list only with explicit consumer evidence and tests, never a basename
pattern such as `**/test*`. The helper is a build input, not a runtime module.

Preserve file bytes, relative paths, executable permissions and safe internal
symlinks. Fail on links escaping the source tree; do not follow them. Sort traversal
and normalize output timestamps so a test-only edit does not change projected
runtime content. A full input bind can still invalidate the cheap projection
step; this design does not claim that step itself stays cached. Unchanged
projected content should allow downstream runtime-copy layers to remain cached.

Copy the complete `multimodal_gen/test` subtree explicitly into `checks` and
`room-checks`, retaining fixture JSON, Node helpers and test package structure.
Keep the current CPU suite list, `--confcutdir`, SDK environment and success
markers. A changed or failing test must still affect the release dependency.
Do not exclude tests from the build context and then silently skip them.
Keep SDK source, native codec and RTC inputs on their current role/stage edges.
No dependency pin changes are part of this projection.

## Task 2: fixed-digest cross-commit layer candidates

Two viable approaches are an independently writable registry cache exporter, or
bounded discovery of inline cache already embedded in successful private images.
Recommend the latter for this first change: it preserves the current writer and
trust boundary and needs no new repository, writable global tag or bundle store.
It may retain fewer intermediate layers than a mode=max exporter; measure this
limit rather than introducing another cache service prematurely.

Retain the existing same-SHA candidate path and final-image reuse rules. Add
cross-SHA candidates only for Whitebox control/Room jobs. UI jobs retain existing
behavior. Discovery uses the existing ECR image repository and credential scope;
Thailand remains ACR-only for outputs and may read the existing trusted ECR cache.
No new ACR cache discovery/auth flow or ECR publication is introduced.

Add OCI cache labels for source branch, logical target and recipe SHA256 to newly
built images. Compute recipe identity after the managed Dockerfile is downloaded:
its exact bytes, Dockerfile-specific ignore bytes, requirements bytes, selected
target, platform, effective base-image references, BuildKit/frontend identity,
and Room SDK pin (Room only). Record mutable base references honestly; BuildKit's
base resolution governs actual layer validity. Do not invent a resolved digest
without registry evidence. Source SHA and general runtime/test source bytes are
not part of this recipe hash; BuildKit keys those COPY/RUN inputs itself.

Inspect at most 20 recent image records per build, retaining at most 3 candidates.
A candidate must come from the fixed private repository, match `white_box`,
logical target (`release` or `room`) and recipe labels, have a full source SHA,
and pass manifest/config digest and linux/amd64 checks using the existing trusted
registry-reading conventions. Missing labels, wrong target/recipe/platform,
malformed or unavailable records are cache misses, never final-image reuse.
Pin accepted imports as `repository@sha256:...`; deduplicate by digest. Do not
accept user-provided repositories, cache URLs or blob bundles. Existing registry
preflight/auth errors retain their failure semantics; only optional candidate
lookup/validation failure degrades to a cold build with a bounded diagnostic.

Pass accepted candidates into the actual `buildctl-daemonless.sh build` invocation.
A new source commit still executes a build graph, preserves checks, labels its
new output with the true checked-out source SHA, and resolves its actual output
digest. Importing a cache image neither relabels that image nor satisfies final
image reuse. Inline export, current final lookup and callback verification stay.
Old images lacking labels simply do not seed this new cross-SHA path.

## Verification and rollout boundary

Focused local tests must prove projection contents and actual consumer imports,
full test fixture availability, stage marker dependencies, candidate scope and
actual Jenkins command construction. Negative cases include runtime resources
with test/demo names, dynamic factories, changed tests, corrupt cache manifests,
unrelated targets and different source commits. Cloud calls remain mocked.

Later bounded candidate builds must compare unchanged-runtime/test-only changes,
runtime-source changes, dependency/recipe changes and Room SDK-pin changes.
Record BuildKit cached vertices, real imported digests, stage execution, role
startup and resulting source labels. Cache candidates alone prove no speedup.
Inspect each actual hosted Dockerfile and installed Jenkins job before any
operational activation; compare exact hashes with reviewed sources. Preserve
current regional routes and record rollback identities. Native image acceptance,
GPU/model/ICE playback, 480/736 performance and live service activation remain
unverified and separate from these local checks. Existing known acceptance
failures remain open; this change does not declare them fixed.

[Execution plan](../plans/2026-11-01-release-dependency-decoupling.md)
