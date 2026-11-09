# Whitebox release dependency decoupling implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` for the two independent tasks. Steps use checkboxes. This plan is not an implementation or deployment receipt.

**Goal:** Keep test-only source out of runtime layers while preserving release tests, and import trusted cross-commit layer candidates in actual Jenkins builds.

**Architecture:** Project runtime source into a build-only stage; restore tests only in check stages. Discover bounded inline cache candidates from existing trusted private images and pin imports to verified digests without changing final-image reuse.

**Tech stack:** Python standard library/pytest, Docker multi-stage builds, existing rootless BuildKit 0.17.2, Jenkins Groovy, private ECR/ACR.

**Spec:** [Release dependency decoupling](../specs/2026-11-01-release-dependency-decoupling.md).

## Global constraints

- Base: `509dc194ca4253b481fb724cbec2b433425aec9b`; integration target `white_box`.
- SDK pin remains `e99e03be52410a37732e2007969de862ecfb75b5` and Room-only.
- Preserve runtime dynamic imports/assets, native x264/RTC, checks, model/profile/media contracts, current private registry/BuildKit trust and final-image reuse.
- No WTP edits, production deployment, service restart, runtime/asset publication or GPU test in this work.
- Read current root/nested AGENTS and applicable project skills before execution. Follow their existing source/CI and infra-skill setup conventions in the isolated worktree; never modify the user's original checkout.

## Review focus

1. A resource named demo/test is still a runtime input: projection must retain it.
2. A check imports a sibling helper/Node fixture: explicit test copy must retain the entire test subtree.
3. A changed test with identical runtime bytes: runtime projection remains identical but the check dependency invalidates and must pass.
4. A mutable tag moves or a manifest/config is corrupt: use verified fixed digest or cold build, never tag-following cross-SHA imports.
5. Same recipe but different SHA/role: layers may be candidates only for the matching target; never reuse another commit's final image as the new source.

---

### Task 1: separate runtime and check inputs

**Files:**
- Create `deploy/whitebox/project_runtime_source.py` (build-only projection).
- Create `deploy/whitebox/test_runtime_source_projection.py`.
- Create `deploy/whitebox/test_runtime_image_contract.py` (stage graph and candidate image checks).
- Modify `docker/whitebox.Dockerfile` and its existing comment in `docker/whitebox.Dockerfile.dockerignore`.
- Update `benchmark/whitebox/README.md` with actual changed behavior and verification limits.

**Interfaces:** `project_runtime_source(source: Path, destination: Path) -> None`; CLI `--source PATH --destination PATH`. Initially exclude only relative `multimodal_gen/test/`. Produce a deterministic runtime tree consumed by both runtime COPY instructions; no interface to Task 2.

- [ ] Add failing projection fixtures: excluded test file changes leave output bytes/path/mode inventory identical; runtime file changes alter it; `runtime/demo.py`, runtime assets and `apps/.../testset` remain; safe internal symlink preserved, escaping symlink rejected. Verify missing source and pre-existing nonempty destination fail without modifying source.
- [ ] Add stage/consumer checks: both runtime targets consume the projected output; both check targets explicitly restore complete tests; release/Room still depend on their original success markers; control target has no SDK-stage dependency. Assert existing fixtures including `fixtures/capture-member-media-commands.mjs` remain visible in checks. Baseline should fail because broad runtime copies remain.
- [ ] Implement the projection helper and stage with stable timestamps, sorted traversal and preserved executable modes. Mount source read-only; copy projected bytes into runtime stages. Restore full tests only in checks, preserve all existing CPU/native test commands and make the new focused projection tests part of `checks` with its deployment helper inputs.
- [ ] Run `python -m pytest deploy/whitebox/test_runtime_source_projection.py deploy/whitebox/test_runtime_image_contract.py --confcutdir=deploy/whitebox -q`. Expect all tests pass; source diff must show no SDK/model/protocol changes.
- [ ] In a separately scheduled bounded image gate, build `release` and `room` with the exact existing SDK checkout, inspect absence of excluded tests and presence of native tools, and exercise actual launcher dispatch for coordinator/gateway/mock-worker/Room using local protocol fixtures. Resolve `session.load_plugin` for current configured factories, including CLI/worldkit Room and UI renderer. Start only local test-owned processes and verify teardown. Preserve current real SDK CLI/native-codec image checks; a skip or known native failure is not acceptance.
- [ ] Compare baseline versus test-only and runtime-source edits in a disposable context; record projection inventory and BuildKit vertex results. Expect check invalidation on test change, stable projected runtime output for that change, and changed runtime copy for runtime change. No numeric speedup claim without timings.
- [ ] Update owning documentation with measured scope, run `git diff --check` and relative-link/source-claim checks, then commit only this task's files. Do not push, merge or deploy without the owning execution scope.

### Task 2: connect cross-SHA cache candidates to Jenkins

**Files:**
- Modify `benchmark/whitebox/Jenkinsfile` (recipe labels, bounded lookup and actual command wiring).
- Create `benchmark/whitebox/test_ci_cache.groovy` (execute real helper/stage command wiring with stubbed Jenkins steps).
- Extend `benchmark/whitebox/test_ci_registry.py` (mock registry integrity/scope cases).
- Keep `benchmark/whitebox/test_ci_routing.groovy` and `test_ci_post.groovy` as regression consumers.
- Update `benchmark/whitebox/CI-REGISTRIES.md` and `README.md` with cache eligibility and hosted-job installation boundary.

**Interfaces:** In the Jenkinsfile, define `String whiteboxCacheRecipe(Map inputs)` for the spec's canonical recipe inputs and `List<String> selectWhiteboxCacheCandidates(Map expected, List<Map> records)` for validated metadata (repository, digest, sourceRevision, branch, target, recipeSha256, platform, integrityVerified). Return at most 3 distinct `repository@sha256` refs from at most 20 records. Production retrieval must produce these records only after actual manifest/config validation; caller-supplied booleans are not a security proof. Existing `/build-secrets/world-model-cache-imports` remains the build command input. Use the existing registry verifier's validation semantics; do not create a second auth/realm policy.

- [ ] Add failing Groovy cases for same recipe/different SHA accepted as a layer candidate, wrong role/recipe/branch/repository/platform rejected, missing labels and corrupt/unavailable metadata skipped, digest deduplication and 20/3 bounds. Stub shell/registry transport; include a moved tag fixture and assert only the verified digest reaches imports.
- [ ] Test recipe construction from downloaded Dockerfile bytes (not repository defaults), ignore/requirements/base/platform/BuildKit/frontend inputs and Room-only SDK pin. Runtime/test source and SG SHA changes leave recipe identity unchanged; requirements/managed Dockerfile/platform changes alter it. Room pin changes alter Room identity only.
- [ ] Implement recipe labels after `Platform Dockerfile`, bounded discovery within existing ECR scope and validation, and append accepted refs to the original imports file. Preserve same-SHA candidates and current inline exporter. UI paths, regional final push/reuse and credential setup retain their current rules. Optional cache lookup failure logs a bounded miss; mandatory registry preflight still fails as before.
- [ ] Execute pipeline command assembly in the Groovy harness: accepted ref appears in actual `--import-cache`, new SHA still sets `SOURCE_REVISION`/OCI revision and runs build when the requested final tag is absent; an existing final tag follows the existing reuse branch. Confirm neither dormant shared-cache helper is accidentally enabled and no global cache tag/repository/exporter is introduced.
- [ ] Run `groovy benchmark/whitebox/test_ci_cache.groovy benchmark/whitebox/Jenkinsfile`, `groovy benchmark/whitebox/test_ci_routing.groovy benchmark/whitebox/Jenkinsfile`, `groovy benchmark/whitebox/test_ci_post.groovy benchmark/whitebox/Jenkinsfile`, and `python -m pytest benchmark/whitebox/test_ci_registry.py -q`. Use the documented Java/Groovy classpath form if Groovy CLI is unavailable. Expect all pass with no cloud writes; then use the existing non-executing Jenkins pipeline model converter before later installation.
- [ ] In later controlled builds, seed one trusted labelled image and build a different SG SHA with the same recipe, then change recipe/target. Capture discovered/accepted/rejected digests, actual cached vertices and current output revision. Cold build with no candidates must also succeed; old unlabelled images are not a failure. Evidence must separate cache reuse from final-image reuse and account for inline-cache intermediate-stage limits.
- [ ] Update docs, check links/diff, and commit independently from Task 1. Before any later operational rollout, read and hash installed inline Jenkins and managed Dockerfiles, reconcile reviewed bytes, and run the original regional output/callback verification. Do not claim repo edits update installed jobs or live services.

## Completion evidence required later

Record task commits, exact recipe/source/pin identities, focused test results,
candidate image stage and role-startup results, cache hit/miss traces, and the
remaining native/GPU/playback/production gates. Repository source tasks are implemented and independently reviewed through `1e06954efd`: projection `025c6de9ab`, required RTC fixture repair `bf4dd7c4f3`, verified cache wiring `1e06954efd`. Local source checks passed; real image/native/launcher gates, installed Jenkins/model converter, cached vertices and operational release remain unfinished. The step checkboxes remain the complete acceptance boundary; source review does not close those unexecuted gates. Preserve historical acceptance failures separately.
