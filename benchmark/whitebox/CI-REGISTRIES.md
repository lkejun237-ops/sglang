# Whitebox registry routing

The Thailand `whitebox-th-room` and `whitebox-th-control` Jenkins jobs publish
only to `lwdp-codex-th-registry.ap-southeast-7.cr.aliyuncs.com/lwdp/whitebox-runtime`.
Platform `IMAGE` parameters and successful CD callbacks use the corresponding
`lwdp-codex-th-registry-vpc.ap-southeast-7.cr.aliyuncs.com` address. The tag is
preserved exactly. An ECR, Beijing ACR or public-endpoint runtime target is
rejected before building. The Thailand UI runs in Beijing and retains its
existing Beijing publication behavior.

The build executor remains on the existing Jenkins infrastructure. Its ECR
credentials are used for existing base images and optional build caches, not
for publishing Thailand runtime artifacts. Thailand pushes use the existing
ACR credential binding with region `ap-southeast-7`, instance
`cri-mqlfg5m1ei9q84qx`, and a repository-scoped RAM grant. The Thailand public
registry endpoint must admit the verified CI NAT address. The Beijing-specific
proxy is not used for Thailand. ACK workloads pull through their VPC endpoint
using the managed ACR credential helper.

BuildKit metadata supplies the output digest. Before a success callback,
verification checks the registry digest, the raw manifest and image-config
SHA256 values, `linux/amd64`, and the complete OCI source revision label.
The verified ENTRYPOINT/CMD are included in the callback. Missing or mismatched
verification aborts the success callback. Unique immutable platform tags are
required; no `latest` overwrite or silent ECR fallback is supported.

The Loopit service image is the source of the repository for subsequent builds.
Migrate each service and owned init container to a verified VPC digest before
resuming ordinary builds. Keep rollback images in the same Thailand repository.
Update both the versioned Jenkinsfile and the two installed inline Jenkins jobs;
editing the repository alone does not update those jobs. Preserve managed
Dockerfiles and unrelated regional jobs.

Local checks:

```sh
python -m pytest benchmark/whitebox/test_ci_registry.py
java -cp /path/to/groovy.jar groovy.ui.GroovyMain \
  benchmark/whitebox/test_ci_routing.groovy benchmark/whitebox/Jenkinsfile
```

The tests execute registry verification against simulated valid and corrupt
artifacts and execute the real Groovy routing functions. Also validate the full
file with Jenkins' non-executing pipeline model converter before installation.

## Cross-commit layer candidates

Control (`release`) and Room (`room`) builds now label their outputs with
`org.worldkit.whitebox.cache.branch`, `.target` and `.recipe-sha256`. The recipe
hash uses the exact **downloaded managed Dockerfile** bytes, effective
Dockerfile-specific ignore file (or root `.dockerignore`), Whitebox requirements,
selected target, `linux/amd64`, effective external `FROM` references,
BuildKit image reference and Dockerfile frontend identity. Only Room includes
`sdk-revision.txt`. The same helper supplies the pod's BuildKit image reference.
Mutable base/frontend/BuildKit references remain references, not claimed resolved
digests; BuildKit resolves and keys actual layers. Unresolved base arguments fail
recipe construction rather than inventing an effective base identity. Canonical
UTF-8 JSON is written under the job temporary workspace, hashed by `sha256sum`
in the existing AWS container, then removed in `finally`. JSON never enters
shell text or the image build context; no new Jenkins hash-method approval or
plugin is required.

General runtime/test source bytes and the SGLang source SHA are excluded from
this recipe. BuildKit still keys their COPY/RUN inputs; every new build retains
its actual `SOURCE_REVISION` and OCI revision. The existing final-tag lookup and
immutable-image reuse branch are unchanged. Layer candidates never satisfy that
lookup and do not establish that the current checks ran or that a layer was hit.

For `white_box` builds, the existing ECR credential scope discovers the 20 newest
tagged records in `leap-world/minwm-realtime`. At most three distinct candidates
are retained after checking the fixed account/repository, actual manifest/config
SHA256 bytes, full source revision, branch, logical target, recipe and platform.
Old unlabelled images, indexes without an eligible platform manifest, unrelated
roles and corrupt/unavailable records are cache misses. References use verified
`repository@sha256:...` digests so a moved tag cannot change the imported artifact.
Only the scoped registry reader creates production integrity records; the pure
metadata selector's `integrityVerified` field is not an independent trust proof.

Accepted references append to `/build-secrets/world-model-cache-imports`, which
the actual build command reads as `--import-cache`. Existing same-SHA candidates
and the inline exporter remain. UI behavior, regional output routing and ACR
verification/callbacks are unchanged. Thailand still reads only the existing ECR
cache and publishes only to its approved ACR output. Neither dormant shared-cache
helper is called; no global writable cache tag/repository/exporter is enabled.
Optional discovery failures retain existing imports or produce a cold build;
mandatory registry preflight still fails as before. Interruptions propagate.

Run the real helper/command tests without cloud writes:

```sh
python -m pytest benchmark/whitebox/test_ci_registry.py -q
java -cp '/path/to/groovy-3.0.24.jar:/path/to/groovy-json-3.0.24.jar' \
  groovy.ui.GroovyMain benchmark/whitebox/test_ci_cache.groovy benchmark/whitebox/Jenkinsfile
java -cp '/path/to/groovy-3.0.24.jar:/path/to/groovy-json-3.0.24.jar' \
  groovy.ui.GroovyMain benchmark/whitebox/test_ci_routing.groovy benchmark/whitebox/Jenkinsfile
java -cp '/path/to/groovy-3.0.24.jar:/path/to/groovy-json-3.0.24.jar' \
  groovy.ui.GroovyMain benchmark/whitebox/test_ci_post.groovy benchmark/whitebox/Jenkinsfile
```

These are repository CPU checks. Before installing any job, run the existing
non-executing Jenkins model converter, read/hash the installed inline Jenkins
and managed Dockerfiles, and reconcile those bytes with the reviewed version.
Repository edits do not install hosted jobs. Later controlled cold/warm builds
must record accepted/rejected digests, actual cached vertices, checks and output
revision/digest, including changed-recipe and wrong-target misses. Inline cache
may omit intermediate-stage layers; candidate discovery alone proves no speedup.
Actual candidate-image/native/GPU/playback and regional callback acceptance remain
separate from this change and require their original operational authorization.
