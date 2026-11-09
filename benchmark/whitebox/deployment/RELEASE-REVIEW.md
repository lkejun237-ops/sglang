# Historical Whitebox release review — 2026-10-12

This was an approved rollout **in progress**, not an online acceptance receipt.
Current operations start at [the release entry](../../../deploy/whitebox/README.md).
The original complete request and review are preserved in the
[private archive index](../history/20261109-evidence-index.json).

The historical candidate added CPU Coordinator/mock worker/Gateway to the Beijing
inference cluster, two slots, 60s play / 90s startup / 30s drain, an isolated Redis
prefix, retained 50Gi recording PVC and new mock-only OSS paths. Existing Zing,
platform tasks, models and old recordings were outside that scope. These are dated
facts about the review, not parameters for a new deployment.

At that checkpoint Jenkins syntax validation passed; the AMD64 image ran 145
contract tests; platform regression ran 139 tests. SDK public-client/typecheck,
census and prebuild passed. Source/payload identities are recoverable through the
original archive; this record did not freeze a complete live runtime version.

Activation encountered the existing ACR helper namespace allowlist: the new Pod
could not pull its Istio image. A separately approved narrow allowlist addition
was requested; this record does not prove that activation or acceptance completed.

Rollback was limited to the newly added routes and services, restoring reviewed
images/config while retaining PVC/OSS records and letting isolated lease keys
expire. No Redis flush, namespace-wide cleanup, or existing inference shutdown
was authorized by that procedure. Historical request JSON/YAML is no longer an
executable repository input; it must not be reused as a current rollback plan.
