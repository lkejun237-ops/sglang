"""Exact exporter contract shared by host projection and the native FC relay.

No inference imports or I/O. Identity labels belong to the proved collector
target, never to the exposition. Unknown families (including _created) are
ignored; malformed samples in supported families reject the collection.
"""

PREFIX = "world_model_conditioning_"
OUTCOMES = frozenset({"hit", "miss", "unknown"})
DURATION_BUCKETS = frozenset(
    {0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30}
)
TOKEN_BUCKETS = frozenset({0, 128, 512, 1024, 4096, 16384, 65536})
LABELS = {
    PREFIX + "encodes_total": {"prefix_outcome": OUTCOMES},
    PREFIX + "tokens_total": {"kind": frozenset({"reused", "encoded", "total"})},
    PREFIX + "results_total": {
        "source": frozenset({"encoding", "joined", "result_cache"})
    },
}
BUCKETS = {}
for family, labels, buckets in (
    ("reused_tokens", {}, TOKEN_BUCKETS),
    ("encode_duration_seconds", {"prefix_outcome": OUTCOMES}, DURATION_BUCKETS),
    ("queue_duration_seconds", {}, DURATION_BUCKETS),
):
    for suffix in ("bucket", "sum", "count"):
        name = PREFIX + family + "_" + suffix
        LABELS[name] = dict(labels)
        if suffix == "bucket":
            LABELS[name]["le"] = None
            BUCKETS[name] = frozenset(str(float(bound)) for bound in buckets) | {"+Inf"}


def validate_labels(name, labels):
    """Require every semantic dimension; do not silently erase extra labels."""
    expected = LABELS[name]
    if set(labels) != set(expected):
        raise ValueError("conditioning_label_set")
    for key, allowed in expected.items():
        value = labels[key]
        if key != "le":
            if value not in allowed:
                raise ValueError("conditioning_label_value")
        elif value not in BUCKETS[name]:
            raise ValueError("conditioning_bucket")


def is_test_conditioning_target(identity):
    return identity == {
        "business_line_id": "world-model-data",
        "region": "cn-shanghai",
        "deployment_id": "test-shanghai-copy",
        "pool": "test-shanghai-copy",
        "service": "sglang",
    }
