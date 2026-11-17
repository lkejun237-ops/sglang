# SPDX-License-Identifier: Apache-2.0
"""Bounded-label metrics; encoder ownership is separate from result delivery."""

from prometheus_client import CollectorRegistry, Counter, Histogram

DURATION_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)


class ConditioningMetrics:
    """One registry per service lifetime; aggregates tenants behind admin auth."""

    def __init__(self):
        self.registry = CollectorRegistry()
        self.encodes = Counter(
            "world_model_conditioning_encodes_total",
            "Successful encoder tasks, counted once before result fanout.",
            ("prefix_outcome",),
            registry=self.registry,
        )
        self.tokens = Counter(
            "world_model_conditioning_tokens_total",
            "Qwen input tokens including visual and template tokens; known statistics only.",
            ("kind",),
            registry=self.registry,
        )
        self.reused = Histogram(
            "world_model_conditioning_reused_tokens",
            "Reused prefix length per successful task with known statistics.",
            buckets=(0, 128, 512, 1024, 4096, 16384, 65536),
            registry=self.registry,
        )
        self.duration = Histogram(
            "world_model_conditioning_encode_duration_seconds",
            "Encoder task time including preprocessing and result staging, excluding queue wait.",
            ("prefix_outcome",),
            buckets=DURATION_BUCKETS,
            registry=self.registry,
        )
        self.queue = Histogram(
            "world_model_conditioning_queue_duration_seconds",
            "Queue wait for successful encoder tasks.",
            buckets=DURATION_BUCKETS,
            registry=self.registry,
        )
        self.results = Counter(
            "world_model_conditioning_results_total",
            "Completed result deliveries, distinct from encoder executions.",
            ("source",),
            registry=self.registry,
        )
        for outcome in ("hit", "miss", "unknown"):
            self.encodes.labels(outcome)
            self.duration.labels(outcome)
        for kind in ("reused", "encoded", "total"):
            self.tokens.labels(kind)
        for source in ("encoding", "joined", "result_cache"):
            self.results.labels(source)

    def encoded(self, stats, encode_ms, queue_ms):
        outcome = "unknown" if stats is None else "hit" if stats["hit"] else "miss"
        self.encodes.labels(outcome).inc()
        self.duration.labels(outcome).observe(encode_ms / 1000)
        if queue_ms is not None:
            self.queue.observe(queue_ms / 1000)
        if stats is not None:
            for kind in ("reused", "encoded", "total"):
                self.tokens.labels(kind).inc(stats[kind + "_tokens"])
            self.reused.observe(stats["reused_tokens"])


def install_metrics_route(app, metrics, token):
    """Expose only counters to a dedicated read-only bearer credential."""
    import hmac

    from fastapi import Header
    from fastapi.responses import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    if token is not None and (
        not isinstance(token, str)
        or not token
        or len(token) > 4096
        or any(not 33 <= ord(c) <= 126 for c in token)
        or metrics is None
    ):
        raise ValueError("Invalid conditioning metrics credential or registry")

    @app.get("/metrics")
    async def conditioning_metrics(authorization: str = Header(default="")):
        if token is None:
            return Response(status_code=404)
        if not hmac.compare_digest(
            authorization.encode(), ("Bearer " + token).encode()
        ):
            return Response(status_code=403)
        return Response(
            generate_latest(metrics.registry),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )
