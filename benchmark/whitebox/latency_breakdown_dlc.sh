#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# CPU preflight: pass --prepare-only. GPU run: no arguments.
set -eu
LATENCY_WORK="${LATENCY_AB_WORK:-/work}"
LATENCY_BOOT="$(mktemp -d /tmp/sglang-latency-boot-XXXXXX)"
tar --no-same-owner -xzf "$LATENCY_WORK/source_bundle.tar.gz" -C "$LATENCY_BOOT"
exec /opt/sglang/bin/python -u "$LATENCY_BOOT/benchmark/whitebox/latency_breakdown_gpu_ab.py" "$@"
