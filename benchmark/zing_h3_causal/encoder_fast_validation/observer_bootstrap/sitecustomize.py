# SPDX-License-Identifier: Apache-2.0
"""Install the existing allocator observer in owned benchmark workers and spawn children."""

import os

if os.environ.get("ZING_BENCHMARK_MEMORY_DIR"):
    import memory_probe

    memory_probe.install()
