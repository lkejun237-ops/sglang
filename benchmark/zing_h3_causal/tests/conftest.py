# SPDX-License-Identifier: Apache-2.0
"""Make standalone benchmark helpers importable by their contract tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
