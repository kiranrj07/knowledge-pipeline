"""Pytest configuration for knowledge-pipeline tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config: object) -> None:
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (require external services; deselect with -m 'not slow')",
    )
