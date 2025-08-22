#!/usr/bin/env python3
"""Shared pytest fixtures and configuration for tests."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch


# Ensure project root is on sys.path so that `src` is importable in all envs
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def devices() -> Dict[str, str | None]:
    """Return available device mapping for tests."""
    return {
        "cpu": "cpu",
        "cuda": "cuda" if torch.cuda.is_available() else None,
    }


@pytest.fixture(scope="session")
def shapes() -> List[Tuple[int, ...]]:
    """Common tensor shapes used in tests."""
    return [
        (100, 100),
        (1000, 1000),
        (4, 3, 64, 64),
    ]


@pytest.fixture(scope="session")
def cosine_threshold() -> float:
    """Similarity threshold used to assert numerical closeness across devices."""
    # Allow override via env var if needed (e.g., CI flakiness)
    env_value = os.getenv("COSINE_SIM_THRESHOLD")
    if env_value is not None:
        try:
            return float(env_value)
        except ValueError:
            pass
    return 0.999


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cuda: tests that require CUDA if available")
    config.addinivalue_line("markers", "cpu: tests that run on CPU")


