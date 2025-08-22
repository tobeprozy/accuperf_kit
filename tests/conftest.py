#!/usr/bin/env python3
"""Shared pytest fixtures and configuration for tests."""

import os
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import pytest
import torch
from src.results import ResultsCollector, BenchmarkRecord
from datetime import datetime


@pytest.fixture(autouse=True, scope="function")
def set_deterministic_seed() -> None:
    """Set global RNG seeds for determinism across CPU/CUDA each test."""
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)


@pytest.fixture(scope="session")
def devices() -> Dict[str, Optional[str]]:
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


@pytest.fixture(scope="session")
def results_collector(tmp_path_factory: pytest.TempPathFactory) -> ResultsCollector:
    """Session-scoped results collector that writes outputs at the end."""
    # Save under ./artifacts/<YYYYmmdd_HHMMSS>
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = PROJECT_ROOT / "artifacts" / timestamp
    collector = ResultsCollector(artifacts_dir)
    yield collector
    # Write outputs at session end
    json_file = collector.write_json()
    csv_file = collector.write_csv()
    print(f"Results written to: {json_file} and {csv_file}")

