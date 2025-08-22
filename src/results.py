#!/usr/bin/env python3
"""Results collection and persistence utilities for benchmarks."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkRecord:
    test_name: str
    operation: str
    shape: str
    device_pair: str  # e.g., "cpu-vs-cuda" or "cpu-only"
    cpu_time_s: Optional[float]
    cuda_time_s: Optional[float]
    speedup: Optional[float]
    cosine_similarity: Optional[float]


class ResultsCollector:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.records: List[BenchmarkRecord] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add(self, record: BenchmarkRecord) -> None:
        self.records.append(record)

    def to_json(self) -> str:
        payload = [asdict(r) for r in self.records]
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def write_json(self, filename: str = "results.json") -> Path:
        target = self.output_dir / filename
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    def write_csv(self, filename: str = "results.csv") -> Path:
        target = self.output_dir / filename
        fieldnames = list(BenchmarkRecord.__annotations__.keys())
        with target.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow(asdict(r))
        return target


