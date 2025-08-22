#!/usr/bin/env python3
"""
Shared utilities for PyTorch benchmarking and tests.
"""

from typing import Callable, Tuple, Any

import torch
import torch.nn.functional as F
import time


def generate_random_data(shape: Tuple[int, ...], device: str = "cpu") -> torch.Tensor:
    """Generate random tensor data on the specified device."""
    return torch.randn(shape, device=device)


def measure_time(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[float, Any]:
    """Measure wall-clock execution time of a callable and return (seconds, result)."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result


def cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute cosine similarity of two tensors after flattening."""
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()
    cos_sim = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0), dim=1)
    return float(cos_sim.item())


