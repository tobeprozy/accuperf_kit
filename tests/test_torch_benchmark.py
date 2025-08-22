#!/usr/bin/env python3
"""
PyTorch interface and performance sanity tests using pytest.
"""

from typing import Tuple, List

import pytest
import torch
import torch.nn.functional as F

from src.torch_utils import (
    generate_random_data,
    measure_time,
    cosine_similarity,
)
from src.results import BenchmarkRecord


def to_cuda_or_skip(t: torch.Tensor) -> torch.Tensor:
    """Move tensor to CUDA or skip the test if backend/device is not usable."""
    try:
        return t.to("cuda")
    except Exception as exc:  # noqa: BLE001 - test infra should be defensive
        pytest.skip(f"CUDA not usable in this environment: {exc}")
        raise


@pytest.mark.cpu
@pytest.mark.parametrize("shape", [(100, 100), (1000, 1000)])
def test_matrix_multiplication_cpu_only(shape: Tuple[int, int], results_collector) -> None:
    duration, result = measure_time(
        lambda: torch.mm(
            generate_random_data(shape, "cpu"),
            generate_random_data(shape, "cpu"),
        )
    )
    assert result.shape == (shape[0], shape[1])
    assert duration >= 0.0

    print(f"[CPU] matmul shape={shape} time={duration:.6f}s")

    results_collector.add(
        BenchmarkRecord(
            test_name="test_matrix_multiplication_cpu_only",
            operation="matmul",
            shape=str(shape),
            device_pair="cpu-only",
            cpu_time_s=duration,
            cuda_time_s=None,
            speedup=None,
            cosine_similarity=None,
        )
    )


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(100, 100), (1000, 1000)])
def test_matrix_multiplication_cpu_vs_cuda(shape: Tuple[int, int], cosine_threshold: float, results_collector) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Use identical inputs for CPU and CUDA comparisons
    a_cpu = generate_random_data(shape, "cpu")
    b_cpu = generate_random_data(shape, "cpu")
    cpu_time, cpu_result = measure_time(lambda: torch.mm(a_cpu, b_cpu))

    a_cuda = to_cuda_or_skip(a_cpu)
    b_cuda = to_cuda_or_skip(b_cpu)
    cuda_time, cuda_result = measure_time(lambda: torch.mm(a_cuda, b_cuda))

    similarity = cosine_similarity(cuda_result, cpu_result.to("cuda"))
    assert similarity > cosine_threshold
    assert cuda_time >= 0.0 and cpu_time >= 0.0

    speedup = (cpu_time / cuda_time) if cuda_time > 0 else None
    print(
        f"[CPU↔CUDA] matmul shape={shape} cpu={cpu_time:.6f}s cuda={cuda_time:.6f}s "
        f"speedup={speedup:.2f}x sim={similarity:.6f}"
    )

    results_collector.add(
        BenchmarkRecord(
            test_name="test_matrix_multiplication_cpu_vs_cuda",
            operation="matmul",
            shape=str(shape),
            device_pair="cpu-vs-cuda",
            cpu_time_s=cpu_time,
            cuda_time_s=cuda_time,
            speedup=speedup,
            cosine_similarity=similarity,
        )
    )


@pytest.mark.cpu
def test_conv2d_cpu(shapes: List[Tuple[int, ...]], results_collector) -> None:
    batch_size, channels, height, width = shapes[2]
    input_cpu = generate_random_data((batch_size, channels, height, width), "cpu")
    kernel_cpu = generate_random_data((channels, channels, 3, 3), "cpu")
    duration, result = measure_time(lambda: F.conv2d(input_cpu, kernel_cpu, padding=1))
    assert result.shape == (batch_size, channels, height, width)
    assert duration >= 0.0

    print(
        f"[CPU] conv2d input={(batch_size, channels, height, width)} time={duration:.6f}s"
    )

    results_collector.add(
        BenchmarkRecord(
            test_name="test_conv2d_cpu",
            operation="conv2d",
            shape=str((batch_size, channels, height, width)),
            device_pair="cpu-only",
            cpu_time_s=duration,
            cuda_time_s=None,
            speedup=None,
            cosine_similarity=None,
        )
    )


@pytest.mark.cuda
def test_conv2d_cpu_vs_cuda(shapes: List[Tuple[int, ...]], cosine_threshold: float, results_collector) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    batch_size, channels, height, width = shapes[2]
    # Use identical tensors on both devices
    input_cpu = generate_random_data((batch_size, channels, height, width), "cpu")
    kernel_cpu = generate_random_data((channels, channels, 3, 3), "cpu")
    cpu_time, cpu_result = measure_time(lambda: F.conv2d(input_cpu, kernel_cpu, padding=1))

    input_cuda = to_cuda_or_skip(input_cpu)
    kernel_cuda = to_cuda_or_skip(kernel_cpu)
    cuda_time, cuda_result = measure_time(lambda: F.conv2d(input_cuda, kernel_cuda, padding=1))

    similarity = cosine_similarity(cuda_result, cpu_result.to("cuda"))
    assert similarity > cosine_threshold
    assert cuda_time >= 0.0 and cpu_time >= 0.0

    speedup = (cpu_time / cuda_time) if cuda_time > 0 else None
    print(
        f"[CPU↔CUDA] conv2d input={(batch_size, channels, height, width)} cpu={cpu_time:.6f}s "
        f"cuda={cuda_time:.6f}s speedup={speedup:.2f}x sim={similarity:.6f}"
    )

    results_collector.add(
        BenchmarkRecord(
            test_name="test_conv2d_cpu_vs_cuda",
            operation="conv2d",
            shape=str((batch_size, channels, height, width)),
            device_pair="cpu-vs-cuda",
            cpu_time_s=cpu_time,
            cuda_time_s=cuda_time,
            speedup=speedup,
            cosine_similarity=similarity,
        )
    )


@pytest.mark.cpu
@pytest.mark.parametrize("shape", [(1000, 1000)])
def test_softmax_cpu(shape: Tuple[int, int], results_collector) -> None:
    input_cpu = generate_random_data(shape, "cpu")
    duration, result = measure_time(lambda: F.softmax(input_cpu, dim=1))
    assert result.shape == shape
    assert duration >= 0.0

    print(f"[CPU] softmax shape={shape} time={duration:.6f}s")

    results_collector.add(
        BenchmarkRecord(
            test_name="test_softmax_cpu",
            operation="softmax",
            shape=str(shape),
            device_pair="cpu-only",
            cpu_time_s=duration,
            cuda_time_s=None,
            speedup=None,
            cosine_similarity=None,
        )
    )


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(1000, 1000)])
def test_softmax_cpu_vs_cuda(shape: Tuple[int, int], cosine_threshold: float, results_collector) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    input_cpu = generate_random_data(shape, "cpu")
    cpu_time, cpu_result = measure_time(lambda: F.softmax(input_cpu, dim=1))
    input_cuda = to_cuda_or_skip(input_cpu)
    cuda_time, cuda_result = measure_time(lambda: F.softmax(input_cuda, dim=1))
    similarity = cosine_similarity(cuda_result, cpu_result.to("cuda"))
    assert similarity > cosine_threshold
    assert cuda_time >= 0.0 and cpu_time >= 0.0

    speedup = (cpu_time / cuda_time) if cuda_time > 0 else None
    print(
        f"[CPU↔CUDA] softmax shape={shape} cpu={cpu_time:.6f}s cuda={cuda_time:.6f}s "
        f"speedup={speedup:.2f}x sim={similarity:.6f}"
    )

    results_collector.add(
        BenchmarkRecord(
            test_name="test_softmax_cpu_vs_cuda",
            operation="softmax",
            shape=str(shape),
            device_pair="cpu-vs-cuda",
            cpu_time_s=cpu_time,
            cuda_time_s=cuda_time,
            speedup=speedup,
            cosine_similarity=similarity,
        )
    )


@pytest.mark.cpu
@pytest.mark.parametrize("shape", [(1000, 1000)])
def test_relu_cpu(shape: Tuple[int, int], results_collector) -> None:
    input_cpu = generate_random_data(shape, "cpu")
    duration, result = measure_time(lambda: F.relu(input_cpu))
    assert result.shape == shape
    assert duration >= 0.0

    print(f"[CPU] relu shape={shape} time={duration:.6f}s")

    results_collector.add(
        BenchmarkRecord(
            test_name="test_relu_cpu",
            operation="relu",
            shape=str(shape),
            device_pair="cpu-only",
            cpu_time_s=duration,
            cuda_time_s=None,
            speedup=None,
            cosine_similarity=None,
        )
    )


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(1000, 1000)])
def test_relu_cpu_vs_cuda(shape: Tuple[int, int], cosine_threshold: float, results_collector) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    input_cpu = generate_random_data(shape, "cpu")
    cpu_time, cpu_result = measure_time(lambda: F.relu(input_cpu))
    input_cuda = to_cuda_or_skip(input_cpu)
    cuda_time, cuda_result = measure_time(lambda: F.relu(input_cuda))
    similarity = cosine_similarity(cuda_result, cpu_result.to("cuda"))
    assert similarity > cosine_threshold
    assert cuda_time >= 0.0 and cpu_time >= 0.0

    speedup = (cpu_time / cuda_time) if cuda_time > 0 else None
    print(
        f"[CPU↔CUDA] relu shape={shape} cpu={cpu_time:.6f}s cuda={cuda_time:.6f}s "
        f"speedup={speedup:.2f}x sim={similarity:.6f}"
    )

    results_collector.add(
        BenchmarkRecord(
            test_name="test_relu_cpu_vs_cuda",
            operation="relu",
            shape=str(shape),
            device_pair="cpu-vs-cuda",
            cpu_time_s=cpu_time,
            cuda_time_s=cuda_time,
            speedup=speedup,
            cosine_similarity=similarity,
        )
    )


