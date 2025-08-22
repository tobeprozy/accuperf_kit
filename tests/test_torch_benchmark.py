#!/usr/bin/env python3
"""
PyTorch interface and performance sanity tests using pytest.
"""

from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

from src.torch_utils import (
    generate_random_data,
    measure_time,
    cosine_similarity,
)


@pytest.mark.cpu
@pytest.mark.parametrize("shape", [(100, 100), (1000, 1000)])
def test_matrix_multiplication_cpu_only(shape: Tuple[int, int]) -> None:
    duration, result = measure_time(
        lambda: torch.mm(
            generate_random_data(shape, "cpu"),
            generate_random_data(shape, "cpu"),
        )
    )
    assert result.shape == (shape[0], shape[1])
    assert duration >= 0.0


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(100, 100), (1000, 1000)])
def test_matrix_multiplication_cpu_vs_cuda(shape: Tuple[int, int], cosine_threshold: float) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cpu_time, cpu_result = measure_time(
        lambda: torch.mm(
            generate_random_data(shape, "cpu"),
            generate_random_data(shape, "cpu"),
        )
    )
    cuda_time, cuda_result = measure_time(
        lambda: torch.mm(
            generate_random_data(shape, "cuda"),
            generate_random_data(shape, "cuda"),
        )
    )

    similarity = cosine_similarity(cuda_result, cpu_result.to("cuda"))
    assert similarity > cosine_threshold
    assert cuda_time >= 0.0 and cpu_time >= 0.0


@pytest.mark.cpu
def test_conv2d_cpu(shapes: list[tuple[int, ...]]) -> None:
    batch_size, channels, height, width = shapes[2]
    input_cpu = generate_random_data((batch_size, channels, height, width), "cpu")
    kernel_cpu = generate_random_data((channels, channels, 3, 3), "cpu")
    duration, result = measure_time(lambda: F.conv2d(input_cpu, kernel_cpu, padding=1))
    assert result.shape == (batch_size, channels, height, width)
    assert duration >= 0.0


@pytest.mark.cuda
def test_conv2d_cpu_vs_cuda(shapes: list[tuple[int, ...]], cosine_threshold: float) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    batch_size, channels, height, width = shapes[2]
    input_cpu = generate_random_data((batch_size, channels, height, width), "cpu")
    kernel_cpu = generate_random_data((channels, channels, 3, 3), "cpu")
    cpu_time, cpu_result = measure_time(lambda: F.conv2d(input_cpu, kernel_cpu, padding=1))

    input_cuda = input_cpu.to("cuda")
    kernel_cuda = kernel_cpu.to("cuda")
    cuda_time, cuda_result = measure_time(lambda: F.conv2d(input_cuda, kernel_cuda, padding=1))

    similarity = cosine_similarity(cuda_result, cpu_result.to("cuda"))
    assert similarity > cosine_threshold
    assert cuda_time >= 0.0 and cpu_time >= 0.0


@pytest.mark.cpu
@pytest.mark.parametrize("shape", [(1000, 1000)])
def test_softmax_cpu(shape: Tuple[int, int]) -> None:
    input_cpu = generate_random_data(shape, "cpu")
    duration, result = measure_time(lambda: F.softmax(input_cpu, dim=1))
    assert result.shape == shape
    assert duration >= 0.0


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(1000, 1000)])
def test_softmax_cpu_vs_cuda(shape: Tuple[int, int], cosine_threshold: float) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    input_cpu = generate_random_data(shape, "cpu")
    cpu_time, cpu_result = measure_time(lambda: F.softmax(input_cpu, dim=1))
    cuda_time, cuda_result = measure_time(lambda: F.softmax(input_cpu.to("cuda"), dim=1))
    similarity = cosine_similarity(cuda_result, cpu_result.to("cuda"))
    assert similarity > cosine_threshold
    assert cuda_time >= 0.0 and cpu_time >= 0.0


@pytest.mark.cpu
@pytest.mark.parametrize("shape", [(1000, 1000)])
def test_relu_cpu(shape: Tuple[int, int]) -> None:
    input_cpu = generate_random_data(shape, "cpu")
    duration, result = measure_time(lambda: F.relu(input_cpu))
    assert result.shape == shape
    assert duration >= 0.0


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(1000, 1000)])
def test_relu_cpu_vs_cuda(shape: Tuple[int, int], cosine_threshold: float) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    input_cpu = generate_random_data(shape, "cpu")
    cpu_time, cpu_result = measure_time(lambda: F.relu(input_cpu))
    cuda_time, cuda_result = measure_time(lambda: F.relu(input_cpu.to("cuda")))
    similarity = cosine_similarity(cuda_result, cpu_result.to("cuda"))
    assert similarity > cosine_threshold
    assert cuda_time >= 0.0 and cpu_time >= 0.0


