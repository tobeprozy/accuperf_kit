#!/usr/bin/env python3
"""
简单的PyTorch接口测试脚本
直接运行，无需pytest，测试CPU和CUDA的运行时间以及结果相似度
"""

import torch
import torch.nn.functional as F
import time
import numpy as np


def generate_random_data(shape, device='cpu'):
    """生成随机数据"""
    return torch.randn(shape, device=device)


def measure_time(func, *args, **kwargs):
    """测量函数执行时间"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result


def cosine_similarity(tensor1, tensor2):
    """计算两个张量的余弦相似度"""
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()
    cos_sim = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0), dim=1)
    return cos_sim.item()


def test_matrix_multiplication():
    """测试矩阵乘法"""
    print("=" * 50)
    print("测试矩阵乘法")
    print("=" * 50)
    
    shapes = [(100, 100), (1000, 1000)]
    
    for shape in shapes:
        if shape[0] != shape[1]:  # 跳过非方阵
            continue
            
        print(f"\n矩阵大小: {shape}")
        
        # CPU测试
        cpu_time, cpu_result = measure_time(
            lambda: torch.mm(
                generate_random_data(shape, 'cpu'),
                generate_random_data(shape, 'cpu')
            )
        )
        
        # CUDA测试（如果可用）
        if torch.cuda.is_available():
            cuda_time, cuda_result = measure_time(
                lambda: torch.mm(
                    generate_random_data(shape, 'cuda'),
                    generate_random_data(shape, 'cuda')
                )
            )
            
            # 验证结果相似度
            cpu_result_cuda = cpu_result.to('cuda')
            similarity = cosine_similarity(cuda_result, cpu_result_cuda)
            
            print(f"  CPU时间: {cpu_time:.4f}s")
            print(f"  CUDA时间: {cuda_time:.4f}s")
            print(f"  加速比: {cpu_time/cuda_time:.2f}x")
            print(f"  余弦相似度: {similarity:.6f}")
            
            # 检查相似度
            if similarity > 0.999:
                print(f"  ✓ 结果一致")
            else:
                print(f"  ✗ 结果不一致！相似度: {similarity}")
        else:
            print(f"  CPU时间: {cpu_time:.4f}s")
            print(f"  CUDA不可用")


def test_convolution_2d():
    """测试2D卷积"""
    print("\n" + "=" * 50)
    print("测试2D卷积")
    print("=" * 50)
    
    batch_size, channels, height, width = 4, 3, 64, 64
    kernel_size = 3
    
    print(f"\n输入形状: ({batch_size}, {channels}, {height}, {width})")
    
    # 生成输入和卷积核
    input_cpu = generate_random_data((batch_size, channels, height, width), 'cpu')
    kernel_cpu = generate_random_data((channels, channels, kernel_size, kernel_size), 'cpu')
    
    # CPU测试
    cpu_time, cpu_result = measure_time(
        lambda: F.conv2d(input_cpu, kernel_cpu, padding=1)
    )
    
    # CUDA测试（如果可用）
    if torch.cuda.is_available():
        input_cuda = input_cpu.to('cuda')
        kernel_cuda = kernel_cpu.to('cuda')
        
        cuda_time, cuda_result = measure_time(
            lambda: F.conv2d(input_cuda, kernel_cuda, padding=1)
        )
        
        # 验证结果相似度
        cpu_result_cuda = cpu_result.to('cuda')
        similarity = cosine_similarity(cuda_result, cpu_result_cuda)
        
        print(f"  CPU时间: {cpu_time:.4f}s")
        print(f"  CUDA时间: {cuda_time:.4f}s")
        print(f"  加速比: {cpu_time/cuda_time:.2f}x")
        print(f"  余弦相似度: {similarity:.6f}")
        
        if similarity > 0.999:
            print(f"  ✓ 结果一致")
        else:
            print(f"  ✗ 结果不一致！相似度: {similarity}")
    else:
        print(f"  CPU时间: {cpu_time:.4f}s")
        print(f"  CUDA不可用")


def test_activation_functions():
    """测试激活函数"""
    print("\n" + "=" * 50)
    print("测试激活函数")
    print("=" * 50)
    
    shape = (1000, 1000)
    print(f"\n张量形状: {shape}")
    
    # 测试ReLU
    print("\nReLU激活函数:")
    input_cpu = generate_random_data(shape, 'cpu')
    cpu_time, cpu_result = measure_time(lambda: F.relu(input_cpu))
    
    if torch.cuda.is_available():
        input_cuda = input_cpu.to('cuda')
        cuda_time, cuda_result = measure_time(lambda: F.relu(input_cuda))
        
        cpu_result_cuda = cpu_result.to('cuda')
        similarity = cosine_similarity(cuda_result, cpu_result_cuda)
        
        print(f"  CPU时间: {cpu_time:.4f}s")
        print(f"  CUDA时间: {cuda_time:.4f}s")
        print(f"  加速比: {cpu_time/cuda_time:.2f}x")
        print(f"  余弦相似度: {similarity:.6f}")
        
        if similarity > 0.999:
            print(f"  ✓ 结果一致")
        else:
            print(f"  ✗ 结果不一致！相似度: {similarity}")
    else:
        print(f"  CPU时间: {cpu_time:.4f}s")
        print(f"  CUDA不可用")
    
    # 测试Softmax
    print("\nSoftmax激活函数:")
    input_cpu = generate_random_data(shape, 'cpu')
    cpu_time, cpu_result = measure_time(lambda: F.softmax(input_cpu, dim=1))
    
    if torch.cuda.is_available():
        input_cuda = input_cpu.to('cuda')
        cuda_time, cuda_result = measure_time(lambda: F.softmax(input_cuda, dim=1))
        
        cpu_result_cuda = cpu_result.to('cuda')
        similarity = cosine_similarity(cuda_result, cpu_result_cuda)
        
        print(f"  CPU时间: {cpu_time:.4f}s")
        print(f"  CUDA时间: {cuda_time:.4f}s")
        print(f"  加速比: {cpu_time/cuda_time:.2f}x")
        print(f"  余弦相似度: {similarity:.6f}")
        
        if similarity > 0.999:
            print(f"  ✓ 结果一致")
        else:
            print(f"  ✗ 结果不一致！相似度: {similarity}")
    else:
        print(f"  CPU时间: {cpu_time:.4f}s")
        print(f"  CUDA不可用")


def test_custom_operation():
    """测试自定义操作 - 在这里添加你的新测试"""
    print("\n" + "=" * 50)
    print("测试自定义操作")
    print("=" * 50)
    
    # 示例：测试tanh激活函数
    print("\nTanh激活函数:")
    shape = (1000, 1000)
    input_cpu = generate_random_data(shape, 'cpu')
    cpu_time, cpu_result = measure_time(lambda: torch.tanh(input_cpu))
    
    if torch.cuda.is_available():
        input_cuda = input_cpu.to('cuda')
        cuda_time, cuda_result = measure_time(lambda: torch.tanh(input_cuda))
        
        cpu_result_cuda = cpu_result.to('cuda')
        similarity = cosine_similarity(cuda_result, cpu_result_cuda)
        
        print(f"  张量形状: {shape}")
        print(f"  CPU时间: {cpu_time:.4f}s")
        print(f"  CUDA时间: {cuda_time:.4f}s")
        print(f"  加速比: {cpu_time/cuda_time:.2f}x")
        print(f"  余弦相似度: {similarity:.6f}")
        
        if similarity > 0.999:
            print(f"  ✓ 结果一致")
        else:
            print(f"  ✗ 结果不一致！相似度: {similarity}")
    else:
        print(f"  张量形状: {shape}")
        print(f"  CPU时间: {cpu_time:.4f}s")
        print(f"  CUDA不可用")


def main():
    """主函数"""
    print("PyTorch接口性能测试")
    print("测试CPU和CUDA的运行时间以及结果一致性")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n开始测试...\n")
    
    try:
        # 运行所有测试
        test_matrix_multiplication()
        test_convolution_2d()
        test_activation_functions()
        test_custom_operation()
        
        print("\n" + "=" * 50)
        print("所有测试完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
