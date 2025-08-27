#!/usr/bin/env python3
"""
简单的PyTorch接口测试脚本
"""

import torch
import torch.nn.functional as F
import time

Wq=torch.nn.Linear(4096,4096)
Wk=torch.nn.Linear(4096,4096)
Wv=torch.nn.Linear(4096,4096)

def generate_random_data(shape, device='cpu'):
    """生成随机数据"""
    return torch.randn(shape, device=device)


def measure_time(func, *args, **kwargs):
    """测量函数执行时间"""
    start_time = time.time()
    # 修正：将函数调用改为传入函数本身和参数
    result = func(*args, **kwargs)
    end_time = time.time()
    # 将秒转换为毫秒
    return (end_time - start_time) * 1000, result


def cosine_similarity(tensor1, tensor2):
    """计算两个张量的余弦相似度"""
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()
    cos_sim = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0), dim=1)
    return cos_sim.item()


def test_linear():
    """测试线性层"""
    print("=" * 50)
    print("测试线性层")
    print("=" * 50)
    hidden_states=generate_random_data((128,1024,4096))
    # CPU测试
    # 
    print("CPU测试")
    qcpu_time, q_result = measure_time(lambda: Wq(hidden_states))
    kcpu_time, k_result = measure_time(lambda: Wk(hidden_states))
    vcpu_time, v_result = measure_time(lambda: Wv(hidden_states))

    print(f"  QKV时间: {qcpu_time:.4f}ms, {kcpu_time:.4f}ms, {vcpu_time:.4f}ms")
    print(f"  QKV大小: {q_result.shape}, {k_result.shape}, {v_result.shape}")

    print("CPU测试")
    qcpu_time, q_result = measure_time(lambda: Wq(hidden_states.view(1,-1,4096)))
    kcpu_time, k_result = measure_time(lambda: Wk(hidden_states.view(1,-1,4096)))
    vcpu_time, v_result = measure_time(lambda: Wv(hidden_states.view(1,-1,4096)))



    print(f"  QKV时间: {qcpu_time:.4f}ms, {kcpu_time:.4f}ms, {vcpu_time:.4f}ms")
    print(f"  QKV大小: {q_result.shape}, {k_result.shape}, {v_result.shape}")    
    if torch.cuda.is_available():
        # 为了避免在函数内重新赋值导致外部变量不可用，这里使用新变量
        Wq_cuda = Wq.to('cuda')
        Wk_cuda = Wk.to('cuda')
        Wv_cuda = Wv.to('cuda')

        hidden_states=hidden_states.to('cuda')
        # 修正：将函数调用改为传入函数本身和参数
        print("CUDA测试")
        qcuda_time, q_result = measure_time(lambda: Wq_cuda(hidden_states))
        kcuda_time, k_result = measure_time(lambda: Wk_cuda(hidden_states))
        vcuda_time, v_result = measure_time(lambda: Wv_cuda(hidden_states))

        print(f"  QKV时间: {qcuda_time:.4f}ms, {kcuda_time:.4f}ms, {vcuda_time:.4f}ms")
        print(f"  QKV大小: {q_result.shape}, {k_result.shape}, {v_result.shape}")

        qcuda_time, q_result = measure_time(lambda: Wq_cuda(hidden_states.view(1,-1,4096)))
        kcuda_time, k_result = measure_time(lambda: Wk_cuda(hidden_states.view(1,-1,4096)))
        vcuda_time, v_result = measure_time(lambda: Wv_cuda(hidden_states.view(1,-1,4096)))

        start_time = time.time()
        # 修正：将函数调用改为传入函数本身和参数
        Wq_cuda(hidden_states.view(1,-1,4096))
        end_time = time.time()
        qcuda_time = (end_time - start_time) * 1000
        print(f"  Q时间: {qcuda_time:.4f}ms")

        print(f"  QKV时间: {qcuda_time:.4f}ms, {kcuda_time:.4f}ms, {vcuda_time:.4f}ms")
        print(f"  QKV大小: {q_result.shape}, {k_result.shape}, {v_result.shape}")

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
        test_linear()
        print("\n" + "=" * 50)
        print("所有测试完成！")
        print("=" * 50)  
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
if __name__ == "__main__":
    main()
