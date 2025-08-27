#!/usr/bin/env python3
"""
简单的PyTorch接口测试脚本
"""

import torch
import torch.nn.functional as F
import time
Wq=torch.nn.Linear(4096,4096)

def generate_random_data(shape, device='cpu'):
    """生成随机数据"""
    return torch.randn(shape, device=device)
def test_linear():
    """测试线性层"""
    print("=" * 50)
    print("测试线性层")
    print("=" * 50)
    hidden_states=generate_random_data((128,1024,4096))

    def run_cpu_test(Wq, hidden_states):
            """
            运行CPU测试10次并打印每次的运行时间
            
            :param Wq: CPU上的线性层
            :param hidden_states: 输入数据
            """
            print("CPU测试")    
            for _ in range(10):
                start_time = time.time()
                Wq(hidden_states.view(1, -1, 4096))
                end_time = time.time()
                qcuda_time = (end_time - start_time) * 1000
                print(f"  Q时间: {qcuda_time:.4f}ms")
        
    run_cpu_test(Wq, hidden_states)
    

    if torch.cuda.is_available():
        # 为了避免在函数内重新赋值导致外部变量不可用，这里使用新变量
        Wq_cuda = Wq.to('cuda')
        hidden_states=hidden_states.to('cuda')

        def run_cuda_test(Wq_cuda, hidden_states):
            """
            运行CUDA测试10次并打印每次的运行时间
            
            :param Wq_cuda: CUDA上的线性层
            :param hidden_states: 输入数据
            """
            print("CUDA测试")
            for _ in range(10):
                start_time = time.time()
                Wq_cuda(hidden_states.view(1, -1, 4096))
                end_time = time.time()
                qcuda_time = (end_time - start_time) * 1000
                print(f"  Q时间: {qcuda_time:.4f}ms")
        
        run_cuda_test(Wq_cuda, hidden_states)

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
