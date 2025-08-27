
    
import torch
import torch.nn as nn
import numpy as np
import time

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

class AddModule(nn.Module):
    def __init__(self):
        super(AddModule, self).__init__()

    def forward(self, a, b):
        # 简单的加法操作
        return a + b

def test(device='auto'):
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"使用设备: {device}")

    # 创建模型
    model = AddModule().to(device)
    a = torch.tensor([1.2, 2.3], device=device)
    b = torch.tensor([2.1, 2.9], device=device)

    for i in range(10000):
        c = model(a, b)
        print(f"第{i+1}次: a={a.cpu().numpy()}, b={b.cpu().numpy()}, c={c.cpu().detach().numpy()}")

 # 生成数据并获取结果
if __name__ == "__main__":
    test()