# PyTorch接口测试脚本

这是一个简单的PyTorch接口测试脚本，使用pytest进行测试，包括CPU和CUDA的运行时间测试以及余弦相似度验证。

## 功能特性

- 自动生成随机数据进行测试
- 比较CPU和CUDA的运行时间
- 验证CPU和CUDA结果的余弦相似度
- 支持多种PyTorch操作（矩阵乘法、卷积、激活函数等）
- 易于扩展新的测试用例

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行测试

现在项目采用更清晰的 pytest 结构：

```
src/
  torch_utils.py
tests/
  conftest.py
  test_torch_benchmark.py
docs/
  README.md          # 详细运行逻辑说明
pytest.ini
```

> 📖 **详细文档**: 查看 [docs/README.md](docs/README.md) 了解完整的运行逻辑和架构说明

使用下面任一方式运行：

```bash
pytest -v
```

或使用脚本：

```bash
bash run_tests.sh
```

## 测试内容

当前包含的测试：
1. **矩阵乘法** - 测试不同大小的矩阵乘法
2. **2D卷积** - 测试卷积操作的性能
3. **Softmax** - 测试softmax激活函数
4. **ReLU** - 测试ReLU激活函数

## 添加新的测试用例

在 `tests/` 中添加新的测试：

```python
import torch
from src.torch_utils import measure_time

def test_your_operation():
    input_tensor = torch.randn(100, 100)
    duration, result = measure_time(your_operation, input_tensor)
    assert result.shape == input_tensor.shape
    assert duration >= 0.0
```

## 标记与选择性运行

- 使用 `@pytest.mark.cpu` 和 `@pytest.mark.cuda` 标记测试
- 只运行 CPU 测试：`pytest -m cpu`
- 只运行 CUDA 测试：`pytest -m cuda`

## 输出示例

```
Matrix multiplication (100, 100):
  CPU time: 0.0001s
  CUDA time: 0.0002s
  Speedup: 0.50x
  Cosine similarity: 1.000000

2D Convolution torch.Size([4, 3, 64, 64]) -> torch.Size([4, 3, 64, 64]):
  CPU time: 0.0012s
  CUDA time: 0.0003s
  Speedup: 4.00x
  Cosine similarity: 1.000000
```

## 注意事项

- 确保系统已安装PyTorch
- 如果有CUDA设备，脚本会自动检测并使用
- 余弦相似度阈值设置为0.999，确保结果一致性
- 测试使用随机数据，每次运行结果可能略有不同
