#!/bin/bash

echo "PyTorch接口测试脚本"
echo "=================="

export PYTHONPATH=$PYTHONPATH:$(pwd)


# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查是否安装了依赖
echo "检查依赖..."
python3 -c "import torch, pytest, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "错误: 依赖安装失败"
        exit 1
    fi
fi



echo "依赖检查完成"
echo ""

echo "开始运行测试..."
echo ""

echo "使用pytest运行测试..."
pytest -v | cat

echo ""
echo "测试完成！"
