#!/bin/bash

# 设置脚本为可执行
chmod +x "$(dirname "$0")/run.sh"
chmod +x "$(dirname "$0")/test_services.py"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请安装Python3"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    echo "虚拟环境创建完成"
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 运行应用或测试
if [ "$1" == "test" ]; then
    echo "运行服务测试..."
    python3 test_services.py
else
    echo "启动XAI-MBTI应用..."
    python app.py
fi