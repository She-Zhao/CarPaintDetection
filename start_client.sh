#!/bin/bash

# ==========================================
# 1. 硬件配置阶段 (需要 sudo 密码)
# ==========================================
echo "[System] 正在配置网络巨帧 (MTU 9000)..."
# 注意：这里会要求输入密码，除非你配置了 sudoers 免密
sudo ifconfig eth0 down
sudo ifconfig eth0 mtu 9000
sudo ifconfig eth0 up
echo "✅ 网络配置完成"

# ==========================================
# 2. 环境准备阶段
# ==========================================
# 获取当前脚本所在的目录 (项目根目录)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "[System] 项目根目录: $PROJECT_ROOT"

# 切换到项目根目录 (解决找不到文件的问题)
cd "$PROJECT_ROOT"

# 激活 Conda 环境 (根据你的路径 /home/nvidia/anaconda3/envs/torch2/)
# 注意：source 的路径根据你的实际安装位置可能微调，通常是下面这个
echo "[System] 正在激活 Conda 环境 (torch2)..."
source /home/nvidia/anaconda3/etc/profile.d/conda.sh
conda activate torch2

# 设置 PYTHONPATH，确保 python 能找到 framework 包
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# ==========================================
# 3. 启动业务程序
# ==========================================
echo "[System] 启动 Client..."
# 这里使用 $@ 允许你给脚本传参，比如 ./start_client.sh --delay 2000
python framework/core/client.py "$@"

# 脚本结束时保持窗口（可选，方便看报错）
# read -p "按回车键退出..."