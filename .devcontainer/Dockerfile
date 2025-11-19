# 使用NVIDIA官方的CUDA基础镜像
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel

# 设置环境变量，避免安装过程中的交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 更新apt并安装必要的工具和Python
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# [可选] 创建一个非root用户
# RUN useradd -ms /bin/bash vscode
# USER vscode