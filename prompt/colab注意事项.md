# 如何在colab上面安装旧版本的nvcc

有时候显卡的cuda版本（`!nvidia-smi`）小于nvcc的版本(`!nvcc --version`)导致编译出来的算法运行报错，或者运行结果不符合预期。可以降低nvcc的版本：

1. 查看当前系统版本，可以使用screenfetch查看（`!apt-get install screenfetch`）
2. 去nvidia官网下载对应版本的nvcc，官网上通常有下面的命令，直接运行即可.下面是ubuntu22.04， cuda版本12.4的nvcc下载命令
```jupyter
!wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
!sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
!wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
!sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
!sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
!sudo apt-get update
!sudo apt-get -y install cuda-toolkit-12-4
```
3. 配置环境变量,安装完新版本后，必须告诉系统使用新安装的 CUDA 路径，否则它可能还在用旧的。
```python
import os

# 设置环境变量，确保系统优先找到新安装的 CUDA 11.8
# 注意：如果你安装的是其他版本，请将 11.8 替换为你的版本号
os.environ['PATH'] = '/usr/local/cuda-12.4/bin:' + os.environ['PATH']
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.4/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

# 验证是否生效
!nvcc --version
```