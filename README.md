# CUDA学习之旅

计算能力是GPU的硬件规格，决定硬件支持的指令和功能
CUDA版本是软件开发平台提供的版本，他为不同计算能力的硬件提供了软件层面的新特性

## 概述

本项目旨在探索一种经济高效的CUDA学习路径并进行对cuda的学习，尤其适合没有NVIDIA显卡的开发者。

传统的CUDA学习面临两大挑战：
*   **本地开发环境配置复杂**：在没有NVIDIA GPU的机器上，配置一个带智能提示的CUDA C++开发环境非常困难。
*   **学习成本高昂**：租用云服务器进行初步的代码学习和调试，成本较高。

为了解决这些问题，我设计了如下三步走的学习和开发工作流：

1.  **本地开发与编码**：利用VS Code和Devcontainer构建一个隔离的、功能齐全的开发环境，实现CUDA C++和Python代码的智能提示和静态检查。
2.  **云端编译与测试**：借助Kaggle和Google Colab等免费的GPU计算平台，对学习阶段的CUDA代码进行编译、运行和验证。同时，可以利用LeetGPU等在线平台进行算法验证和性能测试。
3.  **远程部署与训练**：将最终开发完成并通过测试的代码通过Git部署到租用的高性能GPU服务器上，进行大规模的训练或计算任务。

这种方法的核心优势在于，将代码编写、代码测试和最终部署三个阶段解耦，并为每个阶段选择最合适的工具，从而最大限度地降低了学习门槛和经济成本。

## 工作流详解

### 步骤一：本地开发 (Local Development)

为了解决在macOS或其他没有NVIDIA GPU的系统上编写CUDA代码的难题，我们采用VS Code的Devcontainer功能。

**核心优势**:
*   **隔离的环境**：所有开发依赖（如CUDA Toolkit、编译器、Python库等）都封装在Docker容器中，不污染本地系统。
*   **代码智能提示**：通过在`devcontainer.json`中配置C/C++和CUDA相关的插件，可以在编写CUDA C++代码时获得语法高亮、代码补全和错误检查等功能。
*   **跨平台一致性**：无论你使用Windows、macOS还是Linux，Devcontainer都能保证开发环境的一致性。

**如何配置？**
1.  **安装Docker Desktop**: 确保系统已安装并运行Docker。
2.  **安装VS Code及Dev Containers插件**: 这是使用Devcontainer的前提。
3.  **配置创建开发容器目录**: 在vscode里面使用 `Dev Containers: Reopen Folder Locally` 命令打开开发容器，会自动读取 `.devcontainer` 下面的 `devcontainer.json` 和 `Dockerfile` 构造开发容器，并使用vscode连接开发。

### 步骤二：云端编译与测试 (Cloud-based Compilation & Testing)

当完成了代码编写，需要一个真实的环境来编译和运行它。这时，免费的云平台是最佳选择。

*   **Kaggle Notebooks**:
    *   **免费GPU**: Kaggle提供免费的GPU使用时长，非常适合学习和实验。
    *   **便捷的环境**: 通过安装`nvcc4jupyter`插件，可以直接在Jupyter环境中编写、编译和运行CUDA C++代码。
    *   **操作步骤**:
        1.  在Kaggle创建一个新的Notebook。
        2.  在设置中，将"Accelerator"选项切换为GPU。
        3.  使用`!pip install nvcc4jupyter`安装插件，并通过`%load_ext nvcc4jupyter`加载它。
        4.  之后就可以使用`%%cuda`魔术命令来编写和执行CUDA代码了。
        > 如果不想安装依赖，可以使用 `%%write_file cuda_file.cu` 把cuda文件写入到文件里面，接着使用来编译和运行cuda文件：
        >
        > `!nvcc cuda_file.cu -o cuda_file`
        >
        > `!./cuda_file` 
    > google colab也是一样的

*   **LeetGPU**:
    *   **在线CUDA练习场**: LeetGPU是一个在线的CUDA代码编写和测试平台，甚至可以在没有GPU的情况下通过模拟器运行代码。但是免费用户每天只有5次提交机会且看不到运行的时间和别人的解法。
    *   **算法验证**: 它提供了一系列CUDA编程挑战，非常适合用来检验和提升你的并行编程技能。

### 步骤三：租用服务器进行训练 (Deployment & Training on Rented Servers)

对于已经开发和测试完毕的大型项目或需要长时间运行的训练任务，可以租用专业的GPU服务器。

*   **云服务提供商**: AWS、Google Cloud和Microsoft Azure等都提供多种配置的GPU实例。
*   **部署流程**:
    1.  **代码同步**: 使用Git将本地开发完成的代码推送到代码托管平台（如GitHub）。
    2.  **远程拉取**: 在租用的云服务器上，通过SSH连接，并使用Git拉取最新的代码。
    3.  **编译运行**: 在服务器上使用`nvcc`编译代码，并执行生成的可执行文件。
    4.  **远程开发与调试**: 可以配置VS Code的Remote-SSH插件，直接在本地VS Code中编辑和调试运行在远程服务器上的代码，获得几乎和本地开发一致的体验。

## 总结

通过上述工作流，我们可以有效地将CUDA的学习和开发过程分解，并为每个环节匹配最优的工具，从而实现一个低成本、高效率的学习路径。希望这个仓库和这份文档能对同样在学习CUDA的你有所帮助！