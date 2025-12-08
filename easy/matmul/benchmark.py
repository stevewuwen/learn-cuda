import os
import subprocess
import torch
import ctypes
import numpy as np
import time
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path_dir = os.path.dirname(__file__)
os.chdir(path_dir)
PARTTERN_COMPOLER = re.compile(r"(sm_\d+)")


for v in os.listdir(os.path.abspath(path_dir)):
    print(v)
    prefix, end = os.path.splitext(v)
    arch = re.search(PARTTERN_COMPOLER, prefix)
    arch = arch.group(0) if arch else "sm_75"
    print(arch)
    if end == ".cu":
        subprocess.run(
            f"nvcc -arch={arch} -shared -o {prefix}.so {prefix}.cu -Xcompiler -fPIC",
            shell=True,
        )
print(os.listdir())


# --- 配置参数 ---
SIZES = [128, 512, 1024, 2048, 4096]  # 测试的矩阵大小
N_WARMUP = 5  # 预热次数
N_ITERS = 40  # 计时的平均运行次数
ATOL = 1  # 结果验证的容差
torch.manual_seed(42)


# --- 辅助函数：自然排序 (v1, v2, v10...) ---
def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("v([0-9_]+).*", s)
    ]


# --- 核心：CUDA 库包装器 ---
class CUDALibWrapper:
    def __init__(self, lib_path):
        self.lib_path = lib_path
        self.name = os.path.splitext(os.path.basename(lib_path))[0]
        self.lib = ctypes.CDLL(lib_path)
        # 配置 solve 函数参数: A_ptr, B_ptr, C_ptr, M, N, K
        self.lib.solve.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

    def __call__(self, a_tensor, b_tensor, c_tensor):
        M, N = a_tensor.shape
        _, K = b_tensor.shape
        self.lib.solve(
            ctypes.c_void_p(a_tensor.data_ptr()),
            ctypes.c_void_p(b_tensor.data_ptr()),
            ctypes.c_void_p(c_tensor.data_ptr()),
            ctypes.c_int(M),
            ctypes.c_int(N),
            ctypes.c_int(K),
        )


# --- 核心：基准测试函数 ---
def benchmark_kernel(func, args, n_warmup, n_iters):
    # 预热
    for _ in range(n_warmup):
        func(*args)
    torch.cuda.synchronize()

    # 计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_iters):
        func(*args)
    end_event.record()
    torch.cuda.synchronize()

    # 返回平均耗时 (毫秒)
    return start_event.elapsed_time(end_event) / n_iters


# --- 主程序 ---
def main():
    # 1. 扫描并加载所有 .so 文件
    so_files = [f for f in os.listdir(".") if f.endswith(".so")]
    so_files.sort(key=natural_sort_key)

    kernels = []
    for f in so_files:
        try:
            kernels.append(CUDALibWrapper(f"./{f}"))
            print(f"已加载内核: {f}")
        except Exception as e:
            print(f"加载 {f} 失败: {e}")

    # 存储结果: { 'MethodName': [perf_size1, perf_size2, ...] }
    results = {k.name: [] for k in kernels}
    results["PyTorch"] = []

    # 2. 循环测试不同大小
    for size in SIZES:
        M, N, K = size, size, size
        print(f"\n====== 正在测试尺寸: [{M}x{N}] * [{N}x{K}] ======")

        # 准备数据
        device = torch.device("cuda")
        A = torch.randn(M, N, device=device, dtype=torch.float32)
        B = torch.randn(N, K, device=device, dtype=torch.float32)
        C_ref = torch.matmul(A, B)  # PyTorch 结果作为基准

        # 计算 FLOPs (2 * M * N * K)
        flops = 2 * M * N * K

        # --- 测试 PyTorch ---
        def run_torch():
            torch.matmul(A, B)

        avg_ms = benchmark_kernel(run_torch, (), N_WARMUP, N_ITERS)
        tflops = (flops / (avg_ms / 1000)) / 1e12
        results["PyTorch"].append(tflops)
        print(f"PyTorch \t: {avg_ms:.3f} ms | {tflops:.3f} TFLOPS")

        # --- 测试自定义内核 ---
        for kernel in kernels:
            C_custom = torch.zeros((M, K), device=device, dtype=torch.float32)

            # 正确性检查
            kernel(A, B, C_custom)
            torch.cuda.synchronize()

            if not torch.allclose(C_custom, C_ref, atol=ATOL):
                print(
                    f"{kernel.name} \t: ❌ 结果错误 (Max Diff: {(C_custom - C_ref).abs().max().item():.4f})"
                )
                results[kernel.name].append(0.0)  # 错误记为 0 分
                continue

            # 性能测试
            def run_custom():
                kernel(A, B, C_custom)

            avg_ms = benchmark_kernel(run_custom, (), N_WARMUP, N_ITERS)
            tflops = (flops / (avg_ms / 1000)) / 1e12
            results[kernel.name].append(tflops)
            print(f"{kernel.name} \t: {avg_ms:.3f} ms | {tflops:.3f} TFLOPS ✅")

    # 3. 绘制图表
    plot_results_seaborn(SIZES, results)


def plot_results_seaborn(sizes, results):
    data = []
    for name, perfs in results.items():
        for s, p in zip(sizes, perfs):
            data.append({"Matrix Size": s, "TFLOPS": p, "Method": name})
    df = pd.DataFrame(data)
    sns.set_theme(style="ticks", context="talk", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(14, 9))
    methods = df["Method"].unique()
    n_methods = len(methods)
    palette = sns.color_palette("husl", n_methods)
    color_dict = dict(zip(methods, palette))
    dash_dict = {m: (1, 0) for m in methods}  # 默认实线
    size_dict = {m: 2.0 for m in methods}  # 默认宽度

    if "PyTorch" in color_dict:
        color_dict["PyTorch"] = "#333333"
        dash_dict["PyTorch"] = (4, 2)
        size_dict["PyTorch"] = 4.0

    sns.lineplot(
        data=df,
        x="Matrix Size",
        y="TFLOPS",
        hue="Method",
        style="Method",
        size="Method",
        palette=color_dict,
        dashes=dash_dict,
        sizes=size_dict,
        markers=True,
        markersize=8,
        ax=ax,
        legend=False,
    )

    last_points = []
    for name in methods:
        subset = df[df["Method"] == name]
        last_row = subset.iloc[-1]
        last_points.append(
            {
                "y": last_row["TFLOPS"],
                "x": last_row["Matrix Size"],
                "label": name,
                "color": color_dict[name],
            }
        )
    last_points.sort(key=lambda x: x["y"])

    all_y = df["TFLOPS"].values
    y_span = all_y.max() - all_y.min()
    min_dist = y_span * 0.04
    last_text_y = -float("inf")
    x_max = sizes[-1]
    ax.set_xlim(left=0, right=x_max * 1.35)

    for point in last_points:
        current_y = point["y"]
        text_y = max(current_y, last_text_y + min_dist)
        last_text_y = text_y
        ax.annotate(
            text=point["label"],
            xy=(point["x"], point["y"]),
            xytext=(x_max * 1.02, text_y),
            color=point["color"],
            fontweight="bold",
            fontsize=12,
            va="center",
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.4, lw=1, shrinkB=5),
        )

    ax.set_title("CUDA GEMM Performance Benchmark", pad=20, fontweight="bold")
    ax.set_xlabel("Matrix Size (M=N=K)")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.grid(True, which="major", ls="--", c="gray", alpha=0.2)

    sns.despine(trim=True, offset=10)

    plt.tight_layout()
    plt.savefig("gemm_benchmark.png", dpi=300, bbox_inches="tight")
    plt.show()


main()
