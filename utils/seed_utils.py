import torch
import numpy as np
import random
import os
from accelerate.utils import set_seed as hf_set_seed


def set_global_seed(seed: int = 42):
    """
    设置全局随机种子，确保实验的可复现性。

    Args:
        seed (int): 要设置的随机种子值。
    """
    print(f"--- 🚀 Setting Global Seed to {seed} ---")

    # 1. 使用 accelerate 的内置函数 (推荐)
    # 它会设置 python, numpy, torch, cuda 的随机种子
    hf_set_seed(seed)

    # 2. 额外设置一些环境变量和 GPU 配置，以确保最大限度的复现性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 确保 Python 哈希操作一致

    # 针对 PyTorch CUDA/CUDNN
    if torch.cuda.is_available():
        # 设置 CUDA 操作的确定性
        torch.backends.cudnn.deterministic = True
        # 关闭 CUDNN 自动选择最优算法
        torch.backends.cudnn.benchmark = False

    print("---------------------------------------")