import torch
import os


class Config:
    # 路径配置
    DATA_DIR = "./data/stock_lake"
    OUTPUT_DIR = "./output/checkpoints"

    # 股票数据配置
    START_DATE = "20210101"

    # 模型超参数
    CONTEXT_LEN = 30  # 回看窗口
    PRED_LEN = 5  # 预测未来5天收益
    PATCH_LEN = 8
    STRIDE = 4
    D_MODEL = 128
    DROPOUT = 0.2

    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 确保目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)