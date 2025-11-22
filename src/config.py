import torch
import os


class Config:
    # 路径配置
    DATA_DIR = "./data/stock_lake"
    OUTPUT_DIR = "./output/checkpoints"

    # 确保目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 股票数据配置
    # 建议设为 20050101 以获取完整历史
    START_DATE = "20050101"

    # 模型超参数
    CONTEXT_LEN = 30  # 回看窗口 (Past Sequence Length)
    PRED_LEN = 5  # 预测窗口 (Future Sequence Length)

    # PatchTST 参数
    PATCH_LEN = 8
    STRIDE = 4
    DROPOUT = 0.2  # 【新增】Dropout 比率

    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 1e-4

    # 损失函数参数
    # 控制 Hybrid Loss 中 MSE 的权重
    # Loss = (1 - IC) + MSE_WEIGHT * MSE
    MSE_WEIGHT = 0.5

    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"