import torch
import os


class Config:
    """
    【全局配置中心】
    管理所有超参数、路径配置、网络设置和常量。
    """
    # --- 基础路径 ---
    DATA_DIR = "./data/stock_lake"
    OUTPUT_DIR = "./output/checkpoints"
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 数据参数 ---
    START_DATE = "20050101"

    # --- 网络与代理配置 (新增) ---
    # 本地代理地址 (供 requests/akshare 使用)
    PROXY_URL = "http://127.0.0.1:7890"
    # Clash 控制器地址 (供 vpn_rotator 使用)
    CLASH_API_URL = "http://127.0.0.1:49812"
    # Clash 密钥
    CLASH_SECRET = "b342ba26-2ae3-47bb-a057-6624e171d5c6"

    # --- 市场符号配置 (新增) ---
    BENCHMARK_SYMBOL = "sh000300"  # 沪深300 (回测基准)
    MARKET_INDEX_SYMBOL = "sh000001"  # 上证指数 (交易日校准)

    # --- 因子系统配置 (新增) ---
    # 所有需要保留的特征列前缀，统一管理防止漏改
    FEATURE_PREFIXES = [
        'style_', 'tech_', 'alpha_', 'adv_', 'ind_', 'fund_',
        'cs_rank_', 'mkt_', 'rel_', 'time_'
    ]

    # --- 模型参数 ---
    CONTEXT_LEN = 30
    PRED_LEN = 5
    PATCH_LEN = 8
    STRIDE = 4
    DROPOUT = 0.2

    # --- 训练参数 ---
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 1e-4
    MSE_WEIGHT = 0.5

    # --- 推理与分析参数 (新增) ---
    INFERENCE_BATCH_SIZE = 128  # 实盘推理 Batch
    ANALYSIS_BATCH_SIZE = 2048  # 历史回溯 Batch (通常可以更大)
    MIN_SCORE_THRESHOLD = 0.6  # 选股最低分阈值

    # --- 交易风控参数 ---
    MIN_VOLUME_PERCENT = 0.02  # 流动性风控
    RISK_FREE_RATE = 0.02  # 无风险利率 (用于夏普比率计算)

    # --- 硬件 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"