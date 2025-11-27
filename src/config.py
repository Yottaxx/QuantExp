import torch
import os


class Config:
    """
    【全局配置中心 - Production Grade (Split Fixed)】
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "stock_lake")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output", "checkpoints")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 数据参数 ---
    START_DATE = "20200101"

    # [CRITICAL UPDATE] 数据集划分比例 (Train / Valid / Test)
    # Valid 用于 Training 过程中的 Early Stopping
    # Test 用于 Analysis 和 Backtest (完全隔离)
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1  # 剩下的是 Test

    # --- 网络配置 ---
    PROXY_URL = "http://127.0.0.1:7890"
    CLASH_API_URL = "http://127.0.0.1:49812"
    CLASH_SECRET = "b342ba26-2ae3-47bb-a057-6624e171d5c6"

    BENCHMARK_SYMBOL = "sh000300"
    MARKET_INDEX_SYMBOL = "sh000001"

    FEATURE_PREFIXES = [
        'style_', 'tech_', 'alpha_', 'adv_', 'ind_', 'fund_',
        'cs_rank_', 'mkt_', 'rel_', 'time_'
    ]

    SEED = 42

    # --- 模型参数 ---
    CONTEXT_LEN = 64
    PRED_LEN = 5
    PATCH_LEN = 8
    STRIDE = 4
    DROPOUT = 0.2
    d_model = 128
    D_MODEL = 128

    # --- 训练参数 ---
    BATCH_SIZE = 128
    EPOCHS = 20
    LR = 1e-4
    MSE_WEIGHT = 0.5
    MAX_GRAD_NORM = 1.0

    # --- 推理与风控 ---
    INFERENCE_BATCH_SIZE = 256
    ANALYSIS_BATCH_SIZE = 2048
    MIN_SCORE_THRESHOLD = 0.6
    TOP_K = 5
    CASH_BUFFER = 0.95

    MIN_VOLUME_PERCENT = 0.02
    RISK_FREE_RATE = 0.02
    SLIPPAGE = 0.002
    STOP_LOSS_PCT = 0.08

    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"