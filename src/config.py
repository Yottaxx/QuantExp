import torch
import os


class Config:
    """
    【全局配置中心 - Production Grade】
    管理所有超参数、路径配置、网络设置和常量。
    """
    # --- 基础路径 (使用绝对路径确保稳定) ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "stock_lake")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output", "checkpoints")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 数据参数 ---
    START_DATE = "20240101"  # 扩大训练窗口覆盖牛熊周期

    # --- 网络与代理配置 ---
    PROXY_URL = "http://127.0.0.1:7890"
    CLASH_API_URL = "http://127.0.0.1:49812"
    CLASH_SECRET = ""

    # --- 市场符号配置 ---
    BENCHMARK_SYMBOL = "sh000300"  # 沪深300
    MARKET_INDEX_SYMBOL = "sh000001"  # 上证指数

    # --- 因子系统配置 ---
    FEATURE_PREFIXES = [
        'style_', 'tech_', 'alpha_', 'adv_', 'ind_', 'fund_',
        'cs_rank_', 'mkt_', 'rel_', 'time_'
    ]

    # --- 随机数种子 ---
    SEED = 42

    # --- 模型参数 (PatchTST) ---
    CONTEXT_LEN = 64  # 增加上下文长度以捕捉中期趋势
    PRED_LEN = 5  # 预测未来5日收益
    PATCH_LEN = 8
    STRIDE = 4
    DROPOUT = 0.2

    # --- 训练参数 ---
    BATCH_SIZE = 128
    EPOCHS = 20  # 增加轮数，依赖 EarlyStopping 控制
    LR = 1e-4
    MSE_WEIGHT = 0.5  # IC Loss 与 MSE 的平衡权重
    MAX_GRAD_NORM = 1.0  # 梯度裁剪阈值
    D_MODEL = 32
    # --- 推理与分析参数 ---
    INFERENCE_BATCH_SIZE = 256
    ANALYSIS_BATCH_SIZE = 2048
    MIN_SCORE_THRESHOLD = 0.6
    TOP_K = 10  # 增加持仓分散度

    # --- 交易风控参数 ---
    MIN_VOLUME_PERCENT = 0.02  # 2% 流动性上限
    RISK_FREE_RATE = 0.02
    SLIPPAGE = 0.002  # 双边万分之二滑点，模拟冲击成本

    # [Added] 资金缓冲系数
    # 防止 T+1 开盘低开导致回款不足以覆盖 T 日预估买单金额
    # 0.98 意味着只使用 98% 的可用资金进行开仓
    CASH_BUFFER = 0.98

    # --- 硬件 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"