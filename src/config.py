# -*- coding: utf-8 -*-
import os
import torch


class Config:
    """
    【全局配置中心 - Production Grade】
    - 已融合 data_provider.py v22 所需配置项
    - 对外接口不改；新增项均提供默认值
    """

    # =============================================================================
    # Paths / Logging
    # =============================================================================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "stock_lake")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output", "checkpoints")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    LOG_LEVEL = "INFO"
    PARQUET_COMPRESSION = "zstd"

    # =============================================================================
    # Data Range / Split
    # =============================================================================
    START_DATE = "20250101"

    # v22 Dataset split uses TRAIN_RATIO / VAL_RATIO (test is remainder)
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1  # 你系统其他模块可用；v22 会用 remainder 做 test

    # =============================================================================
    # Network / Proxy / VPN
    # =============================================================================
    PROXY_URL = "http://127.0.0.1:7890"
    CLASH_API_URL = "http://127.0.0.1:49812"
    CLASH_SECRET = "b342ba26-2ae3-47bb-a057-6624e171d5c6"

    USE_VPN_ROTATOR = True
    VPN_ROTATE_COOLDOWN_SEC = 60
    VPN_POST_ROTATE_SLEEP_SEC = 3.0

    # AkShare retry
    AK_RETRIES = 10
    AK_RETRY_BASE_SLEEP = 1.0
    AK_RETRY_MAX_SLEEP = 15.0

    # =============================================================================
    # Market / Calendar
    # =============================================================================
    BENCHMARK_SYMBOL = "sh000300"
    MARKET_INDEX_SYMBOL = "sh000001"
    CALENDAR_TTL_SEC = 7 * 24 * 3600

    # =============================================================================
    # Feature / Factor
    # =============================================================================
    FEATURE_PREFIXES = [
        "raw_", "style_", "tech_", "fund_",  # 基础类
        "ind_", "adv_",  # 进阶类
        "ccf_", "int_",  # SOTA与交互类
        "cs_", "csn_",   # 截面类
        "time_",         # 时间嵌入
        "meta_"
    ]

    # backend for per-code factor building
    DEBUG = False
    ALPHA_BACKEND = "process"  # "process" | "serial"
    DEBUG_MAX_FILES = 10
    FAIL_FAST = True

    # process pool details
    ALPHA_WORKERS = max(1, (os.cpu_count() or 8) - 1)
    ALPHA_MAP_CHUNKSIZE = 8

    # panel write/merge
    PANEL_FLUSH_N = 200
    PANEL_MERGE_BATCH = 24
    USE_CROSS_SECTIONAL = True

    # =============================================================================
    # Reproducibility / Seeds
    # =============================================================================
    SEED = 42

    # =============================================================================
    # Storage / Download switches
    # =============================================================================
    STORE_RAW_PRICE = True
    STORE_QFQ_PRICE = True

    ALIGN_TO_CALENDAR = False  # 开启更严谨，但更慢；建议最终回测开启 True
    ENTRY_PRICE_MODE = "open"  # open / close / vwap

    # v22 price download workers (映射你原来的 DL_WORKERS)
    DL_WORKERS = 16
    PRICE_WORKERS = DL_WORKERS

    FIN_WORKERS = 8

    # price schema / incremental policy
    PRICE_SCHEMA_VER = 2
    PRICE_ADJUSTS = ["qfq"]
    PRICE_BACKFILL_YEARS = 10
    PRICE_OVERLAP_DAYS = 3

    SKIP_IF_FRESH = True
    FRESH_LAG_DAYS = 0

    STRICT_PRICE_META = True
    ALLOW_LEGACY_PRICE_CACHE = False

    # =============================================================================
    # Fundamentals (optional)
    # =============================================================================
    # v22: 只有 SYNC_FUNDAMENTAL=True 才会在 download_data() 中同步财务指标文件
    SYNC_FUNDAMENTAL = False  # 如你希望 download_data() 同步基本面，改 True
    USE_FUNDAMENTAL = False   # 如你希望面板合并 PIT 基本面，改 True
    FUND_LAG_DAYS = 90        # 你系统可用（v22 未强依赖）

    # =============================================================================
    # Universe (optional filters)
    # =============================================================================
    UNIVERSE_TTL_SEC = 24 * 3600
    FORCE_UNIVERSE_REFRESH = False

    INCLUDE_BSE = False
    INCLUDE_BSHARE = False
    UNIVERSE_ALLOW_PREFIXES = None
    UNIVERSE_EXTRA_CODES_FILE = ""

    # =============================================================================
    # Tradable mask / constraints
    # =============================================================================
    MIN_DOLLAR_VOL_FOR_TRADE = 1e6
    MIN_LIST_DAYS = 60
    MIN_PRICE = 1.0

    UNIVERSE_MIN_PRICE = 2.0
    UNIVERSE_MIN_LIST_DAYS = 60

    # board-wise涨跌停阈值
    LIMIT_RATE_MAIN = 0.10
    LIMIT_RATE_ST = 0.05
    LIMIT_RATE_GEM = 0.20
    LIMIT_RATE_STAR = 0.20
    LIMIT_RATE_BSE = 0.30
    LIMIT_RATE_BSHARE = 0.10
    LIMIT_EPS = 0.002

    # label gating (P0: entry-day tradable)
    GATE_TARGET_WITH_ENTRY = True
    GATE_TARGET_WITH_EXIT = False

    # =============================================================================
    # Model params
    # =============================================================================
    CONTEXT_LEN = 64
    PRED_LEN = 5
    PATCH_LEN = 8
    STRIDE = 4
    DROPOUT = 0.2
    d_model = 128
    D_MODEL = 128

    # =============================================================================
    # Train params
    # =============================================================================
    BATCH_SIZE = 128
    EPOCHS = 20
    LR = 1e-4
    MSE_WEIGHT = 0.5
    MAX_GRAD_NORM = 1.0

    # =============================================================================
    # Inference / Analysis / Risk control
    # =============================================================================
    INFERENCE_BATCH_SIZE = 256
    ANALYSIS_BATCH_SIZE = 2048
    MIN_SCORE_THRESHOLD = 0.6
    TOP_K = 5
    CASH_BUFFER = 0.95

    # 严格回测开关与细节
    BACKTEST_LONG_SHORT = True  # 是否做 long-short spread
    BACKTEST_SHORT_K = 5  # short 选股数，默认=TOP_K
    BACKTEST_MAX_SELL_DELAY = 5  # 跌停/不可卖时最多延迟几天
    ANALYSIS_MIN_CROSS_SECTION = 50  # 每日至少多少股票才计算 IC

    # 交易成本（可选）
    COMMISSION_RATE = 0.0003  # 双边手续费(示例)
    STAMP_DUTY = 0.001  # 卖出印花税(A股示例)

    MIN_VOLUME_PERCENT = 0.02
    RISK_FREE_RATE = 0.02
    SLIPPAGE = 0.002
    STOP_LOSS_PCT = 0.08

    # =============================================================================
    # Dataset memmap (v22)
    # =============================================================================
    DATASET_BACKEND = "hf_memmap"   # 推荐：hf_memmap / memmap
    FORCE_DATASET_REBUILD = False
    MEMMAP_WRITE_CHUNK = 250_000

    # =============================================================================
    # Cache fingerprint keys (v22 uses these to version panel cache filename)
    # 这里不仅列出 keys，也已在本 Config 里给出对应属性默认值（避免缺字段）
    # =============================================================================
    PANEL_CACHE_FINGERPRINT_KEYS = [
        "ALIGN_TO_CALENDAR",
        "ENTRY_PRICE_MODE",
        "DELIST_GAP_DAYS",
        "CONTEXT_LEN",
        "PRED_LEN",
        "STRIDE",
        "PRICE_SCHEMA_VER",
        "FEATURE_PREFIXES",
        "ALPHA_BACKEND",
        "USE_FUNDAMENTAL",
        "USE_CROSS_SECTIONAL",
        "UNIVERSE_MIN_PRICE",
        "UNIVERSE_MIN_LIST_DAYS",
        "MIN_LIST_DAYS",
        "MIN_DOLLAR_VOL_FOR_TRADE",
        "MIN_PRICE",
        "LIMIT_RATE_MAIN",
        "LIMIT_RATE_ST",
        "LIMIT_RATE_GEM",
        "LIMIT_RATE_STAR",
        "LIMIT_RATE_BSE",
        "LIMIT_RATE_BSHARE",
        "LIMIT_EPS",
        "DATASET_BACKEND",
        "INCLUDE_BSE",
        "INCLUDE_BSHARE",
        "GATE_TARGET_WITH_ENTRY",
        "GATE_TARGET_WITH_EXIT",
    ]

    # =============================================================================
    # Device
    # =============================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
