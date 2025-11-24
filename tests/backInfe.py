import sys
import os
import shutil
import pandas as pd
import numpy as np
from unittest.mock import patch

# ==========================================
# 1. 核心修复：路径设置
# ==========================================
# 获取当前脚本所在目录 (tests/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (QuantExp/) - 也就是 tests 的上一级
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 【关键】将 项目根目录 加入 sys.path，而不是 src
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"🚀 启动脚本测试...")
print(f"📂 项目根目录: {PROJECT_ROOT}")

# ==========================================
# 2. 导入模块 (注意加上 src. 前缀)
# ==========================================
try:
    # 必须以 src. 开头，这样 backtest.py 里的 from .config 才能正确识别 src 包
    from src.config import Config
    from src.backtest import run_walk_forward_backtest
    # 如果需要 akshare 做 mock，这里可以导入，不需要则跳过
    import akshare as ak
except ImportError as e:
    print(f"\n❌ 导入失败: {e}")
    print(
        "💡 请检查：\n1. src/ 目录下是否有 __init__.py 文件？(如果没有，请新建一个空文件)\n2. 你的代码是否在 src/ 目录下？")
    sys.exit(1)

# ==========================================
# 3. 测试配置与主逻辑
# ==========================================
TEST_ENV_DIR = os.path.join(CURRENT_DIR, "temp_env_script")
TEST_DATA_DIR = os.path.join(TEST_ENV_DIR, "data")
TEST_OUTPUT_DIR = os.path.join(TEST_ENV_DIR, "output")


def setup_environment():
    """搭建临时测试环境"""
    if os.path.exists(TEST_ENV_DIR):
        shutil.rmtree(TEST_ENV_DIR)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    # 修改全局配置
    Config.DATA_DIR = TEST_DATA_DIR
    Config.OUTPUT_DIR = TEST_OUTPUT_DIR


def create_mock_data(codes, start_date, end_date):
    """生成伪造数据"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    for code in codes:
        base = 100 if code == '000001' else 50
        prices = base + np.cumsum(np.random.randn(len(dates)))
        df = pd.DataFrame({
            'date': dates,
            'open': prices, 'high': prices + 1, 'low': prices - 1, 'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates)) * 100.0
        })
        df['volume'] = df['volume'].astype(float)
        df.set_index('date', inplace=True)
        df.to_parquet(os.path.join(Config.DATA_DIR, f"{code}.parquet"))


def main():
    start_date = "2025-01-01"
    end_date = "2025-01-10"
    codes = ["000001", "000002"]

    print("\n[1/3] 🛠️  准备环境...")
    setup_environment()
    create_mock_data(codes, start_date, end_date)

    # 模拟信号和基准
    mock_signals = pd.DataFrame(index=pd.date_range(start_date, end_date), columns=codes,dtype=float).fillna(0)
    mock_signals['000001'] = 0.9  # 给高分

    mock_bench = pd.DataFrame({'date': pd.date_range(start_date, end_date).date, 'close': 3000.0})

    print("\n[2/3] 🏃 运行回测 (Mock AI & AkShare)...")

    # 注意：patch 的路径也必须带上 src.
    try:
        with patch('src.backtest.WalkForwardBacktester.generate_signal_matrix', return_value=mock_signals), \
                patch('akshare.stock_zh_index_daily', return_value=mock_bench):

            run_walk_forward_backtest(start_date, end_date, 500000, top_k=1)

        print("\n[3/3] ✅ 运行结束")
        if os.path.exists(os.path.join(TEST_OUTPUT_DIR, "walk_forward_result.png")):
            print(f"   -> 成功生成图表: {TEST_OUTPUT_DIR}/walk_forward_result.png")
        else:
            print("   -> ⚠️ 未找到结果文件")

    except Exception as e:
        print(f"❌ 运行出错: {e}")
    finally:
        # 清理
        if os.path.exists(TEST_ENV_DIR):
            shutil.rmtree(TEST_ENV_DIR)


if __name__ == "__main__":
    main()