import argparse
import warnings
import os

# 忽略 Pandas 和 Torch 的非关键警告，保持界面整洁
warnings.filterwarnings("ignore")

# 设置环境变量，确保在导入 data_provider 前清理代理
# (虽然 DataProvider 内部也有清理，但入口处清理更保险)
for k in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    if k in os.environ: del os.environ[k]

from src.data_provider import DataProvider
from src.train import run_training
from src.inference import run_inference
from src.backtest import run_backtest
from src.analysis import BacktestAnalyzer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOTA Quant System v3.0")

    # 核心模式选择
    parser.add_argument('--mode', type=str, required=False, default="train",
                        choices=['download', 'train', 'predict', 'analysis'],
                        help='运行模式: download(下载数据) | train(训练模型) | predict(选股+回测) | analysis(历史回溯分析)')

    # 回测参数
    parser.add_argument('--cash', type=float, default=50000.0, help='回测初始资金 (默认 5万)')
    parser.add_argument('--top_k', type=int, default=5, help='持仓只数 (默认 5)')

    # 分析参数
    parser.add_argument('--start_date', type=str, default='2024-01-01', help='分析开始日期')
    parser.add_argument('--end_date', type=str, default='2025-12-31', help='分析结束日期')

    args = parser.parse_args()

    print(f"🚀 启动量化系统 Mode: [{args.mode}]")

    if args.mode == 'download':
        # 下载/更新数据
        DataProvider.download_data()

    elif args.mode == 'train':
        # 训练模型
        # 注意：这里会自动调用 load_and_process_panel 加载全量数据
        run_training()

    elif args.mode == 'predict':
        # 1. AI 选股
        top_stocks = run_inference(top_k=args.top_k)

        # 2. 策略回测 (基于选出的股票进行模拟交易)
        if top_stocks:
            run_backtest(top_stocks, initial_cash=args.cash)
        else:
            print("⚠️ 未选出有效股票，跳过回测。")

    elif args.mode == 'analysis':
        # 全历史回溯分析 (IC/IR/分层回测)
        print(f"启动回溯预测分析模块 ({args.start_date} ~ {args.end_date})...")
        analyzer = BacktestAnalyzer(start_date=args.start_date, end_date=args.end_date)
        analyzer.generate_historical_predictions()
        analyzer.analyze_performance()