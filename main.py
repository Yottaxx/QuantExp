import argparse
import warnings

# 忽略 UserWarning (Pandas/Torch 产生的非关键警告)
warnings.filterwarnings("ignore")

from src.data_provider import DataProvider
from src.train import run_training
from src.inference import run_inference
from src.backtest import run_backtest
from src.analysis import BacktestAnalyzer  # [NEW]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOTA Quant System")
    parser.add_argument('--mode', type=str, required=False, default='train',
                        choices=['download', 'train', 'predict', 'analysis'],
                        help='运行模式: download | train | predict | analysis')

    # 可以添加日期参数
    parser.add_argument('--start_date', type=str, default='2024-01-01', help='回溯分析开始日期')
    parser.add_argument('--end_date', type=str, default='2025-11-20', help='回溯分析结束日期')

    args = parser.parse_args()

    if args.mode == 'download':
        DataProvider.download_data()

    elif args.mode == 'train':
        run_training()

    elif args.mode == 'predict':
        top_stocks = run_inference(top_k=5)
        run_backtest(top_stocks)

    elif args.mode == 'analysis':
        print(f"启动回溯预测分析模块 ({args.start_date} ~ {args.end_date})...")
        analyzer = BacktestAnalyzer(start_date=args.start_date, end_date=args.end_date)

        # 1. 生成历史预测数据
        analyzer.generate_historical_predictions()

        # 2. 计算指标并绘图
        analyzer.analyze_performance()



