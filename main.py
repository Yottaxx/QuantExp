import argparse
import warnings
import os

warnings.filterwarnings("ignore")
for k in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    if k in os.environ: del os.environ[k]

from src.data_provider import DataProvider
from src.train import run_training
from src.inference import run_inference
# 引入新的回测入口
from src.backtest import run_walk_forward_backtest, run_backtest
from src.analysis import BacktestAnalyzer
from src.config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOTA Quant System v8.2")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['download', 'train', 'predict', 'analysis', 'backtest'],
                        help='运行模式')
    parser.add_argument('--cash', type=float, default=1000000.0)

    # 【优化】使用 Config 默认值
    parser.add_argument('--top_k', type=int, default=Config.TOP_K)

    parser.add_argument('--start_date', type=str, default='2024-01-01')
    parser.add_argument('--end_date', type=str, default='2025-12-31')
    parser.add_argument('--force_refresh', action='store_true')
    parser.add_argument('--mse_weight', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.2)

    args = parser.parse_args()
    Config.MSE_WEIGHT = args.mse_weight
    Config.DROPOUT = args.dropout

    print(f"🚀 Mode: [{args.mode}]")

    if args.mode == 'download':
        DataProvider.download_data()

    elif args.mode == 'train':
        if args.force_refresh:
            p = DataProvider._get_cache_path('train')
            if os.path.exists(p): os.remove(p)
        run_training()

    elif args.mode == 'predict':
        if args.force_refresh:
            p = DataProvider._get_cache_path('predict')
            if os.path.exists(p): os.remove(p)

        top_stocks = run_inference(top_k=args.top_k)

        if top_stocks:
            run_backtest(top_stocks, initial_cash=args.cash, top_k=args.top_k)

    elif args.mode == 'backtest':
        run_walk_forward_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            initial_cash=args.cash,
            top_k=args.top_k
        )

    elif args.mode == 'analysis':
        an = BacktestAnalyzer(start_date=args.start_date, end_date=args.end_date)
        an.generate_historical_predictions()
        an.analyze_performance()