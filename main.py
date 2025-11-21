import argparse
import warnings
import os

warnings.filterwarnings("ignore")

for k in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    if k in os.environ: del os.environ[k]

from src.data_provider import DataProvider
from src.train import run_training
from src.inference import run_inference
from src.backtest import run_backtest
from src.analysis import BacktestAnalyzer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOTA Quant System v3.0")

    parser.add_argument('--mode', type=str, required=True,
                        choices=['download', 'train', 'predict', 'analysis'],
                        help='运行模式')
    parser.add_argument('--cash', type=float, default=1000000.0, help='回测初始资金')
    parser.add_argument('--top_k', type=int, default=5, help='持仓只数')
    parser.add_argument('--start_date', type=str, default='2024-01-01', help='分析开始日期')
    parser.add_argument('--end_date', type=str, default='2025-12-31', help='分析结束日期')

    # 【新增】强制刷新参数
    parser.add_argument('--force_refresh', action='store_true', help='强制重新计算因子(忽略缓存)')

    args = parser.parse_args()

    print(f"🚀 启动量化系统 Mode: [{args.mode}]")

    if args.mode == 'download':
        DataProvider.download_data()
    elif args.mode == 'train':
        # 传递 force_refresh 信号到 DataProvider
        if args.force_refresh:
            # Hack: 这里我们得修改 run_training 或者 DataProvider.load_and_process_panel 的调用
            # 为了简单，我们直接删除缓存文件即可
            cache_path = DataProvider._get_cache_path('train')
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"🗑️ 已删除缓存: {cache_path}")
        run_training()
    elif args.mode == 'predict':
        if args.force_refresh:
            cache_path = DataProvider._get_cache_path('predict')
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"🗑️ 已删除缓存: {cache_path}")
        top_stocks = run_inference(top_k=args.top_k)
        if top_stocks:
            run_backtest(top_stocks, initial_cash=args.cash)
    elif args.mode == 'analysis':
        analyzer = BacktestAnalyzer(start_date=args.start_date, end_date=args.end_date)
        analyzer.generate_historical_predictions()
        analyzer.analyze_performance()