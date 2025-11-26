import argparse
import warnings
import os
from utils.seed_utils import set_global_seed

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
    parser = argparse.ArgumentParser(description="SOTA Quant System v8.3 (Fixed)")

    # 增加 test 和 debug_proxy 模式
    parser.add_argument('--mode', type=str, required=False,default="analysis",
                        choices=['download', 'train', 'predict', 'analysis', 'backtest', 'test', 'debug_proxy'],
                        help='运行模式: [download|train|predict|analysis|backtest|test|debug_proxy]')

    parser.add_argument('--cash', type=float, default=1000000.0, help='回测初始资金')
    parser.add_argument('--top_k', type=int, default=Config.TOP_K, help='持仓数量')

    parser.add_argument('--start_date', type=str, default='2024-01-01', help='开始日期')
    parser.add_argument('--end_date', type=str, default='2025-12-31', help='结束日期')

    parser.add_argument('--force_refresh', action='store_true', help='强制重新生成缓存')
    parser.add_argument('--mse_weight', type=float, default=0.5, help='Loss中MSE的权重')
    parser.add_argument('--dropout', type=float, default=0.2, help='模型Dropout比率')

    args = parser.parse_args()

    # 覆盖全局配置
    Config.MSE_WEIGHT = args.mse_weight
    Config.DROPOUT = args.dropout
    # 如果命令行传入了 top_k，也更新 Config (虽然函数调用时已传参，但保持一致性更好)
    Config.TOP_K = args.top_k

    SEED = Config.SEED
    set_global_seed(SEED)

    print(f"\n🚀 System Launching... Mode: [{args.mode}]")
    print(f"🔧 Config: TopK={args.top_k}, MSE_Weight={args.mse_weight}, Dropout={args.dropout}")

    # --------------------------------------------------------------------------
    # 模式分发
    # --------------------------------------------------------------------------

    if args.mode == 'download':
        DataProvider.download_data()

    elif args.mode == 'train':
        if args.force_refresh:
            p = DataProvider._get_cache_path('train')
            if os.path.exists(p):
                print(f"清理旧缓存: {p}")
                os.remove(p)
        run_training()

    elif args.mode == 'predict':
        if args.force_refresh:
            p = DataProvider._get_cache_path('predict')
            if os.path.exists(p):
                print(f"清理旧缓存: {p}")
                os.remove(p)

        top_stocks = run_inference(top_k=args.top_k)

        # 预测后自动跑一次简单回测验证
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
        an = BacktestAnalyzer(use_test_set_only=True)
        an.generate_historical_predictions()
        an.analyze_performance()
