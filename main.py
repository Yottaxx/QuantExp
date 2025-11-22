# This script is the main entry point for the SOTA Quant System v7.2.
# It handles different modes such as data download, training, prediction, and analysis.

import argparse  # For parsing command-line arguments
import warnings  # For ignoring warnings
import os  # For environment variable manipulation

# Ignore all warnings to keep the console clean
warnings.filterwarnings("ignore")
# Remove any proxy environment variables to avoid conflicts with the data provider
for k in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    if k in os.environ: del os.environ[k]

# Import necessary modules from the src package
from src.data_provider import DataProvider  # Handles data downloading and caching
from src.train import run_training  # Handles the training process
from src.inference import run_inference  # Handles the inference process
from src.backtest import run_backtest  # Handles the backtesting process
from src.analysis import BacktestAnalyzer  # Handles the performance analysis
from src.config import Config  # Global configuration settings

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="SOTA Quant System v7.2")
    # Add arguments for different modes and parameters
    parser.add_argument('--mode', type=str, required=True, choices=['download', 'train', 'predict', 'analysis'],
                        help="Mode to run the system in")
    parser.add_argument('--cash', type=float, default=1000000.0, help="Initial cash for backtesting")
    parser.add_argument('--top_k', type=int, default=5, help="Number of top stocks to select")
    parser.add_argument('--start_date', type=str, default='2024-01-01', help="Start date for backtesting")
    parser.add_argument('--end_date', type=str, default='2025-12-31', help="End date for backtesting")
    parser.add_argument('--force_refresh', action='store_true', help="Force refresh of cached data")
    parser.add_argument('--mse_weight', type=float, default=0.5, help="Weight for MSE in the hybrid loss function")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate for the model")

    # Parse the command-line arguments
    args = parser.parse_args()
    # Update the global configuration with the provided arguments
    Config.MSE_WEIGHT = args.mse_weight
    Config.DROPOUT = args.dropout

    # Print the current mode
    print(f"🚀 Mode: [{args.mode}]")

    # Handle different modes
    if args.mode == 'download':
        # Download and cache the data
        DataProvider.download_data()
    elif args.mode == 'train':
        # If force refresh is enabled, delete the cached training data
        if args.force_refresh:
            p = DataProvider._get_cache_path('train')
            if os.path.exists(p): os.remove(p)
        # Run the training process
        run_training()
    elif args.mode == 'predict':
        # If force refresh is enabled, delete the cached prediction data
        if args.force_refresh:
            p = DataProvider._get_cache_path('predict')
            if os.path.exists(p): os.remove(p)
        # Run the inference process and get the top-k predictions
        top = run_inference(top_k=args.top_k)
        # If there are valid predictions, run the backtest
        if top: run_backtest(top, initial_cash=args.cash)
    elif args.mode == 'analysis':
        # Initialize the backtest analyzer with the specified date range
        an = BacktestAnalyzer(start_date=args.start_date, end_date=args.end_date)
        # Generate historical predictions
        an.generate_historical_predictions()
        # Analyze the performance of the backtest
        an.analyze_performance()