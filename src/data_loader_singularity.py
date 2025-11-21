import akshare as ak
import pandas as pd
import os
import glob
import numpy as np
import time
import random
import requests
import threading
import datetime
import concurrent.futures
from datasets import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from .config import Config
from .vpn_rotator import vpn_rotator


class DataProvider:
    _vpn_lock = threading.Lock()
    _last_switch_time = 0

    # ==============================================================================
    #   配置区：数据粒度
    #   可选值: 'daily' (日线), '1' (1分钟), '5' (5分钟), '15', '30', '60'
    # ==============================================================================
    DATA_PERIOD = '5'  # <--- 修改这里来改变粒度

    @staticmethod
    def _setup_proxy_env():
        """设置代理环境"""
        proxy_url = "http://127.0.0.1:7890"
        os.environ['http_proxy'] = proxy_url
        os.environ['https_proxy'] = proxy_url
        os.environ['all_proxy'] = proxy_url
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        os.environ['ALL_PROXY'] = proxy_url

    @classmethod
    def _safe_switch_vpn(cls):
        with cls._vpn_lock:
            if time.time() - cls._last_switch_time < 5:
                return
            vpn_rotator.switch_random()
            cls._last_switch_time = time.time()
            time.sleep(2)

    @staticmethod
    def _download_worker(code):
        """
        通用下载单元 (支持 日线/分钟线 自动切换)
        """
        # 根据粒度区分文件名，避免覆盖
        # 例如: 000001_daily.parquet 或 000001_5m.parquet
        suffix = "daily" if DataProvider.DATA_PERIOD == 'daily' else f"{DataProvider.DATA_PERIOD}m"
        path = os.path.join(Config.DATA_DIR, f"{code}_{suffix}.parquet")

        for attempt in range(5):
            try:
                time.sleep(random.uniform(0.05, 0.2))

                df = None

                # --- 分支 1: 下载日线数据 ---
                if DataProvider.DATA_PERIOD == 'daily':
                    df = ak.stock_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=Config.START_DATE,
                        adjust="qfq"
                    )
                    if df is not None and not df.empty:
                        df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close',
                                           '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)

                # --- 分支 2: 下载分钟级数据 ---
                else:
                    # 分钟线接口: period 可选 '1', '5', '15', '30', '60'
                    df = ak.stock_zh_a_hist_min_em(
                        symbol=code,
                        start_date=f"{Config.START_DATE} 09:00:00",  # 格式兼容
                        period=DataProvider.DATA_PERIOD,
                        adjust="qfq"
                    )
                    if df is not None and not df.empty:
                        # 分钟线列名通常是 '时间', '开盘', ...
                        df.rename(columns={'时间': 'date', '开盘': 'open', '收盘': 'close',
                                           '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)

                if df is None or df.empty:
                    return code, True, "Empty"

                # 统一处理索引
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

                # 存盘
                if len(df) > 0:
                    df.to_parquet(path)

                return code, True, "Success"

            except Exception as e:
                # print(f"Err {code}: {e}")
                DataProvider._safe_switch_vpn()
                continue

        return code, False, "Failed"

    @staticmethod
    def download_data():
        """下载入口"""
        print(f">>> [Phase 1] 初始化下载引擎 (粒度: {DataProvider.DATA_PERIOD})...")
        DataProvider._setup_proxy_env()

        # 获取股票列表
        codes = []
        for _ in range(5):
            try:
                stock_info = ak.stock_zh_a_spot_em()
                codes = stock_info['代码'].tolist()
                break
            except:
                vpn_rotator.switch_random()
                time.sleep(2)

        if not codes:
            print("❌ 无法获取股票列表")
            return

        if not os.path.exists(Config.DATA_DIR):
            os.makedirs(Config.DATA_DIR)

        # 智能断点续传 (根据当前粒度后缀过滤)
        suffix = "daily" if DataProvider.DATA_PERIOD == 'daily' else f"{DataProvider.DATA_PERIOD}m"
        print(f">>> 扫描本地已下载数据 (后缀: _{suffix}.parquet)...")

        files = os.listdir(Config.DATA_DIR)
        # 只检查当前粒度的文件
        existing_codes = {
            f.split('_')[0] for f in files
            if f.endswith(f"_{suffix}.parquet") and os.path.getsize(os.path.join(Config.DATA_DIR, f)) > 1024
        }

        all_codes_set = set(codes)
        todo_codes = list(all_codes_set - existing_codes)
        todo_codes.sort()

        print(f"📊 任务统计: 总数 {len(codes)} | 已完成 {len(existing_codes)} | 待下载 {len(todo_codes)}")

        if not todo_codes:
            print("✅ 当前粒度数据已全部下载完毕。")
            return

        MAX_WORKERS = 8  # 分钟线数据量大，建议降低并发数防止内存溢出或封锁过快
        print(f"🚀 启动 {MAX_WORKERS} 线程并发下载 (分钟线速度较慢请耐心等待)...")

        failed_codes = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_code = {executor.submit(DataProvider._download_worker, code): code for code in todo_codes}
            progress_bar = tqdm(concurrent.futures.as_completed(future_to_code), total=len(todo_codes), unit="it")

            for future in progress_bar:
                code = future_to_code[future]
                try:
                    _, is_success, _ = future.result()
                    if not is_success:
                        failed_codes.append(code)
                except:
                    failed_codes.append(code)

        print(f"下载结束。失败数: {len(failed_codes)}")

    # ==============================================================================
    #   Phase 2: 数据处理
    # ==============================================================================

    @staticmethod
    def process_single_stock(df):
        from .alpha_lib import AlphaFactory
        # 分钟级预测通常预测未来 N 个 Bar，比如未来 12 个 5分钟(1小时)
        df['target'] = df['close'].shift(-Config.PRED_LEN) / df['close'] - 1
        factory = AlphaFactory(df)
        df = factory.make_factors()
        factor_cols = [c for c in df.columns if c.startswith('alpha_')]
        keep_cols = factor_cols + ['target']
        df.dropna(subset=keep_cols, inplace=True)
        return df, factor_cols

    def generator(self):
        # 这里也要适配文件名
        suffix = "daily" if DataProvider.DATA_PERIOD == 'daily' else f"{DataProvider.DATA_PERIOD}m"
        pattern = f"*_{suffix}.parquet"

        files = glob.glob(os.path.join(Config.DATA_DIR, pattern))
        target_files = files

        for fpath in target_files:
            try:
                df = pd.read_parquet(fpath)
                if len(df) < 100: continue

                df_proc, factor_cols = self.process_single_stock(df)

                scaler = StandardScaler()
                x_data = scaler.fit_transform(df_proc[factor_cols].values)
                y_data = df_proc['target'].values
                for i in range(0, len(x_data) - Config.CONTEXT_LEN, 5):
                    yield {
                        "past_values": x_data[i: i + Config.CONTEXT_LEN].astype(np.float32),
                        "labels": y_data[i + Config.CONTEXT_LEN - 1].astype(np.float32)
                    }
            except:
                continue
