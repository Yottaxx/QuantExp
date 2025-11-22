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
import pickle
from datasets import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from .config import Config
from .vpn_rotator import vpn_rotator
from .alpha_lib import AlphaFactory


class DataProvider:
    _vpn_lock = threading.Lock()
    _last_switch_time = 0

    # ... [PART 1: 下载模块保持不变，省略以节省篇幅] ...
    # 请保留原有的 _setup_proxy_env, _safe_switch_vpn, _get_latest_trading_date
    # _download_finance_worker, _download_worker, download_data, _get_cache_path
    # _filter_universe

    @staticmethod
    def _setup_proxy_env():
        proxy_url = Config.PROXY_URL
        for k in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
            os.environ[k] = proxy_url

    @classmethod
    def _safe_switch_vpn(cls):
        with cls._vpn_lock:
            if time.time() - cls._last_switch_time < 5: return
            vpn_rotator.switch_random()
            cls._last_switch_time = time.time()
            time.sleep(2)

    @staticmethod
    def _get_latest_trading_date():
        try:
            df = ak.stock_zh_index_daily(symbol=Config.MARKET_INDEX_SYMBOL)
            return pd.to_datetime(df['date']).max().date().strftime("%Y-%m-%d")
        except:
            return datetime.date.today().strftime("%Y-%m-%d")

    @staticmethod
    def _download_finance_worker(code):
        fund_dir = os.path.join(Config.DATA_DIR, "fundamental")
        if not os.path.exists(fund_dir): os.makedirs(fund_dir)
        path = os.path.join(fund_dir, f"{code}.parquet")
        if os.path.exists(path):
            if (time.time() - os.path.getmtime(path)) < 7 * 24 * 3600: return code, True, "Skipped"
        for attempt in range(3):
            try:
                time.sleep(random.uniform(0.1, 0.5))
                df = ak.stock_financial_analysis_indicator_em(symbol=code)
                if df is None or df.empty: return code, True, "Empty"
                df['date'] = pd.to_datetime(df['日期'])
                cols_map = {'加权净资产收益率': 'roe', '主营业务收入增长率(%)': 'rev_growth',
                            '净利润增长率(%)': 'profit_growth', '资产负债率(%)': 'debt_ratio', '市盈率(动态)': 'pe_ttm',
                            '市净率': 'pb'}
                valid_cols = [c for c in cols_map.keys() if c in df.columns]
                df = df[['date'] + valid_cols].copy()
                df.rename(columns=cols_map, inplace=True)
                for c in df.columns:
                    if c != 'date': df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float32)
                df.set_index('date', inplace=True)
                df.to_parquet(path)
                return code, True, "Success"
            except:
                DataProvider._safe_switch_vpn()
                continue
        return code, False, "Failed"

    @staticmethod
    def _download_worker(code):
        path = os.path.join(Config.DATA_DIR, f"{code}.parquet")
        for attempt in range(5):
            try:
                time.sleep(random.uniform(0.05, 0.2))
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=Config.START_DATE, adjust="qfq")
                if df is None or df.empty: return code, True, "Empty"
                df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close',
                                   '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
                df.dropna(inplace=True)
                if not df.empty:
                    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
                    df = df.reindex(full_idx)
                    if 'volume' in df.columns: df['volume'] = df['volume'].fillna(0)
                    df = df.ffill()
                    df.dropna(inplace=True)
                    df = df[df.index.dayofweek < 5]
                if len(df) > 0: df.to_parquet(path)
                return code, True, "Success"
            except:
                DataProvider._safe_switch_vpn()
                continue
        return code, False, "Failed"

    @staticmethod
    def download_data():
        print(">>> [Phase 1] 启动全量数据下载...")
        DataProvider._setup_proxy_env()
        if not os.path.exists(Config.DATA_DIR): os.makedirs(Config.DATA_DIR)
        try:
            stock_info = ak.stock_zh_a_spot_em()
            codes = stock_info['代码'].tolist()
        except:
            print("❌ 无法获取股票列表")
            return
        target_date_str = DataProvider._get_latest_trading_date()
        print(f"📅 市场最新交易日: {target_date_str}")
        existing_fresh = set()
        files = os.listdir(Config.DATA_DIR)
        for fname in files:
            if fname.endswith(".parquet"):
                fpath = os.path.join(Config.DATA_DIR, fname)
                if os.path.getsize(fpath) > 1024:
                    mtime = os.path.getmtime(fpath)
                    file_date = datetime.date.fromtimestamp(mtime).strftime("%Y-%m-%d")
                    if file_date >= target_date_str: existing_fresh.add(fname.replace(".parquet", ""))
        todo_price = list(set(codes) - existing_fresh)
        todo_price.sort()
        if todo_price:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = {executor.submit(DataProvider._download_worker, c): c for c in todo_price}
                for _ in tqdm(concurrent.futures.as_completed(futures), total=len(todo_price), desc="Price"): pass
        else:
            print("✅ 日线行情已是最新。")
        fund_dir = os.path.join(Config.DATA_DIR, "fundamental")
        if not os.path.exists(fund_dir): os.makedirs(fund_dir)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(DataProvider._download_finance_worker, c): c for c in codes}
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(codes), desc="Finance"): pass
        print("同步完成。")

    @staticmethod
    def _get_cache_path(mode):
        today_str = datetime.date.today().strftime("%Y%m%d")
        return os.path.join(Config.OUTPUT_DIR, f"panel_cache_{mode}_{today_str}.pkl")

    @staticmethod
    def _filter_universe(panel_df):
        print(">>> [Filtering] 正在执行动态股票池过滤...")
        original_len = len(panel_df)
        panel_df = panel_df[panel_df['volume'] > 0]
        panel_df = panel_df[panel_df['close'] >= 2.0]
        panel_df['list_days'] = panel_df.groupby('code').cumcount()
        panel_df = panel_df[panel_df['list_days'] > 60]
        panel_df = panel_df.drop(columns=['list_days'])
        new_len = len(panel_df)
        print(f"过滤完成。移除样本: {original_len - new_len} ({1 - new_len / original_len:.2%})")
        return panel_df

    # --------------------------------------------------------------------------
    # PART 3: Panel 加载与融合
    # --------------------------------------------------------------------------
    @staticmethod
    def load_and_process_panel(mode='train', force_refresh=False):
        cache_path = DataProvider._get_cache_path(mode)
        if not force_refresh and os.path.exists(cache_path):
            print(f"⚡️ [Cache Hit] 加载缓存: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass

        print(f"\n>>> [Phase 2] 构建全内存 Panel 数据 (Mode: {mode})...")
        price_files = glob.glob(os.path.join(Config.DATA_DIR, "*.parquet"))
        fund_dir = os.path.join(Config.DATA_DIR, "fundamental")

        # 读取行情
        print("加载行情数据...")

        def _read_price(f):
            try:
                df = pd.read_parquet(f)
                code = os.path.basename(f).replace(".parquet", "")
                float_cols = df.select_dtypes(include=['float64']).columns
                df[float_cols] = df[float_cols].astype(np.float32)
                df['code'] = code
                return df
            except:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(_read_price, price_files), total=len(price_files), desc="Reading Price"))

        data_frames = [df for df in results if df is not None and len(df) > Config.CONTEXT_LEN]
        if not data_frames: raise ValueError("有效数据为空")

        panel_df = pd.concat(data_frames, ignore_index=False)
        del data_frames
        panel_df['code'] = panel_df['code'].astype(str)

        # 合并财务
        print("合并财务数据...")
        fund_files = glob.glob(os.path.join(fund_dir, "*.parquet"))

        def _read_fund(f):
            try:
                df = pd.read_parquet(f)
                df['code'] = os.path.basename(f).replace(".parquet", "")
                return df
            except:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            fund_frames = [df for df in executor.map(_read_fund, fund_files) if df is not None]

        if fund_frames:
            fund_df = pd.concat(fund_frames)
            fund_df = fund_df.reset_index().sort_values(['code', 'date'])
            fund_df['announce_date'] = fund_df['date'] + pd.Timedelta(days=60)
            fund_df = fund_df.drop(columns=['date']).rename(columns={'announce_date': 'date'})
            panel_df = panel_df.reset_index().sort_values(['code', 'date'])
            panel_df = pd.merge_asof(panel_df, fund_df, on='date', by='code', direction='backward')
            for c in ['roe', 'rev_growth', 'profit_growth', 'debt_ratio', 'pe_ttm', 'pb']:
                if c in panel_df.columns: panel_df[c] = panel_df[c].fillna(0).astype(np.float32)

        if 'date' in panel_df.columns: panel_df = panel_df.set_index('date')
        panel_df = panel_df.reset_index().sort_values(['code', 'date'])

        print("计算时序因子...")
        panel_df = panel_df.groupby('code', group_keys=False).apply(lambda x: AlphaFactory(x).make_factors())

        print("构造 Target...")
        panel_df['next_open'] = panel_df.groupby('code')['open'].shift(-1)
        panel_df['future_close'] = panel_df.groupby('code')['close'].shift(-Config.PRED_LEN)
        panel_df['target'] = panel_df['future_close'] / panel_df['next_open'] - 1
        panel_df.drop(columns=['next_open', 'future_close'], inplace=True)

        if mode == 'train':
            panel_df.dropna(subset=['target'], inplace=True)

        panel_df = DataProvider._filter_universe(panel_df)

        print("计算截面因子...")
        panel_df = panel_df.set_index('date')
        panel_df = AlphaFactory.add_cross_sectional_factors(panel_df)

        # 【核心修复】增加 'time_' 前缀，确保日历因子被包含
        feature_cols = [c for c in panel_df.columns
                        if any(c.startswith(p) for p in Config.FEATURE_PREFIXES)]

        panel_df[feature_cols] = panel_df[feature_cols].fillna(0).astype(np.float32)
        panel_df = panel_df.reset_index()

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump((panel_df, feature_cols), f)
        except:
            pass

        return panel_df, feature_cols

    # ... [make_dataset 保持不变] ...
    @staticmethod
    def make_dataset(panel_df, feature_cols):
        print(">>> [Phase 3] 转换 Dataset...")
        panel_df = panel_df.sort_values(['code', 'date'])
        feature_matrix = panel_df[feature_cols].values.astype(np.float32)
        target_col = 'rank_label' if 'rank_label' in panel_df.columns else 'target'
        target_array = panel_df[target_col].fillna(0.5).values.astype(np.float32)

        codes = panel_df['code'].values
        code_changes = np.where(codes[:-1] != codes[1:])[0] + 1
        start_indices = np.concatenate(([0], code_changes))
        end_indices = np.concatenate((code_changes, [len(codes)]))
        valid_indices = []
        seq_len = Config.CONTEXT_LEN
        stride = 5
        for start, end in zip(start_indices, end_indices):
            length = end - start
            if length <= seq_len: continue
            for i in range(start, end - seq_len + 1, stride): valid_indices.append(i)

        dates = panel_df['date'].unique()
        dates.sort()
        split_idx = int(len(dates) * 0.9)
        split_date = dates[split_idx]
        sample_dates = panel_df['date'].values[np.array(valid_indices) + seq_len - 1]
        train_mask = sample_dates < split_date
        train_indices = np.array(valid_indices)[train_mask]
        valid_indices = np.array(valid_indices)[~train_mask]

        def gen_train():
            np.random.shuffle(train_indices)
            for idx in train_indices:
                yield {"past_values": feature_matrix[idx: idx + seq_len], "labels": target_array[idx + seq_len - 1]}

        def gen_valid():
            for idx in valid_indices:
                yield {"past_values": feature_matrix[idx: idx + seq_len], "labels": target_array[idx + seq_len - 1]}

        from datasets import DatasetDict
        ds = DatasetDict({'train': Dataset.from_generator(gen_train), 'test': Dataset.from_generator(gen_valid)})
        return ds, len(feature_cols)


def get_dataset(force_refresh=False):
    panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train', force_refresh=force_refresh)
    ds, num_features = DataProvider.make_dataset(panel_df, feature_cols)
    return ds, num_features