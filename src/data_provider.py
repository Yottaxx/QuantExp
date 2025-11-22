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

    # ... [PART 1 基础设置 保持不变] ...
    @staticmethod
    def _setup_proxy_env():
        proxy_url = "http://127.0.0.1:7890"
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
            df = ak.stock_zh_index_daily(symbol="sh000001")
            return pd.to_datetime(df['date']).max().date().strftime("%Y-%m-%d")
        except:
            return datetime.date.today().strftime("%Y-%m-%d")

    # --------------------------------------------------------------------------
    # PART 1.5: 财务数据下载 (新增)
    # --------------------------------------------------------------------------
    @staticmethod
    def _download_finance_worker(code):
        """下载单只股票的财务指标"""
        # 财务数据存放在单独的目录，避免和行情数据混淆
        fund_dir = os.path.join(Config.DATA_DIR, "fundamental")
        if not os.path.exists(fund_dir): os.makedirs(fund_dir)
        path = os.path.join(fund_dir, f"{code}.parquet")

        # 财务数据更新频率低，如果文件存在且较新(7天内)，可以跳过
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if (time.time() - mtime) < 7 * 24 * 3600:  # 7天过期
                return code, True, "Skipped"

        for attempt in range(3):
            try:
                time.sleep(random.uniform(0.1, 0.5))
                # 接口：东方财富-个股-财务分析-主要指标
                # 包含: 每股收益, 净资产收益率(ROE), 资产负债率, 营收增长率, 净利润增长率等
                df = ak.stock_financial_analysis_indicator_em(symbol=code)

                if df is None or df.empty: return code, True, "Empty"

                # 核心清洗：日期处理
                # 东财返回的日期是 "2023-03-31"，这是报告期，不是公告期。
                # 为了防止未来函数，我们需要做一个简单的处理：
                # 默认假设财报在报告期后 45 天发布 (保守估计)
                # 或者保留原始日期，在合并时做 shift
                df['date'] = pd.to_datetime(df['日期'])

                # 筛选核心字段
                cols_map = {
                    '加权净资产收益率': 'roe',
                    '主营业务收入增长率(%)': 'rev_growth',
                    '净利润增长率(%)': 'profit_growth',
                    '资产负债率(%)': 'debt_ratio',
                    '市盈率(动态)': 'pe_ttm',  # 注意：东财这个接口里的PE可能不准，最好用实时行情算
                    '市净率': 'pb'
                }

                # 并非所有列都存在，取交集
                valid_cols = [c for c in cols_map.keys() if c in df.columns]
                df = df[['date'] + valid_cols].copy()
                df.rename(columns=cols_map, inplace=True)

                # 转 float32
                for c in df.columns:
                    if c != 'date':
                        df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float32)

                df.set_index('date', inplace=True)
                df.to_parquet(path)
                return code, True, "Success"
            except:
                DataProvider._safe_switch_vpn()
                continue
        return code, False, "Failed"

    # --------------------------------------------------------------------------
    # PART 2: 下载模块 (含财务数据调度)
    # --------------------------------------------------------------------------
    @staticmethod
    def _download_worker(code):
        # ... [保持原有的行情下载逻辑不变] ...
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
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
                df.dropna(inplace=True)

                # 对齐时间
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
        print(">>> [Phase 1] 启动全量数据下载 (行情 + 财务)...")
        DataProvider._setup_proxy_env()
        if not os.path.exists(Config.DATA_DIR): os.makedirs(Config.DATA_DIR)

        try:
            stock_info = ak.stock_zh_a_spot_em()
            codes = stock_info['代码'].tolist()
        except:
            print("❌ 无法获取股票列表")
            return

        # 1. 下载日线行情
        print(">>> (1/2) 正在同步日线行情...")
        target_date_str = DataProvider._get_latest_trading_date()
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

        # 2. 下载财务数据 (独立线程池)
        print(">>> (2/2) 正在同步财务数据...")
        # 财务数据不需要每天下，上面的 worker 内部有7天过期检查
        # 这里直接把所有 code 扔进去，worker 会自己判断是否跳过
        # 但为了效率，也可以在这里检查文件存在性
        fund_dir = os.path.join(Config.DATA_DIR, "fundamental")
        if not os.path.exists(fund_dir): os.makedirs(fund_dir)

        # 简单起见，全量检查一遍
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(DataProvider._download_finance_worker, c): c for c in codes}
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(codes), desc="Finance"): pass

        print("所有数据同步完成。")

    # ... [缓存路径和过滤逻辑保持不变] ...
    @staticmethod
    def _get_cache_path(mode):
        today_str = datetime.date.today().strftime("%Y%m%d")
        return os.path.join(Config.OUTPUT_DIR, f"panel_cache_{mode}_{today_str}.pkl")

    @staticmethod
    def _filter_universe(panel_df):
        # ... [复用之前的过滤逻辑] ...
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
    # PART 3: Panel 加载与融合 (Merge Price & Fund)
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

        # 1. 读取行情数据
        print("正在加载行情数据...")
        price_files = glob.glob(os.path.join(Config.DATA_DIR, "*.parquet"))
        fund_dir = os.path.join(Config.DATA_DIR, "fundamental")

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

        # 2. 读取财务数据并合并
        print("正在加载并合并财务数据...")
        fund_files = glob.glob(os.path.join(fund_dir, "*.parquet"))
        fund_frames = []

        def _read_fund(f):
            try:
                df = pd.read_parquet(f)
                code = os.path.basename(f).replace(".parquet", "")
                df['code'] = code
                return df
            except:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            fund_results = list(executor.map(_read_fund, fund_files))

        fund_frames = [df for df in fund_results if df is not None]
        if fund_frames:
            fund_df = pd.concat(fund_frames)
            fund_df = fund_df.reset_index().sort_values(['code', 'date'])

            # 【关键步骤】财务数据对齐
            # 1. 财务数据是低频的，行情是高频的
            # 2. 我们需要把财务数据根据 date 映射到行情数据上
            # 3. 为了防未来函数，我们将财务数据的 date 向后推迟 2 个月 (60天)
            #    作为"公告日"的保守估计 (因为一季报3.31，通常4月底发，甚至更晚)
            #    或者更简单的做法：merge_asof

            fund_df['announce_date'] = fund_df['date'] + pd.Timedelta(days=60)
            fund_df = fund_df.drop(columns=['date']).rename(columns={'announce_date': 'date'})

            # 重置索引以便 merge
            panel_df = panel_df.reset_index().sort_values(['code', 'date'])

            # merge_asof 需要按 key 排序
            panel_df = pd.merge_asof(
                panel_df,
                fund_df,
                on='date',
                by='code',
                direction='backward'  # 向后查找最近的一个财报
            )

            # 填充财务数据的空值 (上市初期可能没财报)
            fund_cols = ['roe', 'rev_growth', 'profit_growth', 'debt_ratio']
            for c in fund_cols:
                if c in panel_df.columns:
                    panel_df[c] = panel_df[c].fillna(0).astype(np.float32)

            print(f"财务数据合并完成。")
        else:
            print("⚠️ 未找到财务数据，跳过合并。")

        panel_df = panel_df.set_index('date')  # 恢复索引
        panel_df = panel_df.reset_index().sort_values(['code', 'date'])  # 确保顺序

        # --- 后续流程 (保持不变) ---
        print("计算时序因子...")
        panel_df = panel_df.groupby('code', group_keys=False).apply(lambda x: AlphaFactory(x).make_factors())

        print("构造 Target...")
        panel_df['target'] = panel_df.groupby('code')['close'].shift(-Config.PRED_LEN) / panel_df['close'] - 1
        if mode == 'train': panel_df.dropna(subset=['target'], inplace=True)

        panel_df = DataProvider._filter_universe(panel_df)

        print("计算截面因子...")
        panel_df = panel_df.set_index('date')
        panel_df = AlphaFactory.add_cross_sectional_factors(panel_df)

        feature_cols = [c for c in panel_df.columns
                        if any(c.startswith(p) for p in
                               ['style_', 'tech_', 'alpha_', 'adv_', 'ind_', 'fund_', 'cs_rank_', 'mkt_',
                                'rel_'])]  # 增加 fund_

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