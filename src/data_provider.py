import akshare as ak
import pandas as pd
import os
import glob
import numpy as np
import time
import random
import threading
import datetime
import concurrent.futures
import pickle
import warnings
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from .config import Config
from .vpn_rotator import vpn_rotator
from .alpha_lib import AlphaFactory
from pandarallel import pandarallel

# 忽略 pandas 的性能警告
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# 初始化并行计算
pandarallel.initialize(progress_bar=True, nb_workers=os.cpu_count())


class DataProvider:
    _vpn_lock = threading.Lock()
    _last_switch_time = 0

    @staticmethod
    def _setup_proxy_env():
        proxy_url = Config.PROXY_URL
        for k in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
            os.environ[k] = proxy_url

    @classmethod
    def _safe_switch_vpn(cls):
        with cls._vpn_lock:
            if time.time() - cls._last_switch_time < 5: return
            try:
                vpn_rotator.switch_random()
            except:
                pass
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
    def _fetch_pub_date_map(code):
        """获取公告日映射，用于PIT"""
        try:
            df = ak.stock_financial_abstract(symbol=code)
            if df is None or df.empty: return None
            if '截止日期' in df.columns and '公告日期' in df.columns:
                res = df[['截止日期', '公告日期']].copy()
                res.columns = ['date', 'pub_date']
                res['date'] = pd.to_datetime(res['date'], errors='coerce')
                res['pub_date'] = pd.to_datetime(res['pub_date'], errors='coerce')
                return res.dropna()
        except:
            pass
        return None

    @staticmethod
    def _download_finance_worker(code):
        fund_dir = os.path.join(Config.DATA_DIR, "fundamental")
        if not os.path.exists(fund_dir): os.makedirs(fund_dir)
        path = os.path.join(fund_dir, f"{code}.parquet")

        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            # 3天内不重复下载
            if (time.time() - mtime) < 3 * 24 * 3600: return code, True, "Skipped"

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

                pub_df = DataProvider._fetch_pub_date_map(code)
                if pub_df is not None:
                    df = pd.merge(df, pub_df, on='date', how='left')
                else:
                    df['pub_date'] = pd.NaT

                for c in df.columns:
                    if c not in ['date', 'pub_date']:
                        df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float32)

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

                df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
                                   '成交量': 'volume', '成交额': 'amount'}, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

                for col in ['open', 'close', 'high', 'low', 'volume', 'amount']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

                df.dropna(inplace=True)

                # 简单的数据清洗
                if 'amount' in df.columns and 'volume' in df.columns and 'close' in df.columns:
                    valid_sample = df[(df['volume'] > 0) & (df['amount'] > 0)].tail(20)
                    if not valid_sample.empty:
                        multiplier = (
                                    valid_sample['amount'] / (valid_sample['close'] * valid_sample['volume'])).median()
                        if multiplier > 50: df['volume'] = df['volume'] * 100

                if 'amount' in df.columns: df.drop(columns=['amount'], inplace=True)
                if not df.empty:
                    df.sort_index(inplace=True)
                    df.to_parquet(path)
                return code, True, "Success"
            except:
                DataProvider._safe_switch_vpn()
                continue
        return code, False, "Failed"

    @staticmethod
    def download_data():
        """ETL 主入口"""
        print(">>> [ETL] 启动数据下载流水线...")
        DataProvider._setup_proxy_env()
        if not os.path.exists(Config.DATA_DIR): os.makedirs(Config.DATA_DIR)

        try:
            stock_info = ak.stock_zh_a_spot_em()
            codes = stock_info['代码'].tolist()
        except:
            print("❌ 无法获取股票列表")
            return

        target_date_str = DataProvider._get_latest_trading_date()
        existing_fresh = set()

        for fname in os.listdir(Config.DATA_DIR):
            if fname.endswith(".parquet"):
                fpath = os.path.join(Config.DATA_DIR, fname)
                if os.path.getsize(fpath) > 1024:
                    mtime = os.path.getmtime(fpath)
                    file_date = datetime.date.fromtimestamp(mtime).strftime("%Y-%m-%d")
                    if file_date >= target_date_str: existing_fresh.add(fname.replace(".parquet", ""))

        todo_price = sorted(list(set(codes) - existing_fresh))
        print(f"📊 股票池总数: {len(codes)} | 待更新: {len(todo_price)}")

        if todo_price:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = {executor.submit(DataProvider._download_worker, c): c for c in todo_price}
                for _ in tqdm(concurrent.futures.as_completed(futures), total=len(todo_price),
                              desc="Downloading Price"): pass

        print("正在同步财务数据...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(DataProvider._download_finance_worker, c): c for c in codes}
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(codes), desc="Downloading Finance"): pass

        print("✅ 数据同步完成。")

    @staticmethod
    def _get_cache_path(mode):
        today_str = datetime.date.today().strftime("%Y%m%d")
        return os.path.join(Config.OUTPUT_DIR, f"panel_cache_{mode}_{today_str}.pkl")

    @staticmethod
    def _tag_universe(panel_df):
        """
        [CRITICAL FIX] 标记 Universe，彻底修复未来数据泄漏
        原逻辑 transform('count') 会导致 2020 年知道 2024 年该股票还活着。
        新逻辑 cumcount() 仅统计截至当天的上市天数。
        """
        print(">>> [Tagging] 标记动态股票池 (Universe Mask)...")

        # 1. 必须确保按日期排序
        panel_df = panel_df.sort_values(['code', 'date'])

        # 2. 使用 expanding count (cumcount) 统计累计上市天数
        panel_df['list_days_count'] = panel_df.groupby('code')['date'].cumcount() + 1

        # 3. 筛选条件
        cond_vol = panel_df['volume'] > 0
        cond_price = panel_df['close'] >= 2.0
        # 必须上市超过 60 天才纳入 (严格剔除次新股)
        cond_list = panel_df['list_days_count'] > 60

        panel_df['is_universe'] = cond_vol & cond_price & cond_list

        # 清理临时列
        panel_df.drop(columns=['list_days_count'], inplace=True)

        valid_count = panel_df['is_universe'].sum()
        total_count = len(panel_df)
        print(f"Universe 覆盖率: {valid_count}/{total_count} ({valid_count / total_count:.2%})")
        return panel_df

    @staticmethod
    def load_and_process_panel(mode='train', force_refresh=False):
        """
        加载并处理面板数据 (计算因子、标签、截面处理)
        mode='train' 表示加载带 Label 的全量数据，并不代表只加载训练集
        """
        cache_path = DataProvider._get_cache_path(mode)
        if not force_refresh and os.path.exists(cache_path):
            print(f"⚡️ [Cache Hit] {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        print(f"\n>>> [Processing] 构建 Panel 数据 (Mode: {mode})...")
        price_files = glob.glob(os.path.join(Config.DATA_DIR, "*.parquet"))
        fund_dir = os.path.join(Config.DATA_DIR, "fundamental")

        def _read_price(f):
            try:
                df = pd.read_parquet(f)
                if isinstance(df.index, pd.DatetimeIndex) and 'date' not in df.columns:
                    df = df.reset_index()
                df['code'] = os.path.basename(f).replace(".parquet", "")
                float_cols = df.select_dtypes(include=['float64']).columns
                df[float_cols] = df[float_cols].astype(np.float32)
                return df
            except:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(_read_price, price_files), total=len(price_files), desc="Reading Price"))

        data_frames = [df for df in results if df is not None and len(df) > Config.CONTEXT_LEN]
        if not data_frames: raise ValueError("没有足够的有效行情数据，请先运行 download_data()")

        panel_df = pd.concat(data_frames, ignore_index=True)
        del data_frames

        panel_df['code'] = panel_df['code'].astype(str)
        panel_df['date'] = pd.to_datetime(panel_df['index'] if 'index' in panel_df.columns else panel_df['date'])

        # 读取财务并进行 PIT 合并
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

            # 使用公告日对齐
            if 'pub_date' in fund_df.columns:
                fund_df['merge_date'] = fund_df['pub_date']
                mask_na = fund_df['merge_date'].isna()
                report_months = fund_df.loc[mask_na, 'date'].dt.month
                delays = report_months.apply(lambda m: 120 if m == 12 else 60)
                fund_df.loc[mask_na, 'merge_date'] = fund_df.loc[mask_na, 'date'] + pd.to_timedelta(delays, unit='D')
            else:
                fund_df['merge_date'] = fund_df['date'] + pd.Timedelta(days=90)

            fund_df = fund_df.drop(columns=['date', 'pub_date'], errors='ignore')
            fund_df.rename(columns={'merge_date': 'date'}, inplace=True)

            panel_df = panel_df.reset_index().sort_values(['code', 'date'])
            panel_df = pd.merge_asof(panel_df, fund_df, on='date', by='code', direction='backward')

            for c in ['roe', 'rev_growth', 'profit_growth', 'debt_ratio', 'pe_ttm', 'pb']:
                if c in panel_df.columns: panel_df[c] = panel_df[c].fillna(0).astype(np.float32)
            print("✅ 财务数据 PIT 对齐完成。")

        if 'date' in panel_df.columns: panel_df = panel_df.set_index('date')
        panel_df = panel_df.reset_index().sort_values(['code', 'date'])

        print("计算时序因子...")
        panel_df = panel_df.groupby('code', group_keys=False).parallel_apply(lambda x: AlphaFactory(x).make_factors())

        print("构造预测目标 (Labels)...")
        panel_df['next_open'] = panel_df.groupby('code')['open'].shift(-1)
        panel_df['future_close'] = panel_df.groupby('code')['close'].shift(-Config.PRED_LEN)
        panel_df['target'] = panel_df['future_close'] / panel_df['next_open'] - 1
        panel_df.drop(columns=['next_open', 'future_close'], inplace=True)

        if mode == 'train':
            panel_df.dropna(subset=['target'], inplace=True)

        # [修复] 标记 Universe
        panel_df = DataProvider._tag_universe(panel_df)

        print("计算截面因子与标准化...")
        panel_df = panel_df.set_index('date')
        panel_df = AlphaFactory.add_cross_sectional_factors(panel_df)

        feature_cols = [c for c in panel_df.columns if any(c.startswith(p) for p in Config.FEATURE_PREFIXES)]
        panel_df[feature_cols] = panel_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
        panel_df = panel_df.reset_index()

        with open(cache_path, 'wb') as f:
            pickle.dump((panel_df, feature_cols), f)

        return panel_df, feature_cols

    @staticmethod
    def make_dataset(panel_df, feature_cols):
        """
        [CRITICAL UPDATE] 实现 Train/Valid/Test 三段式严格切分，带 Gap 防止泄漏
        """
        print(">>> [Dataset] 转换张量格式 (Train/Valid/Test Split)...")
        panel_df = panel_df.sort_values(['code', 'date']).reset_index(drop=True)

        feature_matrix = panel_df[feature_cols].values.astype(np.float32)
        if 'rank_label' in panel_df.columns:
            target_array = panel_df['rank_label'].fillna(0.5).values.astype(np.float32)
        else:
            target_array = panel_df['target'].fillna(0).values.astype(np.float32)

        universe_mask = panel_df['is_universe'].values
        dates = panel_df['date'].values
        codes = panel_df['code'].values

        # 1. 生成滑动窗口索引
        code_changes = np.where(codes[:-1] != codes[1:])[0] + 1
        start_indices = np.concatenate(([0], code_changes))
        end_indices = np.concatenate((code_changes, [len(codes)]))

        valid_indices = []
        seq_len = Config.CONTEXT_LEN
        stride = Config.STRIDE

        for start, end in zip(start_indices, end_indices):
            if end - start <= seq_len: continue
            for i in range(start + seq_len - 1, end, stride):
                if universe_mask[i]:
                    valid_indices.append(i - seq_len + 1)

        valid_indices = np.array(valid_indices)

        # 2. 基于时间的切分 (Train / Valid / Test)
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)

        # Config 中定义了 TRAIN_RATIO (0.8), VAL_RATIO (0.1), TEST_RATIO (0.1)
        train_end_idx = int(n_dates * Config.TRAIN_RATIO)
        val_end_idx = int(n_dates * (Config.TRAIN_RATIO + Config.VAL_RATIO))

        # 切分日期点
        train_date_limit = unique_dates[train_end_idx]

        # Gap: 防止 Train 尾部样本看到 Valid 头部样本的未来
        # Valid Start = Train End + Context Length
        val_start_idx = min(train_end_idx + Config.CONTEXT_LEN, n_dates - 1)
        val_start_date = unique_dates[val_start_idx]
        val_date_limit = unique_dates[val_end_idx]

        # Test Start = Valid End + Context Length
        test_start_idx = min(val_end_idx + Config.CONTEXT_LEN, n_dates - 1)
        test_start_date = unique_dates[test_start_idx]

        print(f"\n📊 数据集切分详情 (Gap={Config.CONTEXT_LEN} days):")
        print(f"   Train : {unique_dates[0]} ~ {train_date_limit}")
        print(f"   Valid : {val_start_date} ~ {val_date_limit}")
        print(f"   Test  : {test_start_date} ~ {unique_dates[-1]}")

        # 获取样本对应的预测时间点
        sample_pred_dates = dates[valid_indices + seq_len - 1]

        # 生成 Mask
        train_mask = sample_pred_dates < train_date_limit
        valid_mask = (sample_pred_dates >= val_start_date) & (sample_pred_dates < val_date_limit)
        test_mask = sample_pred_dates >= test_start_date

        idx_train = valid_indices[train_mask]
        idx_valid = valid_indices[valid_mask]
        idx_test = valid_indices[test_mask]

        print(f"   样本数 : Train={len(idx_train)}, Valid={len(idx_valid)}, Test={len(idx_test)}")

        # 3. 生成器工厂
        def create_gen(indices, shuffle=False):
            def _gen():
                if shuffle: np.random.shuffle(indices)
                for start_idx in indices:
                    end_idx = start_idx + seq_len
                    yield {
                        "past_values": feature_matrix[start_idx: end_idx],
                        "labels": target_array[end_idx - 1]
                    }

            return _gen

        # 4. 返回 DatasetDict
        ds = DatasetDict({
            'train': Dataset.from_generator(create_gen(idx_train, shuffle=True)),
            'validation': Dataset.from_generator(create_gen(idx_valid, shuffle=False)),
            'test': Dataset.from_generator(create_gen(idx_test, shuffle=False))
        })
        return ds, len(feature_cols)


def get_dataset(force_refresh=False):
    """对外接口"""
    panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train', force_refresh=force_refresh)
    ds, num_features = DataProvider.make_dataset(panel_df, feature_cols)
    return ds, num_features