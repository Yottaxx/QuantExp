import akshare as ak
import pandas as pd
import numpy as np
import os
import glob
import time
import random
import threading
import datetime
import concurrent.futures
import pickle
import warnings
import shutil
from typing import Tuple, List, Optional, Union, Dict
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from pandarallel import pandarallel

# 内部模块依赖
from .config import Config
from .vpn_rotator import vpn_rotator
from .alpha_lib import AlphaFactory

# --- 全局配置 ---
# 忽略 Pandas 的碎片化警告和性能警告
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# 初始化并行计算 (用于因子计算)
# verbose=0 静默启动
pandarallel.initialize(progress_bar=True, nb_workers=os.cpu_count(), verbose=0)


class DataProvider:
    """
    【SOTA Data Engine v11.0 - Industrial Grade】

    Architecture:
    1. ETL Layer: Atomic IO, Smart Caching, Deep Probing.
    2. Data Lake: Parquet with Snappy compression.
    3. Serving Layer: Zero-Copy Lazy Mapping (HuggingFace Arrow).
    """

    _vpn_lock = threading.Lock()
    _last_switch_time = 0

    # ==========================================================================
    # 1. 基础 I/O 与网络设施 (Infrastructure)
    # ==========================================================================

    @staticmethod
    def _setup_proxy_env():
        """配置系统级代理"""
        if Config.PROXY_URL:
            for k in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
                os.environ[k] = Config.PROXY_URL

    @classmethod
    def _safe_switch_vpn(cls):
        """线程安全的 VPN 轮询"""
        with cls._vpn_lock:
            if time.time() - cls._last_switch_time < 5: return
            try:
                # Debug 模式下可开启 print
                # print("🔄 [Network] Switching Proxy Node...")
                vpn_rotator.switch_random()
            except Exception:
                pass
            cls._last_switch_time = time.time()
            time.sleep(2)

    @staticmethod
    def _atomic_save(df: pd.DataFrame, file_path: str):
        """
        【原子写入】
        先写入 .tmp，再执行原子替换。防止进程崩溃导致 0 字节文件。
        """
        tmp_path = file_path + ".tmp"
        try:
            df.to_parquet(tmp_path, index=True)
            # POSIX 原子操作 (Windows Python 3.3+ 支持)
            os.replace(tmp_path, file_path)
        except Exception as e:
            if os.path.exists(tmp_path): os.remove(tmp_path)
            raise e

    @staticmethod
    def _is_data_fresh(file_path: str, target_date_str: str) -> bool:
        """
        【深度内容探针】Deep Content Probe
        只读取 Parquet 索引列 (IO 开销极小)，校验数据是否确实包含目标日期。
        """
        if not os.path.exists(file_path): return False
        if os.path.getsize(file_path) < 1024: return False

        try:
            # Column Pruning: 只读一列获取 Index
            df_meta = pd.read_parquet(file_path, columns=['close'])
            if df_meta.empty: return False

            last_dt = df_meta.index.max()
            last_date = last_dt.strftime("%Y-%m-%d") if isinstance(last_dt, pd.Timestamp) else str(last_dt)[:10]

            return last_date >= target_date_str
        except Exception:
            return False

    @staticmethod
    def _get_latest_trading_date() -> str:
        """获取全市场最新交易日 (Benchmark: SH000001)"""
        try:
            df = ak.stock_zh_index_daily(symbol=Config.MARKET_INDEX_SYMBOL)
            return pd.to_datetime(df['date']).max().strftime("%Y-%m-%d")
        except:
            return datetime.date.today().strftime("%Y-%m-%d")

    # ==========================================================================
    # 2. 下载工作流 (Workers)
    # ==========================================================================

    @staticmethod
    def _download_price_worker(code: str) -> Tuple[str, bool, str]:
        """日线行情下载器"""
        path = os.path.join(Config.DATA_DIR, f"{code}.parquet")

        for attempt in range(3):
            try:
                time.sleep(random.uniform(0.1, 0.4))  # Jitter

                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=Config.START_DATE, adjust="qfq")
                if df is None or df.empty: return code, True, "Empty"

                # 规范化
                rename_map = {'日期': 'date', '开盘': 'open', '收盘': 'close',
                              '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount','换手率': 'turnover'}
                df.rename(columns=rename_map, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

                # 类型压缩
                cols = ['open', 'close', 'high', 'low', 'volume', 'amount','turnover']
                for c in cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float32)

                # 单位清洗 (手 vs 股)
                if 'amount' in df.columns and 'volume' in df.columns:
                    sample = df[(df['volume'] > 0) & (df['amount'] > 0)].tail(10)
                    if not sample.empty:
                        vwap = sample['amount'] / sample['volume']
                        ratio = (vwap / sample['close']).median()
                        if 80 < ratio < 120:  # 接近 100
                            df['volume'] *= 100

                df = df[['open', 'high', 'low', 'close', 'volume','turnover']]
                df.sort_index(inplace=True)

                DataProvider._atomic_save(df, path)
                return code, True, "Success"

            except Exception:
                if attempt < 2: DataProvider._safe_switch_vpn()
                continue

        return code, False, "Failed"

    @staticmethod
    def _download_finance_worker(code: str) -> Tuple[str, bool, str]:
        """财务数据下载器 (智能缓存)"""
        fund_dir = os.path.join(Config.DATA_DIR, "fundamental")
        os.makedirs(fund_dir, exist_ok=True)
        path = os.path.join(fund_dir, f"{code}.parquet")

        # --- Smart Seasonality Logic ---
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            curr_month = datetime.date.today().month
            # 财报月(4,8,10)缓存12小时，平时缓存3天
            ttl_seconds = 12 * 3600 if curr_month in [4, 8, 10] else 72 * 3600

            if (time.time() - mtime) < ttl_seconds:
                return code, True, "Skipped (Cache)"

        for attempt in range(2):
            try:
                time.sleep(random.uniform(0.1, 0.5))
                df = ak.stock_financial_analysis_indicator_em(symbol=code)
                if df is None or df.empty: return code, True, "Empty"

                df['date'] = pd.to_datetime(df['日期'])

                col_map = {
                    '加权净资产收益率': 'roe', '主营业务收入增长率(%)': 'rev_growth',
                    '净利润增长率(%)': 'profit_growth', '资产负债率(%)': 'debt_ratio',
                    '市盈率(动态)': 'pe_ttm', '市净率': 'pb'
                }
                valid_cols = [c for c in col_map.keys() if c in df.columns]
                df = df[['date'] + valid_cols].copy()
                df.rename(columns=col_map, inplace=True)

                for c in df.columns:
                    if c != 'date':
                        df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float32)

                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)

                DataProvider._atomic_save(df, path)
                return code, True, "Success"
            except:
                DataProvider._safe_switch_vpn()
        return code, False, "Failed"

    # ==========================================================================
    # 3. ETL 主流程 (Pipeline)
    # ==========================================================================

    @staticmethod
    def download_data():
        """ETL Entry Point"""
        print(f"\n{'=' * 60}\n>>> [ETL] Data Pipeline Initiated\n{'=' * 60}")
        DataProvider._setup_proxy_env()
        os.makedirs(Config.DATA_DIR, exist_ok=True)

        # 1. Sync List
        try:
            print("☁️ Syncing Universe List...")
            stock_info = ak.stock_zh_a_spot_em()
            codes = stock_info['代码'].tolist()
        except Exception as e:
            print(f"❌ Critical Error: Failed to fetch stock list. {e}")
            return

        target_date = DataProvider._get_latest_trading_date()
        print(f"📅 Target Trading Date: {target_date}")

        # 2. Parallel Probe (Deep Integrity Scan)
        print("🔍 Probing Local Data Integrity...")

        def _check_task(c):
            fpath = os.path.join(Config.DATA_DIR, f"{c}.parquet")
            if not DataProvider._is_data_fresh(fpath, target_date):
                return c
            return None

        scan_workers = min(os.cpu_count() * 4, 64)
        with concurrent.futures.ThreadPoolExecutor(max_workers=scan_workers) as executor:
            results = list(tqdm(executor.map(_check_task, codes), total=len(codes), desc="Scanning"))

        todo_codes = [r for r in results if r is not None]
        print(f"📊 Status: Total={len(codes)} | Fresh={len(codes) - len(todo_codes)} | Stale={len(todo_codes)}")

        # 3. Download Execution
        if todo_codes:
            max_workers = 8 if len(todo_codes) < 500 else 16
            print(f"🚀 Launching Download Engine (Workers={max_workers})...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(DataProvider._download_price_worker, c): c for c in todo_codes}
                success_count = 0
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(todo_codes),
                                   desc="Downloading Price"):
                    try:
                        _, status, _ = future.result()
                        if status: success_count += 1
                    except:
                        pass
            print(f"✅ Price Sync Complete. Success: {success_count}/{len(todo_codes)}")
        else:
            print("✅ Market Data is Up-to-Date.")

        # 4. Finance Sync
        print("📋 Syncing Fundamental Data...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(DataProvider._download_finance_worker, c): c for c in codes}
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(codes), desc="Downloading Finance"):
                pass

        print("✅ ETL Pipeline Completed.")

    # ==========================================================================
    # 4. 数据加载与预处理 (Processing Layer)
    # ==========================================================================

    @staticmethod
    def _get_cache_path(mode):
        today = datetime.date.today().strftime("%Y%m%d")
        return os.path.join(Config.OUTPUT_DIR, f"panel_cache_{mode}_{today}.pkl")

    @staticmethod
    def _tag_universe(panel_df):
        """Universe Selection Logic"""
        print(">>> [Tagging] 标记动态股票池 (Universe Mask)...")
        panel_df = panel_df.sort_values(['code', 'date'])
        panel_df['list_days_count'] = panel_df.groupby('code')['date'].cumcount() + 1

        cond_vol = panel_df['volume'] > 0
        cond_price = panel_df['close'] >= 2.0
        cond_list = panel_df['list_days_count'] > 60

        panel_df['is_universe'] = cond_vol & cond_price & cond_list
        panel_df.drop(columns=['list_days_count'], inplace=True)
        return panel_df

    @staticmethod
    def load_and_process_panel(mode='train', force_refresh=False):
        """
        Build Panel Data: Merge -> Alpha -> Label -> Norm
        """
        cache_path = DataProvider._get_cache_path(mode)
        if not force_refresh and os.path.exists(cache_path):
            print(f"⚡️ [Cache Hit] Loading from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass

        print(f"\n>>> [Processing] Building Panel Data (Mode: {mode})...")

        # 1. Load Price
        price_files = glob.glob(os.path.join(Config.DATA_DIR, "*.parquet"))
        if not price_files:
            raise RuntimeError("❌ No data found! Run `python main.py --mode download` first.")

        def _read_pq(f):
            try:
                df = pd.read_parquet(f)
                if df.empty: return None
                df['code'] = os.path.basename(f).replace(".parquet", "")
                return df
            except:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, os.cpu_count() + 4)) as executor:
            dfs = list(tqdm(executor.map(_read_pq, price_files), total=len(price_files), desc="Reading Price"))

        valid_dfs = [d for d in dfs if d is not None and len(d) > Config.CONTEXT_LEN]
        if not valid_dfs: raise ValueError("Not enough valid data.")

        panel_df = pd.concat(valid_dfs, ignore_index=True)
        if 'date' not in panel_df.columns: panel_df = panel_df.reset_index().rename(columns={'index': 'date'})
        panel_df['date'] = pd.to_datetime(panel_df['date'])
        panel_df['code'] = panel_df['code'].astype(str)

        # Optimization: Downcast
        f_cols = panel_df.select_dtypes(include=['float64']).columns
        panel_df[f_cols] = panel_df[f_cols].astype(np.float32)

        # 2. Merge Fundamental (PIT)
        fund_dir = os.path.join(Config.DATA_DIR, "fundamental")
        fund_files = glob.glob(os.path.join(fund_dir, "*.parquet"))

        # --- Explicit Warning ---
        if not fund_files:
            print("\033[93m" + "=" * 60)
            print("⚠️  [WARNING] MISSING FUNDAMENTAL DATA!")
            print("   The model will lose all valuation (PE/PB) factors.")
            print("   Please run `python main.py --mode download`.")
            print("=" * 60 + "\033[0m")
        else:
            print(f"🔗 Merging Fundamental Data (PIT Mode)... Coverage: {len(fund_files)}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                funds = [f for f in ex.map(_read_pq, fund_files) if f is not None]

            if funds:
                fund_df = pd.concat(funds).sort_values(['code', 'date'])

                # Default visibility lag: 90 days
                fund_df['merge_date'] = fund_df['date'] + pd.Timedelta(days=90)
                if 'pub_date' in fund_df.columns:
                    fund_df['merge_date'] = fund_df['pub_date'].fillna(fund_df['merge_date'])

                fund_df = fund_df.drop(columns=['date', 'pub_date'], errors='ignore').rename(
                    columns={'merge_date': 'date'})

                panel_df = panel_df.sort_values(['code', 'date'])
                fund_df = fund_df.sort_values(['code', 'date'])

                panel_df = pd.merge_asof(panel_df, fund_df, on='date', by='code', direction='backward')

                # Keep NaN for fundamental factors here
                fin_cols = ['roe', 'rev_growth', 'profit_growth', 'debt_ratio', 'pe_ttm', 'pb']
                for c in fin_cols:
                    if c in panel_df.columns:
                        panel_df[c] = panel_df[c].astype(np.float32)

        # 3. Alpha Gen (Parallel)
        print("⚙️  Running AlphaFactory (Parallel)...")
        panel_df.set_index('date', inplace=True)
        panel_df = panel_df.groupby('code', group_keys=False).parallel_apply(lambda x: AlphaFactory(x).make_factors())
        panel_df.reset_index(inplace=True)

        # 4. Labeling
        print("🏷️  Generating Labels...")
        panel_df.sort_values(['code', 'date'], inplace=True)
        panel_df['next_open'] = panel_df.groupby('code')['open'].shift(-1)
        panel_df['future_close'] = panel_df.groupby('code')['close'].shift(-Config.PRED_LEN)
        panel_df['target'] = panel_df['future_close'] / panel_df['next_open'] - 1

        if mode == 'train':
            panel_df.dropna(subset=['target'], inplace=True)

        # Tag Universe
        panel_df = DataProvider._tag_universe(panel_df)

        # 5. CS Norm
        print("🌐 Cross-Sectional Normalization...")
        panel_df.set_index('date', inplace=True)
        panel_df = AlphaFactory.add_cross_sectional_factors(panel_df)
        panel_df.reset_index(inplace=True)

        # 6. Final Clean
        feat_cols = [c for c in panel_df.columns if any(c.startswith(p) for p in Config.FEATURE_PREFIXES)]

        # Fill NaN with 0 for Technical Factors.
        panel_df[feat_cols] = panel_df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)

        with open(cache_path, 'wb') as f:
            pickle.dump((panel_df, feat_cols), f)
        print(f"✅ Panel Ready. Shape: {panel_df.shape}")
        return panel_df, feat_cols

    # ==========================================================================
    # 5. 高性能数据集构建 (Lazy Mapping Layer)
    # ==========================================================================

    @staticmethod
    def make_dataset(panel_df, feature_cols):
        """
        【Zero-Copy Lazy Dataset】
        不再生成 Sample 对象，而是存储索引，使用 set_transform 动态切片。
        极度节省内存，且初始化极快。
        """
        print(">>> [Dataset] Constructing Lazy Mapping (Zero-Copy Mode)...")

        # 1. 内存锁定 (Memory Locking)
        panel_df = panel_df.sort_values(['code', 'date']).reset_index(drop=True)

        # 关键：转为 C-contiguous 内存块，这是高效切片的前提
        print("    > Locking features into contiguous memory block...")
        feature_matrix = np.ascontiguousarray(
            panel_df[feature_cols].values.astype(np.float32)
        )

        if 'rank_label' in panel_df.columns:
            target_array = panel_df['rank_label'].fillna(0.5).values.astype(np.float32)
        else:
            target_array = panel_df['target'].fillna(0).values.astype(np.float32)

        # 2. 索引计算 (Valid Index Calculation)
        universe_mask = panel_df['is_universe'].values
        dates = panel_df['date'].values
        codes = panel_df['code'].values

        # 快速向量化寻找切换点
        code_changes = np.where(codes[:-1] != codes[1:])[0] + 1
        start_indices = np.concatenate(([0], code_changes))
        end_indices = np.concatenate((code_changes, [len(codes)]))

        valid_start_indices = []
        seq_len = Config.CONTEXT_LEN
        stride = Config.STRIDE

        # 计算所有合法的 Window Start Index
        for start, end in zip(start_indices, end_indices):
            length = end - start
            if length <= seq_len: continue

            # 候选起点
            curr_starts = np.arange(start, end - seq_len + 1, stride)
            # 对应的预测点 (切片末尾)
            pred_indices = curr_starts + seq_len - 1

            # Universe 过滤
            mask = universe_mask[pred_indices]
            valid_start_indices.extend(curr_starts[mask])

        valid_start_indices = np.array(valid_start_indices, dtype=np.int64)

        # 3. 严格时间切分 (Strict Time Split)
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)

        train_end_idx = int(n_dates * Config.TRAIN_RATIO)
        val_end_idx = int(n_dates * (Config.TRAIN_RATIO + Config.VAL_RATIO))

        train_date_limit = unique_dates[train_end_idx]
        val_start_date = unique_dates[min(train_end_idx + Config.CONTEXT_LEN, n_dates - 1)]
        val_date_limit = unique_dates[val_end_idx]
        test_start_date = unique_dates[min(val_end_idx + Config.CONTEXT_LEN, n_dates - 1)]

        print(f"\n📊 Dataset Split (Gap={Config.CONTEXT_LEN} days):")
        print(f"   Train : ~ {train_date_limit}")
        print(f"   Valid : {val_start_date} ~ {val_date_limit}")
        print(f"   Test  : {test_start_date} ~")

        # 映射回日期进行筛选
        sample_pred_dates = dates[valid_start_indices + seq_len - 1]

        idx_train = valid_start_indices[sample_pred_dates < train_date_limit]

        valid_mask = (sample_pred_dates >= val_start_date) & (sample_pred_dates < val_date_limit)
        idx_valid = valid_start_indices[valid_mask]

        idx_test = valid_start_indices[sample_pred_dates >= test_start_date]

        print(f"   Samples: Train={len(idx_train)}, Valid={len(idx_valid)}, Test={len(idx_test)}")

        # 4. 闭包 Transform 函数 (Lazy Loader)
        # 此函数在 DataLoader Worker 中被调用
        def lazy_transform(batch):
            """
            batch: {'start_idx': [id1, id2, ...]}
            """
            start_idxs = batch['start_idx']

            past_values = []
            labels = []

            for start in start_idxs:
                end = start + seq_len
                # 这里只产生 View 或极小的 Copy，利用 Shared Memory
                past_values.append(feature_matrix[start:end])
                labels.append(target_array[end - 1])

            return {
                "past_values": past_values,
                "labels": labels
            }

        # 5. 构建 Light-weight Dataset
        ds = DatasetDict({
            'train': Dataset.from_dict({'start_idx': idx_train}),
            'validation': Dataset.from_dict({'start_idx': idx_valid}),
            'test': Dataset.from_dict({'start_idx': idx_test})
        })

        # 注册 On-the-fly Transform
        ds.set_transform(lazy_transform)

        return ds, len(feature_cols)


def get_dataset(force_refresh=False):
    """External API"""
    panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train', force_refresh=force_refresh)
    return DataProvider.make_dataset(panel_df, feature_cols)