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
from datasets import Dataset
from tqdm import tqdm
from .config import Config
from .vpn_rotator import vpn_rotator
from .alpha_lib import AlphaFactory
from pandarallel import pandarallel

# 忽略性能警告
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# 1. 初始化并行环境 (progress_bar=True 开启进度条)
# nb_workers 根据你的CPU核数调整，默认是全部
pandarallel.initialize(progress_bar=True, nb_workers=os.cpu_count())


class DataProvider:
    _vpn_lock = threading.Lock()
    _last_switch_time = 0

    @staticmethod
    def _setup_proxy_env():
        """设置代理环境"""
        proxy_url = Config.PROXY_URL
        for k in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
            os.environ[k] = proxy_url

    @classmethod
    def _safe_switch_vpn(cls):
        """线程安全的 VPN 切换"""
        with cls._vpn_lock:
            # 防止切换过于频繁，冷却时间 5 秒
            if time.time() - cls._last_switch_time < 5:
                return
            try:
                vpn_rotator.switch_random()
            except Exception as e:
                print(f"VPN Switch Warning: {e}")
            cls._last_switch_time = time.time()
            time.sleep(2)

    @staticmethod
    def _get_latest_trading_date():
        """获取最近交易日"""
        try:
            df = ak.stock_zh_index_daily(symbol=Config.MARKET_INDEX_SYMBOL)
            return pd.to_datetime(df['date']).max().date().strftime("%Y-%m-%d")
        except:
            return datetime.date.today().strftime("%Y-%m-%d")

    @staticmethod
    def _fetch_pub_date_map(code):
        """获取财报公告日期，用于 PIT (Point-in-Time) 对齐"""
        try:
            df = ak.stock_financial_abstract(symbol=code)
            if df is None or df.empty:
                return None

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
        """下载单只股票财务数据"""
        fund_dir = os.path.join(Config.DATA_DIR, "fundamental")
        if not os.path.exists(fund_dir):
            os.makedirs(fund_dir)
        path = os.path.join(fund_dir, f"{code}.parquet")

        # 增量更新检查：如果是最近 3 天内更新过的，跳过
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if (time.time() - mtime) < 3 * 24 * 3600:
                return code, True, "Skipped"

        for attempt in range(3):
            try:
                time.sleep(random.uniform(0.1, 0.5))
                df = ak.stock_financial_analysis_indicator_em(symbol=code)
                if df is None or df.empty:
                    return code, True, "Empty"

                df['date'] = pd.to_datetime(df['日期'])
                cols_map = {
                    '加权净资产收益率': 'roe',
                    '主营业务收入增长率(%)': 'rev_growth',
                    '净利润增长率(%)': 'profit_growth',
                    '资产负债率(%)': 'debt_ratio',
                    '市盈率(动态)': 'pe_ttm',
                    '市净率': 'pb'
                }
                valid_cols = [c for c in cols_map.keys() if c in df.columns]
                df = df[['date'] + valid_cols].copy()
                df.rename(columns=cols_map, inplace=True)

                # 获取公告日进行合并
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
        """下载单只股票日线行情"""
        path = os.path.join(Config.DATA_DIR, f"{code}.parquet")
        for attempt in range(5):
            try:
                time.sleep(random.uniform(0.05, 0.2))
                # 使用前复权数据
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=Config.START_DATE, adjust="qfq")

                if df is None or df.empty:
                    return code, True, "Empty"

                df.rename(columns={
                    '日期': 'date', '开盘': 'open', '收盘': 'close',
                    '最高': 'high', '最低': 'low',
                    '成交量': 'volume', '成交额': 'amount'
                }, inplace=True)

                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

                for col in ['open', 'close', 'high', 'low', 'volume', 'amount']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

                df.dropna(inplace=True)

                # 简单清洗：处理 volume 单位异常 (部分接口返回的是手，部分是股)
                if 'amount' in df.columns and 'volume' in df.columns and 'close' in df.columns:
                    valid_sample = df[(df['volume'] > 0) & (df['amount'] > 0)].tail(20)
                    if not valid_sample.empty:
                        # 估算均价
                        multiplier = (
                                    valid_sample['amount'] / (valid_sample['close'] * valid_sample['volume'])).median()
                        # 如果 multiplier 接近 100，说明 volume 是手，需要乘 100
                        if multiplier > 50:
                            df['volume'] = df['volume'] * 100

                if 'amount' in df.columns:
                    df.drop(columns=['amount'], inplace=True)

                if not df.empty:
                    df.sort_index(inplace=True)

                if len(df) > 0:
                    df.to_parquet(path)
                return code, True, "Success"
            except:
                DataProvider._safe_switch_vpn()
                continue
        return code, False, "Failed"

    @staticmethod
    def download_data():
        """主下载入口"""
        print(">>> [ETL] 启动数据下载流水线...")
        DataProvider._setup_proxy_env()
        if not os.path.exists(Config.DATA_DIR):
            os.makedirs(Config.DATA_DIR)

        try:
            stock_info = ak.stock_zh_a_spot_em()
            codes = stock_info['代码'].tolist()
        except:
            print("❌ 无法获取股票列表，请检查网络或代理配置")
            return

        target_date_str = DataProvider._get_latest_trading_date()

        # 增量检查
        existing_fresh = set()
        for fname in os.listdir(Config.DATA_DIR):
            if fname.endswith(".parquet"):
                fpath = os.path.join(Config.DATA_DIR, fname)
                if os.path.getsize(fpath) > 1024:
                    mtime = os.path.getmtime(fpath)
                    file_date = datetime.date.fromtimestamp(mtime).strftime("%Y-%m-%d")
                    if file_date >= target_date_str:
                        existing_fresh.add(fname.replace(".parquet", ""))

        todo_price = sorted(list(set(codes) - existing_fresh))

        print(f"📊 股票池总数: {len(codes)} | 待更新: {len(todo_price)}")

        # 下载行情
        if todo_price:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = {executor.submit(DataProvider._download_worker, c): c for c in todo_price}
                for _ in tqdm(concurrent.futures.as_completed(futures), total=len(todo_price),
                              desc="Downloading Price"):
                    pass

        # 下载财务
        print("正在同步财务数据...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(DataProvider._download_finance_worker, c): c for c in codes}
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(codes), desc="Downloading Finance"):
                pass

        print("✅ 数据同步完成。")

    @staticmethod
    def _get_cache_path(mode):
        today_str = datetime.date.today().strftime("%Y%m%d")
        return os.path.join(Config.OUTPUT_DIR, f"panel_cache_{mode}_{today_str}.pkl")

    @staticmethod
    def _tag_universe(panel_df):
        """
        [Tagging] 标记动态股票池
        CRITICAL FIX: 修复未来数据泄漏
        """
        print(">>> [Tagging] 标记动态股票池 (Universe Mask)...")

        # 1. 确保按日期排序
        panel_df = panel_df.sort_values(['code', 'date'])

        # 2. 计算累计上市天数 (Expanding Window)
        # 错误写法: transform('count') -> 看了未来数据
        # 正确写法: cumcount() -> 只看过去数据
        panel_df['list_days_count'] = panel_df.groupby('code')['date'].cumcount() + 1

        # 3. 筛选条件
        cond_vol = panel_df['volume'] > 0
        cond_price = panel_df['close'] >= 2.0
        # 必须上市超过 60 天才纳入 (剔除次新股)
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

        # 并行读取行情
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(_read_price, price_files), total=len(price_files), desc="Reading Price"))

        data_frames = [df for df in results if df is not None and len(df) > Config.CONTEXT_LEN]
        if not data_frames:
            raise ValueError("没有足够的有效行情数据，请先运行 download_data()")

        panel_df = pd.concat(data_frames, ignore_index=True)
        del data_frames  # 释放内存

        panel_df['code'] = panel_df['code'].astype(str)
        panel_df['date'] = pd.to_datetime(panel_df['index'] if 'index' in panel_df.columns else panel_df['date'])

        # 读取财务数据
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

            # PIT 对齐逻辑
            if 'pub_date' in fund_df.columns:
                # 优先使用公告日作为 merge_date
                fund_df['merge_date'] = fund_df['pub_date']
                # 如果公告日缺失，回退到 报告期 + 滞后时间
                mask_na = fund_df['merge_date'].isna()
                report_months = fund_df.loc[mask_na, 'date'].dt.month
                delays = report_months.apply(lambda m: 120 if m == 12 else 60)  # 年报延迟长，季报延迟短
                fund_df.loc[mask_na, 'merge_date'] = fund_df.loc[mask_na, 'date'] + pd.to_timedelta(delays, unit='D')
            else:
                fund_df['merge_date'] = fund_df['date'] + pd.Timedelta(days=90)

            fund_df = fund_df.drop(columns=['date', 'pub_date'], errors='ignore')
            fund_df.rename(columns={'merge_date': 'date'}, inplace=True)

            panel_df = panel_df.reset_index().sort_values(['code', 'date'])
            # 使用 merge_asof 进行 PIT 合并 (Backward direction)
            panel_df = pd.merge_asof(panel_df, fund_df, on='date', by='code', direction='backward')

            # 填充财务缺失值
            for c in ['roe', 'rev_growth', 'profit_growth', 'debt_ratio', 'pe_ttm', 'pb']:
                if c in panel_df.columns:
                    panel_df[c] = panel_df[c].fillna(0).astype(np.float32)
            print("✅ 财务数据 PIT 对齐完成。")

        if 'date' in panel_df.columns:
            panel_df = panel_df.set_index('date')

        panel_df = panel_df.reset_index().sort_values(['code', 'date'])

        # 计算时序因子
        print("计算时序因子...")
        # 使用 pandarallel 进行并行计算
        panel_df = panel_df.groupby('code', group_keys=False).parallel_apply(lambda x: AlphaFactory(x).make_factors())

        # 构造预测目标 (Labels)
        print("构造预测目标 (Labels)...")
        panel_df['next_open'] = panel_df.groupby('code')['open'].shift(-1)
        panel_df['future_close'] = panel_df.groupby('code')['close'].shift(-Config.PRED_LEN)
        panel_df['target'] = panel_df['future_close'] / panel_df['next_open'] - 1
        panel_df.drop(columns=['next_open', 'future_close'], inplace=True)

        if mode == 'train':
            # 训练模式下，必须去除 Label 为空的行
            panel_df.dropna(subset=['target'], inplace=True)

        # 标记 Universe (包含 Future Leakage Fix)
        panel_df = DataProvider._tag_universe(panel_df)

        # 计算截面因子与标准化
        print("计算截面因子与标准化...")
        panel_df = panel_df.set_index('date')
        panel_df = AlphaFactory.add_cross_sectional_factors(panel_df)

        # 提取特征列
        feature_cols = [c for c in panel_df.columns
                        if any(c.startswith(p) for p in Config.FEATURE_PREFIXES)]

        # 最终清洗
        panel_df[feature_cols] = panel_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
        panel_df = panel_df.reset_index()

        # 缓存
        with open(cache_path, 'wb') as f:
            pickle.dump((panel_df, feature_cols), f)

        return panel_df, feature_cols

    @staticmethod
    def make_dataset(panel_df, feature_cols):
        print(">>> [Dataset] 转换张量格式...")
        panel_df = panel_df.sort_values(['code', 'date']).reset_index(drop=True)

        feature_matrix = panel_df[feature_cols].values.astype(np.float32)
        if 'rank_label' in panel_df.columns:
            target_array = panel_df['rank_label'].fillna(0.5).values.astype(np.float32)
        else:
            target_array = panel_df['target'].fillna(0).values.astype(np.float32)

        universe_mask = panel_df['is_universe'].values
        dates = panel_df['date'].values
        codes = panel_df['code'].values

        # 快速定位每只股票的起始/结束位置
        code_changes = np.where(codes[:-1] != codes[1:])[0] + 1
        start_indices = np.concatenate(([0], code_changes))
        end_indices = np.concatenate((code_changes, [len(codes)]))

        valid_indices = []
        seq_len = Config.CONTEXT_LEN
        stride = Config.STRIDE

        # 生成有效的时间窗口索引
        for start, end in zip(start_indices, end_indices):
            if end - start <= seq_len:
                continue
            # 滑动窗口采样
            for i in range(start + seq_len - 1, end, stride):
                # 只有当预测点属于 Universe 时才加入样本
                if universe_mask[i]:
                    valid_indices.append(i - seq_len + 1)

        valid_indices = np.array(valid_indices)

        # 按时间切分 Train/Test (Time Series Split)
        unique_dates = np.sort(np.unique(dates))
        split_idx = int(len(unique_dates) * 0.9)
        split_date = unique_dates[split_idx]

        # 获取样本对应的预测日期 (T)
        sample_pred_dates = dates[valid_indices + seq_len - 1]

        train_mask = sample_pred_dates < split_date
        # Test 集需留出 Gap，防止数据重叠
        gap_date = unique_dates[min(split_idx + Config.CONTEXT_LEN, len(unique_dates) - 1)]
        test_mask = sample_pred_dates > gap_date

        train_indices = valid_indices[train_mask]
        test_indices = valid_indices[test_mask]

        print(f"样本分割: Train={len(train_indices)}, Test={len(test_indices)}")

        def gen_train():
            np.random.shuffle(train_indices)  # 训练集打乱
            for start_idx in train_indices:
                end_idx = start_idx + seq_len
                yield {
                    "past_values": feature_matrix[start_idx: end_idx],
                    "labels": target_array[end_idx - 1]
                }

        def gen_valid():
            for start_idx in test_indices:
                end_idx = start_idx + seq_len
                yield {
                    "past_values": feature_matrix[start_idx: end_idx],
                    "labels": target_array[end_idx - 1]
                }

        from datasets import DatasetDict
        ds = DatasetDict({
            'train': Dataset.from_generator(gen_train),
            'test': Dataset.from_generator(gen_valid)
        })
        return ds, len(feature_cols)


def get_dataset(force_refresh=False):
    """
    对外暴露的数据获取接口
    """
    panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train', force_refresh=force_refresh)
    ds, num_features = DataProvider.make_dataset(panel_df, feature_cols)
    return ds, num_features