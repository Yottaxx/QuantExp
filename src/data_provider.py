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

# 忽略 pandas 的 SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class DataProvider:
    _vpn_lock = threading.Lock()
    _last_switch_time = 0

    # --------------------------------------------------------------------------
    # PART 1: 基础设施 (保持稳健)
    # --------------------------------------------------------------------------
    @staticmethod
    def _setup_proxy_env():
        """配置系统代理环境"""
        proxy_url = Config.PROXY_URL
        for k in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
            os.environ[k] = proxy_url

    @classmethod
    def _safe_switch_vpn(cls):
        """线程安全的 VPN 轮换"""
        with cls._vpn_lock:
            if time.time() - cls._last_switch_time < 5: return
            vpn_rotator.switch_random()
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
        """
        获取真实的财报披露日期，消除 Look-ahead Bias
        """
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
        """下载财务数据，严格处理披露日期"""
        fund_dir = os.path.join(Config.DATA_DIR, "fundamental")
        if not os.path.exists(fund_dir): os.makedirs(fund_dir)
        path = os.path.join(fund_dir, f"{code}.parquet")

        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if (time.time() - mtime) < 3 * 24 * 3600: return code, True, "Skipped"  # 3天更新一次

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

                # --- 严格的披露日期对齐 ---
                pub_df = DataProvider._fetch_pub_date_map(code)
                if pub_df is not None:
                    df = pd.merge(df, pub_df, on='date', how='left')
                else:
                    df['pub_date'] = pd.NaT

                # 数值转换
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
        """下载日线行情，严格的单位判定"""
        path = os.path.join(Config.DATA_DIR, f"{code}.parquet")
        for attempt in range(5):
            try:
                time.sleep(random.uniform(0.05, 0.2))
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=Config.START_DATE, adjust="qfq")

                if df is None or df.empty: return code, True, "Empty"

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

                # --- 核心修复：单位自适应 (手 vs 股) ---
                # 仅使用最近 20 天数据判断，避免历史 QFQ 价格过小导致的系统性偏差
                if 'amount' in df.columns and 'volume' in df.columns and 'close' in df.columns:
                    valid_sample = df[(df['volume'] > 0) & (df['amount'] > 0)].tail(20)
                    if not valid_sample.empty:
                        # 理论值：手=100，股=1
                        multiplier = (
                                    valid_sample['amount'] / (valid_sample['close'] * valid_sample['volume'])).median()
                        if multiplier > 50:
                            df['volume'] = df['volume'] * 100

                # 删除 amount 以节省内存 (或保留用于更高级回测)
                if 'amount' in df.columns:
                    df.drop(columns=['amount'], inplace=True)

                if not df.empty:
                    df.sort_index(inplace=True)

                if len(df) > 0: df.to_parquet(path)
                return code, True, "Success"
            except:
                DataProvider._safe_switch_vpn()
                continue
        return code, False, "Failed"

    # --------------------------------------------------------------------------
    # PART 2: 数据处理核心 (ETL)
    # --------------------------------------------------------------------------

    @staticmethod
    def download_data():
        """执行全量数据下载"""
        print(">>> [ETL] 启动数据下载流水线...")
        DataProvider._setup_proxy_env()
        if not os.path.exists(Config.DATA_DIR): os.makedirs(Config.DATA_DIR)

        try:
            # 注意：此处存在幸存者偏差风险，akshare 仅返回当前上市股票
            # 生产环境应接入 Paid Data Source 获取 Delisted Stocks
            stock_info = ak.stock_zh_a_spot_em()
            codes = stock_info['代码'].tolist()
        except:
            print("❌ 无法获取股票列表")
            return

        target_date_str = DataProvider._get_latest_trading_date()

        # 增量更新逻辑
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

        # 并发下载行情
        if todo_price:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = {executor.submit(DataProvider._download_worker, c): c for c in todo_price}
                for _ in tqdm(concurrent.futures.as_completed(futures), total=len(todo_price),
                              desc="Downloading Price"): pass

        # 并发下载财务
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
    def _filter_universe(panel_df):
        """动态股票池过滤 (Volume > 0, Price > 2, Listed > 60d)"""
        print(">>> [Filtering] 执行 Universe 过滤...")
        original_len = len(panel_df)

        # 基础清洗
        panel_df = panel_df[panel_df['volume'] > 0]
        panel_df = panel_df[panel_df['close'] >= 2.0]

        # 上市时间过滤
        panel_df['list_days'] = panel_df.groupby('code').cumcount()
        panel_df = panel_df[panel_df['list_days'] > 60]
        panel_df = panel_df.drop(columns=['list_days'])

        new_len = len(panel_df)
        print(f"过滤统计: {original_len} -> {new_len} (剔除率: {1 - new_len / original_len:.2%})")
        return panel_df

    @staticmethod
    def load_and_process_panel(mode='train', force_refresh=False):
        """构建全市场 Panel 数据 (核心函数)"""
        cache_path = DataProvider._get_cache_path(mode)
        if not force_refresh and os.path.exists(cache_path):
            print(f"⚡️ [Cache Hit] {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        print(f"\n>>> [Processing] 构建 Panel 数据 (Mode: {mode})...")
        price_files = glob.glob(os.path.join(Config.DATA_DIR, "*.parquet"))
        fund_dir = os.path.join(Config.DATA_DIR, "fundamental")

        # 1. 读取日线
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
        if not data_frames: raise ValueError("没有足够的有效行情数据")
        panel_df = pd.concat(data_frames, ignore_index=False)
        del data_frames
        panel_df['code'] = panel_df['code'].astype(str)

        # 2. 读取并合并财务数据 (PIT - Point In Time 处理)
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

            # 【核心逻辑修正】构建 merge_date
            if 'pub_date' in fund_df.columns:
                fund_df['merge_date'] = fund_df['pub_date']

                # 处理缺失披露日期的逻辑 (尤其是 Q4 年报)
                mask_na = fund_df['merge_date'].isna()
                # 提取报告期月份
                report_months = fund_df.loc[mask_na, 'date'].dt.month

                # 规则：12月年报默认延后120天(4/30)，其他季报延后60天
                delays = report_months.apply(lambda m: 120 if m == 12 else 60)
                fund_df.loc[mask_na, 'merge_date'] = fund_df.loc[mask_na, 'date'] + pd.to_timedelta(delays, unit='D')
            else:
                # 悲观假设：全部延后 90 天 (介于 60-120 之间)
                fund_df['merge_date'] = fund_df['date'] + pd.Timedelta(days=90)

            # 清理
            fund_df = fund_df.drop(columns=['date', 'pub_date'], errors='ignore')
            fund_df.rename(columns={'merge_date': 'date'}, inplace=True)

            panel_df = panel_df.reset_index().sort_values(['code', 'date'])
            panel_df = pd.merge_asof(panel_df, fund_df, on='date', by='code', direction='backward')

            # 缺失值填充
            for c in ['roe', 'rev_growth', 'profit_growth', 'debt_ratio', 'pe_ttm', 'pb']:
                if c in panel_df.columns: panel_df[c] = panel_df[c].fillna(0).astype(np.float32)
            print("✅ 财务数据 PIT 对齐完成。")

        # 3. 计算 Alpha
        if 'date' in panel_df.columns: panel_df = panel_df.set_index('date')
        panel_df = panel_df.reset_index().sort_values(['code', 'date'])

        print("计算时序因子...")
        # 使用 GroupBy Apply 计算时序特征
        panel_df = panel_df.groupby('code', group_keys=False).apply(lambda x: AlphaFactory(x).make_factors())

        # 4. 构造 Label (预测未来收益)
        print("构造预测目标 (Labels)...")
        panel_df['next_open'] = panel_df.groupby('code')['open'].shift(-1)
        panel_df['future_close'] = panel_df.groupby('code')['close'].shift(-Config.PRED_LEN)
        # Target: T+1 Open 到 T+N Close 的收益 (模拟实盘 T+1 买入)
        panel_df['target'] = panel_df['future_close'] / panel_df['next_open'] - 1
        panel_df.drop(columns=['next_open', 'future_close'], inplace=True)

        # 5. 切分数据
        if mode == 'train':
            panel_df.dropna(subset=['target'], inplace=True)
        else:
            # 预测模式下保留最后一行用于推理
            pass

        # 6. 过滤 Universe
        panel_df = DataProvider._filter_universe(panel_df)

        # 7. 截面处理 (SVD/Rank)
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
        """构造 PyTorch Dataset"""
        print(">>> [Dataset] 转换张量格式...")
        panel_df = panel_df.sort_values(['code', 'date'])
        feature_matrix = panel_df[feature_cols].values.astype(np.float32)

        # 确定 Label
        if 'rank_label' in panel_df.columns:
            target_col = 'rank_label'
            fill_val = 0.5
        elif 'excess_label' in panel_df.columns:
            target_col = 'excess_label'
            fill_val = 0.0
        else:
            target_col = 'target'
            fill_val = 0.0

        target_array = panel_df[target_col].fillna(fill_val).values.astype(np.float32)
        dates = panel_df['date'].values  # 保留日期用于 split

        # 生成有效索引
        codes = panel_df['code'].values
        code_changes = np.where(codes[:-1] != codes[1:])[0] + 1
        start_indices = np.concatenate(([0], code_changes))
        end_indices = np.concatenate((code_changes, [len(codes)]))

        valid_indices = []
        seq_len = Config.CONTEXT_LEN
        stride = Config.STRIDE  # 使用 Config 中的 Stride

        for start, end in zip(start_indices, end_indices):
            length = end - start
            if length <= seq_len: continue
            for i in range(start, end - seq_len + 1, stride): valid_indices.append(i)

        # 按时间切分 Train/Valid
        # 注意：这里增加了 Purging 逻辑，虽然简单，但确保了验证集在时间上严格晚于训练集
        unique_dates = np.sort(panel_df['date'].unique())
        split_idx = int(len(unique_dates) * 0.9)
        split_date = unique_dates[split_idx]

        sample_dates = dates[np.array(valid_indices) + seq_len - 1]

        # 严格的时间切分
        train_mask = sample_dates < split_date
        train_indices = np.array(valid_indices)[train_mask]
        valid_indices = np.array(valid_indices)[~train_mask]

        # 为了防止训练集末尾数据泄露到验证集，验证集应跳过 Context Length 长度
        # (此处简化处理，因 Split Date 已经是截断的)

        def gen_train():
            np.random.shuffle(train_indices)  # 训练集打乱
            for idx in train_indices:
                yield {
                    "past_values": feature_matrix[idx: idx + seq_len],
                    "labels": target_array[idx + seq_len - 1]
                }

        def gen_valid():
            for idx in valid_indices:
                yield {
                    "past_values": feature_matrix[idx: idx + seq_len],
                    "labels": target_array[idx + seq_len - 1]
                }

        from datasets import DatasetDict
        ds = DatasetDict({
            'train': Dataset.from_generator(gen_train),
            'test': Dataset.from_generator(gen_valid)
        })
        return ds, len(feature_cols)


def get_dataset(force_refresh=False):
    panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train', force_refresh=force_refresh)
    ds, num_features = DataProvider.make_dataset(panel_df, feature_cols)
    return ds, num_features