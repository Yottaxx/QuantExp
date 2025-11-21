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
from .alpha_lib import AlphaFactory


class DataProvider:
    _vpn_lock = threading.Lock()
    _last_switch_time = 0

    # --------------------------------------------------------------------------
    # PART 1: 基础设施与辅助函数
    # --------------------------------------------------------------------------

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
        """
        【新增】获取最近的一个交易日
        防止周末/节假日运行脚本时重复下载
        """
        try:
            # 获取上证指数的最新日线数据作为参考
            # 这里的 symbol 是 sh000001 (上证指数)
            df = ak.stock_zh_index_daily(symbol="sh000001")
            latest_date = pd.to_datetime(df['date']).max().date()
            return latest_date.strftime("%Y-%m-%d")
        except:
            # 如果获取失败，退化为使用今天
            return datetime.date.today().strftime("%Y-%m-%d")

    # --------------------------------------------------------------------------
    # PART 2: 下载模块 (Memory & Calendar Optimized)
    # --------------------------------------------------------------------------

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

                # 【优化】存盘前转为 float32 以节省磁盘空间
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    if col in df.columns:
                        df[col] = df[col].astype(np.float32)

                if len(df) > 0: df.to_parquet(path)
                return code, True, "Success"
            except:
                DataProvider._safe_switch_vpn()
                continue
        return code, False, "Failed"

    @staticmethod
    def download_data():
        print(">>> [Phase 1] 启动数据下载 (智能增量模式)...")
        DataProvider._setup_proxy_env()

        if not os.path.exists(Config.DATA_DIR): os.makedirs(Config.DATA_DIR)

        try:
            stock_info = ak.stock_zh_a_spot_em()
            codes = stock_info['代码'].tolist()
        except:
            print("❌ 无法获取股票列表，请检查网络/VPN")
            return

        # 1. 获取市场最新的交易日 (例如今天是周六，target_date 应该是周五)
        print(">>> 正在校对交易日历...")
        target_date_str = DataProvider._get_latest_trading_date()
        print(f"📅 最近交易日锁定为: {target_date_str}")

        # 2. 智能过滤
        existing_fresh = set()
        files = os.listdir(Config.DATA_DIR)

        for fname in files:
            if fname.endswith(".parquet"):
                fpath = os.path.join(Config.DATA_DIR, fname)
                # 检查1: 文件不为空
                if os.path.getsize(fpath) > 1024:
                    # 检查2: 修改时间 >= 目标交易日
                    # 只要文件的修改日期是在目标交易日之后(含)，说明包含了最新数据
                    mtime = os.path.getmtime(fpath)
                    file_date = datetime.date.fromtimestamp(mtime).strftime("%Y-%m-%d")
                    if file_date >= target_date_str:
                        existing_fresh.add(fname.replace(".parquet", ""))

        todo = list(set(codes) - existing_fresh)
        todo.sort()

        print(f"📊 任务: 总数 {len(codes)} | 已是最新 {len(existing_fresh)} | 待更新 {len(todo)}")
        if not todo:
            print("✅ 所有数据已同步至最新交易日。")
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(DataProvider._download_worker, c): c for c in todo}
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(todo)):
                pass
        print("下载完成。")

    # --------------------------------------------------------------------------
    # PART 3: 内存 Panel 处理 (Float32 内存优化版)
    # --------------------------------------------------------------------------

    @staticmethod
    def _filter_universe(panel_df):
        print(">>> [Filtering] 动态过滤...")
        original_len = len(panel_df)
        panel_df = panel_df[panel_df['volume'] > 0]
        panel_df = panel_df[panel_df['close'] >= 2.0]
        # 使用 transform 替代 groupby.cumcount 稍微快一点点，或者保持原样
        panel_df['list_days'] = panel_df.groupby('code')['close'].transform('count')
        # 注意：上面的 list_days 逻辑变了，变成总天数，这不对。
        # 还是保持 cumcount 正确
        panel_df['list_days'] = panel_df.groupby('code').cumcount()

        panel_df = panel_df[panel_df['list_days'] > 60]
        panel_df = panel_df.drop(columns=['list_days'])
        print(f"过滤移除: {original_len - new_len} ({1 - len(panel_df) / original_len:.2%})")
        return panel_df

    @staticmethod
    def load_and_process_panel(mode='train'):
        print(f"\n>>> [Phase 2] 构建全内存 Panel (Mode: {mode}, Opt: Float32)...")

        files = glob.glob(os.path.join(Config.DATA_DIR, "*.parquet"))
        if not files: raise ValueError("无数据文件")

        print(f"正在加载 {len(files)} 个文件...")

        def _read_helper(f):
            try:
                # 【优化】读取时直接指定列类型，大幅减少内存开销
                df = pd.read_parquet(f)
                code = os.path.basename(f).replace(".parquet", "")
                # 强制转 float32
                float_cols = df.select_dtypes(include=['float64']).columns
                df[float_cols] = df[float_cols].astype(np.float32)
                df['code'] = code
                # 将 code 转为 category 类型进一步省内存
                df['code'] = df['code'].astype('category')
                return df
            except:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(_read_helper, files), total=len(files), desc="Reading"))

        data_frames = [df for df in results if df is not None and len(df) > Config.CONTEXT_LEN + 10]
        if not data_frames: raise ValueError("有效数据为空")

        print("合并 DataFrame...")
        panel_df = pd.concat(data_frames, ignore_index=False)
        del data_frames

        # 恢复 code 为 string 以便后续处理 (pandas category 在 groupby apply 有时会有坑)
        panel_df['code'] = panel_df['code'].astype(str)
        panel_df = panel_df.reset_index().sort_values(['code', 'date'])

        print("计算时序因子...")
        # 这里的 apply 依然是内存瓶颈，无法避免，但由于输入已经是 float32，会好很多
        panel_df = panel_df.groupby('code', group_keys=False).apply(lambda x: AlphaFactory(x).make_factors())

        print("构造 Target...")
        panel_df['target'] = panel_df.groupby('code')['close'].shift(-Config.PRED_LEN) / panel_df['close'] - 1

        if mode == 'train':
            panel_df.dropna(subset=['target'], inplace=True)

        # 动态过滤
        print("动态过滤...")
        # 复用代码逻辑...
        original_len = len(panel_df)
        panel_df = panel_df[panel_df['volume'] > 0]
        panel_df = panel_df[panel_df['close'] >= 2.0]
        panel_df['list_days'] = panel_df.groupby('code').cumcount()
        panel_df = panel_df[panel_df['list_days'] > 60]
        panel_df.drop(columns=['list_days'], inplace=True)
        print(f"过滤移除: {original_len - len(panel_df)}")

        # 截面因子
        print("计算截面因子...")
        panel_df = panel_df.set_index('date')
        panel_df = AlphaFactory.add_cross_sectional_factors(panel_df)

        feature_cols = [c for c in panel_df.columns
                        if any(c.startswith(p) for p in ['style_', 'tech_', 'alpha_', 'adv_', 'cs_rank_'])]

        # 最终转 float32
        panel_df[feature_cols] = panel_df[feature_cols].fillna(0).astype(np.float32)

        panel_df = panel_df.reset_index()
        return panel_df, feature_cols

    # ... [make_dataset 等保持不变] ...
    @staticmethod
    def make_dataset(panel_df, feature_cols):
        print(">>> [Phase 3] 转换 Dataset...")
        panel_df = panel_df.sort_values(['code', 'date'])

        feature_matrix = panel_df[feature_cols].values  # 已经是 float32

        target_col = 'excess_label' if 'excess_label' in panel_df.columns else 'target'
        target_array = panel_df[target_col].fillna(0).values.astype(np.float32)

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
            for i in range(start, end - seq_len + 1, stride):
                valid_indices.append(i)

        print(f"样本数: {len(valid_indices)}")

        # 时间切分 (Time-Series Split)
        dates = panel_df['date'].unique()
        dates.sort()
        split_idx = int(len(dates) * 0.9)
        split_date = dates[split_idx]
        print(f"切分日期: {split_date}")

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
        ds = DatasetDict({
            'train': Dataset.from_generator(gen_train),
            'test': Dataset.from_generator(gen_valid)
        })

        return ds, len(feature_cols)


def get_dataset():
    panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train')
    ds, num_features = DataProvider.make_dataset(panel_df, feature_cols)
    return ds, num_features