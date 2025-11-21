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

    # ... [PART 1: _setup_proxy_env, _safe_switch_vpn, _download_worker, download_data 保持不变] ...
    # 请直接保留之前的下载代码
    @staticmethod
    def _setup_proxy_env():
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
            if time.time() - cls._last_switch_time < 5: return
            vpn_rotator.switch_random()
            cls._last_switch_time = time.time()
            time.sleep(2)

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
                    if col in df.columns: df[col] = df[col].astype(np.float32)
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
            print("❌ 无法获取股票列表")
            return

        today_str = datetime.date.today().strftime("%Y-%m-%d")
        existing_fresh = set()
        files = os.listdir(Config.DATA_DIR)
        for fname in files:
            if fname.endswith(".parquet"):
                fpath = os.path.join(Config.DATA_DIR, fname)
                if os.path.getsize(fpath) > 1024:
                    mtime = os.path.getmtime(fpath)
                    file_date = datetime.date.fromtimestamp(mtime).strftime("%Y-%m-%d")
                    if file_date >= today_str: existing_fresh.add(fname.replace(".parquet", ""))

        todo = list(set(codes) - existing_fresh)
        todo.sort()
        print(f"📊 任务: 总数 {len(codes)} | 已是最新 {len(existing_fresh)} | 待更新 {len(todo)}")
        if not todo:
            print("✅ 所有数据已同步至最新交易日。")
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(DataProvider._download_worker, c): c for c in todo}
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(todo)): pass
        print("下载完成。")

    # --------------------------------------------------------------------------
    # PART 2: 核心重构 - 缓存化 Panel 处理 (适配新因子)
    # --------------------------------------------------------------------------

    @staticmethod
    def _get_cache_path(mode):
        today_str = datetime.date.today().strftime("%Y%m%d")
        return os.path.join(Config.OUTPUT_DIR, f"panel_cache_{mode}_{today_str}.pkl")

    @staticmethod
    def load_and_process_panel(mode='train'):
        cache_path = DataProvider._get_cache_path(mode)

        if os.path.exists(cache_path):
            print(f"⚡️ [Cache Hit] 发现今日缓存，正在极速加载: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    panel_df, feature_cols = pickle.load(f)
                print(f"✅ 缓存加载成功，特征数: {len(feature_cols)}")
                return panel_df, feature_cols
            except Exception as e:
                print(f"⚠️ 缓存读取失败 ({e})，将重新计算...")

        print(f"\n>>> [Phase 2] 开始构建全内存 Panel 数据 (Mode: {mode})...")
        files = glob.glob(os.path.join(Config.DATA_DIR, "*.parquet"))
        if not files: raise ValueError("没有找到数据文件")

        print(f"正在加载 {len(files)} 个文件到内存...")

        def _read_helper(f):
            try:
                df = pd.read_parquet(f)
                code = os.path.basename(f).replace(".parquet", "")
                float_cols = df.select_dtypes(include=['float64']).columns
                df[float_cols] = df[float_cols].astype(np.float32)
                df['code'] = code
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

        panel_df['code'] = panel_df['code'].astype(str)
        panel_df = panel_df.reset_index().sort_values(['code', 'date'])

        print("计算时序因子...")
        panel_df = panel_df.groupby('code', group_keys=False).apply(lambda x: AlphaFactory(x).make_factors())

        print("构造 Target...")
        panel_df['target'] = panel_df.groupby('code')['close'].shift(-Config.PRED_LEN) / panel_df['close'] - 1

        if mode == 'train':
            panel_df.dropna(subset=['target'], inplace=True)

        print("动态过滤...")
        original_len = len(panel_df)
        panel_df = panel_df[panel_df['volume'] > 0]
        panel_df = panel_df[panel_df['close'] >= 2.0]
        panel_df['list_days'] = panel_df.groupby('code').cumcount()
        panel_df = panel_df[panel_df['list_days'] > 60]
        panel_df.drop(columns=['list_days'], inplace=True)
        print(f"过滤移除: {original_len - len(panel_df)}")

        print("计算截面与市场交互因子...")
        panel_df = panel_df.set_index('date')
        panel_df = AlphaFactory.add_cross_sectional_factors(panel_df)

        # 【核心更新】: 增加 'mkt_' 和 'rel_' 前缀的特征提取
        feature_cols = [c for c in panel_df.columns
                        if any(
                c.startswith(p) for p in ['style_', 'tech_', 'alpha_', 'adv_', 'ind_', 'cs_rank_', 'mkt_', 'rel_'])]

        panel_df[feature_cols] = panel_df[feature_cols].fillna(0).astype(np.float32)
        panel_df = panel_df.reset_index()

        print(f"💾 正在保存计算结果到缓存: {cache_path} ...")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump((panel_df, feature_cols), f)
            print("✅ 缓存保存完毕。")
        except Exception as e:
            print(f"⚠️ 缓存保存失败: {e}")

        return panel_df, feature_cols

    # ... [PART 3: make_dataset 保持不变，请保留原代码] ...
    @staticmethod
    def make_dataset(panel_df, feature_cols):
        print(">>> [Phase 3] 转换 Dataset...")
        panel_df = panel_df.sort_values(['code', 'date'])
        feature_matrix = panel_df[feature_cols].values
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
        ds = DatasetDict({
            'train': Dataset.from_generator(gen_train),
            'test': Dataset.from_generator(gen_valid)
        })

        return ds, len(feature_cols)


def get_dataset():
    panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train')
    ds, num_features = DataProvider.make_dataset(panel_df, feature_cols)
    return ds, num_features