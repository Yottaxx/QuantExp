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

    # ... [PART 1 下载模块保持不变] ...
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
            if time.time() - cls._last_switch_time < 5:
                return
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

                if len(df) > 0: df.to_parquet(path)
                return code, True, "Success"
            except:
                DataProvider._safe_switch_vpn()
                continue
        return code, False, "Failed"

    @staticmethod
    def download_data():
        """下载全市场数据"""
        print(">>> [Phase 1] 启动数据下载 (每日更新模式)...")
        DataProvider._setup_proxy_env()

        if not os.path.exists(Config.DATA_DIR): os.makedirs(Config.DATA_DIR)

        try:
            stock_info = ak.stock_zh_a_spot_em()
            codes = stock_info['代码'].tolist()
        except:
            print("❌ 无法获取股票列表，请检查网络/VPN")
            return

        print(">>> 正在检查数据新鲜度...")
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        existing_fresh = set()

        files = os.listdir(Config.DATA_DIR)
        for fname in files:
            if fname.endswith(".parquet"):
                fpath = os.path.join(Config.DATA_DIR, fname)
                if os.path.getsize(fpath) > 1024:
                    mtime = os.path.getmtime(fpath)
                    file_date = datetime.date.fromtimestamp(mtime).strftime("%Y-%m-%d")
                    if file_date == today_str:
                        existing_fresh.add(fname.replace(".parquet", ""))

        todo = list(set(codes) - existing_fresh)
        todo.sort()

        print(f"📊 任务: 总数 {len(codes)} | 今日已新 {len(existing_fresh)} | 待更新 {len(todo)}")
        if not todo:
            print("✅ 数据已是最新，无需下载。")
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(DataProvider._download_worker, c): c for c in todo}
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(todo)):
                pass
        print("下载完成。")

    # ... [PART 2 load_and_process_panel 等保持不变] ...
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

    @staticmethod
    def load_and_process_panel(mode='train'):
        print(f"\n>>> [Phase 2] 开始构建全内存 Panel 数据 (Mode: {mode})...")

        files = glob.glob(os.path.join(Config.DATA_DIR, "*.parquet"))
        if not files:
            raise ValueError("没有找到数据文件，请先运行 download")

        print(f"正在加载 {len(files)} 个文件到内存...")

        def _read_helper(f):
            try:
                df = pd.read_parquet(f)
                code = os.path.basename(f).replace(".parquet", "")
                df['code'] = code
                return df
            except:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(_read_helper, files), total=len(files), desc="Reading"))

        data_frames = [df for df in results if df is not None and len(df) > Config.CONTEXT_LEN + 10]
        if not data_frames: raise ValueError("有效数据为空")

        print("正在合并 Panel DataFrame...")
        panel_df = pd.concat(data_frames)
        del data_frames

        panel_df = panel_df.reset_index().sort_values(['code', 'date'])

        print("正在计算时序因子 (TS Factors)...")

        def _process_ts(df_sub):
            factory = AlphaFactory(df_sub)
            return factory.make_factors()

        panel_df = panel_df.groupby('code', group_keys=False).apply(_process_ts)

        print("正在构造预测目标 (Future Returns)...")
        panel_df['target'] = panel_df.groupby('code')['close'].shift(-Config.PRED_LEN) / panel_df['close'] - 1

        if mode == 'train':
            print("训练模式：剔除无标签的尾部数据...")
            panel_df.dropna(subset=['target'], inplace=True)
        else:
            print("预测模式：保留尾部数据用于推理...")

        panel_df = DataProvider._filter_universe(panel_df)

        panel_df = panel_df.set_index('date')
        panel_df = AlphaFactory.add_cross_sectional_factors(panel_df)

        feature_cols = [c for c in panel_df.columns
                        if any(c.startswith(p) for p in ['style_', 'tech_', 'alpha_', 'adv_', 'cs_rank_'])]

        print(f"因子工程完成。特征维度: {len(feature_cols)}")
        panel_df[feature_cols] = panel_df[feature_cols].fillna(0)

        panel_df = panel_df.reset_index()
        return panel_df, feature_cols

    # --------------------------------------------------------------------------
    # PART 3: 核心重构 - 数据集切分 (修复验证集泄露)
    # --------------------------------------------------------------------------

    @staticmethod
    def make_dataset(panel_df, feature_cols):
        """
        转换 Dataset (仅用于训练)
        """
        print(">>> [Phase 3] 转换 Dataset (时间序列切分)...")
        # 确保按时间排序
        panel_df = panel_df.sort_values(['code', 'date'])

        feature_matrix = panel_df[feature_cols].values.astype(np.float32)
        target_col = 'excess_label' if 'excess_label' in panel_df.columns else 'target'
        target_array = panel_df[target_col].fillna(0).values.astype(np.float32)

        codes = panel_df['code'].values
        # 计算每只股票的切分点
        code_changes = np.where(codes[:-1] != codes[1:])[0] + 1
        start_indices = np.concatenate(([0], code_changes))
        end_indices = np.concatenate((code_changes, [len(codes)]))

        # 生成所有合法样本索引
        valid_indices = []
        seq_len = Config.CONTEXT_LEN
        stride = 5

        for start, end in zip(start_indices, end_indices):
            length = end - start
            if length <= seq_len: continue
            for i in range(start, end - seq_len + 1, stride):
                valid_indices.append(i)

        print(f"总样本数量: {len(valid_indices)}")

        # 【核心修复】时间序列切分 (Time-Series Split)
        # 逻辑：为了防止滑动窗口的数据泄露，我们不能随机打乱。
        # 但由于我们是多只股票，按 "总索引" 切分可能把某只股票全部切进 Test。
        # 更好的方法是：对每只股票，前 90% 时间做 Train，后 10% 做 Valid。
        # 但为了实现简单且高效，我们采用全局时间切分：
        # 直接按 valid_indices 的顺序切分（因为 valid_indices 是按 code 排序的，这其实是 GroupKFold 的一种变体）
        # 等等，按 Code 排序切分意味着 Test 集是“全新的几只股票”，而不是“未来的时间”。这是 Cross-Sectional Split。
        # 对于量化模型，我们更想要“未来的时间”做测试。

        # 修正方案：基于日期进行切分
        # 1. 找到分割日期 (Split Date)
        dates = panel_df['date'].unique()
        dates.sort()
        split_idx = int(len(dates) * 0.9)
        split_date = dates[split_idx]
        print(f"训练/验证切分日期: {split_date}")

        # 2. 重新构建索引，分为 Train/Valid
        # 这需要我们在遍历 valid_indices 时知道对应的日期
        # idx 是 feature_matrix 的索引，对应 panel_df 的行号

        # 为了性能，我们直接操作 panel_df 的 date 列
        # 获取所有样本对应的日期 (valid_indices 指向的是窗口的起点，但预测的是终点+预测期)
        # 我们用窗口结束日作为基准
        sample_dates = panel_df['date'].values[np.array(valid_indices) + seq_len - 1]

        train_mask = sample_dates < split_date
        train_indices = np.array(valid_indices)[train_mask]
        valid_indices = np.array(valid_indices)[~train_mask]

        print(f"训练集样本: {len(train_indices)} | 验证集样本: {len(valid_indices)}")

        # 构造生成器
        def gen_train():
            # 训练集可以打乱
            np.random.shuffle(train_indices)
            for idx in train_indices:
                yield {
                    "past_values": feature_matrix[idx: idx + seq_len],
                    "labels": target_array[idx + seq_len - 1]
                }

        def gen_valid():
            # 验证集保持顺序
            for idx in valid_indices:
                yield {
                    "past_values": feature_matrix[idx: idx + seq_len],
                    "labels": target_array[idx + seq_len - 1]
                }

        train_ds = Dataset.from_generator(gen_train)
        valid_ds = Dataset.from_generator(gen_valid)

        # 手动组合成 DatasetDict
        from datasets import DatasetDict
        ds = DatasetDict({
            'train': train_ds,
            'test': valid_ds
        })

        return ds, len(feature_cols)


def get_dataset():
    panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train')
    ds, num_features = DataProvider.make_dataset(panel_df, feature_cols)
    return ds, num_features