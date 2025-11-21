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
    # PART 1: 下载模块 (保持不变)
    # --------------------------------------------------------------------------

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
        print(">>> [Phase 1] 启动数据下载...")
        DataProvider._setup_proxy_env()

        if not os.path.exists(Config.DATA_DIR): os.makedirs(Config.DATA_DIR)

        try:
            stock_info = ak.stock_zh_a_spot_em()
            codes = stock_info['代码'].tolist()
        except:
            print("❌ 无法获取股票列表，请检查网络/VPN")
            return

        files = os.listdir(Config.DATA_DIR)
        existing = {f.replace(".parquet", "") for f in files if
                    f.endswith(".parquet") and os.path.getsize(os.path.join(Config.DATA_DIR, f)) > 1024}
        todo = list(set(codes) - existing)
        todo.sort()

        print(f"📊 任务: 总数 {len(codes)} | 待下载 {len(todo)}")
        if not todo:
            print("✅ 数据已最新。")
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(DataProvider._download_worker, c): c for c in todo}
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(todo)):
                pass
        print("下载完成。")

    # --------------------------------------------------------------------------
    # PART 2: 核心重构 - 内存 Panel 处理 (含 Phase 2 过滤)
    # --------------------------------------------------------------------------

    @staticmethod
    def _filter_universe(panel_df):
        """
        【Phase 2 核心】动态宇宙过滤
        目的：清洗掉不适合交易的脏数据，防止模型学坏。
        注意：必须在时序因子计算完成后调用，但在截面因子计算前调用。
        """
        print(">>> [Filtering] 正在执行动态股票池过滤...")
        original_len = len(panel_df)

        # 1. 剔除停牌 (Volume = 0)
        # 停牌期间无法交易，且复牌后往往会有剧烈跳空，是极大的噪音
        panel_df = panel_df[panel_df['volume'] > 0]

        # 2. 剔除垃圾股/准退市股 (Close < 2.0)
        # 低价股往往伴随流动性陷阱或退市风险，量化策略应尽量避开
        panel_df = panel_df[panel_df['close'] >= 2.0]

        # 3. 剔除上市不满 60 天的次新股
        # 逻辑：按 code 分组，计算累计交易天数。前 60 天的数据不稳，剔除。
        # 使用 cumcount() 高效生成序号
        panel_df['list_days'] = panel_df.groupby('code').cumcount()
        panel_df = panel_df[panel_df['list_days'] > 60]

        # 清理临时列
        panel_df = panel_df.drop(columns=['list_days'])

        new_len = len(panel_df)
        print(f"过滤完成。移除样本: {original_len - new_len} ({1 - new_len / original_len:.2%})")
        return panel_df

    @staticmethod
    def load_and_process_panel():
        """
        全内存加载与处理核心函数
        """
        print("\n>>> [Phase 2] 开始构建全内存 Panel 数据...")

        files = glob.glob(os.path.join(Config.DATA_DIR, "*.parquet"))
        if not files:
            raise ValueError("没有找到数据文件，请先运行 download")

        # --- Step 1: 并行读取 ---
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
            # 使用 list() 强制执行 map
            results = list(tqdm(executor.map(_read_helper, files), total=len(files), desc="Reading"))

        # 过滤无效数据，合并
        data_frames = [df for df in results if df is not None and len(df) > Config.CONTEXT_LEN + 10]
        if not data_frames: raise ValueError("有效数据为空")

        print("正在合并 Panel DataFrame...")
        panel_df = pd.concat(data_frames)
        del data_frames  # 释放内存

        # 重置索引，确保 'date' 是列
        panel_df = panel_df.reset_index().sort_values(['code', 'date'])

        # --- Step 2: 计算时序因子 (TS Factors) ---
        # 注意：必须在过滤之前计算，否则因为某些天被剔除导致 rolling 计算中断
        print("正在计算时序因子 (TS Factors)...")

        def _process_ts(df_sub):
            factory = AlphaFactory(df_sub)
            return factory.make_factors()

        # 优化：只对需要的列进行 groupby 运算，防止内存爆炸
        # group_keys=False 避免索引层级增加
        panel_df = panel_df.groupby('code', group_keys=False).apply(_process_ts)

        # --- Step 3: 构造 Label ---
        # 预测未来 N 天收益
        print("正在构造预测目标 (Future Returns)...")
        panel_df['target'] = panel_df.groupby('code')['close'].shift(-Config.PRED_LEN) / panel_df['close'] - 1

        # 剔除 label 为空的行 (最后 N 天)
        panel_df.dropna(subset=['target'], inplace=True)

        # --- Step 4: 执行动态过滤 (Filtering) ---
        # 【关键】在这里切除垃圾数据，确保后续的截面排名只在优质股票中进行
        panel_df = DataProvider._filter_universe(panel_df)

        # --- Step 5: 计算截面因子 & 超额收益 Label ---
        # 此时 panel_df 已经很干净了，计算 cs_rank 会更准确
        # 重置索引为 date，方便 AlphaFactory 处理
        panel_df = panel_df.set_index('date')
        panel_df = AlphaFactory.add_cross_sectional_factors(panel_df)

        # --- Step 6: 最终清洗 ---
        feature_cols = [c for c in panel_df.columns
                        if any(c.startswith(p) for p in ['style_', 'tech_', 'alpha_', 'adv_', 'cs_rank_'])]

        print(f"因子工程完成。特征维度: {len(feature_cols)}")
        # 填充 NaN
        panel_df[feature_cols] = panel_df[feature_cols].fillna(0)

        # 重置索引回来，方便后续排序
        panel_df = panel_df.reset_index()

        return panel_df, feature_cols

    @staticmethod
    def make_dataset(panel_df, feature_cols):
        """
        将 Panel DataFrame 转换为 PyTorch 友好的 Dataset
        """
        print(">>> [Phase 3] 转换 Dataset...")

        # 1. 排序: 必须按 (code, date) 排序以保证滑动窗口正确
        panel_df = panel_df.sort_values(['code', 'date'])

        # 2. 提取 numpy 数组 (使用 float32 压缩内存)
        feature_matrix = panel_df[feature_cols].values.astype(np.float32)

        # 【关键】使用 'excess_label' (超额收益) 作为训练目标
        # 如果没有 excess_label，回退到 target
        target_col = 'excess_label' if 'excess_label' in panel_df.columns else 'target'
        print(f"使用训练目标: {target_col}")
        target_array = panel_df[target_col].values.astype(np.float32)

        # 3. 构建样本索引
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

            # 滑动窗口切片
            for i in range(start, end - seq_len + 1, stride):
                valid_indices.append(i)

        print(f"生成的样本数量: {len(valid_indices)}")

        def gen():
            for idx in valid_indices:
                yield {
                    "past_values": feature_matrix[idx: idx + seq_len],
                    "labels": target_array[idx + seq_len - 1]
                }

        ds = Dataset.from_generator(gen)
        ds = ds.train_test_split(test_size=0.1, shuffle=True)

        return ds, len(feature_cols)


# 对外接口
def get_dataset():
    panel_df, feature_cols = DataProvider.load_and_process_panel()
    ds, num_features = DataProvider.make_dataset(panel_df, feature_cols)
    return ds, num_features