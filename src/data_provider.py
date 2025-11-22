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

    # --------------------------------------------------------------------------
    # PART 1: 下载模块 (多线程 + VPN 轮询 + 智能日历)
    # --------------------------------------------------------------------------

    @staticmethod
    def _setup_proxy_env():
        """设置当前进程的代理环境变量 (对应 Clash 混合端口 7890)"""
        proxy_url = "http://127.0.0.1:7890"
        os.environ['http_proxy'] = proxy_url
        os.environ['https_proxy'] = proxy_url
        os.environ['all_proxy'] = proxy_url
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        os.environ['ALL_PROXY'] = proxy_url

    @classmethod
    def _safe_switch_vpn(cls):
        """线程安全的 VPN 切换逻辑"""
        with cls._vpn_lock:
            # 防止多个线程同时触发切换，设置 5 秒冷却
            if time.time() - cls._last_switch_time < 5:
                return
            vpn_rotator.switch_random()
            cls._last_switch_time = time.time()
            time.sleep(2)  # 给 Clash 建立连接留出时间

    @staticmethod
    def _get_latest_trading_date():
        """
        【新增】获取最近的一个交易日
        优化：防止周末/节假日运行脚本时重复下载周五的数据
        """
        try:
            # 获取上证指数的最新日线数据作为参考
            # symbol="sh000001" 是上证指数
            df = ak.stock_zh_index_daily(symbol="sh000001")
            if df is not None and not df.empty:
                latest_date = pd.to_datetime(df['date']).max().date()
                return latest_date.strftime("%Y-%m-%d")
        except:
            pass

        # 如果获取失败（比如断网），退化为使用今天
        return datetime.date.today().strftime("%Y-%m-%d")

    @staticmethod
    def _download_worker(code):
        """单个股票下载任务"""
        path = os.path.join(Config.DATA_DIR, f"{code}.parquet")
        for attempt in range(5):
            try:
                # 极速模式：保留微小随机延迟模拟真人
                time.sleep(random.uniform(0.05, 0.2))
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=Config.START_DATE, adjust="qfq")

                if df is None or df.empty: return code, True, "Empty"

                # 标准化列名
                df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close',
                                   '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

                # 内存优化：存盘前转 float32
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    if col in df.columns: df[col] = df[col].astype(np.float32)

                if len(df) > 0: df.to_parquet(path)
                return code, True, "Success"
            except:
                # 遇到封锁，申请切换 VPN
                DataProvider._safe_switch_vpn()
                continue
        return code, False, "Failed"

    @staticmethod
    def download_data():
        """下载全市场数据主入口"""
        print(">>> [Phase 1] 启动数据下载 (智能增量模式)...")
        DataProvider._setup_proxy_env()

        if not os.path.exists(Config.DATA_DIR): os.makedirs(Config.DATA_DIR)

        try:
            stock_info = ak.stock_zh_a_spot_em()
            codes = stock_info['代码'].tolist()
        except:
            print("❌ 无法获取股票列表，请检查网络/VPN")
            return

        # 1. 获取【真正】需要更新到的日期
        print(">>> 正在校对交易日历...")
        target_date_str = DataProvider._get_latest_trading_date()
        print(f"📅 市场最新交易日: {target_date_str}")

        # 2. 智能断点续传
        # 逻辑：如果本地文件的修改日期 >= 市场最新交易日，说明已经包含了最新数据，跳过
        existing_fresh = set()
        files = os.listdir(Config.DATA_DIR)

        for fname in files:
            if fname.endswith(".parquet"):
                fpath = os.path.join(Config.DATA_DIR, fname)
                if os.path.getsize(fpath) > 1024:
                    # 获取文件修改时间
                    mtime = os.path.getmtime(fpath)
                    file_date = datetime.date.fromtimestamp(mtime).strftime("%Y-%m-%d")

                    # 【核心优化】只要文件日期 >= 目标交易日，就算新鲜
                    # 例如：目标日是周五，你在周六运行，文件日期是周五，满足 >=，跳过下载
                    if file_date >= target_date_str:
                        existing_fresh.add(fname.replace(".parquet", ""))

        todo = list(set(codes) - existing_fresh)
        todo.sort()

        print(f"📊 任务: 总数 {len(codes)} | 已是最新 {len(existing_fresh)} | 待更新 {len(todo)}")
        if not todo:
            print("✅ 所有数据已同步至最新交易日，无需下载。")
            return

        # 开启 16 线程并发下载
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(DataProvider._download_worker, c): c for c in todo}
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(todo)):
                pass
        print("下载完成。")

    # --------------------------------------------------------------------------
    # PART 2: 核心重构 - 内存 Panel 处理 (含 Phase 2 过滤与逻辑修复)
    # --------------------------------------------------------------------------

    @staticmethod
    def _get_cache_path(mode):
        today_str = datetime.date.today().strftime("%Y%m%d")
        return os.path.join(Config.OUTPUT_DIR, f"panel_cache_{mode}_{today_str}.pkl")

    @staticmethod
    def _filter_universe(panel_df):
        """
        【动态股票池过滤】
        清洗掉不适合交易的脏数据（停牌、退市、次新股），防止模型学坏。
        """
        print(">>> [Filtering] 正在执行动态股票池过滤...")
        original_len = len(panel_df)

        # 1. 剔除停牌 (Volume = 0)
        panel_df = panel_df[panel_df['volume'] > 0]

        # 2. 剔除垃圾股/准退市股 (Close < 2.0)
        panel_df = panel_df[panel_df['close'] >= 2.0]

        # 3. 剔除上市不满 60 天的次新股
        panel_df['list_days'] = panel_df.groupby('code').cumcount()
        panel_df = panel_df[panel_df['list_days'] > 60]
        panel_df = panel_df.drop(columns=['list_days'])

        new_len = len(panel_df)
        print(f"过滤完成。移除样本: {original_len - new_len} ({1 - new_len / original_len:.2%})")
        return panel_df

    @staticmethod
    def load_and_process_panel(mode='train', force_refresh=False):
        """
        全内存加载与处理核心函数
        :param mode: 'train' (剔除无标签数据) | 'predict' (保留最新数据用于推理)
        :param force_refresh: 强制不使用缓存
        """
        cache_path = DataProvider._get_cache_path(mode)

        # 尝试读取缓存
        if not force_refresh and os.path.exists(cache_path):
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
        if not files:
            raise ValueError("没有找到数据文件，请先运行 download")

        print(f"正在加载 {len(files)} 个文件到内存...")

        def _read_helper(f):
            try:
                df = pd.read_parquet(f)
                code = os.path.basename(f).replace(".parquet", "")

                # 内存优化：强转 float32
                float_cols = df.select_dtypes(include=['float64']).columns
                df[float_cols] = df[float_cols].astype(np.float32)

                df['code'] = code
                # df['code'] = df['code'].astype('category') # 暂时禁用 category 以防 groupby 兼容性问题
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

        # 重置索引
        panel_df = panel_df.reset_index().sort_values(['code', 'date'])

        # --- Step 2: 计算时序因子 (TS Factors) ---
        print("正在计算时序因子 (TS Factors)...")

        def _process_ts(df_sub):
            factory = AlphaFactory(df_sub)
            return factory.make_factors()

        panel_df = panel_df.groupby('code', group_keys=False).apply(_process_ts)

        # --- Step 3: 构造 Label ---
        print("正在构造预测目标 (Future Returns)...")
        panel_df['target'] = panel_df.groupby('code')['close'].shift(-Config.PRED_LEN) / panel_df['close'] - 1

        # 【核心修复逻辑：防止预测日自杀】
        if mode == 'train':
            print("训练模式：剔除无标签的尾部数据...")
            panel_df.dropna(subset=['target'], inplace=True)
        else:
            print("预测模式：保留尾部数据用于推理 (Target为NaN是正常的)...")
            # 不执行 dropna，保留最新的数据行

        # --- Step 4: 执行动态过滤 ---
        panel_df = DataProvider._filter_universe(panel_df)

        # --- Step 5: 计算截面因子 (Cross-Sectional) ---
        # 必须在过滤之后做，确保排名是在可交易股票池中进行的
        panel_df = panel_df.set_index('date')
        panel_df = AlphaFactory.add_cross_sectional_factors(panel_df)

        # --- Step 6: 最终清洗 ---
        feature_cols = [c for c in panel_df.columns
                        if any(
                c.startswith(p) for p in ['style_', 'tech_', 'alpha_', 'adv_', 'ind_', 'cs_rank_', 'mkt_', 'rel_'])]

        print(f"因子工程完成。特征维度: {len(feature_cols)}")
        panel_df[feature_cols] = panel_df[feature_cols].fillna(0).astype(np.float32)

        panel_df = panel_df.reset_index()

        # 保存缓存
        print(f"💾 正在保存计算结果到缓存: {cache_path} ...")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump((panel_df, feature_cols), f)
            print("✅ 缓存保存完毕。")
        except Exception as e:
            print(f"⚠️ 缓存保存失败: {e}")

        return panel_df, feature_cols

    @staticmethod
    def make_dataset(panel_df, feature_cols):
        """转换 Dataset (仅用于训练)"""
        print(">>> [Phase 3] 转换 Dataset...")
        panel_df = panel_df.sort_values(['code', 'date'])

        feature_matrix = panel_df[feature_cols].values.astype(np.float32)

        # 优先使用超额收益作为目标
        # 优先 rank_label -> excess_label -> target
        if 'rank_label' in panel_df.columns:
            target_col = 'rank_label'
        elif 'excess_label' in panel_df.columns:
            target_col = 'excess_label'
        else:
            target_col = 'target'

        # 填充 NaN，防止预测模式报错
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
            for i in range(start, end - seq_len + 1, stride):
                valid_indices.append(i)

        print(f"生成的样本数量: {len(valid_indices)}")

        # 时间序列切分 (Time-Series Split)
        dates = panel_df['date'].unique()
        dates.sort()
        split_idx = int(len(dates) * 0.9)
        split_date = dates[split_idx]
        print(f"切分日期: {split_date}")

        sample_dates = panel_df['date'].values[np.array(valid_indices) + seq_len - 1]
        train_mask = sample_dates < split_date
        train_indices = np.array(valid_indices)[train_mask]
        valid_indices = np.array(valid_indices)[~train_mask]

        print(f"Train: {len(train_indices)} | Valid: {len(valid_indices)}")

        def gen_train():
            np.random.shuffle(train_indices)
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
    # 默认是训练模式
    panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train', force_refresh=force_refresh)
    ds, num_features = DataProvider.make_dataset(panel_df, feature_cols)
    return ds, num_features