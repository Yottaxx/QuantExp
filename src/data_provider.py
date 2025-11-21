import akshare as ak
import pandas as pd
import os
import glob
import numpy as np
import time
import random
import requests
import threading
import concurrent.futures
from datasets import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from .config import Config
# 注意：这里不再在顶层导入 AlphaFactory，证明下载阶段完全不依赖它
# from .alpha_lib import AlphaFactory
from .vpn_rotator import vpn_rotator


class DataProvider:
    # 线程锁
    _vpn_lock = threading.Lock()
    _last_switch_time = 0

    # ==============================================================================
    #   Phase 1: 数据下载 (纯 IO 操作)
    #   目标: 只负责把数据搬运到硬盘，不做任何复杂的数学计算
    # ==============================================================================

    @staticmethod
    def _setup_proxy_env():
        """设置代理环境"""
        proxy_url = "http://127.0.0.1:7890"
        os.environ['http_proxy'] = proxy_url
        os.environ['https_proxy'] = proxy_url
        os.environ['all_proxy'] = proxy_url
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        os.environ['ALL_PROXY'] = proxy_url

    @classmethod
    def _safe_switch_vpn(cls):
        """线程安全的 VPN 切换"""
        with cls._vpn_lock:
            if time.time() - cls._last_switch_time < 5:
                return
            vpn_rotator.switch_random()
            cls._last_switch_time = time.time()
            time.sleep(2)

    @staticmethod
    def _download_worker(code):
        """
        下载单元
        注意：这里只做【最小化格式清洗】，绝对不计算 Alpha
        """
        path = os.path.join(Config.DATA_DIR, f"{code}.parquet")

        for attempt in range(5):
            try:
                time.sleep(random.uniform(0.05, 0.2))

                # 1. 网络请求 (这是最耗时的)
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=Config.START_DATE, adjust="qfq")

                if df is None or df.empty:
                    return code, True, "Empty"

                # 2. 最小化格式清洗 (Standardization)
                # 这不是“处理”，这是为了让数据存下来后更好用。耗时 < 0.001秒。
                # 如果不改名，以后读取时全是中文列名会很麻烦。
                df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close',
                                   '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

                # 3. 存盘 (IO)
                if len(df) > 0:
                    df.to_parquet(path)

                return code, True, "Success"

            except Exception as e:
                DataProvider._safe_switch_vpn()
                continue

        return code, False, "Failed"

    @staticmethod
    def download_data():
        """下载入口"""
        print(">>> [Phase 1] 初始化下载引擎...")
        DataProvider._setup_proxy_env()

        # 获取列表
        codes = []
        for _ in range(5):
            try:
                stock_info = ak.stock_zh_a_spot_em()
                codes = stock_info['代码'].tolist()
                break
            except:
                vpn_rotator.switch_random()
                time.sleep(2)

        if not codes:
            print("❌ 无法获取股票列表")
            return

        if not os.path.exists(Config.DATA_DIR):
            os.makedirs(Config.DATA_DIR)

        # 极速断点续传计算
        files = os.listdir(Config.DATA_DIR)
        existing_codes = {f.replace(".parquet", "") for f in files if f.endswith(".parquet")}

        all_codes_set = set(codes)
        todo_codes = list(all_codes_set - existing_codes)
        todo_codes.sort()

        print(f"📊 任务统计: 总数 {len(codes)} | 已存盘 {len(existing_codes)} | 待下载 {len(todo_codes)}")

        if not todo_codes:
            print("✅ 所有数据已下载完毕。")
            return

        MAX_WORKERS = 16
        print(f"🚀 启动 {MAX_WORKERS} 线程并发下载...")

        failed_codes = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_code = {executor.submit(DataProvider._download_worker, code): code for code in todo_codes}
            progress_bar = tqdm(concurrent.futures.as_completed(future_to_code), total=len(todo_codes), unit="it")

            for future in progress_bar:
                code = future_to_code[future]
                try:
                    _, is_success, _ = future.result()
                    if not is_success:
                        failed_codes.append(code)
                except:
                    failed_codes.append(code)

        print(f"下载结束。失败数: {len(failed_codes)}")

    # ==============================================================================
    #   Phase 2: 数据处理 (CPU 密集型)
    #   目标: 读取硬盘数据，计算 Alpha 因子，生成训练集
    #   注意: 这部分代码只在训练时运行 (main.py --mode train)
    # ==============================================================================

    @staticmethod
    def process_single_stock(df):
        """
        单只股票的 Alpha 计算
        只有在这里，我们才引入 AlphaFactory 进行繁重的数学计算
        """
        # 延迟导入：证明下载阶段绝对没用到它
        from .alpha_lib import AlphaFactory

        df['target'] = df['close'].shift(-Config.PRED_LEN) / df['close'] - 1
        factory = AlphaFactory(df)
        df = factory.make_factors()
        factor_cols = [c for c in df.columns if c.startswith('alpha_')]
        keep_cols = factor_cols + ['target']
        df.dropna(subset=keep_cols, inplace=True)
        return df, factor_cols

    def generator(self):
        """
        训练数据生成器
        它会 'Lazily' (懒加载) 地从硬盘读取 Parquet，处理完一个丢给 GPU，再读下一个
        """
        files = glob.glob(os.path.join(Config.DATA_DIR, "*.parquet"))
        # 生产环境请去掉切片
        target_files = files[:500]
        for fpath in target_files:
            try:
                df = pd.read_parquet(fpath)
                if len(df) < 100: continue

                # 这里才开始处理数据
                df_proc, factor_cols = self.process_single_stock(df)

                scaler = StandardScaler()
                x_data = scaler.fit_transform(df_proc[factor_cols].values)
                y_data = df_proc['target'].values
                for i in range(0, len(x_data) - Config.CONTEXT_LEN, 5):
                    yield {
                        "past_values": x_data[i: i + Config.CONTEXT_LEN].astype(np.float32),
                        "labels": y_data[i + Config.CONTEXT_LEN - 1].astype(np.float32)
                    }
            except:
                continue


def get_dataset():
    provider = DataProvider()
    try:
        pass
    except:
        pass

    ds = Dataset.from_generator(provider.generator)
    ds = ds.train_test_split(test_size=0.1)
    temp_gen = provider.generator()
    try:
        first = next(temp_gen)
        num_features = first['past_values'].shape[1]
    except:
        num_features = 12

    return ds, num_features