import akshare as ak
import pandas as pd
import os
import glob
import numpy as np
from datasets import Dataset
from sklearn.preprocessing import StandardScaler
from .config import Config
from .alpha_lib import AlphaFactory
from .evaluator import AlphaEvaluator
from tqdm import tqdm


class DataProvider:
    @staticmethod
    def download_data():
        """下载全市场数据到 Parquet"""
        print("开始同步 A 股数据...")
        stock_info = ak.stock_zh_a_spot_em()
        codes = stock_info['代码'].tolist()

        for code in tqdm(codes):
            path = os.path.join(Config.DATA_DIR, f"{code}.parquet")
            if os.path.exists(path): continue
            try:
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=Config.START_DATE, adjust="qfq")
                if df.empty: continue
                df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
                                   '成交量': 'volume'}, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.to_parquet(path)
            except:
                pass

    @staticmethod
    def process_single_stock(df):
        """单只股票处理：计算 Target + Alpha"""
        # 1. 构造 Target (未来 N 天收益)
        df['target'] = df['close'].shift(-Config.PRED_LEN) / df['close'] - 1

        # 2. 构造 Factors
        factory = AlphaFactory(df)
        df = factory.make_factors()

        # 3. 筛选列
        factor_cols = [c for c in df.columns if c.startswith('alpha_')]
        keep_cols = factor_cols + ['target']
        df.dropna(subset=keep_cols, inplace=True)

        return df, factor_cols

    def generator(self):
        """HF Dataset 生成器"""
        files = glob.glob(os.path.join(Config.DATA_DIR, "*.parquet"))
        # 训练时只用部分优质股或前 N 个文件演示
        for fpath in files[:200]:
            try:
                df = pd.read_parquet(fpath)
                if len(df) < 100: continue

                df_proc, factor_cols = self.process_single_stock(df)

                # 归一化
                scaler = StandardScaler()
                x_data = scaler.fit_transform(df_proc[factor_cols].values)
                y_data = df_proc['target'].values

                # 切片
                for i in range(0, len(x_data) - Config.CONTEXT_LEN, 5):
                    yield {
                        "past_values": x_data[i: i + Config.CONTEXT_LEN].astype(np.float32),
                        "labels": y_data[i + Config.CONTEXT_LEN - 1].astype(np.float32)
                    }
            except:
                continue


def get_dataset():
    provider = DataProvider()
    # 必须先跑一次生成器获取 feature dim
    temp_gen = provider.generator()
    first_item = next(temp_gen)
    num_features = first_item['past_values'].shape[1]

    ds = Dataset.from_generator(provider.generator)
    ds = ds.train_test_split(test_size=0.1)
    return ds, num_features