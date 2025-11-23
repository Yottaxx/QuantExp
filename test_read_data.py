import os
import glob
import random

import pytest

pytest.importorskip("pandas")
import pandas as pd

# 数据存储目录 (根据之前的配置)
DATA_DIR = "./data/stock_lake"


def inspect_parquet():
    print(f"正在扫描目录: {DATA_DIR} ...")

    # 1. 寻找 parquet 文件
    if not os.path.exists(DATA_DIR):
        print(f"❌ 目录不存在: {DATA_DIR}")
        print("请先运行: python main.py --mode download")
        return

    files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))

    if not files:
        print("❌ 目录下没有找到 .parquet 文件。")
        return

    print(f"✅ 发现 {len(files)} 个数据文件。")

    # 2. 随机选取一个文件进行读取
    target_file = random.choice(files)
    file_name = os.path.basename(target_file)
    print(f"\n{'=' * 40}")
    print(f"正在读取样本文件: {file_name}")
    print(f"{'=' * 40}")

    try:
        # 3. 读取 Parquet
        df = pd.read_parquet(target_file)

        # 4. 展示基础信息
        print("\n【1. 数据预览 (Head 5)】")
        print(df.head())

        print("\n【2. 数据预览 (Tail 5)】")
        print(df.tail())

        print("\n【3. 数据结构 (Info)】")
        print(df.info())

        print("\n【4. 索引检查】")
        print(f"索引名称: {df.index.name}")
        print(f"索引类型: {type(df.index)}")
        print(f"时间范围: {df.index.min()} ~ {df.index.max()}")

        print("\n【5. 列检查】")
        print(f"包含列: {df.columns.tolist()}")

        print("\n【6. 缺失值检查】")
        print(df.isnull().sum())

        # 检查是否包含复权后的异常值
        print("\n【7. 简单统计描述】")
        print(df.describe())

    except Exception as e:
        print(f"❌ 读取失败: {e}")


if __name__ == "__main__":
    inspect_parquet()