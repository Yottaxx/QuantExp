import torch
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from .config import Config
from .model import PatchTSTForStock
from .data_provider import DataProvider


def run_inference(top_k=5):
    """
    全市场选股推理 (Inference)
    基于全内存 Panel 数据，确保截面因子 (CS Rank) 计算正确
    """
    print("\n" + "=" * 50)
    print(">>> 启动全市场每日选股 (Daily Screening)")
    print("=" * 50)

    device = Config.DEVICE
    model_path = f"{Config.OUTPUT_DIR}/final_model"

    # 1. 加载模型
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行: python main.py --mode train")
        return []

    print(f"正在加载模型权重: {model_path} ...")
    try:
        model = PatchTSTForStock.from_pretrained(model_path).to(device)
        model.eval()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return []

    # 2. 加载全量数据并计算因子 (含截面因子)
    # 【核心修复】 必须传入 mode='predict'，否则 DataProvider 会把最新的（没有Target的）数据删掉！
    print("正在加载全市场数据并计算 SOTA 因子...")
    try:
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='predict')
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return []

    if panel_df.empty:
        print("❌ 数据为空")
        return []

    # 3. 提取【最新一个交易日】的数据作为输入
    # 我们需要用 T-29 到 T 的数据，来预测 T+1 的收益
    # 首先找到数据集中最后一个日期
    last_date = panel_df['date'].max()
    print(f"📅 锁定最新交易日: {last_date.date()}")

    # 筛选出在该日期有数据的股票

    print("正在构建推理张量 (Tensor Construction)...")

    results = []
    # 按股票分组，取最后窗口
    grouped = panel_df.groupby('code')

    candidates = []

    # 使用 tqdm 显示进度
    for code, group in tqdm(grouped, desc="Scoring"):
        # 如果该股票最后一天不是选定的日期（说明停牌了），跳过
        if group['date'].iloc[-1] != last_date:
            continue

        # 数据长度不够
        if len(group) < Config.CONTEXT_LEN:
            continue

        # 取最后 30 天
        last_window = group.iloc[-Config.CONTEXT_LEN:]

        # 提取特征矩阵
        input_data = last_window[feature_cols].values.astype(np.float32)

        candidates.append({
            'code': code,
            'input': input_data
        })

    if not candidates:
        print("❌ 没有符合条件的股票（数据不足或全部停牌）")
        return []

    # 4. 批量推理 (Batch Inference)
    # 为了速度，我们可以把 candidates 打包成 batch
    batch_size = 128

    print(f"正在对 {len(candidates)} 只活跃股票进行评分...")

    with torch.no_grad():
        for i in range(0, len(candidates), batch_size):
            batch_items = candidates[i: i + batch_size]

            # 构造 Batch Tensor: [Batch, Seq_Len, Features]
            batch_input = np.array([item['input'] for item in batch_items])
            tensor_input = torch.tensor(batch_input, dtype=torch.float32).to(device)

            # 模型预测
            outputs = model(past_values=tensor_input)
            scores = outputs.logits.squeeze().cpu().numpy()

            # 处理 batch 只有 1 个的情况
            if scores.ndim == 0: scores = [scores]

            for j, score in enumerate(scores):
                results.append((batch_items[j]['code'], float(score)))

    # 5. 排序与输出
    # 分数越高，代表预测的【超额收益】越高
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 40)
    print(f"🚀 【SOTA 模型选股结果 (Top {top_k})】")
    print(f"基准日期: {last_date.date()}")
    print("-" * 40)
    print(f"{'排名':<5} | {'代码':<10} | {'AI 预测得分':<15}")
    print("-" * 40)

    top_stocks = results[:top_k]
    for rank, (code, score) in enumerate(top_stocks, 1):
        print(f"{rank:<5} | {code:<10} | {score:.6f}")
    print("=" * 40)

    return top_stocks