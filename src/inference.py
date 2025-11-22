import torch
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from .config import Config
from .model import PatchTSTForStock
from .data_provider import DataProvider


def check_market_regime(panel_df, last_date):
    """
    【新增】市场环境诊断 (Market Regime)
    利用全市场数据判断当前是 牛市 还是 熊市
    """
    # 取出最新一天的所有股票数据
    daily_slice = panel_df[panel_df['date'] == last_date]

    if daily_slice.empty:
        return "Unknown", 0.0

    # 1. 计算上涨家数占比
    # style_mom_1m 代表过去20天动量
    up_count = (daily_slice['style_mom_1m'] > 0).sum()
    total_count = len(daily_slice)
    up_ratio = up_count / total_count if total_count > 0 else 0

    # 2. 计算市场平均动量 (中位数)
    median_mom = daily_slice['style_mom_1m'].median()

    print(f"📊 市场温度计 (基准日: {last_date.date()})")
    print(f"   - 上涨趋势占比: {up_ratio:.2%}")
    print(f"   - 市场动量中位数: {median_mom:.4f}")

    # 简单择时逻辑：如果超过 60% 的股票处于下跌趋势，或者中位数动量为负，定义为熊市
    if up_ratio < 0.4 or median_mom < -0.02:
        return "Bear", median_mom
    elif up_ratio > 0.6:
        return "Bull", median_mom
    else:
        return "Shock", median_mom


def run_inference(top_k=5, min_score_threshold=0.6):
    """
    全市场选股推理 (带择时风控)
    :param min_score_threshold: 最小得分阈值 (针对 Rank 0~1)，低于此分不买
    """
    print("\n" + "=" * 50)
    print(">>> 启动全市场每日选股 (AI + 择时风控)")
    print("=" * 50)

    device = Config.DEVICE
    model_path = f"{Config.OUTPUT_DIR}/final_model"

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

    # 1. 加载全量数据
    print("正在加载全市场数据 (mode='predict')...")
    try:
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='predict')
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return []

    if panel_df.empty:
        print("❌ 数据为空")
        return []

    # 提取最新日期
    last_date = panel_df['date'].max()

    # 2. 【核心新增】执行大盘择时风控
    regime, mom_val = check_market_regime(panel_df, last_date)

    if regime == "Bear":
        print(f"\n⚠️⚠️ 警告：检测到市场处于【空头/熊市】状态 (动量: {mom_val:.3f})")
        print("🛡️ 触发熔断机制：建议空仓观望，停止买入！")
        # 这里可以选择直接 return [] 强制空仓，或者仅提示
        # 为了演示，我们这里提示但继续，让用户看分
        print("------------------------------------------------")

    # 3. 构建推理输入
    print(f"正在对 {last_date.date()} 的活跃股票进行评分...")
    grouped = panel_df.groupby('code')
    candidates = []

    for code, group in tqdm(grouped, desc="Scoring"):
        if group['date'].iloc[-1] != last_date: continue
        if len(group) < Config.CONTEXT_LEN: continue

        last_window = group.iloc[-Config.CONTEXT_LEN:]
        input_data = last_window[feature_cols].values.astype(np.float32)

        candidates.append({'code': code, 'input': input_data})

    if not candidates:
        print("❌ 无符合条件股票")
        return []

    # 4. 批量推理
    batch_size = 128
    results = []

    with torch.no_grad():
        for i in range(0, len(candidates), batch_size):
            batch_items = candidates[i: i + batch_size]
            batch_input = np.array([item['input'] for item in batch_items])
            tensor_input = torch.tensor(batch_input, dtype=torch.float32).to(device)

            outputs = model(past_values=tensor_input)
            scores = outputs.logits.squeeze().cpu().numpy()
            if scores.ndim == 0: scores = [scores]

            for j, score in enumerate(scores):
                results.append((batch_items[j]['code'], float(score)))

    # 5. 排序与置信度过滤
    results.sort(key=lambda x: x[1], reverse=True)

    # 获取第一名分数
    top_score = results[0][1]

    print("\n" + "-" * 40)
    print(f"🏆 冠军股票得分: {top_score:.4f}")

    # 【核心新增】置信度检查
    # 因为我们训练用的是 Rank [0, 1]，理论上好股票应该接近 1.0
    # 如果第一名只有 0.5，说明全市场都很烂，或者模型看不准
    if top_score < min_score_threshold:
        print(f"⚠️ 警告：最高分低于阈值 ({min_score_threshold})。")
        print("🤖 模型潜台词：'这届股票都不行，我不建议买。'")
        print("🛡️ 建议：空仓或极小仓位尝试。")

    print("-" * 40)
    print(f"🚀 【SOTA 模型最终推荐 (Top {top_k})】")
    print(f"{'排名':<5} | {'代码':<10} | {'预测分':<10} | {'建议'}")
    print("-" * 40)

    top_stocks = results[:top_k]
    final_picks = []

    for rank, (code, score) in enumerate(top_stocks, 1):
        # 结合 大盘风控 和 个股得分 给出最终建议
        advice = "买入"
        if regime == "Bear": advice = "慎买(熊市)"
        if score < min_score_threshold: advice = "观望(分低)"

        print(f"{rank:<5} | {code:<10} | {score:.4f}     | {advice}")

        # 只有在非熊市且分数够高时，才真正返回给回测系统
        # (您可以根据激进程度调整这里的逻辑)
        if advice == "买入":
            final_picks.append((code, score))

    print("=" * 40)

    # 如果您希望严格执行，可以返回 final_picks
    # 这里为了让您看到结果，我们还是返回所有 top_stocks，由您人工决定
    return top_stocks