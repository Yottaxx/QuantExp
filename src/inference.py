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
    daily_slice = panel_df[panel_df['date'] == last_date]

    if daily_slice.empty:
        return "Unknown", 0.0

    # 计算上涨家数占比 (基于短期动量)
    up_count = (daily_slice['style_mom_1m'] > 0).sum()
    total_count = len(daily_slice)
    up_ratio = up_count / total_count if total_count > 0 else 0

    # 计算市场平均动量 (中位数)
    median_mom = daily_slice['style_mom_1m'].median()

    print(f"📊 市场温度计 (基准日: {last_date.date()})")
    print(f"   - 上涨趋势占比: {up_ratio:.2%}")
    print(f"   - 市场动量中位数: {median_mom:.4f}")

    if up_ratio < 0.4 or median_mom < -0.02:
        return "Bear", median_mom
    elif up_ratio > 0.6:
        return "Bull", median_mom
    else:
        return "Shock", median_mom


def run_inference(top_k=5, min_score_threshold=0.6):
    """
    全市场选股推理 (带 PE 展示)
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

    # 2. 加载全量数据
    print("正在加载全市场数据并计算 SOTA 因子...")
    try:
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='predict')
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return []

    if panel_df.empty:
        print("❌ 数据为空")
        return []

    # 3. 提取【最新一个交易日】
    last_date = panel_df['date'].max()
    print(f"📅 锁定最新交易日: {last_date.date()}")

    # 执行大盘择时风控
    regime, mom_val = check_market_regime(panel_df, last_date)

    if regime == "Bear":
        print(f"\n⚠️⚠️ 警告：检测到市场处于【空头/熊市】状态 (动量: {mom_val:.3f})")
        print("🛡️ 触发熔断机制：建议空仓观望，停止买入！")
        print("------------------------------------------------")

    print("正在构建推理张量 (Tensor Construction)...")

    results = []
    grouped = panel_df.groupby('code')
    candidates = []

    for code, group in tqdm(grouped, desc="Scoring"):
        # 确保股票还在交易
        if group['date'].iloc[-1] != last_date:
            continue

        if len(group) < Config.CONTEXT_LEN:
            continue

        # 取最后 30 天数据
        last_window = group.iloc[-Config.CONTEXT_LEN:]
        input_data = last_window[feature_cols].values.astype(np.float32)

        # 【新增】提取 PE (TTM)
        # 如果数据源里有 'pe_ttm' 列则提取，否则为 NaN
        pe_val = np.nan
        if 'pe_ttm' in group.columns:
            pe_val = group['pe_ttm'].iloc[-1]

        candidates.append({
            'code': code,
            'input': input_data,
            'pe': pe_val  # 携带 PE 信息
        })

    if not candidates:
        print("❌ 没有符合条件的股票")
        return []

    # 4. 批量推理
    batch_size = 128
    print(f"正在对 {len(candidates)} 只活跃股票进行评分...")

    with torch.no_grad():
        for i in range(0, len(candidates), batch_size):
            batch_items = candidates[i: i + batch_size]
            batch_input = np.array([item['input'] for item in batch_items])
            tensor_input = torch.tensor(batch_input, dtype=torch.float32).to(device)

            outputs = model(past_values=tensor_input)
            scores = outputs.logits.squeeze().cpu().numpy()

            if scores.ndim == 0: scores = [scores]

            for j, score in enumerate(scores):
                # 保存结果: (代码, 分数, PE)
                results.append((batch_items[j]['code'], float(score), batch_items[j]['pe']))

    # 5. 排序与输出
    results.sort(key=lambda x: x[1], reverse=True)

    # 获取第一名分数
    top_score = results[0][1] if results else 0

    print("\n" + "-" * 60)
    print(f"🏆 冠军股票得分: {top_score:.4f}")

    if top_score < min_score_threshold:
        print(f"⚠️ 警告：最高分低于阈值 ({min_score_threshold})。")
        print("🛡️ 建议：空仓或极小仓位尝试。")

    print("-" * 60)
    print(f"🚀 【SOTA 模型最终推荐 (Top {top_k})】")
    # 增加 PE 列展示
    print(f"{'排名':<5} | {'代码':<10} | {'AI 预测分':<12} | {'PE (TTM)':<10} | {'建议'}")
    print("-" * 60)

    top_stocks = results[:top_k]
    final_picks = []

    for rank, (code, score, pe) in enumerate(top_stocks, 1):
        advice = "买入"
        if regime == "Bear": advice = "慎买(熊市)"
        if score < min_score_threshold: advice = "观望(分低)"

        # 格式化 PE 显示
        pe_str = f"{pe:.2f}" if pd.notna(pe) and pe != 0 else "-"

        print(f"{rank:<5} | {code:<10} | {score:.6f}     | {pe_str:<10} | {advice}")

        if advice == "买入":
            final_picks.append((code, score))

    print("=" * 60)

    return top_stocks