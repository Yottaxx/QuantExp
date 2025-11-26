import torch
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from .config import Config
from .model import PatchTSTForStock
from .data_provider import DataProvider


# ... [check_market_regime 保持不变] ...
def check_market_regime(panel_df, last_date):
    daily_slice = panel_df[panel_df['date'] == last_date]
    if daily_slice.empty: return "Unknown", 0.0
    up_count = (daily_slice['style_mom_1m'] > 0).sum()
    up_ratio = up_count / len(daily_slice)
    median_mom = daily_slice['style_mom_1m'].median()
    print(f"📊 市场状态: 上涨占比 {up_ratio:.2%} | 动量中位数 {median_mom:.4f}")
    if up_ratio < 0.4 or median_mom < -0.02:
        return "Bear", median_mom
    elif up_ratio > 0.6:
        return "Bull", median_mom
    else:
        return "Shock", median_mom


def run_inference(top_k=Config.TOP_K, min_score_threshold=Config.MIN_SCORE_THRESHOLD):
    """
    【优化】使用全局配置默认值
    """
    print("\n" + "=" * 50)
    print(">>> 启动全市场每日选股")
    print("=" * 50)

    device = Config.DEVICE
    model_path = f"{Config.OUTPUT_DIR}/final_model"
    # model_path= "/Users/yotta/PycharmProjects/QuantExp/output/checkpoints/checkpoint-3000"
    if not os.path.exists(model_path):
        print("请先运行 train 模式")
        return []

    model = PatchTSTForStock.from_pretrained(model_path).to(device)
    model.eval()

    print("加载最新数据 (mode='predict')...")
    try:
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='predict')
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return []

    last_date = panel_df['date'].max()
    print(f"📅 最新交易日: {last_date.date()}")

    regime, mom_val = check_market_regime(panel_df, last_date)
    if regime == "Bear":
        print(f"\n⚠️ 警告：熊市特征明显，建议空仓！")

    print("构建推理张量...")
    results = []
    grouped = panel_df.groupby('code')
    candidates = []

    for code, group in tqdm(grouped, desc="Scoring"):
        if group['date'].iloc[-1] != last_date: continue
        if len(group) < Config.CONTEXT_LEN: continue

        last_window = group.iloc[-Config.CONTEXT_LEN:]
        input_data = last_window[feature_cols].values.astype(np.float32)
        pe_val = group['pe_ttm'].iloc[-1] if 'pe_ttm' in group.columns else np.nan

        candidates.append({'code': code, 'input': input_data, 'pe': pe_val})

    if not candidates:
        print("❌ 无符合条件股票")
        return []

    batch_size = Config.INFERENCE_BATCH_SIZE
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
                results.append((batch_items[j]['code'], float(score), batch_items[j]['pe']))

    results.sort(key=lambda x: x[1], reverse=True)
    top_score = results[0][1] if results else 0

    if top_score < min_score_threshold:
        print(f"⚠️ 警告：最高分低于阈值 ({min_score_threshold})")

    print("-" * 60)
    print(f"{'排名':<5} | {'代码':<10} | {'AI预测分':<10} | {'PE(TTM)':<10} | {'建议'}")
    print("-" * 60)

    top_stocks = results[:top_k]
    final_picks = []
    for rank, (code, score, pe) in enumerate(top_stocks, 1):
        advice = "买入"
        if regime == "Bear": advice = "慎买"
        if score < min_score_threshold: advice = "观望"
        pe_str = f"{pe:.2f}" if pd.notna(pe) and pe != 0 else "-"
        print(f"{rank:<5} | {code:<10} | {score:.6f}     | {pe_str:<10} | {advice}")
        if advice == "买入": final_picks.append((code, score, pe))

    print("=" * 60)
    if len(final_picks) < len(top_stocks):
        print(f"💡 风控生效：{len(top_stocks)} -> {len(final_picks)}")
    if not final_picks:
        print("🛡️ 最终决策：空仓")
    return final_picks