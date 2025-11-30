import torch
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from .config import Config
from .model import PatchTSTForStock
from .data_provider import DataProvider


def check_market_regime(panel_df, last_date):
    # 确保日期类型一致
    target_dt = pd.to_datetime(last_date)
    daily_slice = panel_df[panel_df['date'] == target_dt]

    if daily_slice.empty:
        print(f"⚠️ 无法判断市场状态：日期 {target_dt.date()} 无数据")
        return "Unknown", 0.0

    up_count = (daily_slice['style_mom_1m'] > 0).sum()
    up_ratio = up_count / len(daily_slice)
    median_mom = daily_slice['style_mom_1m'].median()
    print(f"📊 市场状态 ({target_dt.date()}): 上涨占比 {up_ratio:.2%} | 动量中位数 {median_mom:.4f}")
    if up_ratio < 0.4 or median_mom < -0.02:
        return "Bear", median_mom
    elif up_ratio > 0.6:
        return "Bull", median_mom
    else:
        return "Shock", median_mom


def run_inference(target_date=None, top_k=Config.TOP_K, min_score_threshold=Config.MIN_SCORE_THRESHOLD):
    """
    运行推理任务
    :param target_date: str or datetime, 指定预测日期 (e.g. '2023-11-20')。若为 None，则使用数据集中最新日期。
    """
    print("\n" + "=" * 50)
    print(f">>> 启动选股预测 (Target: {target_date if target_date else 'Latest'})")
    print("=" * 50)

    device = Config.DEVICE
    model_path = f"{Config.OUTPUT_DIR}/final_model"

    if not os.path.exists(model_path):
        print("请先运行 train 模式")
        return []

    model = PatchTSTForStock.from_pretrained(model_path).to(device)
    model.eval()

    # 处理 target_date 格式
    if target_date:
        target_dt = pd.to_datetime(target_date)
    else:
        target_dt = None

    print(f"加载数据 (End Date: {target_dt.date() if target_dt else 'Auto'})...")

    try:
        # [CRITICAL] 传递 target_date 给 DataProvider 进行数据截断，防止未来数据泄露
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='predict', end_date=target_dt)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return []

    # 确定最终的推理日期
    if target_dt is None:
        last_date = panel_df['date'].max()
    else:
        last_date = target_dt

    print(f"📅 推理基准日期: {last_date.date()}")

    # 检查该日期是否有数据
    if last_date not in panel_df['date'].values:
        print(f"❌ 错误：指定日期 {last_date.date()} 在数据集中不存在（可能是非交易日）。")
        # 可选：寻找最近的前一个交易日
        available_dates = panel_df['date'].unique()
        prev_dates = available_dates[available_dates < last_date]
        if len(prev_dates) > 0:
            last_date = prev_dates.max()
            print(f"🔄 自动回退至最近交易日: {last_date.date()}")
        else:
            return []

    regime, mom_val = check_market_regime(panel_df, last_date)
    if regime == "Bear":
        print(f"\n⚠️ 警告：熊市特征明显，建议空仓！")

    print("构建推理张量...")
    results = []
    grouped = panel_df.groupby('code')
    candidates = []

    # [Optimization] 预先筛选出在 last_date 依然活跃（有数据）的股票
    # 这样可以避免在循环中对非活跃股票进行无意义的检查
    active_codes_at_date = panel_df[panel_df['date'] == last_date]['code'].unique()
    active_codes_set = set(active_codes_at_date)

    for code, group in tqdm(grouped, desc="Scoring"):
        # 1. 股票必须在目标日期有交易数据
        if code not in active_codes_set: continue

        # 2. 确保 group 按时间排序
        # (load_and_process_panel 已经排过序，但为了保险)
        # group = group.sort_values('date')

        # 3. 严格获取截止到 last_date 的窗口
        # 由于 DataProvider 已经根据 end_date 截断，且我们确认 code 在 last_date 有数据
        # 所以 group.iloc[-1] 理论上就是 last_date。但为了双重保险：
        curr_row = group.iloc[-1]
        if curr_row['date'] != last_date:
            continue

        # 4. 检查历史长度是否足够 Context Window
        if len(group) < Config.CONTEXT_LEN: continue

        # 5. 提取输入特征
        last_window = group.iloc[-Config.CONTEXT_LEN:]
        input_data = last_window[feature_cols].values.astype(np.float32)
        pe_val = curr_row['pe_ttm'] if 'pe_ttm' in group.columns else np.nan

        candidates.append({'code': code, 'input': input_data, 'pe': pe_val})

    if not candidates:
        print(f"❌ 日期 {last_date.date()} 无符合条件（历史数据长度充足）的股票")
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
    print(f"预测日期: {last_date.date()}")
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