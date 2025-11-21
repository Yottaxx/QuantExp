import torch
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from .config import Config
from .model import PatchTSTForStock
from .data_provider import DataProvider


class BacktestAnalyzer:
    def __init__(self, start_date='2024-01-01', end_date='2025-11-20'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.device = Config.DEVICE
        self.model_path = f"{Config.OUTPUT_DIR}/final_model"

    def load_model(self):
        print(f"正在加载模型: {self.model_path}")
        self.model = PatchTSTForStock.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

    def generate_historical_predictions(self):
        """
        核心逻辑：
        不使用 for date in dates (太慢)，
        而是 for stock in stocks (批量处理)，一次性生成该股票在整个回测区间的预测值，
        最后再合并成一张大表。
        """
        if not hasattr(self, 'model'):
            self.load_model()

        files = glob.glob(os.path.join(Config.DATA_DIR, "*.parquet"))
        all_predictions = []

        print(f"正在进行全市场回溯预测 ({self.start_date.date()} - {self.end_date.date()})...")

        # 我们可以只抽样部分股票进行演示分析，全市场跑需要较长时间
        # files = files[:500]

        with torch.no_grad():
            for fpath in tqdm(files):
                try:
                    stock_code = os.path.basename(fpath).replace('.parquet', '')
                    df = pd.read_parquet(fpath)

                    # 1. 基础处理 (计算因子)
                    if len(df) < 100: continue
                    df_proc, factor_cols = DataProvider.process_single_stock(df)

                    # 2. 筛选时间范围 (需要多留出 Context_Len 的数据用于生成第一天的预测)
                    mask = (df_proc.index >= self.start_date - pd.Timedelta(days=60)) & (df_proc.index <= self.end_date)
                    df_sub = df_proc[mask].copy()

                    if len(df_sub) < Config.CONTEXT_LEN: continue

                    # 3. 准备批量推理数据
                    # 我们需要构建滑动窗口: [T-30, T], [T-29, T+1] ...
                    # 为了速度，手动构建 Numpy 视图
                    data_values = df_sub[factor_cols].values
                    scaler = StandardScaler()
                    data_values = scaler.fit_transform(data_values)

                    # 使用 stride_tricks 高效切片
                    # Shape: [Num_Days, Context_Len, Features]
                    # 这种方法比一个个 append 快 100倍
                    num_samples = len(data_values) - Config.CONTEXT_LEN
                    if num_samples <= 0: continue

                    # 构造 Tensor (注意显存，如果显存不够需要分 Batch)
                    # 这里为了演示简单，直接转 Tensor (对于单只股票通常没问题)
                    # 实际操作：我们需要对应每一天的 Input
                    input_list = []
                    valid_dates = []
                    valid_targets = []

                    # 对应的日期是窗口的最后一天
                    dates = df_sub.index[Config.CONTEXT_LEN:]
                    targets = df_sub['target'].values[Config.CONTEXT_LEN:]

                    # 滑动窗口生成
                    for i in range(num_samples):
                        window = data_values[i: i + Config.CONTEXT_LEN]
                        input_list.append(window)
                        valid_dates.append(dates[i])
                        valid_targets.append(targets[i])

                    # 转 Batch Tensor
                    input_tensor = torch.tensor(np.array(input_list), dtype=torch.float32).to(self.device)

                    # 4. 模型推理
                    # 如果 input_tensor 很大，建议这里再拆 mini-batch
                    outputs = self.model(past_values=input_tensor)
                    scores = outputs.logits.squeeze().cpu().numpy()

                    # 5. 收集结果
                    # 如果 scores 是标量(只有1天)，转为数组
                    if scores.ndim == 0: scores = [scores]

                    for date, score, true_ret in zip(valid_dates, scores, valid_targets):
                        if date >= self.start_date:  # 再次确保日期在回测区间内
                            all_predictions.append({
                                'date': date,
                                'code': stock_code,
                                'score': float(score),
                                'true_return': float(true_ret)
                            })

                except Exception as e:
                    # print(f"Error {stock_code}: {e}")
                    continue

        self.df_res = pd.DataFrame(all_predictions)
        print(f"回溯完成，生成预测记录 {len(self.df_res)} 条。")
        return self.df_res

    def analyze_performance(self):
        """
        工业级分析：Rank IC, ICIR, 分层收益
        """
        if self.df_res is None or self.df_res.empty:
            print("无预测数据，请先运行 generate_historical_predictions")
            return

        print("\n正在计算绩效指标...")
        df = self.df_res.sort_values(['date', 'score'], ascending=[True, False])

        # --- 1. IC 分析 (Information Coefficient) ---
        # 每天计算一次 预测分 和 真实收益 的相关系数
        daily_ic = df.groupby('date').apply(
            lambda x: spearmanr(x['score'], x['true_return'])[0]
        )

        rank_ic_mean = daily_ic.mean()
        rank_ic_std = daily_ic.std()
        icir = rank_ic_mean / (rank_ic_std + 1e-9)
        win_rate = (daily_ic > 0).sum() / len(daily_ic)

        print(f"{'=' * 30}")
        print(f"【IC 绩效报告】")
        print(f"Rank IC (均值): {rank_ic_mean:.4f}")
        print(f"ICIR (稳定性):  {icir:.4f}")
        print(f"IC 胜率:       {win_rate:.2%}")
        print(f"{'=' * 30}")

        # --- 2. 分层回测 (Group Test) ---
        # 每天将股票分为 5 组 (Quintiles)
        def get_group_ret(day_df):
            # pd.qcut 可能会报错如果数据太少，这里用简单的切片
            n = len(day_df)
            top_n = int(n * 0.2)  # Top 20%
            bottom_n = int(n * 0.2)  # Bottom 20%

            if top_n == 0: return pd.Series([0, 0, 0], index=['top', 'bottom', 'ls'])

            # 假设按 score 降序排好了
            top_ret = day_df.iloc[:top_n]['true_return'].mean()
            bottom_ret = day_df.iloc[-bottom_n:]['true_return'].mean()

            return pd.Series({
                'top': top_ret,
                'bottom': bottom_ret,
                'ls': top_ret - bottom_ret  # 多空收益
            })

        daily_group_ret = df.groupby('date').apply(get_group_ret)

        # 计算累计收益 (单利累加近似，或者复利 (1+r).cumprod())
        # 这里用累加展示 Log Returns 概念，或者简单累积
        cum_ret = daily_group_ret.cumsum()

        self.plot_analysis(daily_ic, cum_ret)

    def plot_analysis(self, daily_ic, cum_ret):
        """绘制专业图表"""
        plt.figure(figsize=(12, 10))

        # 图1: 累计收益曲线
        plt.subplot(2, 1, 1)
        plt.plot(cum_ret.index, cum_ret['top'], label='Top 20% (Long)', color='red')
        plt.plot(cum_ret.index, cum_ret['bottom'], label='Bottom 20% (Short)', color='green', alpha=0.5)
        plt.plot(cum_ret.index, cum_ret['ls'], label='Long-Short (Alpha)', color='blue', linewidth=2)
        plt.title('Group Backtest Performance (Cumulative Return)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 图2: 每日 IC 分布
        plt.subplot(2, 1, 2)
        plt.bar(daily_ic.index, daily_ic.values, color='gray', alpha=0.6, label='Daily IC')
        plt.axhline(daily_ic.mean(), color='red', linestyle='--', label=f'Mean IC: {daily_ic.mean():.3f}')
        plt.title('Daily Rank IC')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(Config.OUTPUT_DIR, "analysis_report.png")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"分析图表已保存至: {save_path}")
        # plt.show() # 如果在服务器运行请注释掉