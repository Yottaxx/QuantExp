import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm
from .config import Config
from .model import PatchTSTForStock
from .data_provider import DataProvider


class BacktestAnalyzer:
    def __init__(self, start_date='2024-01-01', end_date='2025-12-31'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.device = Config.DEVICE
        self.model_path = f"{Config.OUTPUT_DIR}/final_model"
        self.results_df = None

    def generate_historical_predictions(self):
        """
        全量回溯推理：
        加载全历史数据，按日期滚动的形式，对每一天全市场的股票进行打分。
        """
        print("\n" + "=" * 60)
        print(">>> 启动全量截面分析 (Full Cross-Sectional Analysis)")
        print("=" * 60)

        # 1. 加载模型
        if not os.path.exists(self.model_path):
            print(f"❌ 模型未找到: {self.model_path}")
            return

        print(f"正在加载模型: {self.model_path}")
        model = PatchTSTForStock.from_pretrained(self.model_path).to(self.device)
        model.eval()

        # 2. 加载全量 Panel 数据
        # 使用 mode='train'，因为我们需要 Target (真实收益) 来计算 IC，所以剔除最后几天无 Target 的数据是正确的
        print("正在加载全市场 Panel 数据 (用于验证)...")
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train')

        # 3. 时间过滤
        mask = (panel_df['date'] >= self.start_date) & (panel_df['date'] <= self.end_date)
        df_sub = panel_df[mask].copy()

        if df_sub.empty:
            print("❌ 所选时间段无数据")
            return

        print(f"分析区间: {self.start_date.date()} ~ {self.end_date.date()}")
        print(f"样本数量: {len(df_sub)} 行")

        # 4. 按日期分组进行批量推理
        # 这样可以模拟每天“面对全市场股票”的选股场景
        date_groups = df_sub.groupby('date')

        predictions = []

        print("正在进行历史回溯推理...")
        with torch.no_grad():
            for date, group in tqdm(date_groups, desc="Daily Inference"):
                # 跳过样本太少的日期
                if len(group) < 10: continue

                # 检查每只股票是否有足够历史窗口
                # 为了速度，这里假设 DataProvider 已经保证了前面有足够的数据填充
                # 严谨的做法是去原始 panel_df 里找前 30 天

                # 我们需要构建 tensor: [Batch, Seq_Len, Features]
                # 这里有一个难点：df_sub 切片可能导致无法获取前序窗口
                # 优化方案：我们直接遍历 panel_df，但只在目标日期输出结果

                pass  # 逻辑优化见下文

        # --- 优化后的推理逻辑 ---
        # 直接利用 panel_df 的连续性

        results = []
        unique_dates = df_sub['date'].unique()

        # 预处理：将 panel_df 设为 (code, date) 索引以便快速查找窗口
        # 但为了效率，我们采用“滑动窗口生成器”模式

        # 实际上，为了简化代码并保证准确性，我们可以直接利用 'code' group
        # 对每只股票，找出它在分析区间内的所有时间点

        # 更加工程化的做法：
        # 我们复用 DataProvider 的逻辑，但这次我们要记录预测值和真实值

        # 让我们用一种更直接的方法：
        # 遍历所有股票，生成 Tensor，预测，然后把结果拼回去

        codes = df_sub['code'].unique()

        # 提取特征矩阵和 Target
        # 注意：这里为了演示，我们简化处理，直接用当前行作为 Input (假设已经包含了时序特征)
        # 实际上 PatchTST 需要 [Batch, 30, F]

        # 重新利用 groupby code
        full_grouped = panel_df.groupby('code')

        batch_inputs = []
        batch_metas = []  # (date, code, target)

        print("正在构建时序窗口...")
        for code, group in tqdm(full_grouped, desc="Windowing"):
            # 筛选该股票在回测区间内的数据
            in_range_indices = group[(group['date'] >= self.start_date) & (group['date'] <= self.end_date)].index

            for idx in in_range_indices:
                # 获取行号位置
                loc = group.index.get_loc(idx)

                # 如果前面没有足够 30 天数据，跳过
                if loc < Config.CONTEXT_LEN: continue

                # 截取窗口 [loc-30 : loc]
                # 注意：iloc 切片是左闭右开，所以是 loc-Context_Len : loc
                # 但我们需要包含 loc 这一天的数据作为输入序列的最后一天吗？
                # PatchTST 的输入是 Past Values。
                # 假设我们要预测 T+1，我们输入 T-29 ~ T。
                # 这里的 idx 就是 T。

                window = group.iloc[loc - Config.CONTEXT_LEN + 1: loc + 1]
                if len(window) != Config.CONTEXT_LEN: continue

                feature_val = window[feature_cols].values.astype(np.float32)

                target_val = group.loc[idx, 'excess_label']  # 使用超额收益作为验证目标
                if pd.isna(target_val): target_val = group.loc[idx, 'target']

                batch_inputs.append(feature_val)
                batch_metas.append({
                    'date': group.loc[idx, 'date'],
                    'code': code,
                    'label': target_val
                })

                # 显存控制：每 2048 个样本推一次
                if len(batch_inputs) >= 2048:
                    self._run_batch(model, batch_inputs, batch_metas, results)
                    batch_inputs = []
                    batch_metas = []

        # 处理剩余的
        if batch_inputs:
            self._run_batch(model, batch_inputs, batch_metas, results)

        self.results_df = pd.DataFrame(results)
        print(f"推理完成，共生成 {len(self.results_df)} 条预测记录。")

    def _run_batch(self, model, inputs, metas, results_list):
        tensor = torch.tensor(np.array(inputs), dtype=torch.float32).to(self.device)
        scores = model(past_values=tensor).logits.squeeze().detach().cpu().numpy()
        if scores.ndim == 0: scores = [scores]

        for i, score in enumerate(scores):
            rec = metas[i]
            rec['score'] = float(score)
            results_list.append(rec)

    def analyze_performance(self):
        """
        核心：计算 IC, ICIR, 分层收益
        """
        if self.results_df is None or self.results_df.empty:
            print("❌ 无预测数据")
            return

        df = self.results_df.sort_values(['date', 'score'], ascending=[True, False])

        print("\n正在计算截面绩效指标...")

        # 1. Rank IC (相关性)
        # 每天计算 预测分(score) 和 真实下期收益(label) 的 Spearman 相关系数
        daily_ic = df.groupby('date').apply(
            lambda x: spearmanr(x['score'], x['label'])[0]
        )

        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        icir = ic_mean / (ic_std + 1e-9) * np.sqrt(252)  # 年化 ICIR

        print("-" * 40)
        print(f"📊 【因子绩效报告 (IC Analysis)】")
        print("-" * 40)
        print(f"Rank IC (均值) : {ic_mean:.4f} (标准: >0.05 优秀)")
        print(f"ICIR (年化)    : {icir:.4f}   (标准: >3.0 优秀)")
        print(f"IC 胜率        : {(daily_ic > 0).mean():.2%}")
        print("-" * 40)

        # 2. 分层回测 (Layered Backtest)
        # 每天把股票分成 5 组 (Quintiles)
        def get_layer_ret(g):
            # qcut 可能会因为数据少报错，用 numpy split
            try:
                # 按分数降序，分为 5 组
                # 0: Top (分数最高), 4: Bottom (分数最低)
                labels = pd.qcut(g['score'], 5, labels=False, duplicates='drop')
                # qcut 默认是升序 (0是最小)，我们需要反过来或者注意一下
                # score 越大越好，所以 qcut 结果 4 是 Top，0 是 Bottom
                g['group'] = labels
                return g.groupby('group')['label'].mean()
            except:
                return None

        layer_ret = df.groupby('date').apply(get_layer_ret)

        # layer_ret 列名是 0,1,2,3,4。其中 4 是高分层(Top)，0 是低分层(Bottom)
        # 计算累积收益
        cum_ret = (1 + layer_ret).cumprod()

        # 多空收益 (Top - Bottom)
        long_short = (1 + (layer_ret[4] - layer_ret[0])).cumprod()

        # 3. 绘图
        plt.figure(figsize=(14, 8))

        plt.subplot(2, 1, 1)
        for i in range(5):
            label = "Top 20% (Long)" if i == 4 else f"Group {i}"
            label = "Bottom 20% (Short)" if i == 0 else label
            color = 'red' if i == 4 else 'green' if i == 0 else 'grey'
            alpha = 1.0 if i in [0, 4] else 0.3
            plt.plot(cum_ret.index, cum_ret[i], label=label, color=color, alpha=alpha)

        plt.plot(long_short.index, long_short, label='Long-Short (Alpha)', color='blue', linestyle='--', linewidth=2)
        plt.title('Layered Backtest (Cumulative Excess Return)')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.bar(daily_ic.index, daily_ic.values, color='orange', alpha=0.5, label='Daily IC')
        plt.axhline(daily_ic.mean(), color='red', linestyle='--', label=f'Mean IC: {ic_mean:.3f}')
        plt.title('Daily Rank IC Series')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(Config.OUTPUT_DIR, "cross_section_analysis.png")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"📈 截面分析图表已保存至: {save_path}")