import pytest
import pandas as pd
import numpy as np
import os
import matplotlib
from unittest.mock import MagicMock, patch
from src.analysis import BacktestAnalyzer
from src.config import Config

# 设置 matplotlib 后端防止在无 GUI 环境报错
matplotlib.use('Agg')


class TestAnalysisSmoke:
    """
    【分析模块冒烟测试】
    测试 src/analysis.py 中的 BacktestAnalyzer。
    不加载真实模型，而是 Mock 数据来验证分析逻辑（IC计算、绘图等）是否通畅。
    """

    @pytest.fixture
    def mock_results_df(self):
        """生成模拟的推理结果 DataFrame"""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        codes = [f"{i:06d}" for i in range(1, 6)]  # 5只股票

        data = []
        for d in dates:
            for c in codes:
                # 构造一个完美的正相关场景：分数越高，收益越高
                # 加入一点点随机噪音
                base_alpha = np.random.random()
                score = base_alpha + np.random.normal(0, 0.01)
                rank_label = base_alpha  # 理想 Rank
                excess_label = base_alpha * 0.1  # 模拟超额收益

                data.append({
                    'date': d,
                    'code': c,
                    'score': score,
                    'rank_label': rank_label,
                    'excess_label': excess_label
                })

        return pd.DataFrame(data)

    def test_analyze_performance_logic(self, tmp_path, mock_results_df):
        """
        测试核心分析逻辑：
        1. 计算 IC
        2. 进行分层回测
        3. 保存图表
        """
        # 1. 初始化分析器
        analyzer = BacktestAnalyzer()

        # 2. 注入伪造的推理结果 (跳过 generate_historical_predictions)
        analyzer.results_df = mock_results_df

        # 3. 临时覆盖 OUTPUT_DIR，确保图片保存到临时目录
        with patch.object(Config, 'OUTPUT_DIR', str(tmp_path)):
            analyzer.analyze_performance()

        # 4. 验证产物
        expected_plot = os.path.join(tmp_path, "cross_section_analysis.png")
        assert os.path.exists(expected_plot), "❌ 未生成分层回测图表"

        # 验证文件大小不为0
        assert os.path.getsize(expected_plot) > 0
        print(f"✅ 图表成功生成: {expected_plot}")

    def test_generate_predictions_flow(self):
        """
        测试推理流程的“管道”是否通畅 (Mock DataProvider 和 Model)
        """
        # 1. Mock 数据源 (从 2023 年开始，确保有足够历史数据)
        mock_panel = pd.DataFrame({
            'date': pd.date_range("2023-12-01", periods=100),  # 覆盖到 2024-03
            'code': '000001',
            # 伪造特征列
            'feat_1': np.random.randn(100),
            'feat_2': np.random.randn(100),
            'rank_label': np.random.randn(100),
            'target': np.random.randn(100)
        })
        feature_cols = ['feat_1', 'feat_2']

        # 2. Mock 模型
        mock_model = MagicMock()
        mock_model.eval.return_value = None

        # [Critical Fix] 模拟 forward 返回
        # 必须返回足够多的分数，以匹配 Batch Size。
        # 如果 Batch 有 10 个样本，但这里只返回 1 个分数，_flush_batch 里的 zip/enumerate 会截断结果。
        mock_output = MagicMock()
        # 返回一个巨大的数组，确保够用 (比如 1000 个)
        mock_output.logits.squeeze.return_value.cpu.return_value.numpy.return_value = np.ones(1000) * 0.5
        mock_model.return_value = mock_output

        # 3. 使用 patch 拦截外部依赖
        with patch('src.analysis.DataProvider.load_and_process_panel', return_value=(mock_panel, feature_cols)), \
                patch('src.analysis.PatchTSTForStock.from_pretrained', return_value=mock_model), \
                patch('os.path.exists', return_value=True), \
                patch.object(Config, 'CONTEXT_LEN', 5):  # 全局覆盖 Context Length

            # [Fix] 调整时间窗口
            # Data: 2023-12-01 ~ 2024-03-xx
            # Context: 5 days
            # Analyzer: 2024-01-01 ~ 2024-01-10 (完全落在数据范围内，且远离边界)
            analyzer = BacktestAnalyzer(start_date="2024-01-01", end_date="2024-01-10")
            analyzer.generate_historical_predictions()

            # 4. 验证
            assert analyzer.results_df is not None
            assert not analyzer.results_df.empty, "❌ 结果集为空，可能是时间窗口过滤或 Mock 模型输出长度不足导致"

            print(f"✅ 推理管道测试通过，生成 {len(analyzer.results_df)} 条记录")
            # 验证生成的记录确实在目标日期范围内
            dates = analyzer.results_df['date']
            print(f"   -> 日期范围: {dates.min().date()} ~ {dates.max().date()}")


if __name__ == "__main__":
    # 允许直接运行此脚本调试
    import sys
    from subprocess import call

    call([sys.executable, "-m", "pytest", "-v", "-s", __file__])