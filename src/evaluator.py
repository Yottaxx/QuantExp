class AlphaEvaluator:
    @staticmethod
    def evaluate(df, factor_cols, target_col='target'):
        """计算 Rank IC 和 ICIR"""
        valid_factors = []
        print(f"\n{'=' * 10} 因子质量评估 {'=' * 10}")

        for factor in factor_cols:
            # 计算 Rolling IC 均值作为近似
            ic_series = df[factor].rolling(60).corr(df[target_col])
            rank_ic = ic_series.mean()
            icir = rank_ic / (ic_series.std() + 1e-9)

            # 宽松准则：IC绝对值 > 1% 即可 (实际生产要求更高)
            if abs(rank_ic) > 0.01:
                valid_factors.append(factor)
                status = "✅"
            else:
                status = "❌"

            print(f"{factor:<15} | IC: {rank_ic:.4f} | Status: {status}")

        return valid_factors