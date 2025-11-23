import pandas as pd
import numpy as np


def check_volume_logic():
    print(">>> 1. 检查成交量校准逻辑 (高严重级别)")
    # 模拟场景：某股票发生过 10送10，导致前复权因子为 0.5
    # 真实情况：T日 收盘价 20元，成交 100 手 (10000股)，成交额 200,000元
    raw_close = 20.0
    raw_vol_hand = 100
    amount = 200000.0

    # AkShare 下载得到的是前复权价格
    adj_factor = 0.5
    qfq_close = raw_close * adj_factor  # 10.0

    # 现有逻辑模拟
    # 1. 假设 volume 单位是手
    volume_in_df = raw_vol_hand

    # 2. 现在的错误校验逻辑
    # multiplier = amount / (qfq_close * volume)
    multiplier = amount / (qfq_close * volume_in_df)
    # 计算: 200000 / (10.0 * 100) = 200

    print(f"  真实单位: 手")
    print(f"  前复权价: {qfq_close}, 成交额: {amount}")
    print(f"  计算出的 Multiplier: {multiplier}")

    if multiplier > 50:
        print("  [❌ 判定结果] Multiplier > 50，判定为'手'，执行 * 100")
        final_vol = volume_in_df * 100  # 10000 股
        print(f"  修正后成交量: {final_vol} (正确)")
    else:
        print("  [判定结果] 判定为'股'")

    print("  ⚠️ 风险分析: 如果复权因子很大（如老股），multiplier 可能变小导致误判；")
    print("  ⚠️ 更大的风险: 如果数据源单位已经是'股' (10000)，计算结果:")

    volume_in_df_shares = 10000
    multiplier_shares = amount / (qfq_close * volume_in_df_shares)
    # 200000 / (10.0 * 10000) = 2.0
    print(f"  若源数据为股，Multiplier: {multiplier_shares}")
    print("  -> 此时判定为股，不乘100。逻辑在'单位判断'上看似自洽，但依赖QFQ价格是不稳定的。")
    print("  -> 核心错误在于：如果不乘100，代码后面是否还有硬编码的 *100？")


def check_liquidity_logic():
    print("\n>>> 2. 检查回测风控容量计算 (高严重级别)")
    # 假设当前成交量（已转为股）
    vol_shares = 1_000_000  # 100万股
    min_volume_percent = 0.02  # 2%

    # 现有代码逻辑
    # limit_size = int(vol * self.p.min_volume_percent / 100) * 100

    limit_size = int(vol_shares * min_volume_percent / 100) * 100

    real_limit = vol_shares * min_volume_percent

    print(f"  当日成交: {vol_shares} 股")
    print(f"  目标限制: {min_volume_percent * 100}% ({real_limit} 股)")
    print(f"  代码计算: {limit_size} 股")

    if limit_size == real_limit / 100:
        print("  [❌ 严重 Bug] 实际限额被缩小了 100 倍！")
        print("  原因: min_volume_percent 已经是小数(0.02)，不应再除以 100。")
    else:
        print("  逻辑正常。")


if __name__ == "__main__":
    check_volume_logic()
    check_liquidity_logic()