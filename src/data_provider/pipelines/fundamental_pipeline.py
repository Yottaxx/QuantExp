from __future__ import annotations

import concurrent.futures
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import akshare as ak
from tqdm.auto import tqdm

from ..clients.ak_client import AkClient
from ..core.config import DPConfig
from ..utils.code import normalize_code
from ..utils.io import atomic_save_parquet
from ..stores.paths import fundamental_dir, fundamental_path

# 匹配 YYYYMMDD 格式的列名
DATE_COL_RE = re.compile(r"^\d{8}$")
# 宽表中可能包含指标名称的列
IND_COL_CANDIDATES = ("指标", "项目", "科目", "指标名称")
# 长表中可能包含日期的列
DATE_COL_CANDIDATES = ("日期", "报告期", "截止日期", "date")


@dataclass(frozen=True)
class MetricSpec:
    key: str
    # 正则表达式元组，用于匹配行名（宽表）或列名（长表）
    patterns: Tuple[str, ...]


# 针对 ak.stock_financial_abstract 返回的指标名称进行适配
# 该接口返回的数据通常包含：净资产收益率、总资产净利率、销售净利率、以及各类增长率
METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec("roe", (r"净资产收益率", r"加权.*净资产收益率")),
    MetricSpec("rev_growth", (r"营业(总)?收入(同比)?增长率", r"主营业务收入增长率")),
    MetricSpec("profit_growth", (r"(归母)?净利润(同比)?增长率",)),
    MetricSpec("debt_ratio", (r"资产负债率",)),
    MetricSpec("eps", (r"基本每股收益", r"每股收益")),
    MetricSpec("bps", (r"每股净资产",)),
)

# 输出文件的标准列序
OUT_COLS = ("date",) + tuple(m.key for m in METRICS) + ("pub_date",)


def _coerce_dt(x) -> pd.Series:
    """强制转换为 datetime，无效值设为 NaT"""
    return pd.to_datetime(x, errors="coerce")


def _to_float32(s: pd.Series) -> pd.Series:
    """清洗包含 '%' 或非数值字符的数据，并转换为 float32"""
    if s.dtype == object:
        # 去除百分号、逗号，处理 'nan', '--' 等情况
        s = s.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
        # akshare 有时返回 'None' 字符串
        s = s.replace({"None": np.nan, "--": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce").astype(np.float32)


def _estimate_pub_date(series_dates: pd.Series) -> pd.Series:
    """
    根据报告期(report_date)估算法定披露截止日(pub_date)。
    这是为了避免 Look-ahead Bias 的保守策略。

    A股法定披露截止日规则：
    1季报(03-31) -> 04-30
    中报(06-30)  -> 08-31
    3季报(09-30) -> 10-31
    年报(12-31)  -> 次年 04-30
    """

    def _map_one(d):
        if pd.isna(d):
            return pd.NaT
        try:
            m = d.month
            y = d.year
            if m == 3:
                return pd.Timestamp(year=y, month=4, day=30)
            elif m == 6:
                return pd.Timestamp(year=y, month=8, day=31)
            elif m == 9:
                return pd.Timestamp(year=y, month=10, day=31)
            elif m == 12:
                return pd.Timestamp(year=y + 1, month=4, day=30)
            else:
                # 非常规报告期，默认延后 60 天
                return d + pd.Timedelta(days=60)
        except Exception:
            return pd.NaT

    # 使用 apply 对 Series 进行逐个处理
    return series_dates.apply(_map_one)


def _empty_frame() -> pd.DataFrame:
    """返回标准的空 DataFrame"""
    return pd.DataFrame(columns=list(OUT_COLS))


def _detect_wide_date_cols(cols: Iterable[str]) -> List[str]:
    """提取形如 20231231 的日期列"""
    return [c for c in cols if DATE_COL_RE.match(str(c))]


def _detect_indicator_col(df: pd.DataFrame) -> Optional[str]:
    """寻找存放指标名称的列"""
    for c in IND_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _detect_date_col_long(df: pd.DataFrame) -> Optional[str]:
    """寻找存放日期的列（用于长表模式）"""
    for c in DATE_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _wide_to_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理 ak.stock_financial_abstract 返回的宽表数据
    结构示例:
      选项 | 指标 | 20250930 | 20250630 ...
    """
    date_cols = _detect_wide_date_cols(df.columns)
    ind_col = _detect_indicator_col(df)

    if not date_cols or not ind_col:
        return _empty_frame()

    # 只保留指标列和日期列，丢弃 '选项' 列以防止干扰
    # 使用 copy 避免 SettingWithCopyWarning
    m = df[[ind_col] + date_cols].copy()

    # 清洗指标名称：去空格、转字符串
    m[ind_col] = m[ind_col].astype(str).str.strip()

    # 宽表转长表 (Melt)
    # var_name="date_str", value_name="raw_value"
    long = m.melt(id_vars=[ind_col], value_vars=date_cols, var_name="date_str", value_name="raw_value")

    # 转换日期
    long["date"] = _coerce_dt(long["date_str"])
    long = long.dropna(subset=["date"])

    parts = []
    # 遍历每个需要的指标，从 long 表中提取对应的行
    for spec in METRICS:
        # 构建正则：忽略大小写
        pat = re.compile("|".join(spec.patterns), re.IGNORECASE)

        # 筛选符合当前指标正则的行
        mask = long[ind_col].str.contains(pat, na=False)
        sub = long[mask][["date", "raw_value"]].copy()

        if sub.empty:
            continue

        # 如果匹配到多行（例如'每股收益'匹配了'基本每股收益'和'稀释每股收益'），
        # 这里的简单逻辑是保留最后出现的（通常更具体）或取均值？
        # 在金融报表中，通常取第一个匹配项或按优先级匹配。
        # 这里为了防止 duplicate index error，我们在 pivot 前去重
        # 比如：按日期去重，保留第一个匹配到的
        sub = sub.drop_duplicates(subset=["date"], keep="first")

        sub["metric"] = spec.key
        parts.append(sub)

    if not parts:
        return _empty_frame()

    # 合并所有指标片段
    got = pd.concat(parts, ignore_index=True)
    got["value"] = _to_float32(got["raw_value"])

    # 透视表：Index=Date, Columns=Metric
    out = (
        got.pivot_table(index="date", columns="metric", values="value", aggfunc="last")
        .reset_index()
        .sort_values("date")
    )

    # 补全缺失的指标列，填充 NaN
    for spec in METRICS:
        if spec.key not in out.columns:
            out[spec.key] = np.nan

    # 补充 pub_date 列
    # 使用法定截止日期进行估算，防止回测前视偏差
    out["pub_date"] = _estimate_pub_date(out["date"])

    # 整理最终列序并去重
    out = out[list(OUT_COLS)].drop_duplicates("date", keep="last").reset_index(drop=True)
    return out


def normalize_fundamental_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """
    统一数据清洗入口
    """
    if raw is None or raw.empty:
        return _empty_frame()

    df = raw.copy()
    # 规范化列名：转字符串并去除空格
    df.columns = [str(c).strip() for c in df.columns]

    # 策略 1: 宽表模式 (stock_financial_abstract 属于此类)
    # 特征：列名中包含 YYYYMMDD 格式的日期
    if _detect_wide_date_cols(df.columns):
        return _wide_to_metrics(df)

    # 策略 2: 长表模式 (备用，部分历史接口可能返回此格式)
    # 特征：有一列叫 "date" 或 "报告期"
    if _detect_date_col_long(df) is not None:
        # 这里为了代码简洁，暂时移除未使用的 _long_to_metrics 实现，
        # 如果未来需要支持长表接口，可在此处恢复逻辑。
        # 目前 stock_financial_abstract 100% 返回宽表。
        pass

    return _empty_frame()


class FundamentalPipeline:
    """
    下载并缓存个股财务摘要数据 (Quarterly Fundamentals)
    Output: {DATA_DIR}/fundamental/{code}.parquet
    Columns: date, roe, rev_growth, profit_growth, debt_ratio, eps, bps, pub_date
    """
    SCHEMA_VER = 2  # Schema 版本升级

    def __init__(self, cfg: DPConfig, ak_client: AkClient, logger):
        self.cfg = cfg
        self.ak_client = ak_client
        self.logger = logger
        os.makedirs(fundamental_dir(cfg), exist_ok=True)

    def _should_skip(self, path: str) -> bool:
        """检查缓存是否有效"""
        days = int(self.cfg.get("FUND_TTL_DAYS", 5) or 5)
        ttl = max(1, days) * 24 * 3600
        # 检查文件存在且不过期，且大小正常
        return os.path.exists(path) and os.path.getsize(path) > 512 and (time.time() - os.path.getmtime(path)) < ttl

    def _download_one(self, code: str) -> Tuple[str, bool, str, int]:
        """下载单个股票的财务数据"""
        c = normalize_code(code)
        if not c:
            return str(code), True, "BadCode", 0

        path = fundamental_path(self.cfg, c)
        if self._should_skip(path):
            return c, True, "Skipped", -1

        # 注意: stock_financial_abstract 接口通常不需要 start_year，它返回所有摘要数据
        try:
            # 使用新接口: ak.stock_financial_abstract
            raw = self.ak_client.call(ak.stock_financial_abstract, symbol=c)
            out = normalize_fundamental_frame(raw)

            if out.empty:
                return c, True, "Empty", 0

            atomic_save_parquet(
                out,
                path,
                index=False,
                compression=str(self.cfg.get("PARQUET_COMPRESSION", "zstd") or "zstd"),
            )
            return c, True, "Success", int(len(out))
        except Exception as e:
            # 捕获异常，防止单只股票失败影响整体
            return c, False, f"Failed({type(e).__name__})", 0

    def download(self, codes) -> None:
        """批量下载入口"""
        if not bool(self.cfg.get("SYNC_FUNDAMENTAL", False)):
            self.logger.info("🟦 [Fundamental] SYNC_FUNDAMENTAL=False; skip.")
            return

        codes = [normalize_code(c) for c in codes]
        codes = [c for c in codes if c]
        if not codes:
            self.logger.warning("🟦 [Fundamental] empty codes; skip.")
            return

        workers = int(self.cfg.get("FIN_WORKERS", 8) or 8)
        # 限制最大排队任务数，防止内存爆炸
        max_inflight = int(self.cfg.get("FIN_MAX_INFLIGHT", workers * 4) or (workers * 4))

        self.logger.info(f"🟦 [Fundamental] syncing {len(codes)} codes ... workers={workers} inflight={max_inflight}")

        q = deque(codes)
        stats = {"ok": 0, "bad": 0, "empty": 0, "skipped": 0}

        def submit_more(ex, inflight_dict):
            """填充任务队列"""
            while q and len(inflight_dict) < max_inflight:
                c = q.popleft()
                inflight_dict[ex.submit(self._download_one, c)] = c

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            inflight = {}
            submit_more(ex, inflight)

            with tqdm(total=len(codes), dynamic_ncols=True, desc="Fundamental", unit="code") as pbar:
                while inflight:
                    # 等待任意一个任务完成
                    done, _ = concurrent.futures.wait(
                        inflight.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for fut in done:
                        _ = inflight.pop(fut, None)
                        try:
                            code, success, msg, rows = fut.result()
                            if success:
                                if msg == "Skipped":
                                    stats["skipped"] += 1
                                elif msg == "Empty":
                                    stats["empty"] += 1
                                else:
                                    stats["ok"] += 1
                            else:
                                stats["bad"] += 1
                                # 可以在这里记录具体错误日志: self.logger.debug(f"{code} failed: {msg}")
                        except Exception as e:
                            stats["bad"] += 1
                            self.logger.error(f"Unexpected error in future: {e}")

                        pbar.update(1)
                        pbar.set_postfix(**stats, last=code if 'code' in locals() else "")

                    submit_more(ex, inflight)

        self.logger.info(f"🟦 [Fundamental] done. {stats}")