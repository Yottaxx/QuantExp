
#后面的代码运行太慢 前面的estimate要好点
# from __future__ import annotations
#
# import concurrent.futures
# import os
# import re
# import time
# from collections import deque
# from dataclasses import dataclass
# from typing import Iterable, List, Optional, Tuple
#
# import numpy as np
# import pandas as pd
# import akshare as ak
# from tqdm.auto import tqdm
#
# from ..clients.ak_client import AkClient
# from ..core.config import DPConfig
# from ..utils.code import normalize_code
# from ..utils.io import atomic_save_parquet
# from ..stores.paths import fundamental_dir, fundamental_path
#
# # 匹配 YYYYMMDD 格式的列名
# DATE_COL_RE = re.compile(r"^\d{8}$")
# # 宽表中可能包含指标名称的列
# IND_COL_CANDIDATES = ("指标", "项目", "科目", "指标名称")
# # 长表中可能包含日期的列
# DATE_COL_CANDIDATES = ("日期", "报告期", "截止日期", "date")
#
#
# @dataclass(frozen=True)
# class MetricSpec:
#     key: str
#     # 正则表达式元组，用于匹配行名（宽表）或列名（长表）
#     patterns: Tuple[str, ...]
#
#
# # 针对 ak.stock_financial_abstract 返回的指标名称进行适配
# # 该接口返回的数据通常包含：净资产收益率、总资产净利率、销售净利率、以及各类增长率
# METRICS: Tuple[MetricSpec, ...] = (
#     MetricSpec("roe", (r"净资产收益率", r"加权.*净资产收益率")),
#     MetricSpec("rev_growth", (r"营业(总)?收入(同比)?增长率", r"主营业务收入增长率")),
#     MetricSpec("profit_growth", (r"(归母)?净利润(同比)?增长率",)),
#     MetricSpec("debt_ratio", (r"资产负债率",)),
#     MetricSpec("eps", (r"基本每股收益", r"每股收益")),
#     MetricSpec("bps", (r"每股净资产",)),
# )
#
# # 输出文件的标准列序
# OUT_COLS = ("date",) + tuple(m.key for m in METRICS) + ("pub_date",)
#
#
# def _coerce_dt(x) -> pd.Series:
#     """强制转换为 datetime，无效值设为 NaT"""
#     return pd.to_datetime(x, errors="coerce")
#
#
# def _to_float32(s: pd.Series) -> pd.Series:
#     """清洗包含 '%' 或非数值字符的数据，并转换为 float32"""
#     if s.dtype == object:
#         # 去除百分号、逗号，处理 'nan', '--' 等情况
#         s = s.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
#         # akshare 有时返回 'None' 字符串
#         s = s.replace({"None": np.nan, "--": np.nan, "nan": np.nan})
#     return pd.to_numeric(s, errors="coerce").astype(np.float32)
#
#
# def _estimate_pub_date(series_dates: pd.Series) -> pd.Series:
#     """
#     根据报告期(report_date)估算法定披露截止日(pub_date)。
#     这是为了避免 Look-ahead Bias 的保守策略。
#
#     A股法定披露截止日规则：
#     1季报(03-31) -> 04-30
#     中报(06-30)  -> 08-31
#     3季报(09-30) -> 10-31
#     年报(12-31)  -> 次年 04-30
#     """
#
#     def _map_one(d):
#         if pd.isna(d):
#             return pd.NaT
#         try:
#             m = d.month
#             y = d.year
#             if m == 3:
#                 return pd.Timestamp(year=y, month=4, day=30)
#             elif m == 6:
#                 return pd.Timestamp(year=y, month=8, day=31)
#             elif m == 9:
#                 return pd.Timestamp(year=y, month=10, day=31)
#             elif m == 12:
#                 return pd.Timestamp(year=y + 1, month=4, day=30)
#             else:
#                 # 非常规报告期，默认延后 60 天
#                 return d + pd.Timedelta(days=60)
#         except Exception:
#             return pd.NaT
#
#     # 使用 apply 对 Series 进行逐个处理
#     return series_dates.apply(_map_one)
#
#
# def _empty_frame() -> pd.DataFrame:
#     """返回标准的空 DataFrame"""
#     return pd.DataFrame(columns=list(OUT_COLS))
#
#
# def _detect_wide_date_cols(cols: Iterable[str]) -> List[str]:
#     """提取形如 20231231 的日期列"""
#     return [c for c in cols if DATE_COL_RE.match(str(c))]
#
#
# def _detect_indicator_col(df: pd.DataFrame) -> Optional[str]:
#     """寻找存放指标名称的列"""
#     for c in IND_COL_CANDIDATES:
#         if c in df.columns:
#             return c
#     return None
#
#
# def _detect_date_col_long(df: pd.DataFrame) -> Optional[str]:
#     """寻找存放日期的列（用于长表模式）"""
#     for c in DATE_COL_CANDIDATES:
#         if c in df.columns:
#             return c
#     return None
#
#
# def _wide_to_metrics(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     处理 ak.stock_financial_abstract 返回的宽表数据
#     结构示例:
#       选项 | 指标 | 20250930 | 20250630 ...
#     """
#     date_cols = _detect_wide_date_cols(df.columns)
#     ind_col = _detect_indicator_col(df)
#
#     if not date_cols or not ind_col:
#         return _empty_frame()
#
#     # 只保留指标列和日期列，丢弃 '选项' 列以防止干扰
#     # 使用 copy 避免 SettingWithCopyWarning
#     m = df[[ind_col] + date_cols].copy()
#
#     # 清洗指标名称：去空格、转字符串
#     m[ind_col] = m[ind_col].astype(str).str.strip()
#
#     # 宽表转长表 (Melt)
#     # var_name="date_str", value_name="raw_value"
#     long = m.melt(id_vars=[ind_col], value_vars=date_cols, var_name="date_str", value_name="raw_value")
#
#     # 转换日期
#     long["date"] = _coerce_dt(long["date_str"])
#     long = long.dropna(subset=["date"])
#
#     parts = []
#     # 遍历每个需要的指标，从 long 表中提取对应的行
#     for spec in METRICS:
#         # 构建正则：忽略大小写
#         pat = re.compile("|".join(spec.patterns), re.IGNORECASE)
#
#         # 筛选符合当前指标正则的行
#         mask = long[ind_col].str.contains(pat, na=False)
#         sub = long[mask][["date", "raw_value"]].copy()
#
#         if sub.empty:
#             continue
#
#         # 如果匹配到多行（例如'每股收益'匹配了'基本每股收益'和'稀释每股收益'），
#         # 这里的简单逻辑是保留最后出现的（通常更具体）或取均值？
#         # 在金融报表中，通常取第一个匹配项或按优先级匹配。
#         # 这里为了防止 duplicate index error，我们在 pivot 前去重
#         # 比如：按日期去重，保留第一个匹配到的
#         sub = sub.drop_duplicates(subset=["date"], keep="first")
#
#         sub["metric"] = spec.key
#         parts.append(sub)
#
#     if not parts:
#         return _empty_frame()
#
#     # 合并所有指标片段
#     got = pd.concat(parts, ignore_index=True)
#     got["value"] = _to_float32(got["raw_value"])
#
#     # 透视表：Index=Date, Columns=Metric
#     out = (
#         got.pivot_table(index="date", columns="metric", values="value", aggfunc="last")
#         .reset_index()
#         .sort_values("date")
#     )
#
#     # 补全缺失的指标列，填充 NaN
#     for spec in METRICS:
#         if spec.key not in out.columns:
#             out[spec.key] = np.nan
#
#     # 补充 pub_date 列
#     # 使用法定截止日期进行估算，防止回测前视偏差
#     out["pub_date"] = _estimate_pub_date(out["date"])
#
#     # 整理最终列序并去重
#     out = out[list(OUT_COLS)].drop_duplicates("date", keep="last").reset_index(drop=True)
#     return out
#
#
# def normalize_fundamental_frame(raw: pd.DataFrame) -> pd.DataFrame:
#     """
#     统一数据清洗入口
#     """
#     if raw is None or raw.empty:
#         return _empty_frame()
#
#     df = raw.copy()
#     # 规范化列名：转字符串并去除空格
#     df.columns = [str(c).strip() for c in df.columns]
#
#     # 策略 1: 宽表模式 (stock_financial_abstract 属于此类)
#     # 特征：列名中包含 YYYYMMDD 格式的日期
#     if _detect_wide_date_cols(df.columns):
#         return _wide_to_metrics(df)
#
#     # 策略 2: 长表模式 (备用，部分历史接口可能返回此格式)
#     # 特征：有一列叫 "date" 或 "报告期"
#     if _detect_date_col_long(df) is not None:
#         # 这里为了代码简洁，暂时移除未使用的 _long_to_metrics 实现，
#         # 如果未来需要支持长表接口，可在此处恢复逻辑。
#         # 目前 stock_financial_abstract 100% 返回宽表。
#         pass
#
#     return _empty_frame()
#
#
# class FundamentalPipeline:
#     """
#     下载并缓存个股财务摘要数据 (Quarterly Fundamentals)
#     Output: {DATA_DIR}/fundamental/{code}.parquet
#     Columns: date, roe, rev_growth, profit_growth, debt_ratio, eps, bps, pub_date
#     """
#     SCHEMA_VER = 2  # Schema 版本升级
#
#     def __init__(self, cfg: DPConfig, ak_client: AkClient, logger):
#         self.cfg = cfg
#         self.ak_client = ak_client
#         self.logger = logger
#         os.makedirs(fundamental_dir(cfg), exist_ok=True)
#
#     def _should_skip(self, path: str) -> bool:
#         """检查缓存是否有效"""
#         days = int(self.cfg.get("FUND_TTL_DAYS", 5) or 5)
#         ttl = max(1, days) * 24 * 3600
#         # 检查文件存在且不过期，且大小正常
#         return os.path.exists(path) and os.path.getsize(path) > 512 and (time.time() - os.path.getmtime(path)) < ttl
#
#     def _download_one(self, code: str) -> Tuple[str, bool, str, int]:
#         """下载单个股票的财务数据"""
#         c = normalize_code(code)
#         if not c:
#             return str(code), True, "BadCode", 0
#
#         path = fundamental_path(self.cfg, c)
#         if self._should_skip(path):
#             return c, True, "Skipped", -1
#
#         # 注意: stock_financial_abstract 接口通常不需要 start_year，它返回所有摘要数据
#         try:
#             # 使用新接口: ak.stock_financial_abstract
#             raw = self.ak_client.call(ak.stock_financial_abstract, symbol=c)
#             out = normalize_fundamental_frame(raw)
#
#             if out.empty:
#                 return c, True, "Empty", 0
#
#             atomic_save_parquet(
#                 out,
#                 path,
#                 index=False,
#                 compression=str(self.cfg.get("PARQUET_COMPRESSION", "zstd") or "zstd"),
#             )
#             return c, True, "Success", int(len(out))
#         except Exception as e:
#             # 捕获异常，防止单只股票失败影响整体
#             return c, False, f"Failed({type(e).__name__})", 0
#
#     def download(self, codes) -> None:
#         """批量下载入口"""
#         if not bool(self.cfg.get("SYNC_FUNDAMENTAL", False)):
#             self.logger.info("🟦 [Fundamental] SYNC_FUNDAMENTAL=False; skip.")
#             return
#
#         codes = [normalize_code(c) for c in codes]
#         codes = [c for c in codes if c]
#         if not codes:
#             self.logger.warning("🟦 [Fundamental] empty codes; skip.")
#             return
#
#         workers = int(self.cfg.get("FIN_WORKERS", 8) or 8)
#         # 限制最大排队任务数，防止内存爆炸
#         max_inflight = int(self.cfg.get("FIN_MAX_INFLIGHT", workers * 4) or (workers * 4))
#
#         self.logger.info(f"🟦 [Fundamental] syncing {len(codes)} codes ... workers={workers} inflight={max_inflight}")
#
#         q = deque(codes)
#         stats = {"ok": 0, "bad": 0, "empty": 0, "skipped": 0}
#
#         def submit_more(ex, inflight_dict):
#             """填充任务队列"""
#             while q and len(inflight_dict) < max_inflight:
#                 c = q.popleft()
#                 inflight_dict[ex.submit(self._download_one, c)] = c
#
#         with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
#             inflight = {}
#             submit_more(ex, inflight)
#
#             with tqdm(total=len(codes), dynamic_ncols=True, desc="Fundamental", unit="code") as pbar:
#                 while inflight:
#                     # 等待任意一个任务完成
#                     done, _ = concurrent.futures.wait(
#                         inflight.keys(),
#                         return_when=concurrent.futures.FIRST_COMPLETED,
#                     )
#                     for fut in done:
#                         _ = inflight.pop(fut, None)
#                         try:
#                             code, success, msg, rows = fut.result()
#                             if success:
#                                 if msg == "Skipped":
#                                     stats["skipped"] += 1
#                                 elif msg == "Empty":
#                                     stats["empty"] += 1
#                                 else:
#                                     stats["ok"] += 1
#                             else:
#                                 stats["bad"] += 1
#                                 # 可以在这里记录具体错误日志: self.logger.debug(f"{code} failed: {msg}")
#                         except Exception as e:
#                             stats["bad"] += 1
#                             self.logger.error(f"Unexpected error in future: {e}")
#
#                         pbar.update(1)
#                         pbar.set_postfix(**stats, last=code if 'code' in locals() else "")
#
#                     submit_more(ex, inflight)
#
#         self.logger.info(f"🟦 [Fundamental] done. {stats}")
#


from __future__ import annotations

import concurrent.futures
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import akshare as ak
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..clients.ak_client import AkClient
from ..core.config import DPConfig
from ..stores.paths import fundamental_dir, fundamental_path
from ..utils.code import normalize_code
from ..utils.io import atomic_save_parquet


DATE_COL_RE = re.compile(r"^\d{8}$")  # e.g. 20250930

IND_COL_CANDIDATES = ("指标", "项目", "科目", "指标名称")
DATE_COL_CANDIDATES = ("日期", "报告期", "截止日期", "截止日期", "date", "report_date")


# ----------------------------
# 1) Stable schema
# ----------------------------
@dataclass(frozen=True)
class MetricSpec:
    key: str
    patterns: Tuple[str, ...]


# 核心输出字段（稳定 schema）
METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec("roe", (r"净资产收益率", r"\bROE\b", r"加权.*净资产收益率", r"净资产收益率.*\(ROE\)")),
    MetricSpec("debt_ratio", (r"资产负债率", r"负债.*资产", r"Debt\s*Ratio")),
    MetricSpec("eps", (r"每股收益", r"基本每股收益", r"\bEPS\b")),
    MetricSpec("bps", (r"每股净资产", r"\bBPS\b")),
    # growth：优先匹配同比/增长率；匹配不到则用收入/利润自己算 YoY
    MetricSpec("rev_growth", (r"(营|主).*收入.*(同比|增长率|增长)", r"收入.*同比", r"营业总收入.*同比", r"营业收入.*同比")),
    MetricSpec("profit_growth", (r"(归母)?净利润.*(同比|增长率|增长)", r"利润.*同比", r"归母净利润.*同比", r"净利润.*同比")),
)

# 用来计算 YoY 的辅助行（当增长率找不到时）
AUX_ABS = {
    "revenue": (r"营业总收入", r"营业收入", r"主营业务收入"),
    "profit": (r"归母净利润", r"净利润"),
}

OUT_COLS = ("date",) + tuple(m.key for m in METRICS) + ("pub_date",)


# ----------------------------
# 2) utilities
# ----------------------------
def _coerce_dt(x) -> pd.Series:
    return pd.to_datetime(x, errors="coerce")


def _clean_numeric_series(s: pd.Series) -> pd.Series:
    """
    Robust numeric cleanup:
      - strip % / 元 / commas
      - handle '--'
    """
    if s is None:
        return pd.Series(dtype=np.float32)
    if not isinstance(s, pd.Series):
        s = pd.Series(s)

    if s.dtype == object:
        x = (
            s.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("元", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("--", "", regex=False)
            .str.strip()
        )
        s = x

    return pd.to_numeric(s, errors="coerce").astype(np.float32)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(OUT_COLS))


def _detect_wide_date_cols(cols: Iterable[str]) -> List[str]:
    return [c for c in cols if DATE_COL_RE.match(str(c))]


def _detect_indicator_col(df: pd.DataFrame) -> Optional[str]:
    for c in IND_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _detect_date_col_long(df: pd.DataFrame) -> Optional[str]:
    for c in DATE_COL_CANDIDATES:
        if c in df.columns:
            return c
    # 有些接口会叫“截止日期”
    if "截止日期" in df.columns:
        return "截止日期"
    return None


def _best_effort_call_financial_abstract(ak_client: AkClient, code: str) -> pd.DataFrame:
    """
    ak.stock_financial_abstract 老版本参数名叫 stock，新版本有时叫 symbol；
    做一个双尝试，避免被 AkShare 参数改动卡死。
    """
    return ak_client.call(ak.stock_financial_abstract, symbol=code)


def _melt_wide(df: pd.DataFrame, ind_col: str, date_cols: List[str]) -> pd.DataFrame:
    m = df[[ind_col] + date_cols].copy()
    m[ind_col] = m[ind_col].astype(str).str.strip()
    long = m.melt(id_vars=[ind_col], value_vars=date_cols, var_name="date", value_name="value")
    long["date"] = _coerce_dt(long["date"])
    long = long.dropna(subset=["date"])
    return long


def _extract_metric_from_long(long: pd.DataFrame, ind_col: str, patterns: Tuple[str, ...]) -> pd.Series:
    """
    Return Series indexed by date -> value (float32) for best match rows.
    If multiple rows match, we just keep the last per date after melt (stable-ish).
    """
    pat = re.compile("|".join(patterns), re.IGNORECASE)
    sub = long[long[ind_col].str.contains(pat, na=False, regex=True)][["date", "value"]].copy()
    if sub.empty:
        return pd.Series(dtype=np.float32)
    sub["value"] = _clean_numeric_series(sub["value"])
    sub = sub.dropna(subset=["date"]).sort_values("date")
    return sub.groupby("date")["value"].last()


def _yoy_growth_from_abs(abs_series: pd.Series) -> pd.Series:
    """
    YoY on endpoints (quarterly/half/annual cumulative):
      growth(date) = value(date) / value(date - 1y) - 1
    """
    if abs_series is None or abs_series.empty:
        return pd.Series(dtype=np.float32)
    s = abs_series.sort_index()
    idx = s.index
    prev_idx = (idx - pd.DateOffset(years=1)).to_list()
    prev = pd.Series([s.get(d, np.nan) for d in prev_idx], index=idx, dtype=np.float32)
    g = (s / prev) - 1.0
    return g.astype(np.float32)


# ----------------------------
# 3) normalize frames
# ----------------------------
def _wide_to_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide table (your real output):
      cols: 选项, 指标, 20250930, 20250630, ...
      rows: 指标名称
    """
    date_cols = _detect_wide_date_cols(df.columns)
    ind_col = _detect_indicator_col(df)
    if not date_cols or not ind_col:
        return _empty_frame()

    long = _melt_wide(df, ind_col, date_cols)

    series_map: Dict[str, pd.Series] = {}
    for spec in METRICS:
        series_map[spec.key] = _extract_metric_from_long(long, ind_col, spec.patterns)

    # 如果同比增长率没取到，就尝试用收入/利润自己算
    if series_map["rev_growth"].empty:
        rev = _extract_metric_from_long(long, ind_col, AUX_ABS["revenue"])
        series_map["rev_growth"] = _yoy_growth_from_abs(rev)

    if series_map["profit_growth"].empty:
        prof = _extract_metric_from_long(long, ind_col, AUX_ABS["profit"])
        series_map["profit_growth"] = _yoy_growth_from_abs(prof)

    # assemble
    all_dates = pd.Index(sorted({d for s in series_map.values() for d in s.index if pd.notna(d)}))
    if all_dates.empty:
        return _empty_frame()

    out = pd.DataFrame({"date": all_dates})
    for spec in METRICS:
        s = series_map.get(spec.key, pd.Series(dtype=np.float32))
        out[spec.key] = out["date"].map(s).astype(np.float32)

    out["pub_date"] = pd.NaT
    out = out[list(OUT_COLS)].drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return out


def _long_to_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long table (some akshare versions show this kind):
      rows: reporting dates
      columns: many indicators
    We select columns by regex; growth can be computed if needed.
    """
    date_col = _detect_date_col_long(df)
    if not date_col:
        return _empty_frame()

    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]
    x = x.rename(columns={date_col: "date"})
    x["date"] = _coerce_dt(x["date"])
    x = x.dropna(subset=["date"]).sort_values("date")

    out = pd.DataFrame({"date": x["date"].values})
    col_names = list(x.columns)

    # direct matches
    for spec in METRICS:
        pat = re.compile("|".join(spec.patterns), re.IGNORECASE)
        cand = next((c for c in col_names if c != "date" and pat.search(str(c))), None)
        out[spec.key] = _clean_numeric_series(x[cand]) if cand else np.nan

    # fallback compute YoY if growth empty
    if out["rev_growth"].isna().all():
        rev_cand = None
        for p in AUX_ABS["revenue"]:
            pat = re.compile(p, re.IGNORECASE)
            rev_cand = next((c for c in col_names if c != "date" and pat.search(str(c))), None)
            if rev_cand:
                break
        if rev_cand:
            rev = pd.Series(_clean_numeric_series(x[rev_cand]).values, index=out["date"])
            out["rev_growth"] = _yoy_growth_from_abs(rev).reindex(out["date"]).values

    if out["profit_growth"].isna().all():
        prof_cand = None
        for p in AUX_ABS["profit"]:
            pat = re.compile(p, re.IGNORECASE)
            prof_cand = next((c for c in col_names if c != "date" and pat.search(str(c))), None)
            if prof_cand:
                break
        if prof_cand:
            prof = pd.Series(_clean_numeric_series(x[prof_cand]).values, index=out["date"])
            out["profit_growth"] = _yoy_growth_from_abs(prof).reindex(out["date"]).values

    out["pub_date"] = pd.NaT
    out = out[list(OUT_COLS)].drop_duplicates("date", keep="last").reset_index(drop=True)
    return out


def normalize_fundamental_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Unified normalizer:
      - If detect YYYYMMDD columns => wide
      - Else if detect date column => long
      - Else empty
    """
    if raw is None or raw.empty:
        return _empty_frame()

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if _detect_wide_date_cols(df.columns):
        return _wide_to_metrics(df)

    if _detect_date_col_long(df) is not None:
        return _long_to_metrics(df)

    return _empty_frame()


# ----------------------------
# 4) pub_date mapping via cninfo disclosure
# ----------------------------
# 使用非捕获组 (?:...) 避免 pandas UserWarning
_TITLE_BAD_RE = re.compile(r"(?:更正|修订|更新|补充|更正后|修正|取消|摘要更正|澄清)", re.IGNORECASE)


def _title_to_report_end(title: str) -> Optional[pd.Timestamp]:
    """
    Parse cninfo announcement title -> report_end_date.
    We only map: Q1(0331), H1(0630), Q3(0930), Annual(1231).
    """
    t = str(title or "").strip()
    m = re.search(r"(\d{4})年", t)
    if not m:
        return None
    year = int(m.group(1))

    if re.search(r"(年度报告|年报)", t):
        return pd.to_datetime(f"{year}1231", errors="coerce")
    if re.search(r"(半年度报告|半年报|中期报告)", t):
        return pd.to_datetime(f"{year}0630", errors="coerce")
    if re.search(r"(第一季度报告|一季度报告|一季报)", t):
        return pd.to_datetime(f"{year}0331", errors="coerce")
    if re.search(r"(第三季度报告|三季度报告|三季报)", t):
        return pd.to_datetime(f"{year}0930", errors="coerce")
    return None


def _fetch_pub_date_map_cninfo(
    ak_client: AkClient,
    code: str,
    start_year: str,
    logger,
) -> pd.DataFrame:
    """
    Use cninfo disclosure interface to build (report_end_date -> pub_date) map.

    Interface: stock_zh_a_disclosure_report_cninfo
      outputs include 公告标题 / 公告时间.
    """
    start_date = f"{start_year}0101"
    end_date = time.strftime("%Y%m%d")

    cats = ("年报", "半年报", "一季报", "三季报")
    frames: List[pd.DataFrame] = []

    for cat in cats:
        try:
            df = ak_client.call(
                ak.stock_zh_a_disclosure_report_cninfo,
                symbol=code,
                market="沪深京",
                category=cat,
                start_date=start_date,
                end_date=end_date,
                keyword="",
            )
        except TypeError:
            # 兼容少数版本 keyword 可能不是必填
            df = ak_client.call(
                ak.stock_zh_a_disclosure_report_cninfo,
                symbol=code,
                market="沪深京",
                category=cat,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as e:
            logger.debug(f"[Fundamental][PubDate] cninfo failed code={code} cat={cat}: {e}")
            continue

        if df is None or df.empty:
            continue

        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        title_col = next((c for c in ("公告标题", "标题", "公告名称") if c in df.columns), None)
        time_col = next((c for c in ("公告时间", "公告日期", "发布时间") if c in df.columns), None)
        if not title_col or not time_col:
            continue

        df = df[[title_col, time_col]].rename(columns={title_col: "title", time_col: "pub_date"})
        df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce")
        df = df.dropna(subset=["pub_date"])
        df["report_end"] = df["title"].map(_title_to_report_end)
        df = df.dropna(subset=["report_end"])

        # Fix: explicitly use regex=True and ensure _TITLE_BAD_RE is non-capturing
        good = df[~df["title"].astype(str).str.contains(_TITLE_BAD_RE, na=False, regex=True)]
        use = good if not good.empty else df

        frames.append(use[["report_end", "pub_date"]])

    if not frames:
        return pd.DataFrame(columns=["date", "pub_date"])

    x = pd.concat(frames, ignore_index=True)
    # For each report_end, take earliest pub_date (closest to first disclosure)
    m = x.groupby("report_end")["pub_date"].min().reset_index()
    m = m.rename(columns={"report_end": "date"})
    return m


def _estimate_pub_date(report_date: pd.Timestamp) -> pd.Timestamp:
    """
    Estimate pub_date based on statutory deadlines if real date is missing.
    Rules (A-Share):
      Q1 (03-31) -> 04-30
      H1 (06-30) -> 08-31
      Q3 (09-30) -> 10-31
      FY (12-31) -> 04-30 (next year)
    """
    if pd.isna(report_date):
        return pd.NaT

    try:
        m, d = report_date.month, report_date.day
        y = report_date.year
        if m == 3 and d == 31:
            return pd.Timestamp(f"{y}-04-30")
        elif m == 6 and d == 30:
            return pd.Timestamp(f"{y}-08-31")
        elif m == 9 and d == 30:
            return pd.Timestamp(f"{y}-10-31")
        elif m == 12 and d == 31:
            return pd.Timestamp(f"{y + 1}-04-30")
    except Exception:
        pass

    return pd.NaT


def _attach_pub_dates(
    ak_client: AkClient,
    out: pd.DataFrame,
    code: str,
    start_year: str,
    cfg: DPConfig,
    logger,
) -> pd.DataFrame:
    if out is None or out.empty:
        return out

    sync_pub = bool(cfg.get("SYNC_FUNDAMENTAL_PUBDATE", True))

    out2 = out.copy()
    if "pub_date" not in out2.columns:
        out2["pub_date"] = pd.NaT

    if sync_pub:
        # 1. Try real fetch
        mp = _fetch_pub_date_map_cninfo(ak_client, code, start_year=start_year, logger=logger)
        if mp is not None and not mp.empty:
            # Drop old (empty) pub_date and merge new one
            out2 = out2.drop(columns=["pub_date"], errors="ignore").merge(mp, on="date", how="left")

    # 2. Fill missing with estimate
    # 无论是没开启同步、同步失败、还是同步了但缺某几期，都对 NaT 进行补全
    mask_missing = out2["pub_date"].isna()
    if mask_missing.any():
        out2.loc[mask_missing, "pub_date"] = out2.loc[mask_missing, "date"].apply(_estimate_pub_date)

    # ensure schema order
    for c in OUT_COLS:
        if c not in out2.columns:
            out2[c] = np.nan
    return out2[list(OUT_COLS)]


# ----------------------------
# 5) pipeline
# ----------------------------
class FundamentalPipeline:
    """
    Download & cache quarterly fundamentals per-code.
    Output: {DATA_DIR}/fundamental/{code}.parquet
      columns: date(report_end), roe, rev_growth, profit_growth, debt_ratio, eps, bps, pub_date

    Fundamentals from: ak.stock_financial_abstract
    Pub dates from: ak.stock_zh_a_disclosure_report_cninfo
    """
    SCHEMA_VER = 2

    def __init__(self, cfg: DPConfig, ak_client: AkClient, logger):
        self.cfg = cfg
        self.ak_client = ak_client
        self.logger = logger
        os.makedirs(fundamental_dir(cfg), exist_ok=True)

    def _should_skip(self, path: str) -> bool:
        days = int(self.cfg.get("FUND_TTL_DAYS", 5) or 5)
        ttl = max(1, days) * 24 * 3600
        return (
            os.path.exists(path)
            and os.path.getsize(path) > 512
            and (time.time() - os.path.getmtime(path)) < ttl
        )

    def _download_one(self, code: str) -> Tuple[str, bool, str, int]:
        c = normalize_code(code)
        if not c:
            return str(code), True, "BadCode", 0

        path = fundamental_path(self.cfg, c)
        if self._should_skip(path):
            return c, True, "Skipped", -1

        start_year = str(self.cfg.get("FUNDAMENTAL_START_YEAR", "2010") or "2010")

        try:
            raw = _best_effort_call_financial_abstract(self.ak_client, c)
            out = normalize_fundamental_frame(raw)
            out = _attach_pub_dates(self.ak_client, out, code=c, start_year=start_year, cfg=self.cfg, logger=self.logger)

            atomic_save_parquet(
                out,
                path,
                index=False,
                compression=str(self.cfg.get("PARQUET_COMPRESSION", "zstd") or "zstd"),
            )
            return c, True, ("Empty" if out.empty else "Success"), int(len(out))
        except Exception as e:
            return c, False, f"Failed({type(e).__name__})", 0

    def download(self, codes) -> None:
        if not bool(self.cfg.get("SYNC_FUNDAMENTAL", False)):
            self.logger.info("🟦 [Fundamental] SYNC_FUNDAMENTAL=False; skip.")
            return

        codes = [normalize_code(c) for c in codes]
        codes = [c for c in codes if c]
        if not codes:
            self.logger.warning("🟦 [Fundamental] empty codes; skip.")
            return

        workers = int(self.cfg.get("FIN_WORKERS", 8) or 8)
        max_inflight = int(self.cfg.get("FIN_MAX_INFLIGHT", workers * 4) or (workers * 4))

        self.logger.info(f"🟦 [Fundamental] syncing {len(codes)} codes ... workers={workers} inflight={max_inflight}")

        q = deque(codes)
        ok = bad = empty = skipped = 0

        def submit_more(ex, inflight):
            while q and len(inflight) < max_inflight:
                c = q.popleft()
                inflight[ex.submit(self._download_one, c)] = c

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            inflight = {}
            submit_more(ex, inflight)

        with tqdm(total=len(codes), dynamic_ncols=True, desc="Fundamental", unit="code") as pbar:
            while inflight:
                done, _ = concurrent.futures.wait(
                    inflight.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done:
                    _ = inflight.pop(fut, None)
                    code, success, msg, rows = fut.result()
                    if success:
                        ok += 1
                        if msg == "Empty":
                            empty += 1
                        if msg == "Skipped":
                            skipped += 1
                    else:
                        bad += 1

                    pbar.update(1)
                    pbar.set_postfix(ok=ok, bad=bad, empty=empty, skipped=skipped, last=code)

                submit_more(ex, inflight)

        self.logger.info(f"🟦 [Fundamental] done. ok={ok}, fail={bad}, empty={empty}, skipped={skipped}")