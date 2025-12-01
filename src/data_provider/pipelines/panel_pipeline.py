from __future__ import annotations
import concurrent.futures
import hashlib
import itertools
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..core.config import DPConfig
from ..core.logging import setup_logger
from ..core.qc import RejectReport, validate_price_frame
from ..core.version import VERSION
from ..rules.normalize import coerce_cached_price_df
from ..rules.trading_rules import TradingParams, add_liquidity_features, add_trade_masks, compute_hold_all_days_mask
from ..rules.split_policy import SplitPolicy
from ..stores.paths import norm_adjust
from ..stores.price_store import price_glob, read_price
from ..stores.panel_store import PanelMeta, PanelStore
from ..stores.info_store import InfoStore
from ..stores.fundamental_store import FundamentalStore
from ..utils.io import atomic_save_parquet, atomic_save_json
from ..utils.code import normalize_code, norm_code_series

# external deps (kept injectable in facade)
from .. import __init__ as _noop  # noqa: F401

def _fingerprint(cfg: DPConfig, AlphaFactory: Any) -> str:
    keys = cfg.get("PANEL_CACHE_FINGERPRINT_KEYS", None)
    if not keys:
        keys = [
            "CONTEXT_LEN","PRED_LEN","STRIDE","PRICE_SCHEMA_VER",
            "FEATURE_PREFIXES","ALPHA_BACKEND","USE_FUNDAMENTAL",
            "USE_CROSS_SECTIONAL","UNIVERSE_MIN_PRICE","UNIVERSE_MIN_LIST_DAYS",
            "MIN_LIST_DAYS","MIN_DOLLAR_VOL_FOR_TRADE","MIN_PRICE",
            "LIMIT_RATE_MAIN","LIMIT_RATE_ST","LIMIT_RATE_GEM","LIMIT_RATE_STAR","LIMIT_RATE_BSE","LIMIT_RATE_BSHARE",
            "LIMIT_EPS","DATASET_BACKEND","INCLUDE_BSE","INCLUDE_BSHARE",
            "GATE_TARGET_WITH_ENTRY","GATE_TARGET_WITH_EXIT","ALIGN_TO_CALENDAR",
            "ENTRY_PRICE_MODE","DELIST_GAP_DAYS","GATE_TARGET_WITH_HOLD_ALL_DAYS",
        ]
    payload = {"dp_ver": VERSION}
    for k in keys:
        payload[k] = cfg.get(k, None)
    payload["alpha_factory"] = getattr(AlphaFactory, "VERSION", None) or getattr(AlphaFactory, "__name__", "AlphaFactory")
    raw = str(sorted(payload.items())).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]

def _align_to_calendar(df: pd.DataFrame, trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    df = df.sort_values(["code","date"]).copy()
    out_parts = []
    for code, g in df.groupby("code", sort=False):
        g = g.copy()
        dmin, dmax = pd.to_datetime(g["date"]).min(), pd.to_datetime(g["date"]).max()
        idx = trade_dates[(trade_dates >= dmin) & (trade_dates <= dmax)]
        base = g.set_index("date")
        base.index = pd.to_datetime(base.index, errors="coerce")
        base = base[~base.index.isna()]
        base = base[~base.index.duplicated(keep="last")].sort_index()
        base = base.reindex(idx)
        base["code"] = code
        base = base.reset_index().rename(columns={"index":"date"})
        out_parts.append(base)
    return pd.concat(out_parts, ignore_index=True) if out_parts else df

def _attach_code_meta(df: pd.DataFrame, meta_map: Dict[str, Dict[str, Any]], sec_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    code = normalize_code(df["code"].iloc[0]) or str(df["code"].iloc[0])
    m = meta_map.get(code, {})
    s = sec_map.get(code, {})
    out = df.copy()
    out["code"] = norm_code_series(out["code"])
    out = out.dropna(subset=["code"]).copy()
    out["code"] = out["code"].astype(str)
    out["board"] = str(m.get("board", s.get("board", "")))
    out["is_st"] = bool(m.get("is_st", False))
    out["limit_rate"] = float(m.get("limit_rate", np.nan))
    out["name"] = str(m.get("name", s.get("name","")))
    out["first_date_hint"] = s.get("first_date", pd.NaT)
    out["last_date_hint"] = s.get("last_date", pd.NaT)
    out["is_delisted_guess"] = bool(s.get("is_delisted_guess", False))
    return out


def _attach_static_info(df: pd.DataFrame, info_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    code = normalize_code(df["code"].iloc[0]) or str(df["code"].iloc[0])
    m = info_map.get(code, {})
    out = df.copy()
    out["industry_cat"] = int(m.get("industry_cat", 0) or 0)
    # for neutralization/groupby, ints are fine
    out["industry"] = out["industry_cat"].astype(int, copy=False)
    out["list_date"] = str(m.get("list_date", "") or "")
    out["total_mkt_cap"] = float(m.get("total_mkt_cap", np.nan) or np.nan)
    # prefer static-info name when universe name missing
    if "name" in out.columns:
        out.loc[out["name"].astype(str).str.len() == 0, "name"] = str(m.get("name", "") or "")
    else:
        out["name"] = str(m.get("name", "") or "")
    return out

def _attach_fundamentals_pit(df: pd.DataFrame, fund_store: FundamentalStore, code: str, cfg: DPConfig) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not bool(cfg.get("USE_FUNDAMENTAL", False)):
        return df
    f = fund_store.load_one(code)
    if f is None:
        # ensure columns exist for downstream factor code (avoid KeyError)
        for c in ["roe", "rev_growth", "profit_growth", "debt_ratio", "eps", "bps", "pe_ttm", "pb"]:
            if c not in df.columns:
                df[c] = np.nan
        return df
    pit = fund_store.to_pit(f)
    if pit is None or pit.empty:
        for c in ["roe", "rev_growth", "profit_growth", "debt_ratio", "eps", "bps", "pe_ttm", "pb"]:
            if c not in df.columns:
                df[c] = np.nan
        return df

    x = df.sort_values("date").copy()
    pit = pit.sort_values("date").copy()

    # merge_asof on date (per-code already)
    merged = pd.merge_asof(x, pit, on="date", direction="backward")

    # derived ratios (best-effort: report-level eps/bps carried forward)
    limit_eps = float(cfg.get("LIMIT_EPS", 0.002) or 0.002)
    if "pe_ttm" not in merged.columns:
        eps = pd.to_numeric(merged.get("eps", np.nan), errors="coerce")
        pe = np.where(eps > limit_eps, merged["close"].astype(float) / eps, np.nan)
        merged["pe_ttm"] = pe.astype(np.float32)
    if "pb" not in merged.columns:
        bps = pd.to_numeric(merged.get("bps", np.nan), errors="coerce")
        pb = np.where(bps > limit_eps, merged["close"].astype(float) / bps, np.nan)
        merged["pb"] = pb.astype(np.float32)

    # cast core fundamentals
    for c in ["roe", "rev_growth", "profit_growth", "debt_ratio", "eps", "bps", "pe_ttm", "pb"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").astype(np.float32)
        else:
            merged[c] = np.nan
    return merged


def _tag_universe(df: pd.DataFrame, cfg: DPConfig) -> pd.DataFrame:
    df = df.sort_values(["code","date"]).copy()
    df["list_days_count"] = df.groupby("code")["date"].cumcount() + 1
    cond_vol = df.get("volume", 0.0).fillna(0.0) > 0
    cond_price = df["close"].fillna(0.0) >= float(cfg.get("UNIVERSE_MIN_PRICE", 2.0) or 2.0)
    cond_list = df["list_days_count"] > int(cfg.get("UNIVERSE_MIN_LIST_DAYS", 60) or 60)
    df["is_universe"] = (cond_vol & cond_price & cond_list).astype(bool)
    df.drop(columns=["list_days_count"], inplace=True)
    return df

def _compute_cross_sectional_stats(parts: List[str], factor_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Two-pass stats: returns (mean, std) per date x col. Memory bounded."""
    acc_sum = None
    acc_sumsq = None
    acc_cnt = None
    for p in tqdm(parts, desc="CS-Stats", leave=False):
        df = pd.read_parquet(p, columns=["date"] + factor_cols)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        vals = df[factor_cols]
        cnt = vals.notna().astype(np.int32)
        vv = vals.fillna(0.0).astype(np.float64)
        g_sum = vv.groupby(df["date"]).sum()
        g_sumsq = (vv * vv).groupby(df["date"]).sum()
        g_cnt = cnt.groupby(df["date"]).sum()
        if acc_sum is None:
            acc_sum, acc_sumsq, acc_cnt = g_sum, g_sumsq, g_cnt
        else:
            acc_sum = acc_sum.add(g_sum, fill_value=0.0)
            acc_sumsq = acc_sumsq.add(g_sumsq, fill_value=0.0)
            acc_cnt = acc_cnt.add(g_cnt, fill_value=0.0)

    acc_cnt = acc_cnt.fillna(0.0)
    mean = acc_sum.divide(acc_cnt.replace(0, np.nan)).fillna(0.0)
    var = acc_sumsq.divide(acc_cnt.replace(0, np.nan)).fillna(0.0) - mean * mean
    std = np.sqrt(np.maximum(var, 1e-12))
    std = std.replace(0.0, 1.0).fillna(1.0)
    return mean, std

def _add_cross_sectional_to_parts(parts: List[str], factor_cols: List[str], cfg: DPConfig) -> List[str]:
    mean, std = _compute_cross_sectional_stats(parts, factor_cols)
    out_parts = []
    for p in tqdm(parts, desc="CS-Apply", leave=False):
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).reset_index(drop=True)
        if df.empty:
            continue
        idx = df["date"]
        mu = mean.reindex(idx).to_numpy(dtype=np.float32, copy=False)
        sd = std.reindex(idx).to_numpy(dtype=np.float32, copy=False)
        base = df[factor_cols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        z = (base - mu) / sd
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        for j, c in enumerate(factor_cols):
            df[f"cs_{c}"] = z[:, j]
        atomic_save_parquet(df, p, index=False, compression=str(cfg.get("PARQUET_COMPRESSION","zstd") or "zstd"))
        out_parts.append(p)
    return out_parts

class PanelPipeline:
    def __init__(self, cfg: DPConfig, logger, calendar_store, universe_store, secmaster_store, AlphaFactory):
        self.cfg = cfg
        self.logger = logger
        self.calendar = calendar_store
        self.universe = universe_store
        self.secmaster = secmaster_store
        self.AlphaFactory = AlphaFactory
        self.info_store = InfoStore(cfg, logger)
        self.fund_store = FundamentalStore(cfg, logger)

    def _process_one_code_file(self, price_file: str, mode: str, adj_norm: str,
                              meta_map: Dict[str, Dict[str, Any]], sec_map: Dict[str, Dict[str, Any]],
                              info_map: Dict[str, Dict[str, Any]],
                              trade_dates: Optional[pd.DatetimeIndex]) -> Tuple[Optional[pd.DataFrame], str]:
        df = read_price(self.cfg, price_file, expected_adj=adj_norm, logger=self.logger)
        if df is None or df.empty:
            return None, "EmptyPrice"
        df = coerce_cached_price_df(df)
        code = normalize_code(df.get("code", "").iloc[0] if "code" in df.columns else None) or normalize_code(os.path.basename(price_file)) or ""
        if not code:
            return None, "BadCode"
        df["code"] = code

        df = _attach_code_meta(df, meta_map, sec_map)

        if bool(self.cfg.get("ALIGN_TO_CALENDAR", False)) and trade_dates is not None:
            df = _align_to_calendar(df, trade_dates)

        ok, reason = validate_price_frame(df)
        if not ok:
            return None, reason

        context_len = int(self.cfg.get("CONTEXT_LEN", 60) or 60)
        if len(df) <= context_len:
            return None, "TooShort"

        df = add_liquidity_features(df)
        tp = TradingParams(
            min_list_days=int(self.cfg.get("MIN_LIST_DAYS", 60) or 60),
            min_dollar_vol=float(self.cfg.get("MIN_DOLLAR_VOL_FOR_TRADE", 1e6) or 1e6),
            min_price=float(self.cfg.get("MIN_PRICE", 1.0) or 1.0),
            limit_eps=float(self.cfg.get("LIMIT_EPS", 0.002) or 0.002),
        )
        df = add_trade_masks(df, tp)

        # Static info & PIT fundamentals (no look-ahead)
        df = _attach_static_info(df, info_map)
        df = _attach_fundamentals_pit(df, self.fund_store, code, self.cfg)

        df = af.make_factors()  # type: ignore

        pred_len = int(self.cfg.get("PRED_LEN", 5) or 5)
        g = df.groupby("code", sort=False)

        entry_price_mode = str(self.cfg.get("ENTRY_PRICE_MODE", "open") or "open").lower().strip()
        if entry_price_mode == "open":
            df["entry_price"] = g["open"].shift(-1)
        elif entry_price_mode == "close":
            df["entry_price"] = g["close"].shift(-1)
        else:
            df["entry_price"] = g["open"].shift(-1)

        df["future_close"] = g["close"].shift(-pred_len)
        df["target"] = df["future_close"] / (df["entry_price"] + 1e-9) - 1.0

        gate_entry = bool(self.cfg.get("GATE_TARGET_WITH_ENTRY", True))
        gate_exit = bool(self.cfg.get("GATE_TARGET_WITH_EXIT", False))
        gate_hold_all = bool(self.cfg.get("GATE_TARGET_WITH_HOLD_ALL_DAYS", False))

        if gate_entry:
            entry_ok = g["buyable_mask"].shift(-1).notna()
            df.loc[~entry_ok, "target"] = np.nan

        if gate_exit and pred_len > 0:
            exit_ok = g["sellable_mask"].shift(-pred_len).notna()
            df.loc[~exit_ok, "target"] = np.nan

        if gate_hold_all and pred_len > 0:
            hold_ok = compute_hold_all_days_mask(df, pred_len=pred_len)
            df.loc[~hold_ok, "target"] = np.nan

        df.loc[~np.isfinite(df["target"].astype(float)), "target"] = np.nan
        if mode == "train":
            df = df.dropna(subset=["target"]).reset_index(drop=True)

        df = _tag_universe(df, self.cfg)

        return df, "OK"

    def build_parts(self, mode: str, force_refresh: bool, adjust: str, backend: Optional[str], debug: bool,
                    meta_map: Dict[str, Dict[str, Any]], sec_map: Dict[str, Dict[str, Any]],
                    universe_asof: pd.Timestamp) -> Tuple[List[str], List[str], RejectReport, str, str]:
        adj_norm = norm_adjust(adjust)
        fp = _fingerprint(self.cfg, self.AlphaFactory)
        store = PanelStore(self.cfg)
        part_dir = store.parts_dir(mode, adj_norm, fp)
        os.makedirs(part_dir, exist_ok=True)

        # Collect files
        price_files = price_glob(self.cfg, adj_norm, self.logger)
        if not price_files:
            raise RuntimeError(f"No price data found for adjust={adj_norm}. Run download_data() first.")

        # Static info (industry/list-date etc). Build once per panel for speed.
        info_df = self.info_store.load_master(force=False)
        info_map = self.info_store.to_map(info_df)

        if debug or bool(self.cfg.get("DEBUG", False)):
            backend = "serial"
            price_files = price_files[: int(self.cfg.get('DEBUG_MAX_FILES', 10) or 10)]
        backend = (backend or str(self.cfg.get("ALPHA_BACKEND", "process") or "process")).lower().strip()

        trade_dates = None
        if bool(self.cfg.get("ALIGN_TO_CALENDAR", False)):
            trade_dates = self.calendar.get_trade_dates()

        flush_n = int(self.cfg.get("PANEL_FLUSH_N", 200) or 200)
        buffer: List[pd.DataFrame] = []
        part_paths: List[str] = []
        reject = RejectReport(max_samples_per_reason=30)

        def flush():
            if not buffer:
                return
            p = os.path.join(part_dir, f"part_{len(part_paths):05d}.parquet")
            tmp = pd.concat(buffer, ignore_index=True)
            atomic_save_parquet(tmp, p, index=False, compression=str(self.cfg.get("PARQUET_COMPRESSION","zstd") or "zstd"))
            part_paths.append(p)
            buffer.clear()

        fail_fast = bool(self.cfg.get("FAIL_FAST", True))


        if backend == "serial":
            for pf in tqdm(price_files, desc="Panel(serial)"):
                try:
                    out, reason = self._process_one_code_file(pf, mode, adj_norm, meta_map, sec_map, info_map, trade_dates)
                    if out is None or out.empty:
                        reject.add(reason, normalize_code(os.path.basename(pf)) or pf[-10:])
                        continue
                    buffer.append(out)
                except Exception as e:
                    reject.add(f"Exc({type(e).__name__})", normalize_code(os.path.basename(pf)) or pf[-10:])
                    self.logger.exception(f"[serial] Failed on file={pf}: {e}")
                    if fail_fast:
                        raise
                if len(buffer) >= flush_n:
                    flush()
            flush()

        elif backend in ("thread", "threads"):
            workers = int(self.cfg.get("ALPHA_WORKERS", min(32, (os.cpu_count() or 8)+8)) or 16)
            self.logger.info(f"[Panel] backend=threads workers={workers}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(self._process_one_code_file, pf, mode, adj_norm, meta_map, sec_map, info_map, trade_dates) for pf in price_files]
                for pf, fut in tqdm(list(zip(price_files, futs)), total=len(futs), desc="Panel(threads)"):
                    try:
                        out, reason = fut.result()
                        if out is None or out.empty:
                            reject.add(reason, normalize_code(os.path.basename(pf)) or pf[-10:])
                            continue
                        buffer.append(out)
                    except Exception as e:
                        reject.add(f"Exc({type(e).__name__})", normalize_code(os.path.basename(pf)) or pf[-10:])
                        self.logger.exception(f"[threads] Failed on file={pf}: {e}")
                        if fail_fast:
                            raise
                    if len(buffer) >= flush_n:
                        flush()
            flush()

        elif backend == "process":
            # For portability: processes often fail to pickle AlphaFactory. Use threads unless you have a picklable implementation.
            self.logger.warning("backend=process is not enabled in v24 sample; use backend='threads' for parallelization.")
            return self.build_parts(mode, force_refresh, adjust, backend="threads", debug=debug, meta_map=meta_map, sec_map=sec_map, universe_asof=universe_asof)
        else:
            raise ValueError(f"Unknown backend={backend}")

        if not part_paths:
            raise ValueError("Not enough valid price/factor data to build panel parts (no parts).")

        # Determine feature columns by prefixes
        prefixes = list(self.cfg.get("FEATURE_PREFIXES", ["alpha_","fac_","cs_"]) or ["alpha_","fac_","cs_"])
        # Probe first part to list columns
        probe = pd.read_parquet(part_paths[0], nrows=5)
        feat_cols = [c for c in probe.columns if any(str(c).startswith(p) for p in prefixes)]
        base_factor_cols = [c for c in feat_cols if not str(c).startswith("cs_")]

        # Cross-sectional factors added as cs_... and appended to feature_cols (optional)
        if bool(self.cfg.get("USE_CROSS_SECTIONAL", True)) and base_factor_cols:
            _add_cross_sectional_to_parts(part_paths, base_factor_cols, self.cfg)
            # update feat_cols including cs_
            probe2 = pd.read_parquet(part_paths[0], nrows=5)
            feat_cols = [c for c in probe2.columns if any(str(c).startswith(p) for p in prefixes)]

        # Persist reject report for observability
        rej_path = os.path.join(part_dir, f"rejects_{mode}_{adj_norm}_{fp}.json")
        atomic_save_json(reject.to_dict(), rej_path)

        # Write meta
        split_gap = int(self.cfg.get("SPLIT_GAP", int(self.cfg.get("CONTEXT_LEN", 60) or 60)) or 60)
        seq_len = int(self.cfg.get("CONTEXT_LEN", 60) or 60)
        stride = int(self.cfg.get("STRIDE", 1) or 1)
        # Compute unique dates from parts (cheap)
        dates = []
        for df in PanelStore(self.cfg).iter_parts(part_paths, columns=["date"]):
            d = pd.to_datetime(df["date"], errors="coerce").dropna().unique()
            if len(d):
                dates.append(d)
        unique_dates = np.unique(np.concatenate(dates)) if dates else np.array([], dtype="datetime64[ns]")
        split_manifest = SplitPolicy.compute(
            unique_dates,
            seq_len=seq_len, stride=stride, gap=split_gap,
            train_ratio=float(self.cfg.get("TRAIN_RATIO", 0.7) or 0.7),
            val_ratio=float(self.cfg.get("VAL_RATIO", 0.15) or 0.15),
            test_ratio=float(self.cfg.get("TEST_RATIO", None)) if self.cfg.get("TEST_RATIO", None) is not None else None,
            universe_asof=str(pd.Timestamp(universe_asof).date()),
            fingerprint=fp,
            adjust=adj_norm,
        ).to_dict()

        meta = PanelMeta(
            mode=mode,
            adjust=adj_norm,
            fingerprint=fp,
            universe_asof=str(pd.Timestamp(universe_asof).date()),
            feature_cols=feat_cols,
            part_paths=part_paths,
            created_by=VERSION,
            split_manifest=split_manifest,
        )
        store.write_meta(part_dir, meta)

        return part_paths, feat_cols, reject, part_dir, fp

    def materialize_panel_df(self, part_paths: List[str]) -> pd.DataFrame:
        merge_batch = int(self.cfg.get("PANEL_MERGE_BATCH", 24) or 24)
        merged_chunks: List[pd.DataFrame] = []
        for i in range(0, len(part_paths), merge_batch):
            batch = [pd.read_parquet(p) for p in part_paths[i:i+merge_batch]]
            merged_chunks.append(pd.concat(batch, ignore_index=True))
        panel_df = pd.concat(merged_chunks, ignore_index=True) if len(merged_chunks) > 1 else merged_chunks[0]
        panel_df["date"] = pd.to_datetime(panel_df["date"], errors="coerce")
        panel_df = panel_df.dropna(subset=["date"]).reset_index(drop=True)
        return panel_df

