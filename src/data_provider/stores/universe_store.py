from __future__ import annotations
import os, time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from ..core.config import DPConfig
from ..utils.io import atomic_save_parquet
from ..utils.code import normalize_code, norm_code_series

class CodeMeta:
    @staticmethod
    def infer_board(code: str) -> str:
        c = str(code).strip()
        if c.startswith(("300","301")): return "GEM"
        if c.startswith(("688","689")): return "STAR"
        if c.startswith(("8","4")): return "BSE"
        if c.startswith(("200","900")): return "BSHARE"
        return "MAIN"

    @staticmethod
    def is_st(name: Optional[str]) -> bool:
        if not name: return False
        return "ST" in str(name).upper()

    @staticmethod
    def limit_rate(cfg: DPConfig, board: str, is_st: bool) -> float:
        main = float(cfg.get("LIMIT_RATE_MAIN", 0.10) or 0.10)
        st = float(cfg.get("LIMIT_RATE_ST", 0.05) or 0.05)
        gem = float(cfg.get("LIMIT_RATE_GEM", 0.20) or 0.20)
        star = float(cfg.get("LIMIT_RATE_STAR", 0.20) or 0.20)
        bse = float(cfg.get("LIMIT_RATE_BSE", 0.30) or 0.30)
        bshare = float(cfg.get("LIMIT_RATE_BSHARE", 0.10) or 0.10)
        if board == "BSE": return bse
        if board == "GEM": return gem
        if board == "STAR": return star
        if board == "BSHARE": return bshare
        return st if is_st else main

class UniverseStore:
    def __init__(self, cfg: DPConfig, ak_client, logger, ak_module, price_glob_fn):
        self.cfg = cfg
        self.ak = ak_module
        self.ak_client = ak_client
        self.logger = logger
        self.price_glob = price_glob_fn

    def _dir(self) -> str:
        root = str(self.cfg.get("DATA_DIR","./data") or "./data")
        d = os.path.join(root, "universe")
        os.makedirs(d, exist_ok=True)
        return d

    def _snapshot_path(self, date: pd.Timestamp) -> str:
        return os.path.join(self._dir(), f"universe_{pd.Timestamp(date).strftime('%Y%m%d')}.parquet")

    def _master_path(self) -> str:
        return os.path.join(self._dir(), "code_master.parquet")

    def _is_fresh(self, path: str, ttl: int) -> bool:
        return os.path.exists(path) and os.path.getsize(path) >= 512 and (time.time()-os.path.getmtime(path)) < ttl

    @staticmethod
    def _extract_code_name(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["code","name"])
        code_col = next((c for c in ["代码","证券代码","code","symbol"] if c in df.columns), None)
        name_col = next((c for c in ["名称","证券简称","name","short_name"] if c in df.columns), None)
        if code_col is None:
            return pd.DataFrame(columns=["code","name"])
        out = pd.DataFrame({"code": df[code_col]})
        out["name"] = df[name_col].astype(str) if name_col is not None else ""
        out["code"] = norm_code_series(out["code"])
        out = out.dropna(subset=["code"]).drop_duplicates("code").reset_index(drop=True)
        out["code"] = out["code"].astype(str)
        out["name"] = out["name"].fillna("").astype(str)
        return out

    def _try_all_code_lists(self) -> pd.DataFrame:
        # lower priority than spot
        for fn_name in ["stock_info_a_code_name","stock_info_sz_name_code","stock_info_sh_name_code"]:
            fn = getattr(self.ak, fn_name, None)
            if callable(fn):
                try:
                    df = self.ak_client.call(fn)
                    out = self._extract_code_name(df)
                    if not out.empty:
                        out["source"] = fn_name
                        return out
                except Exception:
                    continue
        return pd.DataFrame(columns=["code","name","source"])

    def _cached_price_codes(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for adj in ["raw","qfq","hfq"]:
            for p in self.price_glob(adj):
                code = normalize_code(os.path.basename(p).replace(".parquet",""))
                if code:
                    rows.append({"code": code, "name": "", "source": f"price_cache_{adj}"})
        if not rows:
            return pd.DataFrame(columns=["code","name","source"])
        return pd.DataFrame(rows).drop_duplicates("code").reset_index(drop=True)

    def _extra_codes_file(self) -> pd.DataFrame:
        path = str(self.cfg.get("UNIVERSE_EXTRA_CODES_FILE","") or "").strip()
        if not path:
            return pd.DataFrame(columns=["code","name","source"])
        if not os.path.exists(path):
            self.logger.warning(f"[Universe] UNIVERSE_EXTRA_CODES_FILE not found: {path}")
            return pd.DataFrame(columns=["code","name","source"])
        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
                out = self._extract_code_name(df)
            else:
                with open(path,"r",encoding="utf-8") as f:
                    codes = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
                out = pd.DataFrame({"code": norm_code_series(pd.Series(codes)), "name": ""})
                out = out.dropna(subset=["code"])
                out["code"] = out["code"].astype(str)
            if out.empty:
                return pd.DataFrame(columns=["code","name","source"])
            out["source"] = "extra_codes_file"
            return out
        except Exception as e:
            self.logger.warning(f"[Universe] read extra codes failed: {e}")
            return pd.DataFrame(columns=["code","name","source"])

    def _apply_market_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        include_bse = bool(self.cfg.get("INCLUDE_BSE", False))
        include_bshare = bool(self.cfg.get("INCLUDE_BSHARE", False))
        board = df["code"].map(CodeMeta.infer_board)
        keep = pd.Series(True, index=df.index)
        if not include_bse:
            keep &= board != "BSE"
        if not include_bshare:
            keep &= board != "BSHARE"
        allow_prefixes = self.cfg.get("UNIVERSE_ALLOW_PREFIXES", None)
        if allow_prefixes:
            allow_prefixes = [str(x) for x in allow_prefixes]
            ok = pd.Series(False, index=df.index)
            for p in allow_prefixes:
                ok |= df["code"].str.startswith(p)
            keep &= ok
        return df.loc[keep].reset_index(drop=True)

    def get_snapshot(self, date: pd.Timestamp, force_refresh: bool = False) -> pd.DataFrame:
        date = pd.Timestamp(date).normalize()
        path = self._snapshot_path(date)
        ttl = int(self.cfg.get("UNIVERSE_TTL_SEC", 24*3600) or 24*3600)
        if (not force_refresh) and self._is_fresh(path, ttl):
            try:
                return pd.read_parquet(path)
            except Exception:
                pass

        # Explicit priority pipeline: spot_em > code_lists > price_cache > extra_file
        try:
            spot = self.ak_client.call(self.ak.stock_zh_a_spot_em)
            spot2 = self._extract_code_name(spot)
            if not spot2.empty:
                spot2["source"] = "spot_em"
        except Exception:
            spot2 = pd.DataFrame(columns=["code","name","source"])

        all_codes = self._try_all_code_lists()
        cached = self._cached_price_codes()
        extra = self._extra_codes_file()

        uni = pd.concat([spot2, all_codes, cached, extra], ignore_index=True)
        if uni.empty:
            return uni
        uni["code"] = norm_code_series(uni["code"])
        uni = uni.dropna(subset=["code"])
        uni["code"] = uni["code"].astype(str)
        uni["name"] = uni.get("name","").fillna("").astype(str)

        # Dedup with explicit precedence: first occurrence wins by priority order above.
        uni["_rank"] = uni["source"].map({"spot_em": 0}).fillna(1).astype(int)
        uni = uni.sort_values(["_rank"]).drop_duplicates("code", keep="first").drop(columns=["_rank"])

        uni = self._apply_market_filters(uni)

        uni["board"] = uni["code"].map(CodeMeta.infer_board)
        uni["is_st"] = uni["name"].map(CodeMeta.is_st)
        uni["limit_rate"] = [float(CodeMeta.limit_rate(self.cfg, b, bool(s))) for b, s in zip(uni["board"], uni["is_st"])]
        uni["asof"] = date

        try:
            atomic_save_parquet(uni.reset_index(drop=True), path, index=False, compression=str(self.cfg.get("PARQUET_COMPRESSION","zstd") or "zstd"))
        except Exception as e:
            self.logger.warning(f"[Universe] persist snapshot failed: {e}")
        self._upsert_master(uni)
        return uni.reset_index(drop=True)

    def _upsert_master(self, snapshot: pd.DataFrame) -> None:
        mp = self._master_path()
        cur = None
        if os.path.exists(mp) and os.path.getsize(mp) >= 512:
            try: cur = pd.read_parquet(mp)
            except Exception: cur = None
        out = snapshot if cur is None or cur.empty else pd.concat([cur, snapshot], ignore_index=True)
        out["asof"] = pd.to_datetime(out["asof"], errors="coerce")
        out = out.sort_values(["asof"]).drop_duplicates(["code"], keep="last")
        try:
            atomic_save_parquet(out.reset_index(drop=True), mp, index=False, compression=str(self.cfg.get("PARQUET_COMPRESSION","zstd") or "zstd"))
        except Exception as e:
            self.logger.warning(f"[Universe] persist master failed: {e}")

    def meta_map_for_asof(self, date: pd.Timestamp) -> Dict[str, Dict[str, Any]]:
        date = pd.Timestamp(date).normalize()
        sp = self._snapshot_path(date)
        df = None
        if os.path.exists(sp) and os.path.getsize(sp) >= 512:
            try: df = pd.read_parquet(sp)
            except Exception: df = None
        if df is None or df.empty:
            mp = self._master_path()
            if os.path.exists(mp) and os.path.getsize(mp) >= 512:
                try:
                    df = pd.read_parquet(mp)
                    self.logger.warning(f"[Meta] snapshot missing for {date.date()}, fallback to master (may bias ST/limits).")
                except Exception:
                    df = None
        if df is None or df.empty:
            return {}
        df = df.drop_duplicates("code", keep="last")
        out: Dict[str, Dict[str, Any]] = {}
        for r in df.itertuples(index=False):
            code = normalize_code(getattr(r,"code",None))
            if not code: continue
            out[code] = {
                "name": getattr(r,"name",""),
                "board": getattr(r,"board", CodeMeta.infer_board(code)),
                "is_st": bool(getattr(r,"is_st", False)),
                "limit_rate": float(getattr(r,"limit_rate", CodeMeta.limit_rate(self.cfg, CodeMeta.infer_board(code), False))),
            }
        return out

    def get_codes(self, date: pd.Timestamp) -> List[str]:
        snap = self.get_snapshot(date)
        if snap is None or snap.empty:
            return []
        return snap["code"].astype(str).tolist()
