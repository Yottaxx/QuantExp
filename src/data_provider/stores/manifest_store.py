from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import pandas as pd
from ..utils.io import atomic_save_parquet
from ..utils.code import normalize_code
from .paths import norm_adjust

@dataclass(frozen=True)
class ManifestRow:
    code: str
    adjust: str
    last_date: Optional[pd.Timestamp]
    rows: int
    updated_at: pd.Timestamp
    schema_ver: int

class ManifestStore:
    def __init__(self, path: str, parquet_compression: str | None = "zstd"):
        self.path = path
        self.compression = parquet_compression
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path) or os.path.getsize(self.path) < 512:
            return pd.DataFrame(columns=["code","adjust","last_date","rows","updated_at","schema_ver"])
        try:
            df = pd.read_parquet(self.path)
        except Exception:
            return pd.DataFrame(columns=["code","adjust","last_date","rows","updated_at","schema_ver"])
        df["last_date"] = pd.to_datetime(df.get("last_date"), errors="coerce")
        df["updated_at"] = pd.to_datetime(df.get("updated_at"), errors="coerce")
        return df

    def save(self, df: pd.DataFrame) -> None:
        df = df.sort_values(["updated_at"]).drop_duplicates(["code","adjust"], keep="last")
        atomic_save_parquet(df, self.path, index=False, compression=self.compression)

    def upsert_many(self, rows: List[ManifestRow]) -> None:
        if not rows:
            return
        cur = self.load()
        add = pd.DataFrame([{
            "code": r.code,
            "adjust": r.adjust,
            "last_date": r.last_date,
            "rows": int(r.rows),
            "updated_at": r.updated_at,
            "schema_ver": int(r.schema_ver),
        } for r in rows])
        self.save(pd.concat([cur, add], ignore_index=True))

    @staticmethod
    def to_map(df: pd.DataFrame) -> Dict[Tuple[str,str], Tuple[Optional[pd.Timestamp], int, Optional[pd.Timestamp]]]:
        out: Dict[Tuple[str,str], Tuple[Optional[pd.Timestamp], int, Optional[pd.Timestamp]]] = {}
        if df is None or df.empty:
            return out
        for rr in df.itertuples(index=False):
            c = normalize_code(getattr(rr,"code",None))
            if not c:
                continue
            a = norm_adjust(getattr(rr,"adjust",None))
            ld = pd.to_datetime(getattr(rr,"last_date",None), errors="coerce") if getattr(rr,"last_date",None) is not None else None
            rows = int(getattr(rr,"rows",0) or 0)
            ua = pd.to_datetime(getattr(rr,"updated_at",None), errors="coerce") if getattr(rr,"updated_at",None) is not None else None
            out[(c,a)] = (ld, rows, ua)
        return out
