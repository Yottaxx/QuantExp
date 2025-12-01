from __future__ import annotations
import re
from typing import Any, Optional
import pandas as pd

_CODE6 = re.compile(r"(\d{6})")

def normalize_code(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    m = _CODE6.search(s)
    return m.group(1) if m else None

def norm_code_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_code)
