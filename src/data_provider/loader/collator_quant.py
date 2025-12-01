# src/train/date_batching.py (continue)
from typing import List, Dict

import numpy as np
import torch

class StockDateCollator:
    """
    Collate list[dict] -> torch tensors.
    Expect each item contains:
      - past_values: (L, F) array-like
      - labels: float or shape (1,)
      - date_id: int
    """
    def __init__(
        self,
        *,
        label_dtype: torch.dtype = torch.float32,
        value_dtype: torch.dtype = torch.float32,
        assert_single_date: bool = True,
    ):
        self.label_dtype = label_dtype
        self.value_dtype = value_dtype
        self.assert_single_date = assert_single_date

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # past_values
        pv = [np.asarray(f["past_values"], dtype=np.float32) for f in features]
        past_values = torch.tensor(np.stack(pv, axis=0), dtype=self.value_dtype)

        # labels -> [B, 1]
        y = [float(np.asarray(f["labels"]).reshape(-1)[0]) for f in features]
        labels = torch.tensor(y, dtype=self.label_dtype).view(-1, 1)

        # date_id -> [B]
        did = [int(f.get("date_id", 0)) for f in features]
        date_id = torch.tensor(did, dtype=torch.int64)

        if self.assert_single_date and len(did) > 1:
            # 训练时保证横截面同日
            if any(d != did[0] for d in did[1:]):
                raise ValueError(f"Batch contains multiple date_id values: {sorted(set(did))[:8]} ...")

        return {"past_values": past_values, "labels": labels, "date_id": date_id}
