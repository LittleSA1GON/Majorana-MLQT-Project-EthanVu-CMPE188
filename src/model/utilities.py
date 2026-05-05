from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

try:
    import torch
except Exception:
    torch = None


@dataclass
class QCVVBatch:
    traces: Any
    valid_length: Any
    coord_normalized: Any
    sample_dt: Any
    family_id: Any | None = None
    run_id: Any | None = None
    sample_id: Any | None = None
    sequence_id: Any | None = None
    window_start: Any | None = None
    original_length: Any | None = None
    coord_values: Any | None = None
    finite_mask: Any | None = None
    nan_fraction: Any | None = None


def ensure_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_torch(x: Any, device: str | None = None, dtype=None):
    if torch is None:
        raise ImportError("torch is required for this operation.")
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype) if (device is not None or dtype is not None) else x
    return torch.as_tensor(x, device=device, dtype=dtype)


def make_length_mask(valid_length: Any, max_len: int | None = None):
    if torch is None:
        raise ImportError("torch is required for this operation.")
    valid_length = ensure_torch(valid_length, dtype=torch.long)
    if max_len is None:
        max_len = int(valid_length.max().item())
    rng = torch.arange(max_len, device=valid_length.device)[None, :]
    return rng < valid_length[:, None]


class StringIndexer:
    def __init__(self, classes: Iterable[str] | None = None):
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: list[str] = []
        if classes is not None:
            self.fit(classes)

    def fit(self, classes: Iterable[str]):
        uniq = sorted(set(str(x) for x in classes))
        self.class_to_idx = {c: i for i, c in enumerate(uniq)}
        self.idx_to_class = uniq
        return self

    def transform(self, values: Iterable[str]) -> np.ndarray:
        if not self.class_to_idx:
            raise ValueError("StringIndexer must be fit before transform.")
        return np.asarray([self.class_to_idx[str(v)] for v in values], dtype=np.int64)

    def fit_transform(self, values: Iterable[str]) -> np.ndarray:
        vals = [str(v) for v in values]
        self.fit(vals)
        return self.transform(vals)

    def inverse_transform(self, idx: Iterable[int]) -> list[str]:
        return [self.idx_to_class[int(i)] for i in idx]
