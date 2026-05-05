import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import h5py
import numpy as np

# Conservative CPU thread defaults copied from the stable merged trainer.
# They prevent GRU/DataLoader stalls on some CPU-only machines; users can
# override these environment variables before launching if needed.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

if TYPE_CHECKING:
    from torch import Tensor
    from torch import device as TorchDevice
    from torch.nn import Module as NNModule
else:
    Tensor = Any
    TorchDevice = Any
    NNModule = Any

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset
    from torch.amp import autocast, GradScaler
    from torch.optim.lr_scheduler import ReduceLROnPlateau
except Exception as exc:
    torch = None
    nn = None
    Dataset = object
    DataLoader = object
    GradScaler = None
    ReduceLROnPlateau = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

TRAIN_VERSION = "stable_v22_subset_sanitize_fix"

THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_READY_H5_DIR = REPO_ROOT / "manual-data" / "~ready"
DEFAULT_READY_PT_DIR = REPO_ROOT / "manual-data" / "~ready_torch"
DEFAULT_READY_DIR = DEFAULT_READY_PT_DIR if DEFAULT_READY_PT_DIR.exists() else DEFAULT_READY_H5_DIR
DEFAULT_RUNS_DIR = SRC_ROOT / "training" / "runs"


def _import_cnn_gru():
    errors = []
    for mod_name in [
        "model.cnn_gru",
        "cnn_gru_qcvv_model",
        "model.cnn_gru_qcvv_model",
    ]:
        try:
            mod = __import__(mod_name, fromlist=["CNNGRUQCVVModel", "CNNGRUQCVVConfig"])
            return mod.CNNGRUQCVVModel, mod.CNNGRUQCVVConfig
        except Exception as exc:
            errors.append(f"{mod_name}: {exc}")
    raise ImportError("Could not import CNN+GRU model. Tried:\n" + "\n".join(errors))


def _import_xgboost():
    errors = []
    for mod_name in [
        "model.xgboost",
        "xgboost_qcvv_model",
        "model.xgboost_qcvv_model",
    ]:
        try:
            mod = __import__(mod_name, fromlist=["XGBoostQCVVModel", "XGBoostQCVVConfig"])
            return mod.XGBoostQCVVModel, mod.XGBoostQCVVConfig
        except Exception as exc:
            errors.append(f"{mod_name}: {exc}")
    raise ImportError("Could not import XGBoost model. Tried:\n" + "\n".join(errors))


def _import_hsmm():
    errors = []
    for mod_name in [
        "model.hsmm",
        "hsmm_qcvv_model",
        "model.hsmm_qcvv_model",
    ]:
        try:
            mod = __import__(mod_name, fromlist=["GaussianHSMMQCVV", "HSMMConfig"])
            return mod.GaussianHSMMQCVV, mod.HSMMConfig
        except Exception as exc:
            errors.append(f"{mod_name}: {exc}")
    raise ImportError("Could not import HSMM model. Tried:\n" + "\n".join(errors))


def _import_dmm():
    errors = []
    for mod_name in [
        "model.dmm",
        "dmm",
        "model.deep_markov_model",
    ]:
        try:
            mod = __import__(mod_name, fromlist=["DeepMarkovQCVVModel", "DMMQCVVConfig"])
            return mod.DeepMarkovQCVVModel, mod.DMMQCVVConfig
        except Exception as exc:
            errors.append(f"{mod_name}: {exc}")
    raise ImportError("Could not import DMM model. Tried:\n" + "\n".join(errors))


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


class StringIndexer:
    def __init__(self):
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: list[str] = []

    def fit(self, values: Iterable[str]):
        uniq = sorted(set(str(v) for v in values))
        self.class_to_idx = {v: i for i, v in enumerate(uniq)}
        self.idx_to_class = uniq
        return self

    def transform(self, values: Iterable[str]) -> np.ndarray:
        return np.asarray([self.class_to_idx[str(v)] for v in values], dtype=np.int64)

    def fit_transform(self, values: Iterable[str]) -> np.ndarray:
        vals = [str(v) for v in values]
        self.fit(vals)
        return self.transform(vals)


def decode_str_array(arr: np.ndarray) -> np.ndarray:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object)


def _safe_string_item(value: Any, default: str = "na") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        return ",".join(str(x) for x in value.reshape(-1).tolist())
    if torch is not None and torch.is_tensor(value):
        if value.ndim == 0:
            return str(value.item())
        return ",".join(str(x) for x in value.detach().cpu().reshape(-1).tolist())
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    return str(value)


def build_group_key(source_file: str, source_group: str, run_name: str, operating_index: str) -> str:
    return f"{source_file}|{source_group}|{run_name}|{operating_index}"


def grouped_train_val_split(dataset, val_split: float, seed: int):
    groups = np.asarray(dataset.group_ids_raw, dtype=np.int64)
    all_indices = np.arange(len(dataset), dtype=np.int64)

    if len(np.unique(groups)) < 2:
        raise ValueError("Grouped split requires at least two unique groups.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(splitter.split(all_indices, groups=groups))

    train_groups = set(int(groups[i]) for i in train_idx.tolist())
    val_groups = set(int(groups[i]) for i in val_idx.tolist())
    overlap = train_groups & val_groups

    split_info = {
        "split_strategy": "grouped",
        "num_unique_groups": int(len(set(groups.tolist()))),
        "train_unique_groups": int(len(train_groups)),
        "val_unique_groups": int(len(val_groups)),
        "group_overlap_count": int(len(overlap)),
    }
    return train_idx, val_idx, split_info


def resolve_learning_task(args, num_classes: int) -> str:
    """
    auto:
    - use multitask when there are at least two supervised classes
    - use self_supervised when the family-specific target is one-class
    """
    requested = getattr(args, "learning_task", "auto")
    if requested == "auto":
        return "multitask" if int(num_classes) > 1 else "self_supervised"
    if requested in {"supervised", "multitask"} and int(num_classes) <= 1:
        print(
            "[WARN] Requested supervised classification but only one class is available; "
            "falling back to self_supervised so x_loop/z_loop can train meaningfully.",
            flush=True,
        )
        return "self_supervised"
    return requested


def classification_enabled(learning_task: str) -> bool:
    return learning_task in {"supervised", "multitask"}


def reconstruction_weight_for_task(args, learning_task: str) -> float:
    if learning_task == "supervised":
        return 0.0
    if learning_task == "self_supervised":
        return float(getattr(args, "self_supervised_recon_weight", 1.0))
    return float(getattr(args, "aux_recon_weight", 0.10))


def empty_or_nan_metrics() -> dict[str, float]:
    return {
        "accuracy": float("nan"),
        "precision_macro": float("nan"),
        "recall_macro": float("nan"),
        "f1_macro": float("nan"),
        "f1_weighted": float("nan"),
    }


def get_sanitize_count(ds) -> int:
    """Safely read sanitize_count from Dataset or Subset(Dataset)."""
    if hasattr(ds, "sanitize_count"):
        return int(getattr(ds, "sanitize_count", 0))
    if hasattr(ds, "dataset") and hasattr(ds.dataset, "sanitize_count"):
        return int(getattr(ds.dataset, "sanitize_count", 0))
    return 0


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def discover_prepared_files(ready_dir: str | Path = DEFAULT_READY_DIR) -> list[str]:
    ready_dir = Path(ready_dir)
    if not ready_dir.exists():
        raise FileNotFoundError(f"Ready directory not found: {ready_dir}")

    pt_files = sorted(str(p) for p in ready_dir.glob("*.pt"))
    if pt_files:
        return pt_files

    h5_files = sorted(str(p) for p in ready_dir.glob("*.h5"))
    if h5_files:
        return h5_files

    raise FileNotFoundError(f"No .pt or .h5 files found in: {ready_dir}")



def _torch_load_bundle(path: str | Path) -> dict[str, Any]:
    path = str(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class PreparedPTBundleDataset(Dataset):
    def __init__(
        self,
        file_paths: list[str | Path],
        target: str = "family",
        indices: np.ndarray | None = None,
        family_filter: str = "all",
        label_h5_dataset: str | None = None,
    ):
        if torch is None:
            raise ImportError(f"torch is required for PreparedPTBundleDataset: {_TORCH_IMPORT_ERROR}")

        self.file_paths = [str(Path(p)) for p in file_paths]
        self.target = target
        self.label_h5_dataset = label_h5_dataset
        self.sample_map: list[tuple[int, int]] = []
        self.family_values: list[str] = []
        self.run_values: list[str] = []
        self.family_ids_raw: list[int] = []
        self.run_ids_raw: list[int] = []
        self.group_keys: list[str] = []
        self.group_ids_raw: list[int] = []
        self.sanitize_count = 0
        self.bundles: list[dict[str, Any]] = []

        family_strings: list[str] = []
        run_strings: list[str] = []

        for file_idx, fp in enumerate(self.file_paths):
            bundle = _torch_load_bundle(fp)
            if bundle.get("format") != "prepared_pt_bundle_v1":
                raise ValueError(f"Unsupported PT bundle format in {fp}: {bundle.get('format')}")
            self.bundles.append(bundle)
            ds = bundle["datasets"]

            if "traces" not in ds or "family_name" not in ds or "run_name" not in ds:
                raise KeyError(f"Missing required datasets in PT bundle: {fp}")

            n = int(ds["traces"].shape[0])
            family = ds["family_name"]
            run = ds["run_name"]
            source_files = ds.get("source_file")
            source_groups = ds.get("source_group")
            operating_index = ds.get("operating_index")
            if len(family) != n or len(run) != n:
                raise ValueError(f"family_name/run_name length mismatch in {fp}")

            for local_idx in range(n):
                fam = str(family[local_idx])
                rn = str(run[local_idx])
                if family_filter != "all" and fam != family_filter:
                    continue
                source_file = _safe_string_item(source_files[local_idx] if source_files is not None else Path(fp).name, default=Path(fp).name)
                source_group = _safe_string_item(source_groups[local_idx] if source_groups is not None else "na", default="na")
                op_idx = _safe_string_item(operating_index[local_idx] if operating_index is not None else "na", default="na")
                group_key = build_group_key(source_file, source_group, rn, op_idx)

                self.sample_map.append((file_idx, local_idx))
                family_strings.append(fam)
                run_strings.append(rn)
                self.group_keys.append(group_key)

        self.family_indexer = StringIndexer().fit(family_strings)
        self.run_indexer = StringIndexer().fit(run_strings)
        self.family_values = family_strings
        self.run_values = run_strings
        self.family_ids_raw = self.family_indexer.transform(family_strings).tolist()
        self.run_ids_raw = self.run_indexer.transform(run_strings).tolist()
        self.group_ids_raw = StringIndexer().fit_transform(self.group_keys).tolist()
        self.indices = np.arange(len(self.sample_map), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        file_idx, local_idx = self.sample_map[real_idx]
        bundle = self.bundles[file_idx]["datasets"]

        traces = bundle["traces"][local_idx]
        valid_length = bundle["valid_length"][local_idx]
        coord_norm = bundle["coord_normalized"][local_idx]
        sample_dt = bundle["sample_dt"][local_idx]

        if not torch.is_tensor(traces):
            traces = torch.as_tensor(traces, dtype=torch.float32)
        else:
            traces = traces.to(dtype=torch.float32)
        if not torch.is_tensor(coord_norm):
            coord_norm = torch.as_tensor(coord_norm, dtype=torch.float32)
        else:
            coord_norm = coord_norm.to(dtype=torch.float32)
        if not torch.is_tensor(valid_length):
            valid_length = torch.as_tensor(valid_length, dtype=torch.long)
        else:
            valid_length = valid_length.to(dtype=torch.long)
        if not torch.is_tensor(sample_dt):
            sample_dt = torch.as_tensor(sample_dt, dtype=torch.float32)
        else:
            sample_dt = sample_dt.to(dtype=torch.float32)

        if not torch.isfinite(traces).all():
            self.sanitize_count += 1
            traces = torch.nan_to_num(traces, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(coord_norm).all():
            self.sanitize_count += 1
            coord_norm = torch.nan_to_num(coord_norm, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(sample_dt):
            self.sanitize_count += 1
            sample_dt = torch.tensor(0.0, dtype=torch.float32)

        family_id = np.int64(self.family_ids_raw[real_idx])
        run_id = np.int64(self.run_ids_raw[real_idx])

        if self.label_h5_dataset is not None:
            label_store = bundle[self.label_h5_dataset]
            y = label_store[local_idx] if isinstance(label_store, list) else label_store[local_idx]
            if torch.is_tensor(y):
                y = float(y.item()) if y.ndim == 0 else float(y)
            else:
                y = float(y)
            if not np.isfinite(y):
                y = 0.0
        elif self.target == "family":
            y = int(self.family_ids_raw[real_idx])
        elif self.target == "run":
            y = int(self.run_ids_raw[real_idx])
        else:
            raise ValueError("Unsupported target. Use 'family', 'run', or --label-h5-dataset.")

        return {
            "traces": traces,
            "valid_length": np.int64(valid_length.item()),
            "coord_normalized": coord_norm,
            "sample_dt": np.float32(sample_dt.item()),
            "family_id": family_id,
            "run_id": run_id,
            "target": y,
            "sample_index": np.int64(real_idx),
        }


def build_prepared_dataset(
    file_paths: list[str | Path],
    target: str = "family",
    indices: np.ndarray | None = None,
    family_filter: str = "all",
    label_h5_dataset: str | None = None,
):
    if not file_paths:
        raise ValueError("No prepared files provided.")
    suffixes = {Path(p).suffix.lower() for p in file_paths}
    if suffixes == {".pt"}:
        return PreparedPTBundleDataset(file_paths, target, indices, family_filter, label_h5_dataset)
    if suffixes == {".h5"}:
        return PreparedH5Dataset(file_paths, target, indices, family_filter, label_h5_dataset)
    raise ValueError(f"Mixed or unsupported prepared file types: {sorted(suffixes)}")



class PreparedH5Dataset(Dataset):
    def __init__(
        self,
        file_paths: list[str | Path],
        target: str = "family",
        indices: np.ndarray | None = None,
        family_filter: str = "all",
        label_h5_dataset: str | None = None,
    ):
        if torch is None:
            raise ImportError(f"torch is required for PreparedH5Dataset: {_TORCH_IMPORT_ERROR}")

        self.file_paths = [str(Path(p)) for p in file_paths]
        self.target = target
        self.label_h5_dataset = label_h5_dataset
        self.handles: dict[int, list[h5py.File]] = {}
        self.sample_map: list[tuple[int, int]] = []
        self.family_values: list[str] = []
        self.run_values: list[str] = []
        self.family_ids_raw: list[int] = []
        self.run_ids_raw: list[int] = []
        self.group_keys: list[str] = []
        self.group_ids_raw: list[int] = []
        self.sanitize_count = 0

        family_strings = []
        run_strings = []

        for file_idx, fp in enumerate(self.file_paths):
            with h5py.File(fp, "r") as f:
                n = int(f["traces"].shape[0])
                family = decode_str_array(f["family_name"][:])
                run = decode_str_array(f["run_name"][:])
                source_files = f["source_file"][:] if "source_file" in f else None
                source_groups = f["source_group"][:] if "source_group" in f else None
                operating_index = f["operating_index"][:] if "operating_index" in f else None

                for local_idx in range(n):
                    fam = str(family[local_idx])
                    rn = str(run[local_idx])
                    if family_filter != "all" and fam != family_filter:
                        continue
                    source_file = _safe_string_item(source_files[local_idx] if source_files is not None else Path(fp).name, default=Path(fp).name)
                    source_group = _safe_string_item(source_groups[local_idx] if source_groups is not None else "na", default="na")
                    op_idx = _safe_string_item(operating_index[local_idx] if operating_index is not None else "na", default="na")
                    group_key = build_group_key(source_file, source_group, rn, op_idx)

                    self.sample_map.append((file_idx, local_idx))
                    family_strings.append(fam)
                    run_strings.append(rn)
                    self.group_keys.append(group_key)

        self.family_indexer = StringIndexer().fit(family_strings)
        self.run_indexer = StringIndexer().fit(run_strings)
        self.family_values = family_strings
        self.run_values = run_strings
        self.family_ids_raw = self.family_indexer.transform(family_strings).tolist()
        self.run_ids_raw = self.run_indexer.transform(run_strings).tolist()
        self.group_ids_raw = StringIndexer().fit_transform(self.group_keys).tolist()
        self.indices = np.arange(len(self.sample_map), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def _get_worker_handles(self) -> list[h5py.File]:
        worker_id = 0
        if torch is not None:
            info = torch.utils.data.get_worker_info()
            worker_id = 0 if info is None else info.id
        if worker_id not in self.handles:
            self.handles[worker_id] = [h5py.File(fp, "r") for fp in self.file_paths]
        return self.handles[worker_id]

    def _sanitize_array(self, x: np.ndarray, name: str) -> np.ndarray:
        if not np.isfinite(x).all():
            self.sanitize_count += 1
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        file_idx, local_idx = self.sample_map[real_idx]
        handles = self._get_worker_handles()
        f = handles[file_idx]

        traces = np.asarray(f["traces"][local_idx], dtype=np.float32)
        valid_length = np.int64(f["valid_length"][local_idx])
        coord_norm = np.asarray(f["coord_normalized"][local_idx], dtype=np.float32)
        sample_dt = np.float32(f["sample_dt"][local_idx])

        traces = self._sanitize_array(traces, "traces")
        coord_norm = self._sanitize_array(coord_norm, "coord_normalized")
        if not np.isfinite(sample_dt):
            self.sanitize_count += 1
            sample_dt = np.float32(0.0)

        family_id = np.int64(self.family_ids_raw[real_idx])
        run_id = np.int64(self.run_ids_raw[real_idx])

        if self.label_h5_dataset is not None:
            y = float(f[self.label_h5_dataset][local_idx])
            if not np.isfinite(y):
                y = 0.0
        elif self.target == "family":
            y = int(self.family_ids_raw[real_idx])
        elif self.target == "run":
            y = int(self.run_ids_raw[real_idx])
        else:
            raise ValueError("Unsupported target. Use 'family', 'run', or --label-h5-dataset.")

        return {
            "traces": traces,
            "valid_length": valid_length,
            "coord_normalized": coord_norm,
            "sample_dt": sample_dt,
            "family_id": family_id,
            "run_id": run_id,
            "target": y,
            "sample_index": np.int64(real_idx),
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    trace_list = [x["traces"] if torch.is_tensor(x["traces"]) else torch.as_tensor(x["traces"], dtype=torch.float32) for x in items]
    coord_list = [x["coord_normalized"] if torch.is_tensor(x["coord_normalized"]) else torch.as_tensor(x["coord_normalized"], dtype=torch.float32) for x in items]

    traces = torch.stack([t.to(dtype=torch.float32) for t in trace_list], dim=0)
    valid_length = torch.as_tensor([x["valid_length"] for x in items], dtype=torch.long)
    coord_normalized = torch.stack([c.to(dtype=torch.float32) for c in coord_list], dim=0)
    sample_dt = torch.as_tensor([x["sample_dt"] for x in items], dtype=torch.float32)
    family_id = torch.as_tensor([x["family_id"] for x in items], dtype=torch.long)
    run_id = torch.as_tensor([x["run_id"] for x in items], dtype=torch.long)

    targets_np = np.asarray([x["target"] for x in items])
    target = torch.as_tensor(targets_np, dtype=torch.float32 if np.issubdtype(targets_np.dtype, np.floating) else torch.long)
    sample_index = torch.as_tensor([x["sample_index"] for x in items], dtype=torch.long)

    return {
        "batch": QCVVBatch(
            traces=traces,
            valid_length=valid_length,
            coord_normalized=coord_normalized,
            sample_dt=sample_dt,
            family_id=family_id,
            run_id=run_id,
            sample_id=sample_index,
        ),
        "target": target,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def make_run_dir(base_dir: str | Path, model_name: str, family: str, target: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{stamp}_{model_name}_{family}_{target}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_metrics_csv(path: Path, row: dict[str, Any]):
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_train_epoch_loader(train_ds, loader_kwargs: dict[str, Any], args):
    """
    Build a stochastic train loader for the current epoch.

    - full_epoch: iterate all samples once
    - steps_per_epoch: train on a fixed number of mini-batches
    - epoch_sample_fraction: train on a random subset each epoch
    """
    batch_size = int(loader_kwargs["batch_size"])
    total_n = len(train_ds)

    if getattr(args, "full_epoch", False):
        return DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs), total_n

    requested_num = None
    if getattr(args, "steps_per_epoch", None):
        requested_num = int(args.steps_per_epoch) * batch_size
    else:
        frac = float(getattr(args, "epoch_sample_fraction", 1.0))
        if frac < 1.0:
            requested_num = max(batch_size, int(math.ceil(total_n * frac)))

    if requested_num is None or requested_num >= total_n:
        return DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs), total_n

    if isinstance(train_ds, Subset):
        subset_indices = np.asarray(train_ds.indices, dtype=np.int64)
        chosen_pos = np.random.choice(len(subset_indices), size=requested_num, replace=False)
        chosen_indices = subset_indices[chosen_pos].tolist()
        epoch_ds = Subset(train_ds.dataset, chosen_indices)
        return DataLoader(epoch_ds, shuffle=True, drop_last=True, **loader_kwargs), requested_num

    sampler = RandomSampler(train_ds, replacement=False, num_samples=requested_num)
    loader = DataLoader(train_ds, shuffle=False, sampler=sampler, drop_last=True, **loader_kwargs)
    return loader, requested_num


def print_epoch_summary(epoch: int, train_loss: float, val_loss: float | None, metrics: dict[str, float] | None, epoch_time: float):
    if val_loss is None or metrics is None:
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val=skipped time={epoch_time:.1f}s",
            flush=True,
        )
        return
    print(
        f"[epoch {epoch:03d}] "
        f"train_loss={train_loss:.6f} "
        f"val_loss={val_loss:.6f} "
        f"acc={metrics['accuracy']:.4f} "
        f"f1_macro={metrics['f1_macro']:.4f} "
        f"time={epoch_time:.1f}s",
        flush=True,
    )


def move_batch_to_device(batch: QCVVBatch, device: TorchDevice) -> QCVVBatch:
    return QCVVBatch(
        traces=batch.traces.to(device, non_blocking=True),
        valid_length=batch.valid_length.to(device, non_blocking=True),
        coord_normalized=batch.coord_normalized.to(device, non_blocking=True),
        sample_dt=batch.sample_dt.to(device, non_blocking=True),
        family_id=batch.family_id.to(device, non_blocking=True) if batch.family_id is not None else None,
        run_id=batch.run_id.to(device, non_blocking=True) if batch.run_id is not None else None,
        sample_id=batch.sample_id.to(device, non_blocking=True) if batch.sample_id is not None else None,
        sequence_id=batch.sequence_id.to(device, non_blocking=True) if batch.sequence_id is not None else None,
    )


def move_loader_item_to_device(item: dict[str, Any], device: TorchDevice) -> dict[str, Any]:
    return {
        "batch": move_batch_to_device(item["batch"], device),
        "target": item["target"].to(device, non_blocking=True),
    }


def unpack_loader_item(item: dict[str, Any], device: TorchDevice):
    target = item["target"]
    if torch.is_tensor(target) and target.device.type == device.type:
        return item["batch"], target
    return move_batch_to_device(item["batch"], device), target.to(device, non_blocking=True)


class CUDAPrefetchLoader:
    def __init__(self, loader, device: TorchDevice):
        self.loader = loader
        self.device = device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if torch is None or self.device.type != "cuda":
            for item in self.loader:
                yield item
            return

        stream = torch.cuda.Stream(device=self.device)
        first = True
        next_item = None

        for item in self.loader:
            with torch.cuda.stream(stream):
                prefetched = move_loader_item_to_device(item, self.device)
            if not first:
                torch.cuda.current_stream(self.device).wait_stream(stream)
                yield next_item
            else:
                first = False
            next_item = prefetched

        if next_item is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            yield next_item


def make_cnn_optimizer(model: NNModule, lr: float, weight_decay: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_one_epoch_cnn(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    use_amp,
    log_interval,
    grad_clip_norm,
    learning_task,
    recon_weight,
):
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_recon = 0.0
    n_samples = 0
    y_true_all = []
    y_pred_all = []
    running_loss = 0.0
    running_count = 0
    use_cls = classification_enabled(learning_task)

    iterator = loader if tqdm is None else tqdm(loader, desc="train", leave=False)

    for step, item in enumerate(iterator, start=1):
        batch, target = unpack_loader_item(item, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            out = model(batch)
            logits = out["state_logits"]
            recon_loss = out.get("recon_loss", torch.zeros((), device=device))
            if use_cls:
                cls_loss = criterion(logits, target)
                loss = cls_loss + recon_weight * recon_loss
            else:
                cls_loss = torch.zeros((), device=device)
                loss = recon_weight * recon_loss

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite CNN+GRU loss after sanitization.")

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm is not None and grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        loss_value = float(loss.detach().item())
        bs = int(target.shape[0])
        total_loss += loss_value * bs
        total_cls += float(cls_loss.detach().item()) * bs
        total_recon += float(recon_loss.detach().item()) * bs
        n_samples += bs

        if use_cls:
            pred = torch.argmax(logits.detach(), dim=-1)
            y_true_all.append(target.detach().cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())

        running_loss += loss_value
        running_count += 1
        avg_recent_loss = running_loss / max(1, running_count)
        if tqdm is not None:
            iterator.set_postfix(loss=f"{loss_value:.6f}", avg=f"{avg_recent_loss:.6f}")

        should_print = (step == 1) or (step % log_interval == 0) or (step == len(loader))
        if should_print:
            print(
                f"  train step {step}/{len(loader)} "
                f"loss={loss_value:.6f} avg_recent_loss={avg_recent_loss:.6f} "
                f"cls={float(cls_loss.detach().item()):.6f} recon={float(recon_loss.detach().item()):.6f}",
                flush=True,
            )
            running_loss = 0.0
            running_count = 0

    if use_cls and y_true_all:
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        metrics = classification_metrics(y_true, y_pred)
    else:
        metrics = empty_or_nan_metrics()
    metrics["cls_loss"] = total_cls / max(1, n_samples)
    metrics["recon_loss"] = total_recon / max(1, n_samples)
    return total_loss / max(1, n_samples), metrics


@torch.no_grad()
def evaluate_cnn(model, loader, criterion, device, use_amp, learning_task, recon_weight):
    model.eval()
    total_loss = 0.0
    total_cls = 0.0
    total_recon = 0.0
    n_samples = 0
    y_true_all = []
    y_pred_all = []
    use_cls = classification_enabled(learning_task)

    iterator = loader if tqdm is None else tqdm(loader, desc="val", leave=False)

    for item in iterator:
        batch, target = unpack_loader_item(item, device)

        with autocast(device_type=device.type, enabled=use_amp):
            out = model(batch)
            logits = out["state_logits"]
            recon_loss = out.get("recon_loss", torch.zeros((), device=device))
            if use_cls:
                cls_loss = criterion(logits, target)
                loss = cls_loss + recon_weight * recon_loss
            else:
                cls_loss = torch.zeros((), device=device)
                loss = recon_weight * recon_loss

        bs = int(target.shape[0])
        total_loss += float(loss.detach().item()) * bs
        total_cls += float(cls_loss.detach().item()) * bs
        total_recon += float(recon_loss.detach().item()) * bs
        n_samples += bs

        if use_cls:
            pred = torch.argmax(logits.detach(), dim=-1)
            y_true_all.append(target.detach().cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())

    if use_cls and y_true_all:
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        metrics = classification_metrics(y_true, y_pred)
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    else:
        metrics = empty_or_nan_metrics()
        metrics["confusion_matrix"] = []
    metrics["cls_loss"] = total_cls / max(1, n_samples)
    metrics["recon_loss"] = total_recon / max(1, n_samples)
    return total_loss / max(1, n_samples), metrics


def train_one_epoch_dmm(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    use_amp,
    log_interval,
    grad_clip_norm,
    learning_task,
    recon_weight,
    kl_weight,
):
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_samples = 0
    y_true_all = []
    y_pred_all = []
    running_loss = 0.0
    running_count = 0
    use_cls = classification_enabled(learning_task)

    iterator = loader if tqdm is None else tqdm(loader, desc="train_dmm", leave=False)
    for step, item in enumerate(iterator, start=1):
        batch, target = unpack_loader_item(item, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            out = model(batch, sample_latent=True)
            logits = out["state_logits"]
            cls_loss = criterion(logits, target) if use_cls else torch.zeros((), device=device)
            recon_loss = out["recon_loss"]
            kl_loss = out["kl_loss"]
            loss = cls_loss + recon_weight * recon_loss + kl_weight * kl_loss

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite DMM loss after sanitization.")

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm is not None and grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        loss_value = float(loss.detach().item())
        bs = int(target.shape[0])
        total_loss += loss_value * bs
        total_cls += float(cls_loss.detach().item()) * bs
        total_recon += float(recon_loss.detach().item()) * bs
        total_kl += float(kl_loss.detach().item()) * bs
        n_samples += bs

        if use_cls:
            pred = torch.argmax(logits.detach(), dim=-1)
            y_true_all.append(target.detach().cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())

        running_loss += loss_value
        running_count += 1
        avg_recent = running_loss / max(1, running_count)
        if tqdm is not None:
            iterator.set_postfix(loss=f"{loss_value:.6f}", avg=f"{avg_recent:.6f}")

        should_print = (step == 1) or (step % log_interval == 0) or (step == len(loader))
        if should_print:
            print(
                f"  dmm train step {step}/{len(loader)} "
                f"loss={loss_value:.6f} avg_recent_loss={avg_recent:.6f} "
                f"cls={float(cls_loss.detach().item()):.6f} recon={float(recon_loss.detach().item()):.6f} kl={float(kl_loss.detach().item()):.6f}",
                flush=True,
            )
            running_loss = 0.0
            running_count = 0

    if use_cls and y_true_all:
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        metrics = classification_metrics(y_true, y_pred)
    else:
        metrics = empty_or_nan_metrics()
    metrics["cls_loss"] = total_cls / max(1, n_samples)
    metrics["recon_loss"] = total_recon / max(1, n_samples)
    metrics["kl_loss"] = total_kl / max(1, n_samples)
    return total_loss / max(1, n_samples), metrics


@torch.no_grad()
def evaluate_dmm(model, loader, criterion, device, use_amp, learning_task, recon_weight, kl_weight):
    model.eval()
    total_loss = 0.0
    total_cls = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_samples = 0
    y_true_all = []
    y_pred_all = []
    use_cls = classification_enabled(learning_task)

    iterator = loader if tqdm is None else tqdm(loader, desc="val_dmm", leave=False)
    for item in iterator:
        batch, target = unpack_loader_item(item, device)
        with autocast(device_type=device.type, enabled=use_amp):
            out = model(batch, sample_latent=False)
            logits = out["state_logits"]
            cls_loss = criterion(logits, target) if use_cls else torch.zeros((), device=device)
            recon_loss = out["recon_loss"]
            kl_loss = out["kl_loss"]
            loss = cls_loss + recon_weight * recon_loss + kl_weight * kl_loss

        bs = int(target.shape[0])
        total_loss += float(loss.detach().item()) * bs
        total_cls += float(cls_loss.detach().item()) * bs
        total_recon += float(recon_loss.detach().item()) * bs
        total_kl += float(kl_loss.detach().item()) * bs
        n_samples += bs

        if use_cls:
            pred = torch.argmax(logits.detach(), dim=-1)
            y_true_all.append(target.detach().cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())

    if use_cls and y_true_all:
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        metrics = classification_metrics(y_true, y_pred)
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    else:
        metrics = empty_or_nan_metrics()
        metrics["confusion_matrix"] = []
    metrics["cls_loss"] = total_cls / max(1, n_samples)
    metrics["recon_loss"] = total_recon / max(1, n_samples)
    metrics["kl_loss"] = total_kl / max(1, n_samples)
    return total_loss / max(1, n_samples), metrics


def run_train_cnn_gru(args, run_dir: Path):
    prefetch_to_gpu = getattr(args, "prefetch_to_gpu", True)
    save_every = getattr(args, "save_every", 2)
    if torch is None:
        raise ImportError(f"torch is required for CNN+GRU training: {_TORCH_IMPORT_ERROR}")

    CNNModel, CNNConfig = _import_cnn_gru()

    dataset = build_prepared_dataset(
        file_paths=args.prepared,
        target=args.target,
        family_filter=args.family,
        label_h5_dataset=args.label_h5_dataset,
    )

    if args.split_strategy == "grouped":
        train_idx, val_idx, split_info = grouped_train_val_split(dataset, args.val_split, args.seed)
    else:
        strat = None
        if args.label_h5_dataset is None:
            strat = np.asarray(
                dataset.family_ids_raw if args.target == "family" else dataset.run_ids_raw,
                dtype=np.int64,
            )
        all_indices = np.arange(len(dataset), dtype=np.int64)
        train_idx, val_idx = train_test_split(
            all_indices,
            test_size=args.val_split,
            random_state=args.seed,
            stratify=strat,
        )
        split_info = {"split_strategy": "random", "group_overlap_count": None}

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())

    num_classes = len(dataset.family_indexer.idx_to_class) if args.target == "family" else len(dataset.run_indexer.idx_to_class)
    learning_task = resolve_learning_task(args, num_classes)
    recon_weight = reconstruction_weight_for_task(args, learning_task)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True

    family_vocab_size = max(8, len(dataset.family_indexer.idx_to_class) + 2)
    run_vocab_size = max(16, len(dataset.run_indexer.idx_to_class) + 2)
    cfg = CNNConfig(
        family_vocab_size=family_vocab_size,
        run_vocab_size=run_vocab_size,
        num_states=max(1, num_classes),
        use_family_meta=args.use_family_meta,
        use_run_meta=args.use_run_meta,
    )

    model = CNNModel(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = make_cnn_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler(enabled=(device.type == "cuda" and args.amp))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_reduce_factor, patience=args.lr_patience)

    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = 0 if os.name == "nt" else max(2, min(4, (os.cpu_count() or 4) - 1))

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_batch,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader, train_samples_this_epoch = build_train_epoch_loader(train_ds, loader_kwargs, args)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    metrics_csv = run_dir / "metrics.csv"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    best_f1 = -1.0
    history = []

    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} | VRAM: {props.total_memory / (1024**3):.2f} GB", flush=True)
    print(f"Train samples total: {len(train_ds):,} | Val samples: {len(val_ds):,}", flush=True)
    print(f"Split info: {split_info}", flush=True)
    print(f"Batch size: {args.batch_size} | Num workers: {num_workers} | AMP: {args.amp and device.type == 'cuda'}", flush=True)
    print(f"Validate every: {args.val_every} epoch(s) | Save every: {args.save_every} epoch(s) | Log interval: {args.log_interval}", flush=True)
    print(f"Anti-overfit: weight_decay={args.weight_decay} | label_smoothing={args.label_smoothing} | early_stopping_patience={args.early_stopping_patience}", flush=True)
    if getattr(args, "full_epoch", False):
        print("Training mode: full epoch", flush=True)
    elif getattr(args, "steps_per_epoch", None):
        print(f"Training mode: stochastic fixed-step | steps_per_epoch={args.steps_per_epoch}", flush=True)
    else:
        print(f"Training mode: stochastic fractional | epoch_sample_fraction={args.epoch_sample_fraction:.3f}", flush=True)
    print(f"Target: {args.target} | Family filter: {args.family} | learning_task={learning_task} | recon_weight={recon_weight}", flush=True)
    print(f"Metadata into model: family={args.use_family_meta} | run={args.use_run_meta}", flush=True)
    print(f"AMP enabled: {args.amp} | GPU prefetch: {args.prefetch_to_gpu}", flush=True)
    if os.name == "nt":
        print("Windows note: default num_workers is 0 in this build to reduce RAM pressure. Increase only if the system stays stable.", flush=True)

    print("Starting training loop...", flush=True)
    epochs_without_improve = 0
    for epoch in range(1, args.epochs + 1):
        print(f"Starting epoch {epoch}/{args.epochs}...", flush=True)

        train_loader, train_samples_this_epoch = build_train_epoch_loader(train_ds, loader_kwargs, args)
        if device.type == "cuda" and prefetch_to_gpu:
            train_epoch_loader = CUDAPrefetchLoader(train_loader, device)
            val_epoch_loader = CUDAPrefetchLoader(val_loader, device)
        else:
            train_epoch_loader = train_loader
            val_epoch_loader = val_loader

        print(f"  epoch train samples: {train_samples_this_epoch:,} | steps: {len(train_loader):,}", flush=True)

        t0 = time.time()
        train_loss, train_metrics = train_one_epoch_cnn(
            model, train_epoch_loader, optimizer, criterion, scaler, device,
            use_amp=(args.amp and device.type == "cuda"),
            log_interval=args.log_interval,
            grad_clip_norm=args.grad_clip_norm,
            learning_task=learning_task,
            recon_weight=recon_weight,
        )

        val_loss = None
        val_metrics = None
        if (epoch % args.val_every == 0) or (epoch == args.epochs):
            val_loss, val_metrics = evaluate_cnn(
                model, val_epoch_loader, criterion, device,
                use_amp=(args.amp and device.type == "cuda"),
                learning_task=learning_task,
                recon_weight=recon_weight,
            )
            scheduler.step(val_loss)

        epoch_time = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "train_cls_loss": train_metrics.get("cls_loss"),
            "train_recon_loss": train_metrics.get("recon_loss"),
            "epoch_time_sec": epoch_time,
            "lr": optimizer.param_groups[0]["lr"],
            "batch_size": args.batch_size,
            "train_samples_this_epoch": int(train_samples_this_epoch),
            "train_steps_this_epoch": int(len(train_loader)),
        }
        if val_metrics is not None:
            row.update({
                "val_loss": val_loss,
                "val_acc": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_cls_loss": val_metrics.get("cls_loss"),
                "val_recon_loss": val_metrics.get("recon_loss"),
                "val_precision_macro": val_metrics["precision_macro"],
                "val_recall_macro": val_metrics["recall_macro"],
            })
        append_metrics_csv(metrics_csv, row)
        history.append(row)
        print_epoch_summary(epoch, train_loss, val_loss, val_metrics, epoch_time)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": vars(args),
            "cnn_config": cfg.__dict__,
            "val_metrics": val_metrics,
            "train_metrics": train_metrics,
        }
        if (epoch % save_every == 0) or (epoch == args.epochs):
            torch.save(ckpt, last_path)

        if val_metrics is not None:
            score = float(val_metrics["f1_macro"]) if classification_enabled(learning_task) and np.isfinite(val_metrics["f1_macro"]) else -float(val_loss)
            if score > best_f1:
                best_f1 = score
                epochs_without_improve = 0
                torch.save(ckpt, best_path)
                print(f"  -> saved new best checkpoint: {best_path.name} (selection_score={best_f1:.6f})", flush=True)
            else:
                epochs_without_improve += 1

        if epoch >= args.min_epochs and epochs_without_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}. Best val_f1_macro={best_f1:.4f}", flush=True)
            break

    summary = {
        "mode": "cnn_gru",
        "target": args.target,
        "family": args.family,
        "device": str(device),
        "best_selection_score": best_f1,
        "learning_task": learning_task,
        "history": history,
        "split_info": split_info,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Sanitized samples encountered in shared dataset: {get_sanitize_count(train_ds)}", flush=True)
    print(f"Saved metrics to: {metrics_csv}", flush=True)
    print(f"Saved best checkpoint to: {best_path}", flush=True)
    print(f"Saved last checkpoint to: {last_path}", flush=True)


def run_train_dmm(args, run_dir: Path):
    prefetch_to_gpu = getattr(args, "prefetch_to_gpu", True)
    save_every = getattr(args, "save_every", 2)
    if torch is None:
        raise ImportError(f"torch is required for DMM training: {_TORCH_IMPORT_ERROR}")

    DMMModel, DMMConfig = _import_dmm()

    dataset = build_prepared_dataset(
        file_paths=args.prepared,
        target=args.target,
        family_filter=args.family,
        label_h5_dataset=args.label_h5_dataset,
    )

    if args.split_strategy == "grouped":
        train_idx, val_idx, split_info = grouped_train_val_split(dataset, args.val_split, args.seed)
    else:
        strat = None
        if args.label_h5_dataset is None:
            strat = np.asarray(
                dataset.family_ids_raw if args.target == "family" else dataset.run_ids_raw,
                dtype=np.int64,
            )
        all_indices = np.arange(len(dataset), dtype=np.int64)
        train_idx, val_idx = train_test_split(
            all_indices,
            test_size=args.val_split,
            random_state=args.seed,
            stratify=strat,
        )
        split_info = {"split_strategy": "random", "group_overlap_count": None}

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())

    num_classes = len(dataset.family_indexer.idx_to_class) if args.target == "family" else len(dataset.run_indexer.idx_to_class)
    learning_task = resolve_learning_task(args, num_classes)
    recon_weight = reconstruction_weight_for_task(args, learning_task)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True

    family_vocab_size = max(8, len(dataset.family_indexer.idx_to_class) + 2)
    run_vocab_size = max(16, len(dataset.run_indexer.idx_to_class) + 2)
    cfg = DMMConfig(
        family_vocab_size=family_vocab_size,
        run_vocab_size=run_vocab_size,
        num_states=max(1, num_classes),
        latent_dim=args.dmm_latent_dim,
        conv_channels=args.dmm_conv_channels,
        encoder_hidden=args.dmm_encoder_hidden,
        temporal_downsample=args.dmm_temporal_downsample,
        dropout=args.dmm_dropout,
        use_family_meta=args.use_family_meta,
        use_run_meta=args.use_run_meta,
    )

    model = DMMModel(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = make_cnn_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler(enabled=(device.type == "cuda" and args.amp))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_reduce_factor, patience=args.lr_patience)

    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = 0 if os.name == "nt" else max(2, min(4, (os.cpu_count() or 4) - 1))

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_batch,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader, train_samples_this_epoch = build_train_epoch_loader(train_ds, loader_kwargs, args)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    metrics_csv = run_dir / "metrics.csv"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    best_f1 = -1.0
    history = []

    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} | VRAM: {props.total_memory / (1024**3):.2f} GB", flush=True)
    print(f"Train samples total: {len(train_ds):,} | Val samples: {len(val_ds):,}", flush=True)
    print(f"Split info: {split_info}", flush=True)
    print(f"Batch size: {args.batch_size} | Num workers: {num_workers} | AMP: {args.amp and device.type == 'cuda'}", flush=True)
    print(f"DMM latent_dim={cfg.latent_dim} | encoder_hidden={cfg.encoder_hidden} | temporal_downsample={cfg.temporal_downsample}", flush=True)
    print(f"Validate every: {args.val_every} epoch(s) | Save every: {args.save_every} epoch(s) | Log interval: {args.log_interval}", flush=True)
    print(f"Anti-overfit: weight_decay={args.weight_decay} | label_smoothing={args.label_smoothing} | early_stopping_patience={args.early_stopping_patience}", flush=True)
    if getattr(args, "full_epoch", False):
        print("Training mode: full epoch", flush=True)
    elif getattr(args, "steps_per_epoch", None):
        print(f"Training mode: stochastic fixed-step | steps_per_epoch={args.steps_per_epoch}", flush=True)
    else:
        print(f"Training mode: stochastic fractional | epoch_sample_fraction={args.epoch_sample_fraction:.3f}", flush=True)
    print(f"Target: {args.target} | Family filter: {args.family} | learning_task={learning_task} | recon_weight={recon_weight}", flush=True)
    print(f"Metadata into model: family={args.use_family_meta} | run={args.use_run_meta}", flush=True)
    print(f"AMP enabled: {args.amp} | GPU prefetch: {args.prefetch_to_gpu}", flush=True)
    if os.name == "nt":
        print("Windows note: default num_workers is 0 in this build to reduce RAM pressure. Increase only if the system stays stable.", flush=True)

    print("Starting DMM training loop...", flush=True)
    epochs_without_improve = 0
    for epoch in range(1, args.epochs + 1):
        print(f"Starting epoch {epoch}/{args.epochs}...", flush=True)

        train_loader, train_samples_this_epoch = build_train_epoch_loader(train_ds, loader_kwargs, args)
        if device.type == "cuda" and prefetch_to_gpu:
            train_epoch_loader = CUDAPrefetchLoader(train_loader, device)
            val_epoch_loader = CUDAPrefetchLoader(val_loader, device)
        else:
            train_epoch_loader = train_loader
            val_epoch_loader = val_loader

        print(f"  epoch train samples: {train_samples_this_epoch:,} | steps: {len(train_loader):,}", flush=True)

        t0 = time.time()
        train_loss, train_metrics = train_one_epoch_dmm(
            model, train_epoch_loader, optimizer, criterion, scaler, device,
            use_amp=(args.amp and device.type == "cuda"),
            log_interval=args.log_interval,
            grad_clip_norm=args.grad_clip_norm,
            learning_task=learning_task,
            recon_weight=recon_weight,
            kl_weight=args.dmm_kl_weight,
        )

        val_loss = None
        val_metrics = None
        if (epoch % args.val_every == 0) or (epoch == args.epochs):
            val_loss, val_metrics = evaluate_dmm(
                model, val_epoch_loader, criterion, device,
                use_amp=(args.amp and device.type == "cuda"),
                learning_task=learning_task,
                recon_weight=recon_weight,
                kl_weight=args.dmm_kl_weight,
            )
            scheduler.step(val_loss)

        epoch_time = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_cls_loss": train_metrics["cls_loss"],
            "train_recon_loss": train_metrics["recon_loss"],
            "train_kl_loss": train_metrics["kl_loss"],
            "train_acc": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "epoch_time_sec": epoch_time,
            "lr": optimizer.param_groups[0]["lr"],
            "batch_size": args.batch_size,
            "train_samples_this_epoch": int(train_samples_this_epoch),
            "train_steps_this_epoch": int(len(train_loader)),
        }
        if val_metrics is not None:
            row.update({
                "val_loss": val_loss,
                "val_cls_loss": val_metrics["cls_loss"],
                "val_recon_loss": val_metrics["recon_loss"],
                "val_kl_loss": val_metrics["kl_loss"],
                "val_acc": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_precision_macro": val_metrics["precision_macro"],
                "val_recall_macro": val_metrics["recall_macro"],
            })
        append_metrics_csv(metrics_csv, row)
        history.append(row)
        print_epoch_summary(epoch, train_loss, val_loss, val_metrics, epoch_time)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": vars(args),
            "dmm_config": cfg.__dict__,
            "val_metrics": val_metrics,
            "train_metrics": train_metrics,
        }
        if (epoch % save_every == 0) or (epoch == args.epochs):
            torch.save(ckpt, last_path)

        if val_metrics is not None:
            score = float(val_metrics["f1_macro"]) if classification_enabled(learning_task) and np.isfinite(val_metrics["f1_macro"]) else -float(val_loss)
            if score > best_f1:
                best_f1 = score
                epochs_without_improve = 0
                torch.save(ckpt, best_path)
                print(f"  -> saved new best checkpoint: {best_path.name} (selection_score={best_f1:.6f})", flush=True)
            else:
                epochs_without_improve += 1

        if epoch >= args.min_epochs and epochs_without_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}. Best val_f1_macro={best_f1:.4f}", flush=True)
            break

    summary = {
        "mode": "dmm",
        "target": args.target,
        "family": args.family,
        "device": str(device),
        "best_selection_score": best_f1,
        "learning_task": learning_task,
        "history": history,
        "split_info": split_info,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Sanitized samples encountered in shared dataset: {get_sanitize_count(dataset)}", flush=True)
    print(f"Saved metrics to: {metrics_csv}", flush=True)
    print(f"Saved best checkpoint to: {best_path}", flush=True)
    print(f"Saved last checkpoint to: {last_path}", flush=True)


def run_train_xgboost(args, run_dir: Path):
    XGBModel, XGBConfig = _import_xgboost()

    dataset = build_prepared_dataset(
        file_paths=args.prepared,
        target=args.target,
        family_filter=args.family,
        label_h5_dataset=args.label_h5_dataset,
    )

    limit = len(dataset) if args.max_samples is None else min(len(dataset), int(args.max_samples))
    print(f"Materializing {limit:,} samples for XGBoost feature extraction...", flush=True)

    traces, valid_length, coords, sample_dt, family_id, run_id, y = [], [], [], [], [], [], []
    iterator = range(limit) if tqdm is None else tqdm(range(limit), desc="xgb_features")
    for i in iterator:
        item = dataset[i]
        traces.append(item["traces"])
        valid_length.append(item["valid_length"])
        coords.append(item["coord_normalized"])
        sample_dt.append(item["sample_dt"])
        family_id.append(item["family_id"])
        run_id.append(item["run_id"])
        y.append(item["target"])

    batch = QCVVBatch(
        traces=np.stack(traces, axis=0),
        valid_length=np.asarray(valid_length, dtype=np.int64),
        coord_normalized=np.stack(coords, axis=0).astype(np.float32),
        sample_dt=np.asarray(sample_dt, dtype=np.float32),
        family_id=np.asarray(family_id, dtype=np.int64),
        run_id=np.asarray(run_id, dtype=np.int64),
    )
    y = np.asarray(y)

    model = XGBModel(XGBConfig(use_gpu=args.use_gpu))
    X, _ = model.build_features_from_batch(batch)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=args.seed, stratify=y
    )

    print(f"Training XGBoost on X={X.shape} target={args.target} gpu={args.use_gpu}", flush=True)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    metrics = classification_metrics(y_val, pred)
    metrics["n_train"] = int(X_train.shape[0])
    metrics["n_val"] = int(X_val.shape[0])
    metrics["feature_importances"] = model.feature_importances()

    model.save(run_dir)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("XGBoost results:", flush=True)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}", flush=True)
    print(f"Saved model bundle to: {run_dir}", flush=True)


def extract_embeddings_with_cnn(args, dataset: PreparedH5Dataset, checkpoint_path: str | Path):
    CNNModel, CNNConfig = _import_cnn_gru()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt.get("cnn_config", {})
    cfg = CNNConfig(**cfg_dict) if cfg_dict else CNNConfig()
    model = CNNModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    num_workers = args.num_workers if args.num_workers is not None else (1 if os.name == "nt" else 4)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_batch,
        shuffle=False,
        drop_last=False,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(dataset, **loader_kwargs)
    embeddings, labels, sequence_ids = [], [], []
    with torch.no_grad():
        iterator = loader if tqdm is None else tqdm(loader, desc="embed", leave=False)
        for item in iterator:
            batch = move_batch_to_device(item["batch"], device)
            out = model(batch)
            embeddings.append(out["embedding"].detach().cpu().numpy())
            labels.append(item["target"].cpu().numpy())
            sequence_ids.append(item["batch"].sample_id.cpu().numpy())

    return np.concatenate(embeddings), np.concatenate(labels), np.concatenate(sequence_ids)


def run_train_hsmm(args, run_dir: Path):
    GaussianHSMMQCVV, HSMMConfig = _import_hsmm()

    dataset = build_prepared_dataset(
        file_paths=args.prepared,
        target=args.target,
        family_filter=args.family,
        label_h5_dataset=args.label_h5_dataset,
    )

    if args.hsmm_source == "cnn":
        if args.cnn_checkpoint is None:
            raise ValueError("--hsmm-source cnn requires --cnn-checkpoint")
        print("Extracting CNN embeddings for HSMM...", flush=True)
        X, y, sequence_id = extract_embeddings_with_cnn(args, dataset, args.cnn_checkpoint)
    else:
        XGBModel, XGBConfig = _import_xgboost()
        limit = len(dataset) if args.max_samples is None else min(len(dataset), int(args.max_samples))
        print(f"Building handcrafted features for HSMM from {limit:,} samples...", flush=True)

        traces, valid_length, coords, sample_dt, family_id, run_id, y_list, sequence_id = [], [], [], [], [], [], [], []
        iterator = range(limit) if tqdm is None else tqdm(range(limit), desc="hsmm_features")
        for i in iterator:
            item = dataset[i]
            traces.append(item["traces"])
            valid_length.append(item["valid_length"])
            coords.append(item["coord_normalized"])
            sample_dt.append(item["sample_dt"])
            family_id.append(item["family_id"])
            run_id.append(item["run_id"])
            y_list.append(item["target"])
            sequence_id.append(item["sample_index"])

        batch = QCVVBatch(
            traces=np.stack(traces, axis=0),
            valid_length=np.asarray(valid_length, dtype=np.int64),
            coord_normalized=np.stack(coords, axis=0).astype(np.float32),
            sample_dt=np.asarray(sample_dt, dtype=np.float32),
            family_id=np.asarray(family_id, dtype=np.int64),
            run_id=np.asarray(run_id, dtype=np.int64),
        )
        y = np.asarray(y_list)
        sequence_id = np.asarray(sequence_id)
        xgb_model = XGBModel(XGBConfig(use_gpu=False))
        X, _ = xgb_model.build_features_from_batch(batch)

    unique_y = np.unique(y)
    supervised_ok = bool(args.hsmm_supervised and unique_y.size > 1)
    num_states = args.hsmm_states if args.hsmm_states is not None else int(max(2, min(4, unique_y.size if unique_y.size > 1 else 2)))

    cfg = HSMMConfig(
        num_states=num_states,
        min_duration=args.hsmm_min_duration,
        max_duration=args.hsmm_max_duration,
    )
    model = GaussianHSMMQCVV(cfg)

    if supervised_ok:
        print("Fitting HSMM in supervised mode...", flush=True)
        model.fit_supervised(X, y, sequence_id=sequence_id)
    else:
        print("Fitting HSMM in unsupervised latent-state mode...", flush=True)
        if args.hsmm_supervised and unique_y.size <= 1:
            print("[WARN] HSMM supervised requested but target is one-class; using unsupervised HSMM instead.", flush=True)
        model.fit_unsupervised_init(X)

    pred = model.predict_states(X, sequence_id=sequence_id)

    metrics = {
        "mode": "supervised" if supervised_ok else "unsupervised",
        "num_states": int(cfg.num_states),
        "state_counts": {str(k): int(v) for k, v in Counter(pred.tolist()).items()},
        "n_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
    }
    if y.dtype.kind in ("i", "u") and unique_y.size > 1:
        metrics.update(classification_metrics(y, pred))

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.save(run_dir / "predicted_states.npy", pred)
    np.save(run_dir / "sequence_id.npy", sequence_id)

    np.savez(
        run_dir / "hsmm_params.npz",
        pi=model.pi,
        A=model.A,
        means=model.means,
        vars=model.vars,
        duration_logprob=model.duration_logprob,
    )

    print("HSMM results:", flush=True)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)
    print(f"Saved HSMM outputs to: {run_dir}", flush=True)




def parse_args():
    parser = argparse.ArgumentParser(description="Train QCVV models on prepared .pt/.h5 files with low-RAM grouped splits.")
    parser.add_argument("--mode", choices=["cnn_gru", "dmm", "xgboost", "hsmm"], default="cnn_gru")
    parser.add_argument("--ready-dir", default=str(DEFAULT_READY_DIR), help="Directory containing prepared .pt bundles or prepared .h5 files. If .pt files exist, they are preferred.")
    parser.add_argument("--prepared", nargs="*", default=None)
    parser.add_argument("--family", choices=["all", "parity", "x_loop", "z_loop"], default="all")
    parser.add_argument("--training-scope", choices=["together", "individual", "both"], default="together", help="Train one model on all selected families together, one per family individually, or both.")
    parser.add_argument("--target", choices=["family", "run"], default="family")
    parser.add_argument("--learning-task", choices=["auto", "supervised", "self_supervised", "multitask"], default="auto", help="auto uses multitask when >=2 classes, else self-supervised reconstruction for one-class x_loop/z_loop.")
    parser.add_argument("--use-family-meta", action="store_true", help="Allow family_id metadata into the model. Default off to avoid target shortcut leakage.")
    parser.add_argument("--use-run-meta", action="store_true", help="Allow run_id metadata into the model. Default off to avoid target shortcut leakage.")
    parser.add_argument("--label-h5-dataset", default=None)
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=["grouped", "random"], default="grouped")

    parser.add_argument("--cpu", action="store_true", help="Deprecated for this GPU build; use --allow-cpu-training only for debugging.")
    parser.add_argument("--allow-cpu-training", action="store_true", help="Allow CPU fallback. Off by default so training fails if CUDA is unavailable.")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)

    parser.add_argument(
        "--prefetch-to-gpu",
        dest="prefetch_to_gpu",
        action="store_true",
        help="Prefetch batches to GPU on a side stream.",
    )
    parser.add_argument(
        "--no-prefetch-to-gpu",
        dest="prefetch_to_gpu",
        action="store_false",
        help="Disable GPU batch prefetch.",
    )
    parser.set_defaults(prefetch_to_gpu=True)

    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable mixed precision.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision.")
    parser.set_defaults(amp=False)

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Leave off on Windows unless compile startup cost is worth it.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--aux-recon-weight", type=float, default=0.10, help="Reconstruction weight when supervised labels exist.")
    parser.add_argument("--self-supervised-recon-weight", type=float, default=1.0, help="Reconstruction weight for one-class self-supervised training.")
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=25)

    parser.add_argument(
        "--epoch-sample-fraction",
        type=float,
        default=0.25,
        help="Fraction of the training set sampled each epoch in stochastic mode.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="If set, overrides epoch_sample_fraction and trains for this many mini-batches per epoch.",
    )
    parser.add_argument(
        "--full-epoch",
        action="store_true",
        help="Disable shortened stochastic epochs and iterate the full training set.",
    )

    parser.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--save-every", type=int, default=2, help="Write last.pt every N epochs.")
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--dmm-latent-dim", type=int, default=24)
    parser.add_argument("--dmm-conv-channels", type=int, default=48)
    parser.add_argument("--dmm-encoder-hidden", type=int, default=64)
    parser.add_argument("--dmm-temporal-downsample", type=int, default=4)
    parser.add_argument("--dmm-dropout", type=float, default=0.10)
    parser.add_argument("--dmm-recon-weight", type=float, default=0.25)
    parser.add_argument("--dmm-kl-weight", type=float, default=0.01)

    parser.add_argument("--hsmm-source", choices=["cnn", "handcrafted"], default="handcrafted")
    parser.add_argument("--cnn-checkpoint", default=None)
    parser.add_argument("--hsmm-supervised", action="store_true")
    parser.add_argument("--hsmm-states", type=int, default=None)
    parser.add_argument("--hsmm-min-duration", type=int, default=1)
    parser.add_argument("--hsmm-max-duration", type=int, default=16)
    parser.add_argument("--gpu-hsmm-hidden", type=int, default=64)
    parser.add_argument("--gpu-hsmm-duration-weight", type=float, default=0.20)
    parser.add_argument("--gpu-hsmm-entropy-weight", type=float, default=0.01)
    parser.add_argument("--gpu-hsmm-balance-weight", type=float, default=0.10)
    return parser.parse_args()


def dispatch_train_mode(args, run_dir: Path):
    if args.mode == "cnn_gru":
        run_train_cnn_gru(args, run_dir)
    elif args.mode == "dmm":
        run_train_dmm(args, run_dir)
    elif args.mode == "xgboost":
        run_train_xgboost(args, run_dir)
    elif args.mode == "hsmm":
        run_train_hsmm(args, run_dir)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


def build_training_plan(args) -> list[tuple[str, str]]:
    families_all = ["parity", "x_loop", "z_loop"]
    allowed = [args.family] if args.family != "all" else families_all

    if args.training_scope == "together":
        return [(args.family, "together")]

    if args.training_scope == "individual":
        return [(fam, "individual") for fam in allowed]

    if args.training_scope == "both":
        plan = []
        plan.append((args.family, "together"))
        plan.extend((fam, "individual") for fam in allowed)
        return plan

    raise ValueError(f"Unsupported training_scope: {args.training_scope}")


def run_training_plan(args):
    plan = build_training_plan(args)

    print("=" * 100, flush=True)
    print(f"TRAIN VERSION: {TRAIN_VERSION}", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Training scope: {args.training_scope}", flush=True)
    print(f"Ready dir: {args.ready_dir}", flush=True)
    print("Prepared files:", flush=True)
    for p in args.prepared:
        print(f"  - {p}", flush=True)
    print("Planned runs:", flush=True)
    for fam, scope_label in plan:
        print(f"  - family={fam} | scope={scope_label}", flush=True)
    print("=" * 100, flush=True)

    completed = []
    for fam, scope_label in plan:
        run_args = copy.deepcopy(args)
        run_args.family = fam
        model_name = f"{args.mode}_{scope_label}"
        run_dir = make_run_dir(run_args.runs_dir, model_name, fam, run_args.target)

        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(vars(run_args), f, indent=2)

        print("=" * 100, flush=True)
        print(f"STARTING RUN: family={fam} | scope={scope_label}", flush=True)
        print(f"Run dir: {run_dir}", flush=True)
        print("=" * 100, flush=True)

        dispatch_train_mode(run_args, run_dir)
        completed.append((fam, scope_label, run_dir))

    print("=" * 100, flush=True)
    print("COMPLETED RUNS", flush=True)
    for fam, scope_label, run_dir in completed:
        print(f"  - family={fam} | scope={scope_label} | run_dir={run_dir}", flush=True)
    print("=" * 100, flush=True)


def main():
    args = parse_args()
    _limit_cpu_runtime_threads()
    if not hasattr(args, "prefetch_to_gpu"):
        args.prefetch_to_gpu = True
    if not hasattr(args, "save_every"):
        args.save_every = 2
    seed_everything(args.seed)

    if args.prepared is None or len(args.prepared) == 0:
        args.prepared = discover_prepared_files(args.ready_dir)

    run_training_plan(args)
    print("Done.", flush=True)



# =============================================================================
# Combined-model QCVV pipeline extensions
# =============================================================================
# These extensions deliberately sit after the stable v22 trainer definitions so
# they can reuse the original data loading, batching, neural training utilities,
# and model import paths while adding the missing combined-pipeline pieces:
#   - CT-HMM mode
#   - fixed train/val/test split manifests
#   - sequence_id/window_start metadata for latent-state models
#   - coord_values based normalization when coord_normalized is degenerate
#   - teacher-distillation for CNN+GRU and DMM
#   - prediction export and model-disagreement comparison

TRAIN_VERSION = "qcvv_v25_gpu_1080ti_training"

_ORIGINAL_PT_GETITEM = PreparedPTBundleDataset.__getitem__
_ORIGINAL_H5_GETITEM = PreparedH5Dataset.__getitem__


def _import_cthmm():
    errors = []
    for mod_name in [
        "model.gaussian_cthmm",
        "gaussian_cthmm",
    ]:
        try:
            mod = __import__(mod_name, fromlist=["GaussianContinuousTimeHMM", "CTHMMConfig"])
            return mod.GaussianContinuousTimeHMM, mod.CTHMMConfig
        except Exception as exc:
            errors.append(f"{mod_name}: {exc}")
    raise ImportError("Could not import CT-HMM model. Tried:\n" + "\n".join(errors))


def _to_numpy(x: Any, dtype=np.float32) -> np.ndarray:
    if torch is not None and torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _to_scalar_int(x: Any, default: int = 0) -> int:
    try:
        if torch is not None and torch.is_tensor(x):
            if x.numel() == 1:
                return int(x.detach().cpu().item())
            return int(x.detach().cpu().reshape(-1)[0].item())
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        return int(arr.reshape(-1)[0])
    except Exception:
        return default


def _get_store_value(store: Any, index: int, default: Any = None) -> Any:
    if store is None:
        return default
    try:
        return store[index]
    except Exception:
        return default


def _ensure_coord_stats(ds: Any) -> tuple[np.ndarray, np.ndarray]:
    """Lazily compute coord_values normalization stats for a prepared dataset."""
    if hasattr(ds, "_qcvv_coord_mu") and hasattr(ds, "_qcvv_coord_sigma"):
        return ds._qcvv_coord_mu, ds._qcvv_coord_sigma

    chunks = []
    try:
        if hasattr(ds, "bundles"):
            for bundle in ds.bundles:
                d = bundle.get("datasets", {})
                raw = d.get("coord_values", d.get("coord_normalized", None))
                if raw is not None:
                    chunks.append(_to_numpy(raw, np.float32))
        elif hasattr(ds, "file_paths"):
            for fp in ds.file_paths:
                with h5py.File(fp, "r") as f:
                    key = "coord_values" if "coord_values" in f else "coord_normalized"
                    chunks.append(np.asarray(f[key][:], dtype=np.float32))
    except Exception:
        chunks = []

    if chunks:
        X = np.concatenate([c.reshape(c.shape[0], -1) for c in chunks], axis=0)
        mu = np.nanmean(X, axis=0).astype(np.float32)
        sigma = np.nanstd(X, axis=0).astype(np.float32)
        sigma = np.where(np.isfinite(sigma) & (sigma > 1e-8), sigma, 1.0).astype(np.float32)
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        mu = np.zeros(3, dtype=np.float32)
        sigma = np.ones(3, dtype=np.float32)

    ds._qcvv_coord_mu = mu
    ds._qcvv_coord_sigma = sigma
    return mu, sigma


def _normalize_coord_for_dataset(ds: Any, coord_raw: Any) -> np.ndarray:
    raw = _to_numpy(coord_raw, np.float32).reshape(-1)
    mu, sigma = _ensure_coord_stats(ds)
    if mu.shape[0] != raw.shape[0]:
        # Be conservative for unusual coordinate dimensionalities.
        mu = np.resize(mu, raw.shape[0]).astype(np.float32)
        sigma = np.resize(sigma, raw.shape[0]).astype(np.float32)
        sigma = np.where(np.isfinite(sigma) & (sigma > 1e-8), sigma, 1.0).astype(np.float32)
    out = (raw - mu) / sigma
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _pt_getitem_qcvv(self, idx: int):
    item = _ORIGINAL_PT_GETITEM(self, idx)
    real_idx = int(self.indices[idx])
    file_idx, local_idx = self.sample_map[real_idx]
    bundle = self.bundles[file_idx]["datasets"]

    coord_raw = _get_store_value(bundle.get("coord_values"), local_idx, item.get("coord_normalized"))
    item["coord_values"] = _to_numpy(coord_raw, np.float32).reshape(-1)
    item["coord_normalized"] = torch.as_tensor(_normalize_coord_for_dataset(self, coord_raw), dtype=torch.float32)

    window_start = _get_store_value(bundle.get("window_start"), local_idx, 0)
    operating_index = _get_store_value(bundle.get("operating_index"), local_idx, None)
    source_group = _get_store_value(bundle.get("source_group"), local_idx, "na")
    source_file = _get_store_value(bundle.get("source_file"), local_idx, "na")

    group_id = np.int64(self.group_ids_raw[real_idx])
    item["group_id"] = group_id
    item["sequence_id"] = group_id
    item["window_start"] = np.int64(_to_scalar_int(window_start, 0))
    item["operating_index"] = _to_numpy(operating_index if operating_index is not None else np.asarray([0]), np.int64).reshape(-1)
    item["source_group"] = _safe_string_item(source_group)
    item["source_file"] = _safe_string_item(source_file)
    return item


def _h5_getitem_qcvv(self, idx: int):
    item = _ORIGINAL_H5_GETITEM(self, idx)
    real_idx = int(self.indices[idx])
    file_idx, local_idx = self.sample_map[real_idx]
    handles = self._get_worker_handles()
    f = handles[file_idx]

    coord_key = "coord_values" if "coord_values" in f else "coord_normalized"
    coord_raw = np.asarray(f[coord_key][local_idx], dtype=np.float32)
    item["coord_values"] = coord_raw.reshape(-1)
    item["coord_normalized"] = _normalize_coord_for_dataset(self, coord_raw)

    window_start = f["window_start"][local_idx] if "window_start" in f else 0
    operating_index = f["operating_index"][local_idx] if "operating_index" in f else np.asarray([0], dtype=np.int64)
    source_group = f["source_group"][local_idx] if "source_group" in f else "na"
    source_file = f["source_file"][local_idx] if "source_file" in f else "na"

    group_id = np.int64(self.group_ids_raw[real_idx])
    item["group_id"] = group_id
    item["sequence_id"] = group_id
    item["window_start"] = np.int64(_to_scalar_int(window_start, 0))
    item["operating_index"] = _to_numpy(operating_index, np.int64).reshape(-1)
    item["source_group"] = _safe_string_item(source_group)
    item["source_file"] = _safe_string_item(source_file)
    return item


PreparedPTBundleDataset.__getitem__ = _pt_getitem_qcvv
PreparedH5Dataset.__getitem__ = _h5_getitem_qcvv


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:  # noqa: F811 - intentional override
    trace_list = [x["traces"] if torch.is_tensor(x["traces"]) else torch.as_tensor(x["traces"], dtype=torch.float32) for x in items]
    coord_list = [x["coord_normalized"] if torch.is_tensor(x["coord_normalized"]) else torch.as_tensor(x["coord_normalized"], dtype=torch.float32) for x in items]

    traces = torch.stack([t.to(dtype=torch.float32) for t in trace_list], dim=0)
    valid_length = torch.as_tensor([x["valid_length"] for x in items], dtype=torch.long)
    coord_normalized = torch.stack([c.to(dtype=torch.float32) for c in coord_list], dim=0)
    sample_dt = torch.as_tensor([x["sample_dt"] for x in items], dtype=torch.float32)
    family_id = torch.as_tensor([x["family_id"] for x in items], dtype=torch.long)
    run_id = torch.as_tensor([x["run_id"] for x in items], dtype=torch.long)
    sample_index = torch.as_tensor([x["sample_index"] for x in items], dtype=torch.long)
    sequence_id = torch.as_tensor([x.get("sequence_id", x.get("group_id", x["sample_index"])) for x in items], dtype=torch.long)

    targets_np = np.asarray([x["target"] for x in items])
    target = torch.as_tensor(targets_np, dtype=torch.float32 if np.issubdtype(targets_np.dtype, np.floating) else torch.long)

    return {
        "batch": QCVVBatch(
            traces=traces,
            valid_length=valid_length,
            coord_normalized=coord_normalized,
            sample_dt=sample_dt,
            family_id=family_id,
            run_id=run_id,
            sample_id=sample_index,
            sequence_id=sequence_id,
        ),
        "target": target,
        "sample_index": sample_index,
        "sequence_id": sequence_id,
        "window_start": torch.as_tensor([x.get("window_start", 0) for x in items], dtype=torch.long),
    }


def _indices_overlap_count(a: np.ndarray, b: np.ndarray, groups: np.ndarray | None = None) -> int:
    if groups is None:
        return int(len(set(map(int, a.tolist())) & set(map(int, b.tolist()))))
    return int(len(set(map(int, groups[a].tolist())) & set(map(int, groups[b].tolist()))))


def make_or_load_split_manifest(dataset, args, run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return train/val/test indices, optionally from a manifest."""
    if getattr(args, "split_manifest", None):
        manifest_path = Path(args.split_manifest)
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        train_idx = np.asarray(manifest["train_idx"], dtype=np.int64)
        val_idx = np.asarray(manifest["val_idx"], dtype=np.int64)
        test_idx = np.asarray(manifest.get("test_idx", []), dtype=np.int64)
        if test_idx.size == 0:
            # Backward-compatible two-way manifest.
            test_idx = val_idx.copy()
        max_idx = max(train_idx.max(initial=-1), val_idx.max(initial=-1), test_idx.max(initial=-1))
        if max_idx >= len(dataset):
            raise ValueError(
                f"Split manifest {manifest_path} contains index {max_idx}, but dataset length is {len(dataset)}. "
                "Use a manifest created with the same prepared files and family filter."
            )
        split_info = dict(manifest.get("split_info", {}))
        split_info["loaded_from"] = str(manifest_path)
        return train_idx, val_idx, test_idx, split_info

    all_indices = np.arange(len(dataset), dtype=np.int64)
    groups = np.asarray(dataset.group_ids_raw, dtype=np.int64)
    test_split = float(getattr(args, "test_split", 0.10))
    val_split = float(getattr(args, "val_split", 0.20))

    if getattr(args, "split_strategy", "grouped") == "grouped" and len(np.unique(groups)) >= 3:
        first = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=args.seed)
        train_val_idx, test_idx = next(first.split(all_indices, groups=groups))
        rel_val = val_split / max(1e-9, 1.0 - test_split)
        rel_val = min(max(rel_val, 0.01), 0.95)
        second = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=args.seed + 1)
        train_pos, val_pos = next(second.split(train_val_idx, groups=groups[train_val_idx]))
        train_idx = train_val_idx[train_pos]
        val_idx = train_val_idx[val_pos]
        strategy = "grouped_train_val_test"
    else:
        strat = None
        if getattr(args, "label_h5_dataset", None) is None:
            labels = np.asarray(dataset.family_ids_raw if args.target == "family" else dataset.run_ids_raw, dtype=np.int64)
            strat = labels if np.unique(labels).size > 1 else None
        train_val_idx, test_idx = train_test_split(all_indices, test_size=test_split, random_state=args.seed, stratify=strat)
        strat_tv = strat[train_val_idx] if strat is not None else None
        rel_val = val_split / max(1e-9, 1.0 - test_split)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=rel_val, random_state=args.seed + 1, stratify=strat_tv)
        strategy = "random_train_val_test"

    split_info = {
        "split_strategy": strategy,
        "num_samples": int(len(dataset)),
        "train_samples": int(train_idx.size),
        "val_samples": int(val_idx.size),
        "test_samples": int(test_idx.size),
        "num_unique_groups": int(len(np.unique(groups))),
        "group_overlap_train_val": _indices_overlap_count(train_idx, val_idx, groups),
        "group_overlap_train_test": _indices_overlap_count(train_idx, test_idx, groups),
        "group_overlap_val_test": _indices_overlap_count(val_idx, test_idx, groups),
    }
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_version": TRAIN_VERSION,
        "family": getattr(args, "family", None),
        "target": getattr(args, "target", None),
        "prepared": [str(p) for p in getattr(args, "prepared", [])],
        "split_info": split_info,
        "train_idx": train_idx.astype(int).tolist(),
        "val_idx": val_idx.astype(int).tolist(),
        "test_idx": test_idx.astype(int).tolist(),
    }
    out_path = run_dir / "split_manifest.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    split_info["written_to"] = str(out_path)
    return train_idx, val_idx, test_idx, split_info


class TeacherStore:
    def __init__(self, path: str | Path, min_confidence: float = 0.0):
        data = np.load(path, allow_pickle=True)
        self.path = str(path)
        self.sample_index = np.asarray(data["sample_index"], dtype=np.int64)
        self.state_probs = np.asarray(data["state_probs"], dtype=np.float32)
        conf = data["confidence"] if "confidence" in data.files else np.max(self.state_probs, axis=1)
        self.confidence = np.asarray(conf, dtype=np.float32)
        self.min_confidence = float(min_confidence)
        self.num_states = int(self.state_probs.shape[1])
        max_idx = int(self.sample_index.max(initial=-1))
        self.lookup = np.full(max_idx + 1, -1, dtype=np.int64)
        valid = self.sample_index >= 0
        self.lookup[self.sample_index[valid]] = np.flatnonzero(valid)

    def batch_probs(self, sample_id: Tensor, device: TorchDevice) -> tuple[Tensor | None, Tensor | None]:
        ids = sample_id.detach().cpu().numpy().astype(np.int64)
        row = np.full(ids.shape[0], -1, dtype=np.int64)
        in_range = (ids >= 0) & (ids < self.lookup.shape[0])
        row[in_range] = self.lookup[ids[in_range]]
        mask_np = (row >= 0) & (self.confidence[np.maximum(row, 0)] >= self.min_confidence)
        if not np.any(mask_np):
            return None, None
        probs = np.zeros((ids.shape[0], self.num_states), dtype=np.float32)
        probs[mask_np] = self.state_probs[row[mask_np]]
        return torch.as_tensor(probs, dtype=torch.float32, device=device), torch.as_tensor(mask_np, dtype=torch.bool, device=device)


def _load_teacher(args) -> TeacherStore | None:
    path = getattr(args, "teacher_npz", None)
    if not path:
        return None
    return TeacherStore(path, min_confidence=float(getattr(args, "teacher_min_confidence", 0.0)))


def _teacher_kl_loss(logits: Tensor, batch: QCVVBatch, teacher: TeacherStore | None, temperature: float) -> tuple[Tensor, int]:
    if teacher is None or batch.sample_id is None:
        return torch.zeros((), device=logits.device), 0
    probs, mask = teacher.batch_probs(batch.sample_id, logits.device)
    if probs is None or mask is None or not bool(mask.any().item()):
        return torch.zeros((), device=logits.device), 0
    if logits.shape[-1] != probs.shape[-1]:
        raise ValueError(f"Teacher has {probs.shape[-1]} states but model logits have {logits.shape[-1]} states. Use --force-num-states or a matching checkpoint.")
    T = float(max(temperature, 1e-6))
    logp = torch.log_softmax(logits[mask] / T, dim=-1)
    q = probs[mask].clamp_min(1e-8)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    loss = torch.nn.functional.kl_div(logp, q, reduction="batchmean") * (T * T)
    return loss, int(mask.sum().item())


def _limit_cpu_runtime_threads() -> None:
    """Keep Python/BLAS/DataLoader CPU usage from overwhelming the machine."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    if torch is not None:
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass


def _require_cuda_device(args, stage: str = "training"):
    """Return cuda:0 or fail loudly.

    This build is intentionally GPU-first for the GTX 1080 Ti workflow.  CPU is
    still used for file I/O and lightweight orchestration, but model fitting and
    optimization should not silently fall back to CPU.
    """
    _limit_cpu_runtime_threads()
    if torch is None:
        raise RuntimeError(f"PyTorch is required for {stage}: {_TORCH_IMPORT_ERROR}")
    if getattr(args, "cpu", False) and not getattr(args, "allow_cpu_training", False):
        raise RuntimeError("--cpu was supplied, but this GPU build is configured to train on CUDA only. Remove --cpu or add --allow-cpu-training for debugging.")
    if torch.cuda.is_available() and not getattr(args, "cpu", False):
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        try:
            props = torch.cuda.get_device_properties(device)
            print(f"CUDA training device for {stage}: {props.name} | VRAM={props.total_memory / (1024**3):.2f} GiB", flush=True)
        except Exception:
            print(f"CUDA training device for {stage}: {device}", flush=True)
        return device
    if getattr(args, "allow_cpu_training", False):
        print(f"[WARN] CPU fallback allowed for {stage}. This may overload CPU and is not recommended.", flush=True)
        return torch.device("cpu")
    raise RuntimeError("CUDA is not available. This build requires the GTX 1080 Ti for training. Check NVIDIA driver, CUDA PyTorch install, and CUDA_VISIBLE_DEVICES.")


def _init_device(args):
    _limit_cpu_runtime_threads()
    return _require_cuda_device(args, "neural training")


def _loader_kwargs(args, device):
    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = 0 if os.name == "nt" else max(2, min(4, (os.cpu_count() or 4) - 1))
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_batch,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs, num_workers


def _classification_ok(learning_task: str, target_classes: int, output_states: int) -> bool:
    return learning_task in {"supervised", "multitask", "auto"} and target_classes > 1 and target_classes == output_states


def _neural_train_one_epoch(model, loader, optimizer, criterion, scaler, device, args, learning_task, recon_weight, kl_weight, teacher, model_kind):
    model.train()
    totals = Counter()
    n_samples = 0
    y_true_all, y_pred_all = [], []
    use_amp = args.amp and device.type == "cuda"
    iterator = loader if tqdm is None else tqdm(loader, desc=f"train_{model_kind}", leave=False)
    for step, item in enumerate(iterator, start=1):
        batch, target = unpack_loader_item(item, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            if model_kind == "dmm":
                out = model(batch, sample_latent=True)
                kl_loss = out["kl_loss"]
                recon_loss = out["recon_loss"]
            else:
                out = model(batch)
                kl_loss = torch.zeros((), device=device)
                recon_loss = out.get("recon_loss", torch.zeros((), device=device))
            logits = out["state_logits"]
            do_cls = _classification_ok(learning_task, int(getattr(args, "_target_classes", 1)), logits.shape[-1])
            cls_loss = criterion(logits, target) if do_cls else torch.zeros((), device=device)
            distill_loss, teacher_count = _teacher_kl_loss(logits, batch, teacher, args.teacher_temperature)
            loss = cls_loss + recon_weight * recon_loss + kl_weight * kl_loss + args.teacher_weight * distill_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {model_kind} loss.")
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
        bs = int(target.shape[0])
        n_samples += bs
        totals["loss"] += float(loss.detach().item()) * bs
        totals["cls_loss"] += float(cls_loss.detach().item()) * bs
        totals["recon_loss"] += float(recon_loss.detach().item()) * bs
        totals["kl_loss"] += float(kl_loss.detach().item()) * bs
        totals["distill_loss"] += float(distill_loss.detach().item()) * max(1, teacher_count)
        totals["teacher_count"] += teacher_count
        if do_cls:
            pred = torch.argmax(logits.detach(), dim=-1)
            y_true_all.append(target.detach().cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())
        if step == 1 or step % args.log_interval == 0 or step == len(loader):
            print(
                f"  {model_kind} train step {step}/{len(loader)} loss={float(loss.detach().item()):.6f} "
                f"cls={float(cls_loss.detach().item()):.6f} recon={float(recon_loss.detach().item()):.6f} "
                f"kl={float(kl_loss.detach().item()):.6f} distill={float(distill_loss.detach().item()):.6f} teacher_n={teacher_count}",
                flush=True,
            )
    if y_true_all:
        metrics = classification_metrics(np.concatenate(y_true_all), np.concatenate(y_pred_all))
    else:
        metrics = empty_or_nan_metrics()
    for k in ["cls_loss", "recon_loss", "kl_loss"]:
        metrics[k] = totals[k] / max(1, n_samples)
    metrics["distill_loss"] = totals["distill_loss"] / max(1, totals["teacher_count"])
    metrics["teacher_count"] = int(totals["teacher_count"])
    return totals["loss"] / max(1, n_samples), metrics


@torch.no_grad()
def _neural_evaluate(model, loader, criterion, device, args, learning_task, recon_weight, kl_weight, teacher, model_kind):
    model.eval()
    totals = Counter()
    n_samples = 0
    y_true_all, y_pred_all = [], []
    teacher_true, teacher_pred = [], []
    use_amp = args.amp and device.type == "cuda"
    iterator = loader if tqdm is None else tqdm(loader, desc=f"val_{model_kind}", leave=False)
    for item in iterator:
        batch, target = unpack_loader_item(item, device)
        with autocast(device_type=device.type, enabled=use_amp):
            if model_kind == "dmm":
                out = model(batch, sample_latent=False)
                kl_loss = out["kl_loss"]
                recon_loss = out["recon_loss"]
            else:
                out = model(batch)
                kl_loss = torch.zeros((), device=device)
                recon_loss = out.get("recon_loss", torch.zeros((), device=device))
            logits = out["state_logits"]
            do_cls = _classification_ok(learning_task, int(getattr(args, "_target_classes", 1)), logits.shape[-1])
            cls_loss = criterion(logits, target) if do_cls else torch.zeros((), device=device)
            distill_loss, teacher_count = _teacher_kl_loss(logits, batch, teacher, args.teacher_temperature)
            loss = cls_loss + recon_weight * recon_loss + kl_weight * kl_loss + args.teacher_weight * distill_loss
        bs = int(target.shape[0])
        n_samples += bs
        totals["loss"] += float(loss.detach().item()) * bs
        totals["cls_loss"] += float(cls_loss.detach().item()) * bs
        totals["recon_loss"] += float(recon_loss.detach().item()) * bs
        totals["kl_loss"] += float(kl_loss.detach().item()) * bs
        totals["distill_loss"] += float(distill_loss.detach().item()) * max(1, teacher_count)
        totals["teacher_count"] += teacher_count
        pred = torch.argmax(logits.detach(), dim=-1)
        if do_cls:
            y_true_all.append(target.detach().cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())
        if teacher is not None and batch.sample_id is not None:
            probs, mask = teacher.batch_probs(batch.sample_id, device)
            if probs is not None and mask is not None and bool(mask.any().item()) and probs.shape[-1] == logits.shape[-1]:
                teacher_true.append(torch.argmax(probs[mask], dim=-1).detach().cpu().numpy())
                teacher_pred.append(pred[mask].detach().cpu().numpy())
    if y_true_all:
        metrics = classification_metrics(np.concatenate(y_true_all), np.concatenate(y_pred_all))
        metrics["confusion_matrix"] = confusion_matrix(np.concatenate(y_true_all), np.concatenate(y_pred_all)).tolist()
    else:
        metrics = empty_or_nan_metrics()
        metrics["confusion_matrix"] = []
    if teacher_true:
        metrics["teacher_argmax_accuracy"] = float(accuracy_score(np.concatenate(teacher_true), np.concatenate(teacher_pred)))
    else:
        metrics["teacher_argmax_accuracy"] = float("nan")
    for k in ["cls_loss", "recon_loss", "kl_loss"]:
        metrics[k] = totals[k] / max(1, n_samples)
    metrics["distill_loss"] = totals["distill_loss"] / max(1, totals["teacher_count"])
    metrics["teacher_count"] = int(totals["teacher_count"])
    return totals["loss"] / max(1, n_samples), metrics


def _run_train_neural(args, run_dir: Path, model_kind: str):
    if torch is None:
        raise ImportError(f"torch is required for {model_kind} training: {_TORCH_IMPORT_ERROR}")
    if model_kind == "cnn_gru":
        Model, Config = _import_cnn_gru()
    elif model_kind == "dmm":
        Model, Config = _import_dmm()
    else:
        raise ValueError(model_kind)

    dataset = build_prepared_dataset(args.prepared, args.target, family_filter=args.family, label_h5_dataset=args.label_h5_dataset)
    train_idx, val_idx, test_idx, split_info = make_or_load_split_manifest(dataset, args, run_dir)
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    test_ds = Subset(dataset, test_idx.tolist())

    target_classes = len(dataset.family_indexer.idx_to_class) if args.target == "family" else len(dataset.run_indexer.idx_to_class)
    teacher = _load_teacher(args)
    forced = int(args.force_num_states) if args.force_num_states is not None else None
    output_states = forced or (teacher.num_states if teacher is not None else max(1, target_classes))
    args._target_classes = target_classes

    learning_task = args.learning_task
    if learning_task == "auto":
        learning_task = "multitask" if target_classes > 1 and target_classes == output_states else "self_supervised"
    recon_weight = reconstruction_weight_for_task(args, learning_task)
    if model_kind == "dmm":
        recon_weight = float(args.dmm_recon_weight)
        kl_weight = float(args.dmm_kl_weight)
    else:
        kl_weight = 0.0

    device = _init_device(args)
    family_vocab_size = max(8, len(dataset.family_indexer.idx_to_class) + 2)
    run_vocab_size = max(16, len(dataset.run_indexer.idx_to_class) + 2)
    if model_kind == "cnn_gru":
        cfg = Config(
            family_vocab_size=family_vocab_size,
            run_vocab_size=run_vocab_size,
            num_states=max(1, output_states),
            use_family_meta=args.use_family_meta,
            use_run_meta=args.use_run_meta,
        )
    else:
        cfg = Config(
            family_vocab_size=family_vocab_size,
            run_vocab_size=run_vocab_size,
            num_states=max(1, output_states),
            use_family_meta=args.use_family_meta,
            use_run_meta=args.use_run_meta,
            latent_dim=args.dmm_latent_dim,
            conv_channels=args.dmm_conv_channels,
            encoder_hidden=args.dmm_encoder_hidden,
            temporal_downsample=args.dmm_temporal_downsample,
            dropout=args.dmm_dropout,
        )
    model = Model(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = make_cnn_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler(enabled=(device.type == "cuda" and args.amp))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_reduce_factor, patience=args.lr_patience)
    loader_kwargs, num_workers = _loader_kwargs(args, device)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    metrics_csv = run_dir / "metrics.csv"
    best_score = -float("inf")
    history = []
    epochs_without_improve = 0

    print(f"Device: {device}", flush=True)
    print(f"Model kind: {model_kind} | output_states={output_states} | target_classes={target_classes}", flush=True)
    print(f"Teacher: {teacher.path if teacher is not None else 'none'} | weight={args.teacher_weight}", flush=True)
    print(f"Train/val/test: {len(train_ds):,}/{len(val_ds):,}/{len(test_ds):,} | split={split_info}", flush=True)
    print(f"learning_task={learning_task} recon_weight={recon_weight} kl_weight={kl_weight} workers={num_workers}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loader, train_samples_this_epoch = build_train_epoch_loader(train_ds, loader_kwargs, args)
        if device.type == "cuda" and args.prefetch_to_gpu:
            train_iter = CUDAPrefetchLoader(train_loader, device)
            val_iter = CUDAPrefetchLoader(val_loader, device)
        else:
            train_iter = train_loader
            val_iter = val_loader
        t0 = time.time()
        train_loss, train_metrics = _neural_train_one_epoch(
            model, train_iter, optimizer, criterion, scaler, device, args,
            learning_task, recon_weight, kl_weight, teacher, model_kind,
        )
        val_loss, val_metrics = (None, None)
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_loss, val_metrics = _neural_evaluate(
                model, val_iter, criterion, device, args,
                learning_task, recon_weight, kl_weight, teacher, model_kind,
            )
            scheduler.step(val_loss)
        epoch_time = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "train_cls_loss": train_metrics.get("cls_loss"),
            "train_recon_loss": train_metrics.get("recon_loss"),
            "train_kl_loss": train_metrics.get("kl_loss"),
            "train_distill_loss": train_metrics.get("distill_loss"),
            "train_teacher_count": train_metrics.get("teacher_count"),
            "epoch_time_sec": epoch_time,
            "lr": optimizer.param_groups[0]["lr"],
            "batch_size": args.batch_size,
            "train_samples_this_epoch": int(train_samples_this_epoch),
            "train_steps_this_epoch": int(len(train_loader)),
        }
        if val_metrics is not None:
            row.update({
                "val_loss": val_loss,
                "val_acc": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_teacher_argmax_accuracy": val_metrics.get("teacher_argmax_accuracy"),
                "val_distill_loss": val_metrics.get("distill_loss"),
                "val_teacher_count": val_metrics.get("teacher_count"),
            })
        append_metrics_csv(metrics_csv, row)
        history.append(row)
        print_epoch_summary(epoch, train_loss, val_loss, val_metrics, epoch_time)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": vars(args),
            "val_metrics": val_metrics,
            "train_metrics": train_metrics,
            "model_kind": model_kind,
            "num_states": output_states,
        }
        if model_kind == "cnn_gru":
            ckpt["cnn_config"] = cfg.__dict__
        else:
            ckpt["dmm_config"] = cfg.__dict__
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(ckpt, last_path)
        if val_metrics is not None:
            if teacher is not None and np.isfinite(val_metrics.get("teacher_argmax_accuracy", np.nan)):
                score = float(val_metrics["teacher_argmax_accuracy"])
            elif classification_enabled(learning_task) and np.isfinite(val_metrics.get("f1_macro", np.nan)):
                score = float(val_metrics["f1_macro"])
            else:
                score = -float(val_loss)
            if score > best_score:
                best_score = score
                epochs_without_improve = 0
                torch.save(ckpt, best_path)
                print(f"  -> saved new best checkpoint: {best_path.name} score={best_score:.6f}", flush=True)
            else:
                epochs_without_improve += 1
        if epoch >= args.min_epochs and epochs_without_improve >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}. best_score={best_score:.6f}", flush=True)
            break

    # Locked test evaluation on the final selected model.
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    test_iter = CUDAPrefetchLoader(test_loader, device) if device.type == "cuda" and args.prefetch_to_gpu else test_loader
    test_loss, test_metrics = _neural_evaluate(model, test_iter, criterion, device, args, learning_task, recon_weight, kl_weight, teacher, model_kind)

    summary = {
        "mode": model_kind,
        "target": args.target,
        "family": args.family,
        "device": str(device),
        "best_selection_score": best_score,
        "learning_task": learning_task,
        "target_classes": int(target_classes),
        "output_states": int(output_states),
        "teacher_npz": args.teacher_npz,
        "history": history,
        "split_info": split_info,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved neural run to: {run_dir}", flush=True)


def run_train_cnn_gru(args, run_dir: Path):  # noqa: F811 - intentional override
    return _run_train_neural(args, run_dir, "cnn_gru")


def run_train_dmm(args, run_dir: Path):  # noqa: F811 - intentional override
    return _run_train_neural(args, run_dir, "dmm")


def _collect_cthmm_sequences(dataset, indices: np.ndarray, args) -> tuple[list[np.ndarray], list[float], list[int], list[int]]:
    sequences: list[np.ndarray] = []
    dts: list[float] = []
    sample_indices: list[int] = []
    sequence_ids: list[int] = []
    max_n = len(indices) if args.max_samples is None else min(len(indices), int(args.max_samples))
    use_indices = indices[:max_n]
    iterator = use_indices if tqdm is None else tqdm(use_indices, desc="cthmm_collect", leave=False)
    for idx in iterator:
        item = dataset[int(idx)]
        tr = _to_numpy(item["traces"], np.float32)
        valid = int(item["valid_length"])
        valid = max(2, min(valid, tr.shape[0]))
        y = tr[:valid, : int(args.cthmm_num_features)]
        if y.shape[0] < 2:
            continue
        sequences.append(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
        dts.append(float(item["sample_dt"]) if float(item["sample_dt"]) > 0 else 1.0)
        sample_indices.append(int(item["sample_index"]))
        sequence_ids.append(int(item.get("sequence_id", item.get("group_id", item["sample_index"]))))
    return sequences, dts, sample_indices, sequence_ids


def _write_cthmm_teacher(model, sequences, dts, sample_indices, split_name: str, out_path: Path):
    probs = []
    conf = []
    hard = []
    for y, dt in zip(sequences, dts):
        post = model.posterior(y, dt)
        p = post.mean(axis=0)
        p = p / max(float(np.sum(p)), 1e-12)
        probs.append(p.astype(np.float32))
        conf.append(float(np.max(p)))
        hard.append(int(np.argmax(p)))
    np.savez_compressed(
        out_path,
        sample_index=np.asarray(sample_indices, dtype=np.int64),
        state_probs=np.asarray(probs, dtype=np.float32),
        confidence=np.asarray(conf, dtype=np.float32),
        hard_state=np.asarray(hard, dtype=np.int64),
        split=np.asarray([split_name] * len(sample_indices), dtype=object),
        source_model=np.asarray(["cthmm"]),
    )


def run_train_cthmm(args, run_dir: Path):
    GaussianContinuousTimeHMM, CTHMMConfig = _import_cthmm()
    dataset = build_prepared_dataset(args.prepared, args.target, family_filter=args.family, label_h5_dataset=args.label_h5_dataset)
    train_idx, val_idx, test_idx, split_info = make_or_load_split_manifest(dataset, args, run_dir)
    train_seq, train_dt, train_samples, _ = _collect_cthmm_sequences(dataset, train_idx, args)
    val_seq, val_dt, val_samples, _ = _collect_cthmm_sequences(dataset, val_idx, args)
    test_seq, test_dt, test_samples, _ = _collect_cthmm_sequences(dataset, test_idx, args)
    cfg = CTHMMConfig(
        num_features=int(args.cthmm_num_features),
        max_em_iters=int(args.cthmm_max_em_iters),
        tol=float(args.cthmm_tol),
        min_variance=float(args.cthmm_min_variance),
        random_state=int(args.seed),
        verbose=bool(args.cthmm_verbose),
    )
    model = GaussianContinuousTimeHMM(cfg)
    fit_result = model.fit(train_seq, train_dt)
    metrics = {
        "mode": "cthmm",
        "family": args.family,
        "fit_result": fit_result.to_dict(),
        "rates": model.rates,
        "lifetimes": model.lifetimes,
        "train_score": model.score(train_seq, train_dt),
        "val_score": model.score(val_seq, val_dt),
        "test_score": model.score(test_seq, test_dt),
        "split_info": split_info,
        "n_train_sequences": len(train_seq),
        "n_val_sequences": len(val_seq),
        "n_test_sequences": len(test_seq),
    }
    model.save(run_dir / "cthmm_model.json")
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    _write_cthmm_teacher(model, train_seq, train_dt, train_samples, "train", run_dir / "teacher_train.npz")
    _write_cthmm_teacher(model, val_seq, val_dt, val_samples, "val", run_dir / "teacher_val.npz")
    _write_cthmm_teacher(model, test_seq, test_dt, test_samples, "test", run_dir / "teacher_test.npz")
    # Combined teacher is convenient for downstream distillation; split manifest still prevents leakage.
    parts = [np.load(run_dir / f"teacher_{s}.npz", allow_pickle=True) for s in ["train", "val", "test"]]
    np.savez_compressed(
        run_dir / "teacher_predictions.npz",
        sample_index=np.concatenate([p["sample_index"] for p in parts]),
        state_probs=np.concatenate([p["state_probs"] for p in parts]),
        confidence=np.concatenate([p["confidence"] for p in parts]),
        hard_state=np.concatenate([p["hard_state"] for p in parts]),
        split=np.concatenate([p["split"] for p in parts]),
        source_model=np.asarray(["cthmm"]),
    )
    print("CT-HMM results:", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved CT-HMM outputs to: {run_dir}", flush=True)


def extract_embeddings_with_cnn(args, dataset, checkpoint_path: str | Path):  # noqa: F811 - intentional override
    CNNModel, CNNConfig = _import_cnn_gru()
    device = _init_device(args)
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt.get("cnn_config", {})
    cfg = CNNConfig(**cfg_dict) if cfg_dict else CNNConfig()
    model = CNNModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    loader_kwargs, _ = _loader_kwargs(args, device)
    loader = DataLoader(dataset, shuffle=False, drop_last=False, **loader_kwargs)
    embeddings, labels, sequence_ids = [], [], []
    with torch.no_grad():
        iterator = loader if tqdm is None else tqdm(loader, desc="embed", leave=False)
        for item in iterator:
            batch = move_batch_to_device(item["batch"], device)
            out = model(batch)
            embeddings.append(out["embedding"].detach().cpu().numpy())
            labels.append(item["target"].cpu().numpy())
            # Use physical grouped sequence IDs, not sample IDs.
            sequence_ids.append(item["batch"].sequence_id.cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(labels), np.concatenate(sequence_ids)


def run_train_hsmm(args, run_dir: Path):  # noqa: F811 - intentional override
    GaussianHSMMQCVV, HSMMConfig = _import_hsmm()
    dataset = build_prepared_dataset(args.prepared, args.target, family_filter=args.family, label_h5_dataset=args.label_h5_dataset)
    train_idx, val_idx, test_idx, split_info = make_or_load_split_manifest(dataset, args, run_dir)
    # HSMM fits on train split only; predictions are exported for all splits.
    train_ds = Subset(dataset, train_idx.tolist())
    if args.hsmm_source == "cnn":
        if args.cnn_checkpoint is None:
            raise ValueError("--hsmm-source cnn requires --cnn-checkpoint")
        print("Extracting CNN embeddings for HSMM train split...", flush=True)
        X_train, y_train, seq_train = extract_embeddings_with_cnn(args, train_ds, args.cnn_checkpoint)
    else:
        XGBModel, XGBConfig = _import_xgboost()
        limit = len(train_ds) if args.max_samples is None else min(len(train_ds), int(args.max_samples))
        traces, valid_length, coords, sample_dt, family_id, run_id, y_list, sequence_id = [], [], [], [], [], [], [], []
        for i in range(limit):
            item = train_ds[i]
            traces.append(item["traces"])
            valid_length.append(item["valid_length"])
            coords.append(item["coord_normalized"])
            sample_dt.append(item["sample_dt"])
            family_id.append(item["family_id"])
            run_id.append(item["run_id"])
            y_list.append(item["target"])
            sequence_id.append(item.get("sequence_id", item.get("group_id", item["sample_index"])))
        batch = QCVVBatch(
            traces=np.stack(traces, axis=0),
            valid_length=np.asarray(valid_length, dtype=np.int64),
            coord_normalized=np.stack(coords, axis=0).astype(np.float32),
            sample_dt=np.asarray(sample_dt, dtype=np.float32),
            family_id=np.asarray(family_id, dtype=np.int64),
            run_id=np.asarray(run_id, dtype=np.int64),
            sequence_id=np.asarray(sequence_id, dtype=np.int64),
        )
        y_train = np.asarray(y_list)
        seq_train = np.asarray(sequence_id)
        xgb_model = XGBModel(XGBConfig(use_gpu=False))
        X_train, _ = xgb_model.build_features_from_batch(batch)
    unique_y = np.unique(y_train)
    supervised_ok = bool(args.hsmm_supervised and unique_y.size > 1)
    num_states = args.hsmm_states if args.hsmm_states is not None else int(max(2, min(4, unique_y.size if unique_y.size > 1 else 2)))
    cfg = HSMMConfig(num_states=num_states, min_duration=args.hsmm_min_duration, max_duration=args.hsmm_max_duration, random_state=args.seed)
    model = GaussianHSMMQCVV(cfg)
    if supervised_ok:
        model.fit_supervised(X_train, y_train, sequence_id=seq_train)
    else:
        model.fit_unsupervised_init(X_train)
    pred_train = model.predict_states(X_train, sequence_id=seq_train)
    metrics = {
        "mode": "supervised" if supervised_ok else "unsupervised",
        "num_states": int(cfg.num_states),
        "state_counts_train": {str(k): int(v) for k, v in Counter(pred_train.tolist()).items()},
        "n_train": int(X_train.shape[0]),
        "feature_dim": int(X_train.shape[1]),
        "split_info": split_info,
    }
    if y_train.dtype.kind in ("i", "u") and unique_y.size > 1:
        metrics.update(classification_metrics(y_train, pred_train))
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.save(run_dir / "predicted_states_train.npy", pred_train)
    np.save(run_dir / "sequence_id_train.npy", seq_train)
    np.savez(run_dir / "hsmm_params.npz", pi=model.pi, A=model.A, means=model.means, vars=model.vars, duration_logprob=model.duration_logprob)
    print("HSMM results:", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved HSMM outputs to: {run_dir}", flush=True)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


@torch.no_grad()
def _export_neural_predictions(args, run_dir: Path, checkpoint_path: Path, model_kind: str, out_path: Path):
    if model_kind == "cnn_gru":
        Model, Config = _import_cnn_gru()
        config_key = "cnn_config"
    else:
        Model, Config = _import_dmm()
        config_key = "dmm_config"
    device = _init_device(args)
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = Config(**ckpt.get(config_key, {}))
    model = Model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    dataset = build_prepared_dataset(args.prepared, args.target, family_filter=args.family, label_h5_dataset=args.label_h5_dataset)
    loader_kwargs, _ = _loader_kwargs(args, device)
    loader = DataLoader(dataset, shuffle=False, drop_last=False, **loader_kwargs)
    sample_ids, seq_ids, probs, confidence, hard = [], [], [], [], []
    for item in tqdm(loader, desc="export", leave=False) if tqdm is not None else loader:
        batch = move_batch_to_device(item["batch"], device)
        out = model(batch, sample_latent=False) if model_kind == "dmm" else model(batch)
        p = torch.softmax(out["state_logits"], dim=-1).detach().cpu().numpy()
        probs.append(p)
        confidence.append(np.max(p, axis=1))
        hard.append(np.argmax(p, axis=1))
        sample_ids.append(item["batch"].sample_id.cpu().numpy())
        seq_ids.append(item["batch"].sequence_id.cpu().numpy())
    np.savez_compressed(
        out_path,
        sample_index=np.concatenate(sample_ids).astype(np.int64),
        sequence_id=np.concatenate(seq_ids).astype(np.int64),
        state_probs=np.concatenate(probs).astype(np.float32),
        confidence=np.concatenate(confidence).astype(np.float32),
        hard_state=np.concatenate(hard).astype(np.int64),
        source_model=np.asarray([model_kind]),
    )


def run_export_predictions(args, run_dir: Path):
    out_path = Path(args.export_out) if args.export_out else run_dir / "predictions.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_kind = ckpt.get("model_kind", "cnn_gru" if "cnn_config" in ckpt else "dmm")
        _export_neural_predictions(args, run_dir, ckpt_path, model_kind, out_path)
    elif args.teacher_inputs:
        loaded = [np.load(p, allow_pickle=True) for p in args.teacher_inputs]
        common = set(map(int, loaded[0]["sample_index"].tolist()))
        for d in loaded[1:]:
            common &= set(map(int, d["sample_index"].tolist()))
        common_ids = np.asarray(sorted(common), dtype=np.int64)
        if common_ids.size == 0:
            raise ValueError("No overlapping sample_index values across teacher inputs.")
        prob_parts = []
        for d in loaded:
            idx_to_row = {int(s): i for i, s in enumerate(d["sample_index"].tolist())}
            prob_parts.append(d["state_probs"][[idx_to_row[int(s)] for s in common_ids]])
        avg = np.mean(np.stack(prob_parts, axis=0), axis=0)
        avg = avg / np.maximum(avg.sum(axis=1, keepdims=True), 1e-12)
        np.savez_compressed(
            out_path,
            sample_index=common_ids,
            state_probs=avg.astype(np.float32),
            confidence=np.max(avg, axis=1).astype(np.float32),
            hard_state=np.argmax(avg, axis=1).astype(np.int64),
            source_model=np.asarray(["teacher_ensemble"]),
            teacher_inputs=np.asarray([str(p) for p in args.teacher_inputs], dtype=object),
        )
    else:
        raise ValueError("export mode requires --checkpoint or --teacher-inputs")
    print(f"Exported predictions to: {out_path}", flush=True)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-8
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=1)
    return float(np.mean(0.5 * (kl_pm + kl_qm)))


def run_compare_models(args, run_dir: Path):
    if not args.prediction_npz or len(args.prediction_npz) < 2:
        raise ValueError("compare mode requires at least two --prediction-npz files")
    preds = []
    for p in args.prediction_npz:
        d = np.load(p, allow_pickle=True)
        name = str(d["source_model"][0]) if "source_model" in d.files else Path(p).stem
        preds.append((name, Path(p), d))
    pairs = []
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            name_a, path_a, a = preds[i]
            name_b, path_b, b = preds[j]
            ids_a = {int(s): k for k, s in enumerate(a["sample_index"].tolist())}
            ids_b = {int(s): k for k, s in enumerate(b["sample_index"].tolist())}
            common = sorted(set(ids_a) & set(ids_b))
            if not common:
                pairs.append({"a": name_a, "b": name_b, "n_common": 0})
                continue
            ia = [ids_a[s] for s in common]
            ib = [ids_b[s] for s in common]
            pa = np.asarray(a["state_probs"])[ia]
            pb = np.asarray(b["state_probs"])[ib]
            ka = pa.shape[1]
            kb = pb.shape[1]
            hard_a = np.argmax(pa, axis=1)
            hard_b = np.argmax(pb, axis=1)
            result = {
                "a": name_a,
                "b": name_b,
                "path_a": str(path_a),
                "path_b": str(path_b),
                "n_common": int(len(common)),
                "states_a": int(ka),
                "states_b": int(kb),
                "hard_agreement": float(np.mean(hard_a == hard_b)) if ka == kb else float("nan"),
            }
            if ka == kb:
                result["mean_js_divergence"] = _js_divergence(pa, pb)
                result["mean_abs_confidence_gap"] = float(np.mean(np.abs(np.max(pa, axis=1) - np.max(pb, axis=1))))
            pairs.append(result)
    report = {"created_at": datetime.now().isoformat(timespec="seconds"), "pairs": pairs, "inputs": [str(p) for p in args.prediction_npz]}
    with (run_dir / "model_disagreement.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved comparison to: {run_dir / 'model_disagreement.json'}", flush=True)


# =============================================================================
# GPU-first CT-HMM / HSMM overrides
# =============================================================================
# The earlier merged implementation used NumPy/CPU CT-HMM and CPU Gaussian-HSMM.
# Those are interpretable, but they can saturate the CPU.  The following overrides
# keep the same CLI modes while moving the expensive fitting steps to CUDA.


def _masked_time_points(batch: QCVVBatch, num_features: int, device: TorchDevice) -> tuple[Tensor, Tensor]:
    x = batch.traces.to(device, non_blocking=True).float()
    if x.shape[-1] < num_features:
        x = torch.nn.functional.pad(x, (0, num_features - x.shape[-1]))
    x = x[:, :, :num_features]
    lengths = batch.valid_length.to(device, non_blocking=True).long().clamp(min=1, max=x.shape[1])
    mask = torch.arange(x.shape[1], device=device)[None, :] < lengths[:, None]
    return x, mask


def _gaussian_log_prob_diag(x: Tensor, means: Tensor, vars_: Tensor) -> Tensor:
    # x [B,T,C], means/vars [K,C] -> [B,T,K]
    vars_ = vars_.clamp_min(1e-6)
    diff = x[:, :, None, :] - means[None, None, :, :]
    return -0.5 * ((diff * diff) / vars_[None, None, :, :] + torch.log(vars_)[None, None, :, :] + math.log(2.0 * math.pi)).sum(dim=-1)


def _gpu_loader_for_subset(ds, args, device, *, shuffle: bool = False, batch_size: int | None = None):
    loader_kwargs, _ = _loader_kwargs(args, device)
    loader_kwargs["batch_size"] = int(batch_size or loader_kwargs["batch_size"])
    # Keep the CPU cool: a single-process loader plus pinned memory is usually
    # enough for a 1080 Ti and avoids worker explosions on Windows/Linux.
    loader_kwargs["num_workers"] = int(getattr(args, "num_workers", 0) or 0)
    loader_kwargs["persistent_workers"] = False
    loader_kwargs.pop("prefetch_factor", None)
    return DataLoader(ds, shuffle=shuffle, drop_last=False, **loader_kwargs)


def _gpu_initialize_gaussian_states(loader, args, device, num_features: int, max_points: int = 200000) -> tuple[Tensor, Tensor]:
    vals = []
    seen = 0
    for item in loader:
        batch = item["batch"]
        x, mask = _masked_time_points(batch, num_features, device)
        flat = x[mask]
        if flat.numel() == 0:
            continue
        vals.append(flat.detach())
        seen += int(flat.shape[0])
        if seen >= max_points:
            break
    if not vals:
        raise ValueError("No valid trace points available for GPU CT-HMM initialization.")
    X = torch.cat(vals, dim=0)[:max_points]
    score = X[:, 0]
    thresh = torch.median(score)
    lab = (score > thresh).long()
    if torch.unique(lab).numel() < 2:
        lab = torch.arange(X.shape[0], device=device) % 2
    means = []
    vars_ = []
    for k in range(2):
        Xk = X[lab == k]
        if Xk.numel() == 0:
            Xk = X
        means.append(Xk.mean(dim=0))
        vars_.append(Xk.var(dim=0, unbiased=False).clamp_min(float(getattr(args, "cthmm_min_variance", 1e-6))))
    means = torch.stack(means, dim=0)
    vars_ = torch.stack(vars_, dim=0)
    order = torch.argsort(means[:, 0])
    return means[order].contiguous(), vars_[order].contiguous()


def _gpu_cthmm_epoch_stats(loader, args, device, means: Tensor, vars_: Tensor, num_features: int):
    K = int(means.shape[0])
    sum_w = torch.zeros(K, device=device)
    sum_x = torch.zeros(K, num_features, device=device)
    sum_x2 = torch.zeros(K, num_features, device=device)
    trans = torch.zeros(K, K, device=device)
    dwell = torch.zeros(K, device=device)
    total_ll = torch.zeros((), device=device)
    total_obs = 0
    for item in loader:
        batch = item["batch"]
        x, mask = _masked_time_points(batch, num_features, device)
        logp = _gaussian_log_prob_diag(x, means, vars_)
        probs = torch.softmax(logp, dim=-1) * mask[:, :, None].float()
        total_ll = total_ll + (torch.logsumexp(logp, dim=-1) * mask.float()).sum()
        total_obs += int(mask.sum().item())
        sum_w += probs.sum(dim=(0, 1))
        sum_x += torch.einsum("btk,btc->kc", probs, x)
        sum_x2 += torch.einsum("btk,btc->kc", probs, x * x)
        states = torch.argmax(logp, dim=-1)
        lengths = batch.valid_length.to(device).long().clamp(min=1, max=x.shape[1])
        dt = batch.sample_dt.to(device).float().clamp_min(1e-12)
        if x.shape[1] >= 2:
            pair_mask = mask[:, 1:] & mask[:, :-1]
            a = states[:, :-1]
            b = states[:, 1:]
            for i in range(K):
                dwell[i] += ((states == i).float() * mask.float() * dt[:, None]).sum()
                for j in range(K):
                    trans[i, j] += (((a == i) & (b == j) & pair_mask).float()).sum()
        else:
            for i in range(K):
                dwell[i] += ((states == i).float() * mask.float() * dt[:, None]).sum()
    new_means = sum_x / sum_w[:, None].clamp_min(1e-12)
    new_vars = (sum_x2 / sum_w[:, None].clamp_min(1e-12) - new_means * new_means).clamp_min(float(getattr(args, "cthmm_min_variance", 1e-6)))
    return new_means, new_vars, trans, dwell, float(total_ll.detach().cpu().item()), int(total_obs)


@torch.no_grad()
def _gpu_cthmm_export_teacher(loader, args, device, means: Tensor, vars_: Tensor, split_name: str, out_path: Path, num_features: int):
    sample_ids, probs_out, conf_out, hard_out = [], [], [], []
    for item in loader:
        batch = item["batch"]
        x, mask = _masked_time_points(batch, num_features, device)
        logp = _gaussian_log_prob_diag(x, means, vars_)
        probs = torch.softmax(logp, dim=-1) * mask[:, :, None].float()
        avg = probs.sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        avg = avg / avg.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sample_ids.append(batch.sample_id.detach().cpu().numpy().astype(np.int64))
        probs_np = avg.detach().cpu().numpy().astype(np.float32)
        probs_out.append(probs_np)
        conf_out.append(np.max(probs_np, axis=1).astype(np.float32))
        hard_out.append(np.argmax(probs_np, axis=1).astype(np.int64))
    if sample_ids:
        np.savez_compressed(
            out_path,
            sample_index=np.concatenate(sample_ids),
            state_probs=np.concatenate(probs_out),
            confidence=np.concatenate(conf_out),
            hard_state=np.concatenate(hard_out),
            split=np.asarray([split_name] * int(sum(len(x) for x in sample_ids)), dtype=object),
            source_model=np.asarray(["gpu_cthmm"]),
        )


def run_train_cthmm(args, run_dir: Path):  # noqa: F811 - GPU override
    """GPU Gaussian-emission CT-HMM approximation.

    It preserves the QCVV outputs needed by the pipeline while moving the
    iterative fitting and posterior computation to CUDA.  CPU work is limited to
    dataset indexing, file I/O, and saving NumPy artifacts.
    """
    if torch is None:
        raise RuntimeError(f"PyTorch is required for GPU CT-HMM: {_TORCH_IMPORT_ERROR}")
    device = _require_cuda_device(args, "gpu_cthmm")
    dataset = build_prepared_dataset(args.prepared, args.target, family_filter=args.family, label_h5_dataset=args.label_h5_dataset)
    train_idx, val_idx, test_idx, split_info = make_or_load_split_manifest(dataset, args, run_dir)
    if getattr(args, "max_samples", None) is not None:
        # Preserve split structure while capping work per split.
        cap = int(args.max_samples)
        train_idx = train_idx[:cap]
        val_idx = val_idx[:max(1, min(len(val_idx), cap // 4 or cap))]
        test_idx = test_idx[:max(1, min(len(test_idx), cap // 4 or cap))]
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    test_ds = Subset(dataset, test_idx.tolist())
    num_features = int(getattr(args, "cthmm_num_features", 2))
    train_loader = _gpu_loader_for_subset(train_ds, args, device, shuffle=True)
    val_loader = _gpu_loader_for_subset(val_ds, args, device, shuffle=False)
    test_loader = _gpu_loader_for_subset(test_ds, args, device, shuffle=False)

    means, vars_ = _gpu_initialize_gaussian_states(train_loader, args, device, num_features)
    history = []
    print(f"GPU CT-HMM: train/val/test={len(train_ds):,}/{len(val_ds):,}/{len(test_ds):,} features={num_features}", flush=True)
    for it in range(1, int(getattr(args, "cthmm_max_em_iters", 25)) + 1):
        new_means, new_vars, trans, dwell, ll, n_obs = _gpu_cthmm_epoch_stats(train_loader, args, device, means, vars_, num_features)
        delta = float(torch.max(torch.abs(new_means - means)).detach().cpu().item())
        means, vars_ = new_means, new_vars
        gamma01 = float((trans[0, 1] / dwell[0].clamp_min(1e-12)).detach().cpu().item())
        gamma10 = float((trans[1, 0] / dwell[1].clamp_min(1e-12)).detach().cpu().item())
        row = {"iter": it, "log_likelihood_per_obs": ll / max(n_obs, 1), "gamma_01": gamma01, "gamma_10": gamma10, "mean_delta": delta}
        history.append(row)
        print(f"[gpu_cthmm {it:03d}] ll/obs={row['log_likelihood_per_obs']:.6g} gamma01={gamma01:.6g} gamma10={gamma10:.6g} delta={delta:.3g}", flush=True)
        if delta < float(getattr(args, "cthmm_tol", 1e-5)):
            break

    _, _, trans, dwell, train_ll, train_obs = _gpu_cthmm_epoch_stats(train_loader, args, device, means, vars_, num_features)
    gamma01 = float((trans[0, 1] / dwell[0].clamp_min(1e-12)).detach().cpu().item())
    gamma10 = float((trans[1, 0] / dwell[1].clamp_min(1e-12)).detach().cpu().item())
    tau0 = 1.0 / max(gamma01, 1e-30)
    tau1 = 1.0 / max(gamma10, 1e-30)
    model_payload = {
        "model_type": "gpu_gaussian_cthmm_approx",
        "device": str(device),
        "means": means.detach().cpu().numpy().tolist(),
        "vars": vars_.detach().cpu().numpy().tolist(),
        "gamma_01": gamma01,
        "gamma_10": gamma10,
        "tau_0": tau0,
        "tau_1": tau1,
        "history": history,
        "split_info": split_info,
    }
    with (run_dir / "cthmm_model.json").open("w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=2)
    with (run_dir / "qcvv_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"mode": "gpu_cthmm", "rates_and_lifetimes": {"gamma_01": gamma01, "gamma_10": gamma10, "tau_0": tau0, "tau_1": tau1}, "train_log_likelihood_per_obs": train_ll / max(train_obs, 1), "device": str(device)}, f, indent=2)

    _gpu_cthmm_export_teacher(train_loader, args, device, means, vars_, "train", run_dir / "teacher_train.npz", num_features)
    _gpu_cthmm_export_teacher(val_loader, args, device, means, vars_, "val", run_dir / "teacher_val.npz", num_features)
    _gpu_cthmm_export_teacher(test_loader, args, device, means, vars_, "test", run_dir / "teacher_test.npz", num_features)
    parts = [np.load(run_dir / name, allow_pickle=True) for name in ["teacher_train.npz", "teacher_val.npz", "teacher_test.npz"] if (run_dir / name).exists()]
    if parts:
        np.savez_compressed(
            run_dir / "teacher_predictions.npz",
            sample_index=np.concatenate([p["sample_index"] for p in parts]),
            state_probs=np.concatenate([p["state_probs"] for p in parts]),
            confidence=np.concatenate([p["confidence"] for p in parts]),
            hard_state=np.concatenate([p["hard_state"] for p in parts]),
            split=np.concatenate([p["split"] for p in parts]),
            source_model=np.asarray(["gpu_cthmm"]),
        )
    print(f"Saved GPU CT-HMM outputs to: {run_dir}", flush=True)


class _GPUHSMMDurationNet(nn.Module):
    def __init__(self, input_channels: int = 2, hidden: int = 64, states: int = 2):
        super().__init__()
        self.input_channels = int(input_channels)
        self.states = int(states)
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_channels, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.state_head = nn.Conv1d(hidden, self.states, kernel_size=1)
        self.prototypes = nn.Parameter(torch.randn(self.states, self.input_channels) * 0.05)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        feat = self.encoder(x.transpose(1, 2))
        logits = self.state_head(feat).transpose(1, 2)
        probs = torch.softmax(logits, dim=-1)
        recon = torch.einsum("btk,kc->btc", probs, self.prototypes)
        return {"logits": logits, "probs": probs, "recon": recon}


def _gpu_hsmm_losses(out: dict[str, Tensor], x: Tensor, mask: Tensor, args) -> tuple[Tensor, dict[str, float]]:
    probs = out["probs"]
    recon = out["recon"]
    mask_f = mask.float()
    recon_loss = (((recon - x) ** 2).sum(dim=-1) * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    if x.shape[1] >= 2:
        pair_mask = (mask[:, 1:] & mask[:, :-1]).float()
        duration_loss = (((probs[:, 1:, :] - probs[:, :-1, :]) ** 2).sum(dim=-1) * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
    else:
        duration_loss = torch.zeros((), device=x.device)
    entropy = (-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1) * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    mean_p = (probs * mask_f[:, :, None]).sum(dim=(0, 1)) / mask_f.sum().clamp_min(1.0)
    balance = ((mean_p - 1.0 / probs.shape[-1]) ** 2).sum()
    loss = recon_loss + float(getattr(args, "gpu_hsmm_duration_weight", 0.20)) * duration_loss + float(getattr(args, "gpu_hsmm_entropy_weight", 0.01)) * entropy + float(getattr(args, "gpu_hsmm_balance_weight", 0.10)) * balance
    return loss, {"recon_loss": float(recon_loss.detach().cpu().item()), "duration_loss": float(duration_loss.detach().cpu().item()), "entropy": float(entropy.detach().cpu().item()), "balance": float(balance.detach().cpu().item())}


def _gpu_hsmm_epoch(model, loader, optimizer, device, args, train: bool):
    model.train(train)
    totals = Counter()
    n_batches = 0
    for item in loader:
        batch = item["batch"]
        x, mask = _masked_time_points(batch, int(getattr(args, "cthmm_num_features", 2)), device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss, parts = _gpu_hsmm_losses(out, x, mask, args)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(getattr(args, "grad_clip_norm", 1.0)))
                optimizer.step()
        totals["loss"] += float(loss.detach().cpu().item())
        for k, v in parts.items():
            totals[k] += v
        n_batches += 1
    return {k: float(v / max(n_batches, 1)) for k, v in totals.items()} | {"num_batches": int(n_batches)}


@torch.no_grad()
def _gpu_hsmm_export_states(model, loader, device, args, out_dir: Path):
    states_all, sample_all, seq_all = [], [], []
    for item in loader:
        batch = item["batch"]
        x, mask = _masked_time_points(batch, int(getattr(args, "cthmm_num_features", 2)), device)
        out = model(x)
        state = torch.argmax(out["probs"], dim=-1)
        # sequence-level state proxy for compatibility with old HSMM artifacts.
        counts = []
        for b in range(state.shape[0]):
            valid = state[b][mask[b]]
            if valid.numel() == 0:
                counts.append(0)
            else:
                counts.append(int(torch.mode(valid).values.detach().cpu().item()))
        states_all.append(np.asarray(counts, dtype=np.int64))
        sample_all.append(batch.sample_id.detach().cpu().numpy().astype(np.int64))
        seq_all.append(batch.sequence_id.detach().cpu().numpy().astype(np.int64))
    pred = np.concatenate(states_all) if states_all else np.asarray([], dtype=np.int64)
    sample_id = np.concatenate(sample_all) if sample_all else np.asarray([], dtype=np.int64)
    seq_id = np.concatenate(seq_all) if seq_all else np.asarray([], dtype=np.int64)
    np.save(out_dir / "predicted_states.npy", pred)
    np.save(out_dir / "sequence_id.npy", seq_id)
    np.savez_compressed(out_dir / "hsmm_predictions.npz", sample_index=sample_id, sequence_id=seq_id, hard_state=pred, source_model=np.asarray(["gpu_hsmm_duration"]))
    return pred


def run_train_hsmm(args, run_dir: Path):  # noqa: F811 - GPU override
    """GPU duration-aware HSMM-style training.

    This replaces the CPU Gaussian-HSMM fit with a CUDA neural duration model so
    HSMM-stage fitting does not saturate the CPU.  It keeps the HSMM role in the
    pipeline: duration/smoothness characterization and decoded latent state runs.
    """
    if torch is None or nn is None:
        raise RuntimeError(f"PyTorch is required for GPU HSMM: {_TORCH_IMPORT_ERROR}")
    device = _require_cuda_device(args, "gpu_hsmm")
    dataset = build_prepared_dataset(args.prepared, args.target, family_filter=args.family, label_h5_dataset=args.label_h5_dataset)
    train_idx, val_idx, test_idx, split_info = make_or_load_split_manifest(dataset, args, run_dir)
    if getattr(args, "max_samples", None) is not None:
        cap = int(args.max_samples)
        train_idx = train_idx[:cap]
        val_idx = val_idx[:max(1, min(len(val_idx), cap // 4 or cap))]
        test_idx = test_idx[:max(1, min(len(test_idx), cap // 4 or cap))]
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    test_ds = Subset(dataset, test_idx.tolist())
    train_loader = _gpu_loader_for_subset(train_ds, args, device, shuffle=True)
    val_loader = _gpu_loader_for_subset(val_ds, args, device, shuffle=False)
    test_loader = _gpu_loader_for_subset(test_ds, args, device, shuffle=False)
    states = int(args.hsmm_states or args.force_num_states or 2)
    model = _GPUHSMMDurationNet(input_channels=int(getattr(args, "cthmm_num_features", 2)), hidden=int(getattr(args, "gpu_hsmm_hidden", 64)), states=states).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(getattr(args, "lr", 3e-4)), weight_decay=float(getattr(args, "weight_decay", 1e-4)))
    history = []
    best_loss = float("inf")
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    print(f"GPU HSMM duration model: train/val/test={len(train_ds):,}/{len(val_ds):,}/{len(test_ds):,} states={states}", flush=True)
    for epoch in range(1, int(getattr(args, "epochs", 20)) + 1):
        t0 = time.time()
        train_metrics = _gpu_hsmm_epoch(model, train_loader, optimizer, device, args, train=True)
        val_metrics = _gpu_hsmm_epoch(model, val_loader, optimizer, device, args, train=False) if len(val_ds) else train_metrics
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "epoch_time_sec": time.time() - t0}
        history.append(row)
        print(f"[gpu_hsmm {epoch:03d}] loss={train_metrics['loss']:.6g} val_loss={val_metrics['loss']:.6g} duration={train_metrics.get('duration_loss', 0):.6g} entropy={train_metrics.get('entropy', 0):.6g}", flush=True)
        torch.save({"model_state": model.state_dict(), "config": vars(args), "epoch": epoch, "history": history, "model_type": "gpu_hsmm_duration"}, last_path)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save({"model_state": model.state_dict(), "config": vars(args), "epoch": epoch, "history": history, "model_type": "gpu_hsmm_duration"}, best_path)
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    pred = _gpu_hsmm_export_states(model, test_loader if len(test_ds) else val_loader, device, args, run_dir)
    metrics = {
        "mode": "gpu_duration_hsmm",
        "device": str(device),
        "num_states": states,
        "best_val_loss": best_loss,
        "state_counts": {str(k): int(v) for k, v in Counter(pred.tolist()).items()},
        "history": history,
        "split_info": split_info,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved GPU HSMM outputs to: {run_dir}", flush=True)


def parse_args():  # noqa: F811 - intentional override
    parser = argparse.ArgumentParser(description="Train and combine QCVV models on prepared .pt/.h5 files.")
    parser.add_argument("--mode", choices=["cnn_gru", "dmm", "xgboost", "hsmm", "cthmm", "export", "compare"], default="cnn_gru")
    parser.add_argument("--ready-dir", default=str(DEFAULT_READY_DIR))
    parser.add_argument("--prepared", nargs="*", default=None)
    parser.add_argument("--family", choices=["all", "parity", "x_loop", "z_loop"], default="all")
    parser.add_argument("--training-scope", choices=["together", "individual", "both"], default="together")
    parser.add_argument("--target", choices=["family", "run"], default="family")
    parser.add_argument("--learning-task", choices=["auto", "supervised", "self_supervised", "multitask"], default="auto")
    parser.add_argument("--use-family-meta", action="store_true")
    parser.add_argument("--use-run-meta", action="store_true")
    parser.add_argument("--label-h5-dataset", default=None)
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--split-strategy", choices=["grouped", "random"], default="grouped")
    parser.add_argument("--split-manifest", default=None)

    parser.add_argument("--cpu", action="store_true", help="Deprecated for this GPU build; use --allow-cpu-training only for debugging.")
    parser.add_argument("--allow-cpu-training", action="store_true", help="Allow CPU fallback. Off by default so training fails if CUDA is unavailable.")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--prefetch-to-gpu", dest="prefetch_to_gpu", action="store_true")
    parser.add_argument("--no-prefetch-to-gpu", dest="prefetch_to_gpu", action="store_false")
    parser.set_defaults(prefetch_to_gpu=True)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=False)
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--aux-recon-weight", type=float, default=0.10)
    parser.add_argument("--self-supervised-recon-weight", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--epoch-sample-fraction", type=float, default=0.25)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--full-epoch", action="store_true")
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--force-num-states", type=int, default=None)
    parser.add_argument("--teacher-npz", default=None)
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--teacher-min-confidence", type=float, default=0.90)
    parser.add_argument("--teacher-temperature", type=float, default=2.0)

    parser.add_argument("--dmm-latent-dim", type=int, default=24)
    parser.add_argument("--dmm-conv-channels", type=int, default=48)
    parser.add_argument("--dmm-encoder-hidden", type=int, default=64)
    parser.add_argument("--dmm-temporal-downsample", type=int, default=4)
    parser.add_argument("--dmm-dropout", type=float, default=0.10)
    parser.add_argument("--dmm-recon-weight", type=float, default=0.25)
    parser.add_argument("--dmm-kl-weight", type=float, default=0.01)

    parser.add_argument("--hsmm-source", choices=["cnn", "handcrafted"], default="handcrafted")
    parser.add_argument("--cnn-checkpoint", default=None)
    parser.add_argument("--hsmm-supervised", action="store_true")
    parser.add_argument("--hsmm-states", type=int, default=None)
    parser.add_argument("--hsmm-min-duration", type=int, default=1)
    parser.add_argument("--hsmm-max-duration", type=int, default=16)
    parser.add_argument("--gpu-hsmm-hidden", type=int, default=64)
    parser.add_argument("--gpu-hsmm-duration-weight", type=float, default=0.20)
    parser.add_argument("--gpu-hsmm-entropy-weight", type=float, default=0.01)
    parser.add_argument("--gpu-hsmm-balance-weight", type=float, default=0.10)

    parser.add_argument("--cthmm-num-features", type=int, default=2)
    parser.add_argument("--cthmm-max-em-iters", type=int, default=25)
    parser.add_argument("--cthmm-tol", type=float, default=1e-5)
    parser.add_argument("--cthmm-min-variance", type=float, default=1e-6)
    parser.add_argument("--cthmm-verbose", action="store_true")

    parser.add_argument("--checkpoint", default=None, help="Neural checkpoint for export mode.")
    parser.add_argument("--export-out", default=None)
    parser.add_argument("--teacher-inputs", nargs="*", default=None)
    parser.add_argument("--prediction-npz", nargs="*", default=None)
    return parser.parse_args()


def dispatch_train_mode(args, run_dir: Path):  # noqa: F811 - intentional override
    if args.mode == "cnn_gru":
        run_train_cnn_gru(args, run_dir)
    elif args.mode == "dmm":
        run_train_dmm(args, run_dir)
    elif args.mode == "xgboost":
        run_train_xgboost(args, run_dir)
    elif args.mode == "hsmm":
        run_train_hsmm(args, run_dir)
    elif args.mode == "cthmm":
        run_train_cthmm(args, run_dir)
    elif args.mode == "export":
        run_export_predictions(args, run_dir)
    elif args.mode == "compare":
        run_compare_models(args, run_dir)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(0)
