import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np

DEFAULT_ROOT = r"E:\Software Engineering Stuff\Quantum\Majorana-MLQT-Project-EthanVu-CMPE188\manual-data"
DEFAULT_EXTS = {".h5", ".hdf5", ".nc"}
TEXT_META_NAMES = {
    "family",
    "family_name",
    "run_name",
    "source_file",
    "source_group",
    "coord_names",
    "trace_channels",
    "runs",
    "sources",
}


@dataclass
class NumericSummary:
    sample_count: int
    finite_fraction: float
    nan_count: int
    inf_count: int
    zero_fraction: float
    min_value: Optional[float]
    max_value: Optional[float]
    mean_value: Optional[float]
    std_value: Optional[float]


@dataclass
class DatasetInfo:
    path: str
    shape: Tuple[int, ...]
    dtype: str
    attrs: Dict[str, Any]
    numeric_summary: Optional[NumericSummary]


@dataclass
class FileReport:
    path: str
    readable: bool
    size_bytes: int
    file_kind: str
    groups: int
    datasets: int
    max_depth: int
    root_attrs: Dict[str, Any]
    warnings: List[str]
    datasets_info: List[DatasetInfo]


def decode_if_bytes(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return repr(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "O", "U"}:
            out = []
            for item in value.tolist():
                out.append(decode_if_bytes(item))
            return out
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def safe_attrs(obj: h5py.Dataset | h5py.Group | h5py.File) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in obj.attrs.items():
        try:
            out[str(key)] = decode_if_bytes(value)
        except Exception:
            out[str(key)] = "<unreadable attr>"
    return out


def find_files(root: Path) -> List[Path]:
    matches: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in DEFAULT_EXTS:
            matches.append(path)
    return sorted(matches)


def sample_numeric_array(ds: h5py.Dataset, max_points: int = 200_000) -> Optional[np.ndarray]:
    if ds.dtype.kind not in "iufb":
        return None
    shape = ds.shape
    if any(dim == 0 for dim in shape):
        return np.array([], dtype=np.float64)
    try:
        if ds.size <= max_points:
            arr = ds[...]
        else:
            # Take a representative hyperslab instead of loading everything.
            slices = []
            remaining = max_points
            for dim in shape:
                take = min(dim, max(1, int(round(remaining ** (1 / max(1, len(shape)))))))
                slices.append(slice(0, take))
            arr = ds[tuple(slices)]
        return np.asarray(arr, dtype=np.float64).ravel()
    except Exception:
        return None


def summarize_numeric(ds: h5py.Dataset) -> Optional[NumericSummary]:
    sample = sample_numeric_array(ds)
    if sample is None:
        return None
    if sample.size == 0:
        return NumericSummary(0, 1.0, 0, 0, 0.0, None, None, None, None)
    finite_mask = np.isfinite(sample)
    finite = sample[finite_mask]
    nan_count = int(np.isnan(sample).sum())
    inf_count = int(np.isinf(sample).sum())
    zero_fraction = float(np.mean(sample == 0))
    if finite.size == 0:
        return NumericSummary(
            sample_count=int(sample.size),
            finite_fraction=float(np.mean(finite_mask)),
            nan_count=nan_count,
            inf_count=inf_count,
            zero_fraction=zero_fraction,
            min_value=None,
            max_value=None,
            mean_value=None,
            std_value=None,
        )
    return NumericSummary(
        sample_count=int(sample.size),
        finite_fraction=float(np.mean(finite_mask)),
        nan_count=nan_count,
        inf_count=inf_count,
        zero_fraction=zero_fraction,
        min_value=float(np.min(finite)),
        max_value=float(np.max(finite)),
        mean_value=float(np.mean(finite)),
        std_value=float(np.std(finite)),
    )


def walk_items(group: h5py.Group, prefix: str = "") -> Iterable[Tuple[str, h5py.Group | h5py.Dataset]]:
    for key in group.keys():
        item = group[key]
        path = f"{prefix}/{key}" if prefix else f"/{key}"
        yield path, item
        if isinstance(item, h5py.Group):
            yield from walk_items(item, path)


def classify_file(hf: h5py.File, path: Path) -> str:
    root_keys = set(hf.keys())
    if {"mpr_A1", "mpr_A2", "mpr_B1"}.issubset(root_keys):
        return "parity_combined"
    if {"traces", "valid_length", "family_name", "run_name"}.issubset(root_keys):
        return "prepared"
    if {"Cq", "iCq", "time"}.issubset(root_keys):
        return "original_single"
    if path.name.lower().startswith("prepared_"):
        return "prepared_like"
    return "generic"


def check_original_single(hf: h5py.File, warnings: List[str]) -> None:
    cq = hf.get("Cq")
    icq = hf.get("iCq")
    time = hf.get("time")
    if not isinstance(cq, h5py.Dataset):
        warnings.append("Missing dataset: /Cq")
        return
    if not isinstance(icq, h5py.Dataset):
        warnings.append("Missing dataset: /iCq")
    if not isinstance(time, h5py.Dataset):
        warnings.append("Missing dataset: /time")
    if isinstance(icq, h5py.Dataset) and cq.shape != icq.shape:
        warnings.append(f"Cq/iCq shape mismatch: {cq.shape} vs {icq.shape}")
    if len(cq.shape) != 4:
        warnings.append(f"Expected 4D Cq tensor, found shape {cq.shape}")
    if isinstance(time, h5py.Dataset) and len(cq.shape) >= 1 and cq.shape[-1] != time.shape[0]:
        warnings.append(f"time length {time.shape[0]} does not match Cq last dim {cq.shape[-1]}")

    coord_names = [name for name in hf.keys() if name not in {"Cq", "iCq", "time"}]
    coord_datasets = [name for name in coord_names if isinstance(hf[name], h5py.Dataset) and hf[name].ndim == 1]
    dim_sizes = list(cq.shape[:-1])
    used = []
    for name in coord_datasets:
        n = int(hf[name].shape[0])
        if n in dim_sizes:
            used.append(name)
    if len(used) < 3:
        warnings.append(
            f"Found fewer than 3 coordinate arrays matching Cq non-time dims; matched={used} all_coords={coord_datasets}"
        )


def check_parity_combined(hf: h5py.File, warnings: List[str]) -> None:
    for run in ["mpr_A1", "mpr_A2", "mpr_B1"]:
        grp = hf.get(run)
        if not isinstance(grp, h5py.Group):
            warnings.append(f"Missing run group: /{run}")
            continue
        cq = grp.get("Cq")
        icq = grp.get("iCq")
        time = grp.get("time")
        if not isinstance(cq, h5py.Dataset):
            warnings.append(f"Missing dataset: /{run}/Cq")
            continue
        if not isinstance(icq, h5py.Dataset):
            warnings.append(f"Missing dataset: /{run}/iCq")
        if not isinstance(time, h5py.Dataset):
            warnings.append(f"Missing dataset: /{run}/time")
        if isinstance(icq, h5py.Dataset) and cq.shape != icq.shape:
            warnings.append(f"{run}: Cq/iCq shape mismatch: {cq.shape} vs {icq.shape}")
        if len(cq.shape) != 4:
            warnings.append(f"{run}: expected 4D Cq tensor, found shape {cq.shape}")
        if isinstance(time, h5py.Dataset) and cq.shape[-1] != time.shape[0]:
            warnings.append(f"{run}: time length {time.shape[0]} != Cq last dim {cq.shape[-1]}")

        # Check that some 1D coordinates align with non-time dims.
        coord_hits = []
        dim_sizes = list(cq.shape[:-1])
        for key in grp.keys():
            item = grp[key]
            if isinstance(item, h5py.Dataset) and item.ndim == 1 and key not in {"time"}:
                if item.shape[0] in dim_sizes:
                    coord_hits.append(key)
        if len(coord_hits) < 3:
            warnings.append(f"{run}: fewer than 3 coord arrays matching Cq dims: {coord_hits}")

    if "runs" in hf and isinstance(hf["runs"], h5py.Dataset):
        try:
            runs = [str(decode_if_bytes(x)) for x in hf["runs"][...].tolist()]
            expected = {"mpr_A1", "mpr_A2", "mpr_B1"}
            if set(runs) != expected:
                warnings.append(f"/runs dataset mismatch: {runs}")
        except Exception:
            warnings.append("Could not read /runs dataset")


def check_prepared(hf: h5py.File, warnings: List[str]) -> None:
    required = ["traces", "valid_length", "family_name", "run_name"]
    for name in required:
        if name not in hf:
            warnings.append(f"Missing required prepared dataset: /{name}")
    if "traces" not in hf:
        return
    traces = hf["traces"]
    if traces.ndim != 3:
        warnings.append(f"Expected /traces to be 3D [N,T,C], found {traces.shape}")
        return
    n, t, c = traces.shape
    if c != 2:
        warnings.append(f"Expected /traces channel dim = 2 [Cq,iCq], found {c}")
    if t <= 0 or n <= 0:
        warnings.append(f"Invalid /traces shape {traces.shape}")

    for name in [
        "valid_length",
        "window_start",
        "operating_index",
        "coord_values",
        "coord_normalized",
        "sample_dt",
        "original_length",
        "family_name",
        "run_name",
        "source_file",
        "source_group",
    ]:
        if name in hf and isinstance(hf[name], h5py.Dataset):
            ds = hf[name]
            if ds.shape[0] != n:
                warnings.append(f"/{name} first dimension {ds.shape[0]} != number of samples {n}")

    if "valid_length" in hf:
        valid = np.asarray(hf["valid_length"][...])
        if np.any(valid <= 0):
            warnings.append("/valid_length contains non-positive entries")
        if np.any(valid > t):
            warnings.append(f"/valid_length contains entries > window length {t}")

    if "coord_values" in hf:
        ds = hf["coord_values"]
        if ds.ndim != 2:
            warnings.append(f"Expected /coord_values to be 2D [N,K], found {ds.shape}")

    if "coord_normalized" in hf:
        ds = hf["coord_normalized"]
        if ds.ndim != 2:
            warnings.append(f"Expected /coord_normalized to be 2D [N,K], found {ds.shape}")
        else:
            try:
                sample = sample_numeric_array(ds)
                if sample is not None and sample.size > 0:
                    finite = sample[np.isfinite(sample)]
                    if finite.size > 0 and (np.min(finite) < -1.1 or np.max(finite) > 1.1):
                        warnings.append("/coord_normalized appears outside expected normalized range [-1, 1]")
            except Exception:
                pass

    # Quick padding sanity check: if valid_length < T, tail after valid_length should be close to zero often.
    try:
        valid = np.asarray(hf["valid_length"][...]) if "valid_length" in hf else None
        if valid is not None and n > 0:
            idx = min(n - 1, 5)
            sample_trace = np.asarray(traces[idx])
            vl = int(valid[idx])
            if vl < t:
                tail = sample_trace[vl:]
                if tail.size > 0:
                    tail_abs_mean = float(np.mean(np.abs(tail)))
                    if tail_abs_mean > 1e-6:
                        warnings.append(
                            "Padded region in one checked sample is not near zero; verify masking strategy if padding is intended"
                        )
    except Exception:
        pass


def report_file(path: Path) -> FileReport:
    size_bytes = path.stat().st_size
    warnings: List[str] = []
    datasets_info: List[DatasetInfo] = []
    groups = 0
    datasets = 0
    max_depth = 0
    root_attrs: Dict[str, Any] = {}
    file_kind = "unknown"

    try:
        with h5py.File(path, "r") as hf:
            root_attrs = safe_attrs(hf)
            file_kind = classify_file(hf, path)
            for p, item in walk_items(hf):
                depth = p.count("/")
                max_depth = max(max_depth, depth)
                if isinstance(item, h5py.Group):
                    groups += 1
                else:
                    datasets += 1
                    datasets_info.append(
                        DatasetInfo(
                            path=p,
                            shape=tuple(int(x) for x in item.shape),
                            dtype=str(item.dtype),
                            attrs=safe_attrs(item),
                            numeric_summary=summarize_numeric(item),
                        )
                    )

            if file_kind == "original_single":
                check_original_single(hf, warnings)
            elif file_kind == "parity_combined":
                check_parity_combined(hf, warnings)
            elif file_kind in {"prepared", "prepared_like"}:
                check_prepared(hf, warnings)

        readable = True
    except Exception as exc:
        readable = False
        warnings.append(f"Open/read failed: {exc}")

    return FileReport(
        path=str(path),
        readable=readable,
        size_bytes=size_bytes,
        file_kind=file_kind,
        groups=groups,
        datasets=datasets,
        max_depth=max_depth,
        root_attrs=root_attrs,
        warnings=warnings,
        datasets_info=datasets_info,
    )


def print_dataset_info(ds: DatasetInfo, verbose: bool) -> None:
    print(f"  - {ds.path} | shape={ds.shape} | dtype={ds.dtype}")
    if verbose:
        for k, v in ds.attrs.items():
            print(f"      attr {k}: {v}")
    if ds.numeric_summary is not None:
        ns = ds.numeric_summary
        print(
            "      sample_stats: "
            f"finite={ns.finite_fraction:.4f}, nan={ns.nan_count}, inf={ns.inf_count}, "
            f"zero_frac={ns.zero_fraction:.4f}, min={ns.min_value}, max={ns.max_value}, "
            f"mean={ns.mean_value}, std={ns.std_value}"
        )


def print_report(report: FileReport, verbose: bool) -> None:
    print("=" * 100)
    print(f"FILE: {report.path}")
    print(f"SIZE: {report.size_bytes / (1024**2):.2f} MB")
    print(f"READABLE: {report.readable}")
    print(f"KIND: {report.file_kind}")
    print(f"GROUPS: {report.groups}")
    print(f"DATASETS: {report.datasets}")
    print(f"MAX DEPTH: {report.max_depth}")
    print("ROOT ATTRS:")
    if report.root_attrs:
        for k, v in report.root_attrs.items():
            print(f"  - {k}: {v}")
    else:
        print("  (none)")
    if report.warnings:
        print("WARNINGS:")
        for w in report.warnings:
            print(f"  - {w}")
    else:
        print("WARNINGS: none")
    print("DATASET TREE:")
    for ds in report.datasets_info:
        print_dataset_info(ds, verbose=verbose)


def report_to_jsonable(report: FileReport) -> Dict[str, Any]:
    return {
        "path": report.path,
        "readable": report.readable,
        "size_bytes": report.size_bytes,
        "file_kind": report.file_kind,
        "groups": report.groups,
        "datasets": report.datasets,
        "max_depth": report.max_depth,
        "root_attrs": report.root_attrs,
        "warnings": report.warnings,
        "datasets_info": [
            {
                "path": ds.path,
                "shape": list(ds.shape),
                "dtype": ds.dtype,
                "attrs": ds.attrs,
                "numeric_summary": None if ds.numeric_summary is None else asdict(ds.numeric_summary),
            }
            for ds in report.datasets_info
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check shape and data integrity for manual-data H5 files.")
    parser.add_argument("root", nargs="?", default=DEFAULT_ROOT, help="Root folder to scan")
    parser.add_argument("--json-out", default=None, help="Optional JSON report path")
    parser.add_argument("--summary-only", action="store_true", help="Skip dataset-level printing")
    parser.add_argument("--verbose-attrs", action="store_true", help="Print dataset attributes too")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root does not exist: {root}")
        return 1

    files = find_files(root)
    print(f"Scanning root: {root}")
    print(f"Found {len(files)} file(s).")
    if not files:
        return 1

    reports = [report_file(path) for path in files]

    if not args.summary_only:
        for report in reports:
            print_report(report, verbose=args.verbose_attrs)

    readable = sum(1 for r in reports if r.readable)
    warned = sum(1 for r in reports if r.warnings)
    total_datasets = sum(r.datasets for r in reports)

    print("=" * 100)
    print("OVERALL SUMMARY")
    print(f"  Total files: {len(reports)}")
    print(f"  Readable files: {readable}")
    print(f"  Files with warnings: {warned}")
    print(f"  Total datasets: {total_datasets}")
    by_kind: Dict[str, int] = {}
    for r in reports:
        by_kind[r.file_kind] = by_kind.get(r.file_kind, 0) + 1
    print(f"  By kind: {by_kind}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(root),
            "summary": {
                "total_files": len(reports),
                "readable_files": readable,
                "files_with_warnings": warned,
                "total_datasets": total_datasets,
                "by_kind": by_kind,
            },
            "files": [report_to_jsonable(r) for r in reports],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"JSON report written to: {out_path}")

    if readable != len(reports):
        return 1
    if warned:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
