import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]
DEFAULT_READY_TORCH_DIR = REPO_ROOT / "manual-data" / "~ready_torch"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "manual-data" / "~ready_torch_no_static"


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_numpy_1d(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def as_string_list(value: Any) -> list[str]:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    out = []
    for x in value:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return out


def finite_median_or_default(value: Any, default: float) -> float:
    arr = to_numpy_1d(value).astype(np.float64, copy=False)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.median(arr))


def tensor_full_like(value: Any, fill_value: float) -> Any:
    if torch.is_tensor(value):
        return torch.full_like(value, fill_value=fill_value)
    arr = np.asarray(value)
    return np.full_like(arr, fill_value, dtype=arr.dtype)


def tensor_zeros_like(value: Any) -> Any:
    if torch.is_tensor(value):
        return torch.zeros_like(value)
    arr = np.asarray(value)
    return np.zeros_like(arr)


def summarize_bundle(path: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    ds = bundle.get("datasets", {})
    summary: dict[str, Any] = {"file": path.name}

    traces = ds.get("traces")
    if traces is not None:
        summary["num_samples"] = int(traces.shape[0])
        summary["trace_shape"] = list(traces.shape)

    if "family_name" in ds:
        fam = as_string_list(ds["family_name"])
        summary["families"] = sorted({str(x) for x in fam})
        summary["family_counts"] = {k: int(fam.count(k)) for k in sorted(set(fam))}

    if "sample_dt" in ds:
        arr = to_numpy_1d(ds["sample_dt"]).astype(np.float64, copy=False)
        finite = arr[np.isfinite(arr)]
        summary["sample_dt_unique_count"] = int(np.unique(finite).size) if finite.size else 0
        summary["sample_dt_min"] = float(np.min(finite)) if finite.size else None
        summary["sample_dt_max"] = float(np.max(finite)) if finite.size else None

    if "coord_normalized" in ds:
        coords = ds["coord_normalized"]
        coords_np = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
        if coords_np.size:
            finite = coords_np[np.isfinite(coords_np)]
            summary["coord_normalized_min"] = float(np.min(finite)) if finite.size else None
            summary["coord_normalized_max"] = float(np.max(finite)) if finite.size else None

    return summary


def scrub_bundle(
    path: Path,
    out_dir: Path,
    zero_coords: bool,
    constant_sample_dt: bool,
    sample_dt_value: float | None,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    bundle = torch_load(path)
    if bundle.get("format") != "prepared_pt_bundle_v1":
        raise ValueError(f"Unsupported PT bundle format in {path}: {bundle.get('format')}")

    original_summary = summarize_bundle(path, bundle)
    out_bundle = copy.deepcopy(bundle)
    ds = out_bundle["datasets"]
    changes: list[str] = []

    if zero_coords and "coord_normalized" in ds:
        ds["coord_normalized"] = tensor_zeros_like(ds["coord_normalized"])
        changes.append("coord_normalized -> zeros")
        if "manifest" in out_bundle and "coord_normalized" in out_bundle["manifest"]:
            out_bundle["manifest"]["coord_normalized"]["scrubbed"] = "zeros"

    if constant_sample_dt and "sample_dt" in ds:
        fill = float(sample_dt_value) if sample_dt_value is not None else finite_median_or_default(ds["sample_dt"], default=1e-6)
        if fill <= 0 or not np.isfinite(fill):
            fill = 1e-6
        ds["sample_dt"] = tensor_full_like(ds["sample_dt"], fill)
        changes.append(f"sample_dt -> constant {fill:.12g}")
        if "manifest" in out_bundle and "sample_dt" in out_bundle["manifest"]:
            out_bundle["manifest"]["sample_dt"]["scrubbed"] = f"constant {fill:.12g}"

    out_bundle.setdefault("attrs", {})["metadata_scrubbed"] = True
    out_bundle["attrs"]["metadata_scrub_changes"] = changes
    out_bundle["attrs"]["metadata_scrub_source"] = str(path)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / path.name
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_path}. Use --overwrite to replace it.")

    torch.save(out_bundle, out_path)
    scrubbed_summary = summarize_bundle(out_path, out_bundle)
    report = {
        "input": str(path),
        "output": str(out_path),
        "changes": changes,
        "before": original_summary,
        "after": scrubbed_summary,
    }
    return out_path, report


def discover_pt_files(ready_dir: Path) -> list[Path]:
    if not ready_dir.exists():
        raise FileNotFoundError(f"Ready torch directory not found: {ready_dir}")
    files = sorted(ready_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in: {ready_dir}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create metadata-scrubbed copies of prepared .pt bundles for leakage ablations."
    )
    parser.add_argument(
        "ready_dir",
        nargs="?",
        default=str(DEFAULT_READY_TORCH_DIR),
        help="Directory containing prepared .pt bundles.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where scrubbed .pt copies will be written.",
    )
    parser.add_argument(
        "--keep-coords",
        action="store_true",
        help="Do not zero coord_normalized. By default coord_normalized is zeroed.",
    )
    parser.add_argument(
        "--keep-sample-dt",
        action="store_true",
        help="Do not make sample_dt constant. By default sample_dt is replaced with a per-file median.",
    )
    parser.add_argument(
        "--sample-dt-value",
        type=float,
        default=None,
        help="Use this positive constant for sample_dt instead of each file's median.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing scrubbed .pt files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ready_dir = Path(args.ready_dir)
    out_dir = Path(args.out_dir)
    files = discover_pt_files(ready_dir)

    print("=" * 100, flush=True)
    print(f"Script: {CURRENT_FILE}", flush=True)
    print(f"Input ready_torch dir: {ready_dir}", flush=True)
    print(f"Output dir: {out_dir}", flush=True)
    print(f"Files: {len(files)}", flush=True)
    print(f"Zero coords: {not args.keep_coords}", flush=True)
    print(f"Constant sample_dt: {not args.keep_sample_dt}", flush=True)
    print("=" * 100, flush=True)

    reports = []
    for path in files:
        print(f"Scrubbing: {path.name}", flush=True)
        out_path, report = scrub_bundle(
            path=path,
            out_dir=out_dir,
            zero_coords=not args.keep_coords,
            constant_sample_dt=not args.keep_sample_dt,
            sample_dt_value=args.sample_dt_value,
            overwrite=args.overwrite,
        )
        reports.append(report)
        print(f"  Saved: {out_path}", flush=True)
        for change in report["changes"]:
            print(f"  - {change}", flush=True)

    report_path = out_dir / "metadata_scrub_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    print("=" * 100, flush=True)
    print(f"DONE. Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
