import argparse
import math
from pathlib import Path
from typing import Any

import torch


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]

DEFAULT_INPUT_DIR = REPO_ROOT / "manual-data" / "~ready_torch"

REQUIRED_DATASETS = [
    "traces",
    "valid_length",
    "coord_normalized",
    "sample_dt",
    "family_name",
    "run_name",
]

OPTIONAL_COMMON_DATASETS = [
    "coord_values",
    "operating_index",
    "original_length",
    "window_start",
    "source_file",
    "source_group",
    "trace_channels",
    "coord_names",
    "axis1_alt_names",
    "axis1_alt_values",
]


def safe_float(x: float) -> float | None:
    return float(x) if math.isfinite(float(x)) else None


def preview_list(values: list[Any], max_preview: int) -> list[str]:
    out = []
    for x in values[:max_preview]:
        out.append(str(x))
    return out


def tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    t = t.detach().cpu()
    stats: dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": int(t.numel()),
    }

    if t.numel() == 0:
        stats.update(
            {
                "finite_frac": None,
                "nan_count": 0,
                "inf_count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "zero_frac": None,
            }
        )
        return stats

    if not (torch.is_floating_point(t) or t.dtype == torch.bool):
        tf = t.to(torch.float64)
        finite = torch.isfinite(tf)
    else:
        tf = t
        finite = torch.isfinite(tf)

    nan_count = int(torch.isnan(tf).sum().item()) if torch.is_floating_point(tf) else 0
    inf_count = int(torch.isinf(tf).sum().item()) if torch.is_floating_point(tf) else 0
    finite_count = int(finite.sum().item())
    finite_frac = finite_count / max(1, t.numel())

    stats["finite_frac"] = float(finite_frac)
    stats["nan_count"] = nan_count
    stats["inf_count"] = inf_count

    if finite_count > 0:
        xf = tf[finite].to(torch.float64)
        stats["min"] = safe_float(xf.min().item())
        stats["max"] = safe_float(xf.max().item())
        stats["mean"] = safe_float(xf.mean().item())
        stats["std"] = safe_float(xf.std(unbiased=False).item())
        zero_frac = float((xf == 0).sum().item() / max(1, finite_count))
        stats["zero_frac"] = zero_frac
    else:
        stats["min"] = None
        stats["max"] = None
        stats["mean"] = None
        stats["std"] = None
        stats["zero_frac"] = None

    return stats


def discover_pt_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted(input_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in: {input_dir}")
    return files


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    input_dir = Path(args.input_dir)
    if not args.input:
        return discover_pt_files(input_dir)

    out: list[Path] = []
    for raw in args.input:
        p = Path(raw)
        if not p.is_absolute():
            p = input_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        out.append(p)
    return out


def load_bundle(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def audit_bundle(path: Path, max_preview: int = 5) -> dict[str, Any]:
    bundle = load_bundle(path)

    report: dict[str, Any] = {
        "file": str(path),
        "readable": True,
        "warnings": [],
        "top_level_keys": sorted(bundle.keys()) if isinstance(bundle, dict) else [],
        "format": bundle.get("format") if isinstance(bundle, dict) else None,
        "source_h5": bundle.get("source_h5") if isinstance(bundle, dict) else None,
        "attrs": bundle.get("attrs", {}) if isinstance(bundle, dict) else {},
        "datasets": {},
        "summary": {},
    }

    if not isinstance(bundle, dict):
        report["warnings"].append("Bundle is not a dict.")
        return report

    if "datasets" not in bundle:
        report["warnings"].append("Missing top-level 'datasets' key.")
        return report

    datasets = bundle["datasets"]
    if not isinstance(datasets, dict):
        report["warnings"].append("'datasets' is not a dict.")
        return report

    missing_required = [k for k in REQUIRED_DATASETS if k not in datasets]
    if missing_required:
        report["warnings"].append(f"Missing required datasets: {missing_required}")

    sample_length_candidates: dict[str, int] = {}

    for name in sorted(datasets.keys()):
        value = datasets[name]

        if torch.is_tensor(value):
            stats = tensor_stats(value)
            report["datasets"][name] = {"kind": "tensor", **stats}
            if value.ndim >= 1:
                sample_length_candidates[name] = int(value.shape[0])

        elif isinstance(value, list):
            entry = {
                "kind": "list",
                "length": len(value),
                "preview": preview_list(value, max_preview),
            }
            report["datasets"][name] = entry
            sample_length_candidates[name] = len(value)

        else:
            report["datasets"][name] = {
                "kind": type(value).__name__,
                "value_preview": str(value)[:200],
            }

    # consistency checks
    expected_sample_keys = [
        "traces",
        "valid_length",
        "coord_normalized",
        "sample_dt",
        "family_name",
        "run_name",
    ]
    lengths = {}
    for k in expected_sample_keys:
        if k in sample_length_candidates:
            lengths[k] = sample_length_candidates[k]

    unique_lengths = sorted(set(lengths.values()))
    if len(unique_lengths) > 1:
        report["warnings"].append(f"Inconsistent sample-axis lengths among required datasets: {lengths}")

    # semantic checks
    if "traces" in datasets and torch.is_tensor(datasets["traces"]):
        traces = datasets["traces"]
        if traces.ndim != 3:
            report["warnings"].append(f"'traces' expected ndim=3, found {traces.ndim}")
        elif traces.shape[-1] != 2:
            report["warnings"].append(f"'traces' expected last dim 2, found {traces.shape[-1]}")

    if "coord_normalized" in datasets and torch.is_tensor(datasets["coord_normalized"]):
        coord_norm = datasets["coord_normalized"]
        if coord_norm.ndim != 2:
            report["warnings"].append(f"'coord_normalized' expected ndim=2, found {coord_norm.ndim}")

    if "valid_length" in datasets and torch.is_tensor(datasets["valid_length"]):
        vl = datasets["valid_length"]
        if vl.ndim != 1:
            report["warnings"].append(f"'valid_length' expected ndim=1, found {vl.ndim}")

    if "sample_dt" in datasets and torch.is_tensor(datasets["sample_dt"]):
        dt = datasets["sample_dt"]
        if dt.ndim != 1:
            report["warnings"].append(f"'sample_dt' expected ndim=1, found {dt.ndim}")

    total_datasets = len(datasets)
    tensor_count = sum(1 for v in datasets.values() if torch.is_tensor(v))
    list_count = sum(1 for v in datasets.values() if isinstance(v, list))

    report["summary"] = {
        "total_datasets": total_datasets,
        "tensor_datasets": tensor_count,
        "list_datasets": list_count,
        "sample_axis_lengths": lengths,
    }

    return report


def print_report(report: dict[str, Any]) -> None:
    print("=" * 100, flush=True)
    print(f"FILE: {report['file']}", flush=True)
    print(f"READABLE: {report['readable']}", flush=True)
    print(f"FORMAT: {report.get('format')}", flush=True)
    print(f"SOURCE_H5: {report.get('source_h5')}", flush=True)

    attrs = report.get("attrs", {})
    print("ROOT ATTRS:", flush=True)
    if attrs:
        for k, v in attrs.items():
            print(f"  - {k}: {v}", flush=True)
    else:
        print("  - none", flush=True)

    warnings = report.get("warnings", [])
    print("WARNINGS:", "none" if not warnings else "", flush=True)
    for w in warnings:
        print(f"  - {w}", flush=True)

    print("DATASET TREE:", flush=True)
    for name, info in report["datasets"].items():
        if info["kind"] == "tensor":
            print(f"  - /{name} | shape={tuple(info['shape'])} | dtype={info['dtype']}", flush=True)
            print(
                "      sample_stats: "
                f"finite={info['finite_frac']}, nan={info['nan_count']}, inf={info['inf_count']}, "
                f"zero_frac={info['zero_frac']}, min={info['min']}, max={info['max']}, "
                f"mean={info['mean']}, std={info['std']}",
                flush=True,
            )
        elif info["kind"] == "list":
            print(f"  - /{name} | length={info['length']} | kind=list", flush=True)
            print(f"      preview: {info['preview']}", flush=True)
        else:
            print(f"  - /{name} | kind={info['kind']} | preview={info.get('value_preview')}", flush=True)

    print("SUMMARY:", flush=True)
    for k, v in report["summary"].items():
        print(f"  - {k}: {v}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit prepared PyTorch .pt bundles.")
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Repo-local directory containing prepared .pt bundles.",
    )
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help="Optional specific .pt filenames or paths. If omitted, audits all .pt in --input-dir.",
    )
    parser.add_argument(
        "--max-preview",
        type=int,
        default=5,
        help="How many string/list values to preview per dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = resolve_inputs(args)

    overall = {
        "total_files": 0,
        "readable_files": 0,
        "files_with_warnings": 0,
        "total_datasets": 0,
    }

    for p in input_paths:
        report = audit_bundle(p, max_preview=args.max_preview)
        print_report(report)

        overall["total_files"] += 1
        overall["readable_files"] += 1 if report["readable"] else 0
        overall["files_with_warnings"] += 1 if report["warnings"] else 0
        overall["total_datasets"] += report["summary"]["total_datasets"]

    print("=" * 100, flush=True)
    print("OVERALL SUMMARY", flush=True)
    for k, v in overall.items():
        print(f"  - {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
