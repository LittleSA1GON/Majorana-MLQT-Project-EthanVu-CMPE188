import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def resolve_path(project_root: Path, value: Any) -> Path | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    p = Path(s)
    if not p.is_absolute():
        p = project_root / p
    return p if p.exists() else None


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def hard_from_npz(data: dict[str, np.ndarray]) -> np.ndarray | None:
    if "hard_state" in data:
        return np.asarray(data["hard_state"]).astype(int).reshape(-1)
    if "state_probs" in data:
        return np.argmax(np.asarray(data["state_probs"]), axis=1).astype(int)
    return None


def confidence_from_npz(data: dict[str, np.ndarray]) -> np.ndarray | None:
    if "confidence" in data:
        return np.asarray(data["confidence"], dtype=float).reshape(-1)
    if "state_probs" in data:
        return np.max(np.asarray(data["state_probs"], dtype=float), axis=1)
    return None


def entropy_bits(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return -np.sum(probs * np.log2(probs), axis=1)


def bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    if values.size == 0:
        return np.array([0, 1])
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
        return np.linspace(0, max(1, len(values)), n_bins + 1)
    return np.linspace(vmin, vmax, n_bins + 1)


def compute_drift_for_artifact(row: Any, pred_path: Path, n_bins: int, project_root: Path) -> list[dict[str, Any]]:
    data = load_npz(pred_path)
    hard = hard_from_npz(data)
    if hard is None:
        return []
    n = len(hard)
    conf = confidence_from_npz(data)
    probs = np.asarray(data["state_probs"], dtype=float) if "state_probs" in data else None
    if "sample_index" in data and len(data["sample_index"]) == n:
        axis = np.asarray(data["sample_index"], dtype=float).reshape(-1)
        axis_name = "sample_index"
    elif "sequence_id" in data and len(data["sequence_id"]) == n:
        axis = np.asarray(data["sequence_id"], dtype=float).reshape(-1)
        axis_name = "sequence_id"
    else:
        axis = np.arange(n, dtype=float)
        axis_name = "prediction_order"

    edges = bin_edges(axis, n_bins)
    rows: list[dict[str, Any]] = []
    for b in range(len(edges) - 1):
        left = edges[b]
        right = edges[b + 1]
        if b == len(edges) - 2:
            mask = (axis >= left) & (axis <= right)
        else:
            mask = (axis >= left) & (axis < right)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        h = hard[idx]
        out: dict[str, Any] = {
            "bundle": getattr(row, "bundle", None),
            "family": getattr(row, "family", None),
            "stage": getattr(row, "stage", None),
            "model": getattr(row, "model", None),
            "source_file": str(pred_path.relative_to(project_root)) if pred_path.is_relative_to(project_root) else str(pred_path),
            "bin_index": b,
            "bin_axis": axis_name,
            "bin_start": float(left),
            "bin_end": float(right),
            "n_predictions": int(idx.size),
            "state_0_occupancy": float(np.mean(h == 0)),
            "state_1_occupancy": float(np.mean(h == 1)),
            "switch_count": int(np.sum(h[1:] != h[:-1])) if idx.size > 1 else 0,
            "switch_rate": float(np.mean(h[1:] != h[:-1])) if idx.size > 1 else 0.0,
        }
        if conf is not None and len(conf) == n:
            out["mean_confidence"] = float(np.mean(conf[idx]))
            out["median_confidence"] = float(np.median(conf[idx]))
        if probs is not None and len(probs) == n:
            ent = entropy_bits(probs[idx])
            out["mean_entropy_bits"] = float(np.mean(ent))
        if "sequence_id" in data and len(data["sequence_id"]) == n:
            out["n_sequences"] = int(len(np.unique(np.asarray(data["sequence_id"])[idx])))
        rows.append(out)
    return rows


def summarize_drift(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    metrics = ["state_0_occupancy", "state_1_occupancy", "mean_confidence", "mean_entropy_bits", "switch_rate"]
    out: list[dict[str, Any]] = []
    group_cols = ["bundle", "family", "stage", "model", "source_file"]
    for keys, group in df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        for metric in metrics:
            if metric not in group:
                continue
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            if vals.empty:
                continue
            out.append({
                **base,
                "metric": metric,
                "n_bins": int(vals.size),
                "mean_across_bins": float(vals.mean()),
                "min_bin_value": float(vals.min()),
                "max_bin_value": float(vals.max()),
                "range_max_minus_min": float(vals.max() - vals.min()),
                "std_across_bins": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "relative_range_over_mean_abs": float((vals.max() - vals.min()) / max(abs(vals.mean()), 1e-12)),
            })
    return out


def plot_drift(out_dir: Path, drift_df: pd.DataFrame) -> list[str]:
    outputs: list[str] = []
    if drift_df.empty:
        return outputs
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["state_0_occupancy", "mean_confidence", "mean_entropy_bits", "switch_rate"]
    for family in sorted(drift_df["family"].dropna().unique()):
        fam = drift_df[drift_df["family"] == family]
        for bundle in sorted(fam["bundle"].dropna().unique()):
            sub = fam[fam["bundle"] == bundle]
            for metric in metrics:
                if metric not in sub or sub[metric].dropna().empty:
                    continue
                plt.figure(figsize=(9, 4.8))
                for model_key, model_group in sub.groupby(["stage", "model"], dropna=False):
                    model_group = model_group.sort_values("bin_index")
                    label = "/".join(str(x) for x in model_key if str(x) != "nan")
                    plt.plot(model_group["bin_index"], model_group[metric], marker="o", linewidth=1.5, label=label)
                plt.xlabel("time/order bin")
                plt.ylabel(metric)
                plt.title(f"Drift check: {family} {bundle} {metric}")
                plt.legend(loc="best", fontsize=8)
                plt.tight_layout()
                path = fig_dir / f"drift_{family}_{bundle}_{metric}.png"
                plt.savefig(path, dpi=180)
                plt.close()
                outputs.append(str(path))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute time/order-binned drift diagnostics from QCVV prediction artifacts.")
    parser.add_argument("--project-root", default=".", help="Project root containing artifact paths from artifact_inventory.csv.")
    parser.add_argument("--out-dir", default="qcvv_outputs", help="QCVV output directory from qcvv_extract.py.")
    parser.add_argument("--n-bins", type=int, default=20, help="Number of bins across sample_index/sequence_id/order.")
    parser.add_argument("--skip-figures", action="store_true", help="Do not create drift figures.")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    inventory = load_csv(out_dir / "artifact_inventory.csv")
    if inventory.empty:
        raise FileNotFoundError(f"No artifact inventory found at {out_dir / 'artifact_inventory.csv'}. Run qcvv_extract.py first.")

    rows: list[dict[str, Any]] = []
    for row in inventory.itertuples(index=False):
        pred_path = resolve_path(project_root, getattr(row, "prediction_npz", None))
        if pred_path is None:
            continue
        try:
            rows.extend(compute_drift_for_artifact(row, pred_path, args.n_bins, project_root))
        except Exception as exc:
            rows.append({
                "bundle": getattr(row, "bundle", None),
                "family": getattr(row, "family", None),
                "stage": getattr(row, "stage", None),
                "model": getattr(row, "model", None),
                "source_file": str(pred_path),
                "error": str(exc),
            })

    summary_rows = summarize_drift(rows)
    pd.DataFrame(rows).to_csv(out_dir / "drift_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "drift_summary.csv", index=False)
    figures = [] if args.skip_figures else plot_drift(out_dir, pd.DataFrame(rows))

    summary = {
        "n_drift_rows": len(rows),
        "n_drift_summary_rows": len(summary_rows),
        "n_bins_requested": args.n_bins,
        "figures": figures,
        "interpretation": "Large bin-to-bin ranges indicate possible drift/nonstationarity and should be investigated before official QCVV claims.",
    }
    with (out_dir / "qcvv_drift_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote drift outputs to: {out_dir}")
    print(f"Drift rows: {len(rows)} | Summary rows: {len(summary_rows)} | Figures: {len(figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
