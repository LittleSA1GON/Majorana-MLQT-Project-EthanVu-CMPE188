import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


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


def block_slices(n: int, preferred_block_size: int, max_blocks: int) -> list[slice]:
    if n <= 0:
        return []
    block_size = max(1, int(preferred_block_size))
    if max_blocks > 0 and math.ceil(n / block_size) > max_blocks:
        block_size = int(math.ceil(n / max_blocks))
    return [slice(start, min(n, start + block_size)) for start in range(0, n, block_size)]


def practical_units(data: dict[str, np.ndarray], n: int, preferred_block_size: int, max_blocks: int) -> tuple[str, list[np.ndarray | slice]]:
    """Use sequence groups only if they are not too numerous/tiny; else blocks."""
    if "sequence_id" in data and len(data["sequence_id"]) == n:
        seq = np.asarray(data["sequence_id"]).reshape(-1)
        uniq, inv = np.unique(seq, return_inverse=True)
        avg = n / max(len(uniq), 1)
        if 1 < len(uniq) <= max_blocks and avg >= 5:
            units: list[np.ndarray | slice] = [np.flatnonzero(inv == i) for i in range(len(uniq))]
            return "sequence_id", units
    return f"contiguous_blocks", block_slices(n, preferred_block_size, max_blocks)


def take(arr: np.ndarray | None, unit: np.ndarray | slice) -> np.ndarray | None:
    if arr is None:
        return None
    return arr[unit]


def summarize_values(values: list[float], n_bootstrap: int, rng: np.random.Generator) -> dict[str, Any]:
    vals = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return {"n_blocks": 0, "bootstrap_replicates": 0, "ci_low_2p5": None, "bootstrap_median": None, "ci_high_97p5": None, "bootstrap_mean": None, "bootstrap_std": None}
    if vals.size == 1 or n_bootstrap <= 0:
        return {
            "n_blocks": int(vals.size),
            "bootstrap_replicates": 0,
            "ci_low_2p5": float(vals[0]),
            "bootstrap_median": float(vals[0]),
            "ci_high_97p5": float(vals[0]),
            "bootstrap_mean": float(vals[0]),
            "bootstrap_std": 0.0,
        }
    # Bootstrap the mean of block-level metrics. This is much faster than sample-level bootstrap.
    draws = rng.integers(0, vals.size, size=(n_bootstrap, vals.size))
    boot = vals[draws].mean(axis=1)
    return {
        "n_blocks": int(vals.size),
        "bootstrap_replicates": int(n_bootstrap),
        "ci_low_2p5": float(np.percentile(boot, 2.5)),
        "bootstrap_median": float(np.percentile(boot, 50.0)),
        "ci_high_97p5": float(np.percentile(boot, 97.5)),
        "bootstrap_mean": float(np.mean(boot)),
        "bootstrap_std": float(np.std(boot, ddof=1)) if boot.size > 1 else 0.0,
        "block_min": float(np.min(vals)),
        "block_max": float(np.max(vals)),
    }


def block_metrics(hard: np.ndarray, conf: np.ndarray | None, probs: np.ndarray | None, units: list[np.ndarray | slice]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for unit in units:
        h = take(hard, unit)
        if h is None or len(h) == 0:
            continue
        out.setdefault("state_0_occupancy", []).append(float(np.mean(h == 0)))
        out.setdefault("state_1_occupancy", []).append(float(np.mean(h == 1)))
        if len(h) > 1:
            out.setdefault("window_switch_rate", []).append(float(np.mean(h[1:] != h[:-1])))
        c = take(conf, unit)
        if c is not None and len(c):
            out.setdefault("mean_confidence", []).append(float(np.mean(c)))
        p = take(probs, unit)
        if p is not None and len(p):
            out.setdefault("mean_entropy_bits", []).append(float(np.mean(entropy_bits(p))))
    return out


def dwell_lengths(hard: np.ndarray) -> dict[int, list[int]]:
    out = {0: [], 1: []}
    if hard.size == 0:
        return out
    cur = int(hard[0])
    length = 1
    for value in hard[1:]:
        s = int(value)
        if s == cur:
            length += 1
        else:
            out.setdefault(cur, []).append(length)
            cur = s
            length = 1
    out.setdefault(cur, []).append(length)
    return out


def block_dwell_metrics(hard: np.ndarray, units: list[np.ndarray | slice]) -> dict[tuple[int, str], list[float]]:
    out: dict[tuple[int, str], list[float]] = {}
    for unit in units:
        h = take(hard, unit)
        if h is None or len(h) == 0:
            continue
        for state, lengths in dwell_lengths(h).items():
            if not lengths:
                continue
            arr = np.asarray(lengths, dtype=float)
            out.setdefault((state, "mean_dwell_windows"), []).append(float(np.mean(arr)))
            out.setdefault((state, "median_dwell_windows"), []).append(float(np.median(arr)))
            out.setdefault((state, "p90_dwell_windows"), []).append(float(np.percentile(arr, 90)))
    return out


def bootstrap_predictions(project_root: Path, inventory: pd.DataFrame, n_bootstrap: int, seed: int, block_size: int, max_blocks: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    pred_rows: list[dict[str, Any]] = []
    dwell_rows: list[dict[str, Any]] = []
    if inventory.empty or "prediction_npz" not in inventory:
        return pred_rows, dwell_rows
    for row in inventory.itertuples(index=False):
        pred_path = resolve_path(project_root, getattr(row, "prediction_npz", None))
        if pred_path is None:
            continue
        try:
            data = load_npz(pred_path)
        except Exception as exc:
            pred_rows.append({"bundle": getattr(row, "bundle", None), "family": getattr(row, "family", None), "model": getattr(row, "model", None), "metric": "load_error", "error": str(exc)})
            continue
        hard = hard_from_npz(data)
        if hard is None:
            continue
        conf = confidence_from_npz(data)
        probs = np.asarray(data["state_probs"], dtype=float) if "state_probs" in data else None
        unit_kind, units = practical_units(data, len(hard), block_size, max_blocks)
        base = {
            "bundle": getattr(row, "bundle", None),
            "family": getattr(row, "family", None),
            "stage": getattr(row, "stage", None),
            "model": getattr(row, "model", None),
            "source_file": str(pred_path.relative_to(project_root)) if pred_path.is_relative_to(project_root) else str(pred_path),
            "resample_unit": unit_kind,
            "n_predictions": int(len(hard)),
        }
        for metric, values in block_metrics(hard, conf, probs, units).items():
            pred_rows.append({**base, "metric": metric, **summarize_values(values, n_bootstrap, rng)})
        for (state, metric), values in block_dwell_metrics(hard, units).items():
            dwell_rows.append({**base, "state": state, "metric": metric, **summarize_values(values, n_bootstrap, rng)})
    return pred_rows, dwell_rows


def align_hard(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    lh = hard_from_npz(left)
    rh = hard_from_npz(right)
    if lh is None or rh is None:
        return None
    if len(lh) == len(rh):
        if "sample_index" in left and "sample_index" in right:
            li = np.asarray(left["sample_index"]).reshape(-1)
            ri = np.asarray(right["sample_index"]).reshape(-1)
            if len(li) == len(ri) and np.array_equal(li, ri):
                return lh, rh
        elif "sample_index" not in left or "sample_index" not in right:
            return lh, rh
    if "sample_index" in left and "sample_index" in right:
        li = np.asarray(left["sample_index"]).reshape(-1)
        ri = np.asarray(right["sample_index"]).reshape(-1)
        common, lpos, rpos = np.intersect1d(li, ri, return_indices=True)
        if len(common) > 1:
            return lh[lpos], rh[rpos]
    n = min(len(lh), len(rh))
    return lh[:n], rh[:n] if n > 1 else None


def bootstrap_agreements(project_root: Path, inventory: pd.DataFrame, n_bootstrap: int, seed: int, block_size: int, max_blocks: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + 17)
    rows: list[dict[str, Any]] = []
    items = []
    for row in inventory.itertuples(index=False):
        pred_path = resolve_path(project_root, getattr(row, "prediction_npz", None))
        if pred_path is not None:
            items.append((row, pred_path))
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            left_row, left_path = items[i]
            right_row, right_path = items[j]
            if getattr(left_row, "bundle", None) != getattr(right_row, "bundle", None):
                continue
            if getattr(left_row, "family", None) != getattr(right_row, "family", None):
                continue
            try:
                aligned = align_hard(load_npz(left_path), load_npz(right_path))
            except Exception:
                continue
            if aligned is None:
                continue
            lh, rh = aligned
            if len(lh) < 2:
                continue
            units = block_slices(len(lh), block_size, max_blocks)
            values = [float(np.mean(lh[u] == rh[u])) for u in units if len(lh[u]) > 0]
            rows.append({
                "bundle": getattr(left_row, "bundle", None),
                "family": getattr(left_row, "family", None),
                "left_model": getattr(left_row, "model", None),
                "left_stage": getattr(left_row, "stage", None),
                "right_model": getattr(right_row, "model", None),
                "right_stage": getattr(right_row, "stage", None),
                "n_aligned": int(len(lh)),
                "resample_unit": "contiguous_blocks",
                "point_hard_state_agreement": float(np.mean(lh == rh)),
                **summarize_values(values, n_bootstrap, rng),
            })
    return rows


def lifetime_uncertainty_status(out_dir: Path) -> list[dict[str, Any]]:
    cthmm = load_csv(out_dir / "cthmm_lifetimes.csv")
    rows: list[dict[str, Any]] = []
    for row in cthmm.itertuples(index=False):
        rows.append({
            "bundle": getattr(row, "bundle", None),
            "family": getattr(row, "family", None),
            "model": getattr(row, "model", None),
            "tau_0_model_units": getattr(row, "tau_0_model_units", None),
            "tau_1_model_units": getattr(row, "tau_1_model_units", None),
            "uncertainty_status": "point_estimate_only_from_current_cthmm_summary",
            "official_qcvv_note": "For official lifetime CIs, rerun CT-HMM with parameter covariance, likelihood profiling, posterior samples, or raw-sequence bootstrap.",
        })
    return rows


def plot_uncertainty(out_dir: Path, pred_rows: list[dict[str, Any]], agree_rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame(pred_rows)
    if not pred.empty:
        subset = pred[pred["metric"].isin(["mean_confidence", "state_0_occupancy", "window_switch_rate"])]
        for metric, group in subset.groupby("metric"):
            group = group.dropna(subset=["bootstrap_median", "ci_low_2p5", "ci_high_97p5"])
            if group.empty:
                continue
            labels = [f"{r.family}\n{r.bundle}\n{r.model}" for r in group.itertuples()]
            x = np.arange(len(group))
            y = group["bootstrap_median"].astype(float).to_numpy()
            lo = y - group["ci_low_2p5"].astype(float).to_numpy()
            hi = group["ci_high_97p5"].astype(float).to_numpy() - y
            plt.figure(figsize=(max(8, len(group) * 0.75), 4.5))
            plt.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt="o", capsize=3)
            plt.xticks(x, labels, rotation=45, ha="right")
            plt.ylabel(metric)
            plt.title(f"Block-bootstrap uncertainty: {metric}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"bootstrap_{metric}.png", dpi=180)
            plt.close()
    agree = pd.DataFrame(agree_rows)
    if not agree.empty:
        agree = agree.dropna(subset=["bootstrap_median", "ci_low_2p5", "ci_high_97p5"])
        if not agree.empty:
            labels = [f"{r.family}\n{r.bundle}\n{r.left_model}-{r.right_model}" for r in agree.itertuples()]
            x = np.arange(len(agree))
            y = agree["bootstrap_median"].astype(float).to_numpy()
            lo = y - agree["ci_low_2p5"].astype(float).to_numpy()
            hi = agree["ci_high_97p5"].astype(float).to_numpy() - y
            plt.figure(figsize=(max(9, len(agree) * 0.5), 4.8))
            plt.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt="o", capsize=3)
            plt.ylim(0, 1.02)
            plt.xticks(x, labels, rotation=55, ha="right")
            plt.ylabel("hard-state agreement")
            plt.title("Block-bootstrap uncertainty: model agreement")
            plt.tight_layout()
            plt.savefig(fig_dir / "bootstrap_model_agreement.png", dpi=180)
            plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fast bootstrap uncertainty for QCVV model-derived metrics.")
    parser.add_argument("--project-root", default=".", help="Project root containing artifact paths from artifact_inventory.csv.")
    parser.add_argument("--out-dir", default="qcvv_outputs", help="QCVV output directory from qcvv_extract.py.")
    parser.add_argument("--n-bootstrap", type=int, default=200, help="Bootstrap replicates over block-level metrics.")
    parser.add_argument("--seed", type=int, default=188, help="Random seed.")
    parser.add_argument("--block-size", type=int, default=2048, help="Preferred contiguous block size.")
    parser.add_argument("--max-blocks", type=int, default=1000, help="Maximum number of blocks to keep runtime bounded.")
    parser.add_argument("--skip-figures", action="store_true", help="Do not create uncertainty plots.")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    inventory = load_csv(out_dir / "artifact_inventory.csv")
    if inventory.empty:
        raise FileNotFoundError(f"No artifact inventory found at {out_dir / 'artifact_inventory.csv'}. Run qcvv_extract.py first.")

    pred_rows, dwell_rows = bootstrap_predictions(project_root, inventory, args.n_bootstrap, args.seed, args.block_size, args.max_blocks)
    agree_rows = bootstrap_agreements(project_root, inventory, args.n_bootstrap, args.seed, args.block_size, args.max_blocks)
    lifetime_rows = lifetime_uncertainty_status(out_dir)

    write_csv(out_dir / "bootstrap_prediction_metrics.csv", pred_rows)
    write_csv(out_dir / "bootstrap_dwell_metrics.csv", dwell_rows)
    write_csv(out_dir / "bootstrap_model_agreements.csv", agree_rows)
    write_csv(out_dir / "cthmm_lifetime_uncertainty_status.csv", lifetime_rows)
    if not args.skip_figures:
        plot_uncertainty(out_dir, pred_rows, agree_rows)

    summary = {
        "n_bootstrap": args.n_bootstrap,
        "block_size": args.block_size,
        "max_blocks": args.max_blocks,
        "n_prediction_ci_rows": len(pred_rows),
        "n_dwell_ci_rows": len(dwell_rows),
        "n_agreement_ci_rows": len(agree_rows),
        "interpretation": "Block-bootstrap intervals quantify model-summary uncertainty/variation only; physical QCVV requires calibration and direct data.",
    }
    with (out_dir / "qcvv_uncertainty_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote QCVV uncertainty outputs to: {out_dir}")
    print(f"Prediction CI rows: {len(pred_rows)} | Agreement CI rows: {len(agree_rows)} | Dwell CI rows: {len(dwell_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
