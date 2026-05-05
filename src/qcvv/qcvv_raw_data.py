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


def load_npz(path: Path) -> pd.DataFrame:
    data = np.load(path, allow_pickle=True)
    n = None
    for key in ["sample_index", "hard_state", "confidence", "sequence_id"]:
        if key in data:
            n = len(data[key])
            break
    if n is None and "state_probs" in data:
        n = len(data["state_probs"])
    if n is None:
        raise ValueError("Prediction NPZ does not contain recognizable prediction arrays.")
    out: dict[str, Any] = {}
    if "sample_index" in data:
        out["sample_index"] = np.asarray(data["sample_index"]).reshape(-1)
    else:
        out["sample_index"] = np.arange(n)
    if "sequence_id" in data:
        out["sequence_id"] = np.asarray(data["sequence_id"]).reshape(-1)
    if "hard_state" in data:
        out["hard_state"] = np.asarray(data["hard_state"]).astype(int).reshape(-1)
    elif "state_probs" in data:
        out["hard_state"] = np.argmax(np.asarray(data["state_probs"]), axis=1).astype(int)
    if "confidence" in data:
        out["confidence"] = np.asarray(data["confidence"], dtype=float).reshape(-1)
    elif "state_probs" in data:
        out["confidence"] = np.max(np.asarray(data["state_probs"], dtype=float), axis=1)
    if "state_probs" in data:
        probs = np.asarray(data["state_probs"], dtype=float)
        for k in range(probs.shape[1]):
            out[f"prob_state_{k}"] = probs[:, k]
    return pd.DataFrame(out)


def choose_signal_columns(df: pd.DataFrame) -> list[str]:
    candidates = ["I", "Q", "value", "signal", "quantum_capacitance", "cq", "voltage", "amplitude"]
    return [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def time_column(df: pd.DataFrame) -> str:
    for col in ["time_seconds", "t", "time", "sample_index"]:
        if col in df.columns:
            return col
    return "sample_index"


def align(raw: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["sample_index"]
    if "sequence_id" in raw.columns and "sequence_id" in pred.columns:
        join_cols = ["sequence_id", "sample_index"]
    merged = raw.merge(pred, on=join_cols, how="left", suffixes=("", "_pred"))
    return merged


def summary_metrics(merged: pd.DataFrame, signal_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "hard_state" not in merged.columns:
        return rows
    for signal in signal_cols:
        for state, group in merged.groupby("hard_state", dropna=True):
            vals = pd.to_numeric(group[signal], errors="coerce").dropna()
            if vals.empty:
                continue
            rows.append({
                "signal": signal,
                "hard_state": int(state),
                "n": int(vals.size),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "median": float(vals.median()),
                "p05": float(vals.quantile(0.05)),
                "p95": float(vals.quantile(0.95)),
            })
    return rows


def plot_overlay(merged: pd.DataFrame, out_dir: Path, family: str, model: str, max_points: int) -> list[str]:
    outputs: list[str] = []
    signal_cols = choose_signal_columns(merged)
    if not signal_cols or "hard_state" not in merged.columns:
        return outputs
    tcol = time_column(merged)
    if tcol not in merged.columns:
        merged = merged.copy()
        merged["sample_index"] = np.arange(len(merged))
        tcol = "sample_index"
    plot_df = merged.dropna(subset=["hard_state"]).copy()
    if len(plot_df) > max_points:
        plot_df = plot_df.iloc[:max_points].copy()
    for signal in signal_cols[:4]:
        plt.figure(figsize=(11, 4.5))
        plt.plot(plot_df[tcol], plot_df[signal], linewidth=0.8, label=signal)
        # draw state as a stepped overlay on a secondary axis
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.step(plot_df[tcol], plot_df["hard_state"], where="post", linewidth=1.0, alpha=0.6, label="hard_state")
        ax.set_xlabel(tcol)
        ax.set_ylabel(signal)
        ax2.set_ylabel("hard_state")
        ax.set_title(f"Raw/prepared trace overlay: {family} {model} {signal}")
        plt.tight_layout()
        path = out_dir / f"raw_overlay_{family}_{model}_{signal}.png"
        plt.savefig(path, dpi=180)
        plt.close()
        outputs.append(str(path))
    if "I" in merged.columns and "Q" in merged.columns:
        plt.figure(figsize=(5.8, 5.2))
        plot_df = merged.dropna(subset=["I", "Q", "hard_state"]).copy()
        if len(plot_df) > max_points:
            plot_df = plot_df.iloc[:max_points].copy()
        scatter = plt.scatter(plot_df["I"], plot_df["Q"], c=plot_df["hard_state"], s=4, alpha=0.6)
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.title(f"I/Q by inferred state: {family} {model}")
        plt.colorbar(scatter, label="hard_state")
        plt.tight_layout()
        path = out_dir / f"iq_by_state_{family}_{model}.png"
        plt.savefig(path, dpi=180)
        plt.close()
        outputs.append(str(path))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Align raw/prepared traces with predictions for QCVV visual checks.")
    parser.add_argument("--raw-csv", required=True, help="Raw/prepared trace CSV.")
    parser.add_argument("--prediction-npz", required=True, help="Prediction NPZ from CT-HMM/CNN/DMM/HSMM export.")
    parser.add_argument("--out-dir", default="qcvv_outputs/raw_trace_checks", help="Output directory.")
    parser.add_argument("--family", default="unknown_family", help="Family label for outputs.")
    parser.add_argument("--model", default="unknown_model", help="Model label for outputs.")
    parser.add_argument("--max-points", type=int, default=20000, help="Max points to plot per figure.")
    args = parser.parse_args(argv)

    raw_path = Path(args.raw_csv).resolve()
    pred_path = Path(args.prediction_npz).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path)
    pred = load_npz(pred_path)
    merged = align(raw, pred)
    merged_path = out_dir / f"aligned_raw_predictions_{args.family}_{args.model}.csv"
    merged.to_csv(merged_path, index=False)

    signal_cols = choose_signal_columns(merged)
    metrics = summary_metrics(merged, signal_cols)
    metrics_path = out_dir / f"raw_state_signal_summary_{args.family}_{args.model}.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    figures = plot_overlay(merged, out_dir, args.family, args.model, args.max_points)
    summary = {
        "raw_csv": str(raw_path),
        "prediction_npz": str(pred_path),
        "aligned_rows": int(len(merged)),
        "prediction_coverage_fraction": float(merged["hard_state"].notna().mean()) if "hard_state" in merged else 0.0,
        "signal_columns": signal_cols,
        "outputs": {"aligned_csv": str(merged_path), "signal_summary_csv": str(metrics_path), "figures": figures},
        "interpretation": "Use these overlays to verify that inferred hidden states align with raw or converted trace structure before making physical QCVV claims.",
    }
    with (out_dir / f"raw_data_qcvv_summary_{args.family}_{args.model}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote raw-data QCVV checks to: {out_dir}")
    print(f"Aligned rows: {len(merged)} | Coverage: {summary['prediction_coverage_fraction']:.3f} | Figures: {len(figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
