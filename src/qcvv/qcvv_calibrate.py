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


def normalize_label(x: Any) -> str:
    s = str(x).strip().lower()
    aliases = {
        "0": "0", "state0": "0", "state_0": "0", "even": "0", "false": "0",
        "1": "1", "state1": "1", "state_1": "1", "odd": "1", "true": "1",
    }
    return aliases.get(s, s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute calibration readout QCVV metrics from prediction NPZ + physical labels CSV.")
    parser.add_argument("--prediction-npz", required=True, help="Prediction .npz containing sample_index, state_probs/hard_state/confidence.")
    parser.add_argument("--labels-csv", required=True, help="CSV with sample_index and physical-state label column.")
    parser.add_argument("--label-column", default="physical_state", help="Column in labels CSV containing the known physical state.")
    parser.add_argument("--sample-column", default="sample_index", help="Column in labels CSV used for alignment.")
    parser.add_argument("--out-dir", default="qcvv_outputs/calibration", help="Output directory.")
    parser.add_argument("--state-map-json", default=None, help="Optional mapping from predicted hidden state to physical label, e.g. {'0':'even','1':'odd'}. If omitted, labels are normalized with even/0 -> 0 and odd/1 -> 1.")
    parser.add_argument("--positive-label", default="1", help="Positive state label for confidence/reliability summaries. Example: odd")
    args = parser.parse_args(argv)

    pred_path = Path(args.prediction_npz).resolve()
    labels_path = Path(args.labels_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = np.load(pred_path, allow_pickle=True)
    if "sample_index" not in pred.files or "hard_state" not in pred.files:
        raise ValueError("Prediction NPZ must contain sample_index and hard_state")
    sample_index = np.asarray(pred["sample_index"], dtype=np.int64).ravel()
    hard_state = np.asarray(pred["hard_state"], dtype=np.int64).ravel()
    state_probs = np.asarray(pred["state_probs"], dtype=float) if "state_probs" in pred.files else None
    confidence = np.asarray(pred["confidence"], dtype=float).ravel() if "confidence" in pred.files else None
    if confidence is None and state_probs is not None:
        confidence = state_probs.max(axis=1)

    labels_df = pd.read_csv(labels_path)
    if args.sample_column not in labels_df or args.label_column not in labels_df:
        raise ValueError(f"labels CSV must contain {args.sample_column!r} and {args.label_column!r}")

    state_map = {}
    if args.state_map_json:
        state_map = json.loads(Path(args.state_map_json).read_text(encoding="utf-8"))

    pred_df = pd.DataFrame({"sample_index": sample_index, "pred_state_raw": hard_state})
    if confidence is not None:
        pred_df["confidence"] = confidence
    if state_probs is not None and state_probs.ndim == 2:
        for k in range(state_probs.shape[1]):
            pred_df[f"prob_state_{k}"] = state_probs[:, k]

    def mapped_pred(s: int) -> str:
        return normalize_label(state_map.get(str(int(s)), state_map.get(int(s), str(int(s)))))

    pred_df["pred_state"] = [mapped_pred(s) for s in hard_state]
    lab = labels_df[[args.sample_column, args.label_column]].copy()
    lab.columns = ["sample_index", "physical_state_raw"]
    lab["physical_state"] = lab["physical_state_raw"].map(normalize_label)
    merged = pred_df.merge(lab, on="sample_index", how="inner")
    if merged.empty:
        raise ValueError("No overlapping sample_index values between predictions and labels")

    labels = sorted(set(merged["physical_state"]).union(set(merged["pred_state"])))
    cm = pd.crosstab(merged["physical_state"], merged["pred_state"], dropna=False).reindex(index=labels, columns=labels, fill_value=0)
    cm_norm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
    accuracy = float((merged["physical_state"] == merged["pred_state"]).mean())
    per_state_error = {state: float(1.0 - cm_norm.loc[state, state]) if state in cm_norm.index and state in cm_norm.columns and not math.isnan(cm_norm.loc[state, state]) else math.nan for state in labels}
    readout_fidelity_balanced = float(1.0 - np.nanmean(list(per_state_error.values()))) if per_state_error else math.nan

    metrics = {
        "prediction_npz": str(pred_path),
        "labels_csv": str(labels_path),
        "n_aligned": int(len(merged)),
        "accuracy": accuracy,
        "balanced_readout_fidelity": readout_fidelity_balanced,
        "per_state_error": per_state_error,
        "labels": labels,
    }
    (out_dir / "calibration_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    merged.to_csv(out_dir / "calibrated_predictions.csv", index=False)
    cm.to_csv(out_dir / "confusion_counts.csv")
    cm_norm.to_csv(out_dir / "confusion_normalized.csv")

    plt.figure(figsize=(5.5, 4.5))
    im = plt.imshow(cm_norm.fillna(0).to_numpy(float), vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="P(predicted | prepared)")
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted state")
    plt.ylabel("Prepared/known state")
    plt.title("Calibration confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{cm_norm.iloc[i, j]:.2f}" if not math.isnan(cm_norm.iloc[i, j]) else "", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close()

    if "confidence" in merged:
        bins = np.linspace(0.0, 1.0, 11)
        merged["confidence_bin"] = pd.cut(merged["confidence"], bins=bins, include_lowest=True)
        reliability = merged.groupby("confidence_bin", observed=True).apply(lambda g: pd.Series({
            "n": len(g),
            "mean_confidence": g["confidence"].mean(),
            "empirical_accuracy": (g["physical_state"] == g["pred_state"]).mean(),
        })).reset_index()
        reliability.to_csv(out_dir / "confidence_reliability.csv", index=False)
        plt.figure(figsize=(5.5, 4.5))
        plt.plot([0, 1], [0, 1], linestyle="--", label="ideal")
        plt.plot(reliability["mean_confidence"], reliability["empirical_accuracy"], marker="o", label="model")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Mean confidence")
        plt.ylabel("Empirical accuracy")
        plt.title("Confidence calibration")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "confidence_calibration.png", dpi=180, bbox_inches="tight")
        plt.close()

    print(f"Wrote calibration QCVV metrics to: {out_dir}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
