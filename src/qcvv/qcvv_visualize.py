import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DPI = 180


def ensure_fig_dir(out_dir: Path) -> Path:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def safe_float(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def label_series(df: pd.DataFrame, fields: list[str]) -> list[str]:
    labels: list[str] = []
    for _, r in df.iterrows():
        parts = [str(r[f]) for f in fields if f in df.columns]
        labels.append("\n".join(parts))
    return labels


def plot_cthmm_lifetimes(df: pd.DataFrame, fig_dir: Path) -> list[str]:
    outputs: list[str] = []
    if df.empty or "tau_mean_model_units" not in df:
        return outputs
    pdf = df[df["tau_mean_model_units"].notna()].copy()
    if pdf.empty:
        return outputs
    pdf = pdf.sort_values(["family", "bundle", "model"])
    labels = label_series(pdf, ["family", "bundle"])
    x = np.arange(len(pdf))
    plt.figure(figsize=(max(8, len(pdf) * 1.05), 4.8))
    plt.bar(x - 0.18, pdf["tau_0_model_units"].astype(float), width=0.36, label="tau_0")
    plt.bar(x + 0.18, pdf["tau_1_model_units"].astype(float), width=0.36, label="tau_1")
    plt.yscale("log")
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Effective lifetime, model units")
    plt.title("CT-HMM effective lifetimes: pipeline vs individual")
    plt.legend()
    path = fig_dir / "cthmm_lifetimes_pipeline_vs_individual.png"
    savefig(path)
    outputs.append(str(path))

    if "tau_mean_physical" in pdf and pdf["tau_mean_physical"].notna().any():
        phys = pdf[pdf["tau_mean_physical"].notna()].copy()
        labels = label_series(phys, ["family", "bundle"])
        x = np.arange(len(phys))
        plt.figure(figsize=(max(8, len(phys) * 1.05), 4.8))
        plt.bar(x - 0.18, phys["tau_0_physical"].astype(float), width=0.36, label="tau_0")
        plt.bar(x + 0.18, phys["tau_1_physical"].astype(float), width=0.36, label="tau_1")
        plt.yscale("log")
        plt.xticks(x, labels, rotation=0)
        plt.ylabel("Effective lifetime, physical units")
        plt.title("CT-HMM effective lifetimes with supplied dt")
        plt.legend()
        path = fig_dir / "cthmm_lifetimes_physical_units_pipeline_vs_individual.png"
        savefig(path)
        outputs.append(str(path))

    pivot = pdf.pivot_table(index="family", columns="bundle", values="tau_mean_model_units", aggfunc="mean")
    if {"pipeline", "individual"}.issubset(set(pivot.columns)):
        ratio = pivot["pipeline"] / pivot["individual"]
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        if not ratio.empty:
            plt.figure(figsize=(7, 4.2))
            x = np.arange(len(ratio))
            plt.bar(x, ratio.values.astype(float))
            plt.axhline(1.0, linestyle="--", linewidth=1)
            plt.xticks(x, list(ratio.index))
            plt.ylabel("pipeline tau_mean / individual tau_mean")
            plt.title("CT-HMM lifetime stability across bundles")
            path = fig_dir / "cthmm_lifetime_bundle_ratio.png"
            savefig(path)
            outputs.append(str(path))
    return outputs


def plot_prediction_confidence(df: pd.DataFrame, fig_dir: Path) -> list[str]:
    outputs: list[str] = []
    if df.empty or "mean_confidence" not in df:
        return outputs
    pdf = df[df["mean_confidence"].notna()].copy()
    if pdf.empty:
        return outputs
    for family, g in pdf.groupby("family"):
        g = g.sort_values(["bundle", "model"])
        x = np.arange(len(g))
        labels = [f"{r.bundle}\n{r.model}" for r in g.itertuples()]
        plt.figure(figsize=(max(8, len(g) * 1.15), 4.3))
        plt.bar(x, g["mean_confidence"].astype(float))
        plt.ylim(0.0, 1.05)
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylabel("Mean decoder confidence")
        plt.title(f"{family}: prediction confidence across bundles")
        path = fig_dir / f"{family}_prediction_confidence_all_bundles.png"
        savefig(path)
        outputs.append(str(path))
    return outputs


def plot_state_occupancy(df: pd.DataFrame, fig_dir: Path) -> list[str]:
    outputs: list[str] = []
    if df.empty:
        return outputs
    occ_cols = [c for c in df.columns if c.startswith("state_") and c.endswith("_occupancy")]
    if not occ_cols:
        return outputs
    for family, g in df.groupby("family"):
        g = g.sort_values(["bundle", "model"])
        x = np.arange(len(g))
        bottom = np.zeros(len(g), dtype=float)
        plt.figure(figsize=(max(8, len(g) * 1.15), 4.3))
        for col in sorted(occ_cols):
            vals = pd.to_numeric(g[col], errors="coerce").fillna(0.0).to_numpy(float)
            plt.bar(x, vals, bottom=bottom, label=col.replace("_occupancy", ""))
            bottom += vals
        plt.ylim(0.0, 1.05)
        plt.xticks(x, [f"{r.bundle}\n{r.model}" for r in g.itertuples()], rotation=25, ha="right")
        plt.ylabel("Occupancy fraction")
        plt.title(f"{family}: inferred hidden-state occupancy across bundles")
        plt.legend()
        path = fig_dir / f"{family}_state_occupancy_all_bundles.png"
        savefig(path)
        outputs.append(str(path))
    return outputs


def plot_agreements(df: pd.DataFrame, fig_dir: Path) -> list[str]:
    outputs: list[str] = []
    if df.empty or "hard_state_agreement" not in df:
        return outputs
    for (bundle, family), g in df.groupby(["bundle", "family"]):
        models = sorted(set(g["left_model"]).union(set(g["right_model"])))
        if not models:
            continue
        mat = np.eye(len(models), dtype=float)
        index = {m: i for i, m in enumerate(models)}
        for row in g.itertuples(index=False):
            i = index[getattr(row, "left_model")]
            j = index[getattr(row, "right_model")]
            val = safe_float(getattr(row, "hard_state_agreement"))
            mat[i, j] = val
            mat[j, i] = val
        plt.figure(figsize=(max(6, len(models) * 1.1), max(5, len(models) * 0.9)))
        im = plt.imshow(mat, vmin=0.0, vmax=1.0)
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Hard-state agreement")
        plt.xticks(np.arange(len(models)), models, rotation=35, ha="right")
        plt.yticks(np.arange(len(models)), models)
        for i in range(len(models)):
            for j in range(len(models)):
                text = "" if np.isnan(mat[i, j]) else f"{mat[i, j]:.2f}"
                plt.text(j, i, text, ha="center", va="center")
        plt.title(f"{family}: {bundle} model agreement")
        path = fig_dir / f"{family}_{bundle}_model_agreement_heatmap.png"
        savefig(path)
        outputs.append(str(path))
    return outputs


def plot_cross_bundle_agreement(df: pd.DataFrame, fig_dir: Path) -> list[str]:
    outputs: list[str] = []
    if df.empty or "hard_state_agreement" not in df:
        return outputs
    g = df.copy().sort_values(["family", "left_model"])
    if g.empty:
        return outputs
    labels = [f"{r.family}\n{r.left_model}" for r in g.itertuples()]
    x = np.arange(len(g))
    plt.figure(figsize=(max(8, len(g) * 1.2), 4.3))
    plt.bar(x, g["hard_state_agreement"].astype(float))
    plt.ylim(0.0, 1.05)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Hard-state agreement")
    plt.title("Cross-bundle same-model agreement: pipeline vs individual")
    path = fig_dir / "cross_bundle_same_model_agreement.png"
    savefig(path)
    outputs.append(str(path))
    return outputs


def plot_dwell(df: pd.DataFrame, fig_dir: Path) -> list[str]:
    outputs: list[str] = []
    if df.empty or "mean_dwell_windows" not in df:
        return outputs
    for family, g in df.groupby("family"):
        g = g.sort_values(["bundle", "model", "state"])
        labels = [f"{r.bundle}\n{r.model}\ns{int(r.state)}" for r in g.itertuples()]
        x = np.arange(len(g))
        plt.figure(figsize=(max(8, len(g) * 0.85), 4.3))
        plt.bar(x, g["mean_dwell_windows"].astype(float))
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel("Mean dwell length, prediction windows")
        plt.title(f"{family}: dwell segments from prediction-window states")
        path = fig_dir / f"{family}_dwell_windows_all_bundles.png"
        savefig(path)
        outputs.append(str(path))
    return outputs


def plot_scorecard(df: pd.DataFrame, fig_dir: Path) -> list[str]:
    outputs: list[str] = []
    if df.empty or "qcvv_model_readiness_score_0_to_100" not in df:
        return outputs
    g = df.sort_values(["family", "bundle"])
    labels = [f"{r.family}\n{r.bundle}" for r in g.itertuples()]
    x = np.arange(len(g))
    plt.figure(figsize=(max(8, len(g) * 1.0), 4.4))
    plt.bar(x, g["qcvv_model_readiness_score_0_to_100"].astype(float))
    plt.ylim(0, 100)
    plt.xticks(x, labels)
    plt.ylabel("Readiness score, artifact/model only")
    plt.title("Model-assisted QCVV readiness score")
    path = fig_dir / "qcvv_model_readiness_scorecard.png"
    savefig(path)
    outputs.append(str(path))
    return outputs


def plot_bundle_comparison(df: pd.DataFrame, fig_dir: Path) -> list[str]:
    outputs: list[str] = []
    if df.empty or "ratio_pipeline_over_individual" not in df:
        return outputs
    key_metrics = [
        "tau_mean_model_units",
        "mean_confidence",
        "mean_entropy_bits",
        "state_0_occupancy",
        "state_1_occupancy",
    ]
    g = df[df["metric"].isin(key_metrics)].copy()
    g = g[pd.to_numeric(g["ratio_pipeline_over_individual"], errors="coerce").notna()]
    if g.empty:
        return outputs
    for metric, m in g.groupby("metric"):
        m = m.sort_values(["family", "model"])
        labels = [f"{r.family}\n{r.model}" for r in m.itertuples()]
        x = np.arange(len(m))
        plt.figure(figsize=(max(8, len(m) * 1.1), 4.2))
        plt.bar(x, m["ratio_pipeline_over_individual"].astype(float))
        plt.axhline(1.0, linestyle="--", linewidth=1)
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylabel("pipeline / individual")
        plt.title(f"Bundle comparison ratio: {metric}")
        path = fig_dir / f"bundle_ratio_{metric}.png"
        savefig(path)
        outputs.append(str(path))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create comprehensive QCVV plots from extracted outputs.")
    parser.add_argument("--out-dir", default="qcvv_outputs", help="Directory containing qcvv_extract outputs.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    fig_dir = ensure_fig_dir(out_dir)
    outputs: list[str] = []
    outputs += plot_cthmm_lifetimes(load_csv(out_dir / "cthmm_lifetimes.csv"), fig_dir)
    pred = load_csv(out_dir / "prediction_metrics.csv")
    outputs += plot_prediction_confidence(pred, fig_dir)
    outputs += plot_state_occupancy(pred, fig_dir)
    outputs += plot_agreements(load_csv(out_dir / "model_agreements.csv"), fig_dir)
    outputs += plot_cross_bundle_agreement(load_csv(out_dir / "cross_bundle_model_agreements.csv"), fig_dir)
    outputs += plot_dwell(load_csv(out_dir / "dwell_metrics.csv"), fig_dir)
    outputs += plot_scorecard(load_csv(out_dir / "qcvv_scorecard.csv"), fig_dir)
    outputs += plot_bundle_comparison(load_csv(out_dir / "bundle_comparison.csv"), fig_dir)

    manifest = out_dir / "figures_manifest.txt"
    manifest.write_text("\n".join(outputs) + ("\n" if outputs else ""), encoding="utf-8")
    print(f"Wrote {len(outputs)} figures to: {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
