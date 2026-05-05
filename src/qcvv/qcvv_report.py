import argparse
import json
import math
from pathlib import Path

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def fmt(x, digits: int = 6) -> str:
    try:
        v = float(x)
        if not math.isfinite(v):
            return "N/A"
        if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0):
            return f"{v:.{digits}e}"
        return f"{v:.{digits}g}"
    except Exception:
        return "N/A" if x is None else str(x)


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows available._\n"
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return "_No requested columns available._\n"
    t = df[cols].copy()
    if max_rows is not None:
        t = t.head(max_rows)
    for col in t.columns:
        if pd.api.types.is_float_dtype(t[col]) or pd.api.types.is_integer_dtype(t[col]):
            t[col] = t[col].map(fmt)
    return t.to_markdown(index=False) + "\n"


def add_figures(lines: list[str], out_dir: Path, names: list[str]) -> None:
    fig_dir = out_dir / "figures"
    found = False
    for name in names:
        path = fig_dir / name
        if path.exists():
            found = True
            rel = path.relative_to(out_dir).as_posix()
            lines.append(f"![{name}]({rel})\n")
    if not found:
        lines.append("_No matching figures were generated._\n")


def best_rows(df: pd.DataFrame, group_cols: list[str], sort_col: str, ascending: bool = False) -> pd.DataFrame:
    if df.empty or sort_col not in df.columns:
        return pd.DataFrame()
    g = df.copy()
    g[sort_col] = pd.to_numeric(g[sort_col], errors="coerce")
    g = g[g[sort_col].notna()]
    if g.empty:
        return pd.DataFrame()
    return g.sort_values(sort_col, ascending=ascending).groupby(group_cols, as_index=False).head(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a comprehensive model-assisted QCVV Markdown report.")
    parser.add_argument("--out-dir", default="qcvv_outputs", help="Directory containing qcvv_extract/qcvv_visualize outputs.")
    parser.add_argument("--title", default="Comprehensive Model-assisted QCVV Report", help="Report title.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    cthmm = load_csv(out_dir / "cthmm_lifetimes.csv")
    pred = load_csv(out_dir / "prediction_metrics.csv")
    agree = load_csv(out_dir / "model_agreements.csv")
    cross = load_csv(out_dir / "cross_bundle_model_agreements.csv")
    dwell = load_csv(out_dir / "dwell_metrics.csv")
    training = load_csv(out_dir / "training_metrics.csv")
    bundle_comp = load_csv(out_dir / "bundle_comparison.csv")
    family_summary = load_csv(out_dir / "family_summary.csv")
    scorecard = load_csv(out_dir / "qcvv_scorecard.csv")
    interp = load_csv(out_dir / "interpretation_notes.csv")
    direct = load_csv(out_dir / "direct_data_needed.csv")
    warnings = load_csv(out_dir / "extraction_warnings.csv")
    artifact_inventory = load_csv(out_dir / "artifact_inventory.csv")
    summary_path = out_dir / "qcvv_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    lines: list[str] = []
    lines.append(f"# {args.title}\n")
    lines.append("This report summarizes both the **pipeline** and **individual** trained artifacts. It treats the trained CT-HMM/CNN-GRU/HSMM/DMM outputs as model-assisted QCVV estimators for hidden parity/loop states, readout confidence, state-transition behavior, dwell diagnostics, and bundle-to-bundle robustness.\n")

    lines.append("## Executive interpretation\n")
    lines.append("- **Pipeline** is the connected QCVV workflow: CT-HMM teacher, neural decoders, HSMM duration check, DMM sequence model, and exports.\n")
    lines.append("- **Individual** is the independent baseline/ablation bundle: useful for checking whether the pipeline values are stable and whether teacher assistance improves downstream models.\n")
    lines.append("- The values here are **model-derived QCVV characterization metrics**, not physical certification. Physical QCVV needs raw data, physical timing, calibration labels, and repeated-readout experiments.\n")

    if not scorecard.empty:
        lines.append("## QCVV model-readiness scorecard\n")
        lines.append("This score combines artifact completeness, CT-HMM lifetime availability, prediction-model availability, within-bundle model agreement, and prediction confidence. It is **not** readout fidelity.\n")
        lines.append(md_table(scorecard.sort_values(["family", "bundle"]), [
            "bundle", "family", "qcvv_model_readiness_score_0_to_100", "interpretation",
            "has_cthmm_lifetimes", "has_multiple_prediction_models", "mean_within_bundle_agreement",
            "best_model_confidence", "n_prediction_models"
        ]))
        add_figures(lines, out_dir, ["qcvv_model_readiness_scorecard.png"])

    lines.append("## Artifact inventory\n")
    lines.append(md_table(artifact_inventory.sort_values(["bundle", "family", "stage"]) if not artifact_inventory.empty else artifact_inventory, [
        "bundle", "family", "stage", "model", "prediction_npz", "qcvv_summary_json", "summary_json", "metrics_json", "metrics_csv", "has_checkpoint"
    ], max_rows=80))

    lines.append("## CT-HMM rates and effective lifetimes\n")
    lines.append("CT-HMM lifetime values are the main interpretable lifetime extraction. They are in model units unless `--dt-json` was supplied.\n")
    lines.append(md_table(cthmm.sort_values(["family", "bundle", "model"]) if not cthmm.empty else cthmm, [
        "bundle", "family", "model", "gamma_01", "gamma_10", "tau_0_model_units", "tau_1_model_units",
        "tau_mean_model_units", "tau_asymmetry_tau0_over_tau1", "tau_mean_physical", "train_log_likelihood_per_obs"
    ]))
    add_figures(lines, out_dir, ["cthmm_lifetimes_pipeline_vs_individual.png", "cthmm_lifetime_bundle_ratio.png"])

    if not family_summary.empty:
        lines.append("## Family-level summary\n")
        lines.append(md_table(family_summary.sort_values(["family", "bundle"]), [
            "bundle", "family", "n_cthmm_lifetime_rows", "n_prediction_models", "mean_prediction_confidence",
            "mean_prediction_entropy_bits", "mean_within_bundle_agreement", "mean_cross_bundle_same_model_agreement",
            "cthmm_tau_mean_model_units"
        ]))

    # Tetron X/Z ratio table.
    if not cthmm.empty and {"x_loop", "z_loop"}.issubset(set(cthmm.get("family", []))):
        ratio_rows: list[dict[str, object]] = []
        for bundle in sorted(cthmm["bundle"].dropna().unique()):
            sub = cthmm[(cthmm["bundle"] == bundle) & (cthmm["model"] == "cthmm_teacher")]
            xv = sub[sub["family"] == "x_loop"]
            zv = sub[sub["family"] == "z_loop"]
            if not xv.empty and not zv.empty:
                x_tau = float(xv.iloc[0]["tau_mean_model_units"])
                z_tau = float(zv.iloc[0]["tau_mean_model_units"])
                ratio_rows.append({"bundle": bundle, "x_loop_tau_mean": x_tau, "z_loop_tau_mean": z_tau, "z_over_x_tau_ratio": z_tau / x_tau if x_tau else math.nan})
        if ratio_rows:
            lines.append("## Tetron X/Z lifetime comparison\n")
            lines.append("This compares the CT-HMM effective lifetime scale for `z_loop` vs `x_loop`. It is in model units unless physical dt is supplied.\n")
            lines.append(md_table(pd.DataFrame(ratio_rows), ["bundle", "x_loop_tau_mean", "z_loop_tau_mean", "z_over_x_tau_ratio"]))

    lines.append("## Prediction metrics\n")
    lines.append("These metrics summarize decoder probability/confidence and inferred hidden-state occupancy. They are useful for model diagnostics, but they are not assignment fidelity without physical labels.\n")
    lines.append(md_table(pred.sort_values(["family", "bundle", "model"]) if not pred.empty else pred, [
        "bundle", "family", "model", "n_predictions", "mean_confidence", "median_confidence", "mean_entropy_bits",
        "state_0_occupancy", "state_1_occupancy", "window_switch_rate", "n_sequences"
    ], max_rows=80))
    for fam in ["parity", "x_loop", "z_loop"]:
        add_figures(lines, out_dir, [f"{fam}_prediction_confidence_all_bundles.png", f"{fam}_state_occupancy_all_bundles.png"])

    best_conf = best_rows(pred, ["bundle", "family"], "mean_confidence", ascending=False)
    if not best_conf.empty:
        lines.append("## Highest-confidence prediction model by bundle/family\n")
        lines.append(md_table(best_conf.sort_values(["family", "bundle"]), [
            "bundle", "family", "model", "mean_confidence", "mean_entropy_bits", "state_0_occupancy", "state_1_occupancy", "source_file"
        ]))

    lines.append("## Within-bundle model agreement\n")
    lines.append("Agreement compares hard hidden-state calls between models in the same bundle/family. High agreement indicates model consistency, not necessarily physical correctness.\n")
    lines.append(md_table(agree.sort_values(["family", "bundle", "left_model", "right_model"]) if not agree.empty else agree, [
        "bundle", "family", "left_model", "right_model", "n_aligned", "alignment", "hard_state_agreement", "js_divergence_bits", "mean_l1_probability_distance"
    ], max_rows=100))
    for fam in ["parity", "x_loop", "z_loop"]:
        add_figures(lines, out_dir, [f"{fam}_pipeline_model_agreement_heatmap.png", f"{fam}_individual_model_agreement_heatmap.png"])

    lines.append("## Cross-bundle same-model agreement\n")
    lines.append("This section compares pipeline vs individual predictions for the same family/model when both bundles contain prediction `.npz` files. Some individual neural baselines currently have checkpoints and training metrics but no exported predictions, so they cannot appear here until predictions are exported.\n")
    lines.append(md_table(cross.sort_values(["family", "left_model"]) if not cross.empty else cross, [
        "family", "left_model", "right_model", "n_aligned", "alignment", "hard_state_agreement", "js_divergence_bits", "mean_l1_probability_distance"
    ]))
    add_figures(lines, out_dir, ["cross_bundle_same_model_agreement.png"])

    lines.append("## Bundle comparison metrics\n")
    lines.append("These are direct pipeline-vs-individual deltas/ratios for matched family/model metrics. Ratios near 1 indicate stable extractions across the two workflows.\n")
    lines.append(md_table(bundle_comp.sort_values(["comparison_type", "family", "model", "metric"]) if not bundle_comp.empty else bundle_comp, [
        "comparison_type", "family", "model", "metric", "pipeline_value", "individual_value", "absolute_delta_pipeline_minus_individual", "ratio_pipeline_over_individual"
    ], max_rows=120))
    add_figures(lines, out_dir, [
        "bundle_ratio_tau_mean_model_units.png", "bundle_ratio_mean_confidence.png", "bundle_ratio_mean_entropy_bits.png",
        "bundle_ratio_state_0_occupancy.png", "bundle_ratio_state_1_occupancy.png"
    ])

    lines.append("## Dwell diagnostics\n")
    lines.append("Dwell metrics are computed from prediction-window state runs when `sequence_id` exists. They are a diagnostic for duration/non-Markovian behavior, not a replacement for raw-trace dwell analysis.\n")
    lines.append(md_table(dwell.sort_values(["family", "bundle", "model", "state"]) if not dwell.empty else dwell, [
        "bundle", "family", "model", "state", "n_dwell_segments", "mean_dwell_windows", "median_dwell_windows", "p90_dwell_windows", "max_dwell_windows"
    ], max_rows=100))
    for fam in ["parity", "x_loop", "z_loop"]:
        add_figures(lines, out_dir, [f"{fam}_dwell_windows_all_bundles.png"])

    lines.append("## Training metrics\n")
    lines.append("Training metrics provide baseline/ablation context for the individual models and downstream pipeline stages. Missing prediction exports do not prevent training metrics from being reported.\n")
    lines.append(md_table(training.sort_values(["family", "bundle", "model"]) if not training.empty else training, [
        "bundle", "family", "model", "stage", "uses_teacher", "test_loss", "best_selection_score", "last_train_loss", "last_val_loss", "best_val_loss", "metrics_csv_rows", "has_checkpoint"
    ], max_rows=100))

    lines.append("## Interpretation notes\n")
    lines.append(md_table(interp, ["section", "bundle", "family", "model", "finding", "interpretation"], max_rows=120))

    lines.append("## Direct data still needed for physical QCVV\n")
    lines.append("The current artifacts support model-derived characterization. The following are required for stronger physical QCVV claims.\n")
    lines.append(md_table(direct, ["need", "why", "status", "how_to_use"]))

    if not warnings.empty:
        lines.append("## Extraction warnings\n")
        lines.append(md_table(warnings, ["code", "message", "artifact"], max_rows=100))

    lines.append("## Generated files\n")
    outputs = summary.get("outputs", {}) if isinstance(summary, dict) else {}
    if outputs:
        for name, path in outputs.items():
            lines.append(f"- `{name}`: `{path}`\n")
    manifest = out_dir / "figures_manifest.txt"
    if manifest.exists():
        lines.append("\n### Figures\n")
        for line in manifest.read_text(encoding="utf-8").splitlines():
            lines.append(f"- `{line}`\n")

    report_path = out_dir / "qcvv_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote QCVV report to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
