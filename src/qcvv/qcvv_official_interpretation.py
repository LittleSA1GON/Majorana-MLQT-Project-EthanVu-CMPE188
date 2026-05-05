import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x: Any, digits: int = 6) -> str:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(val):
        return "N/A"
    if abs(val) >= 1e4 or (abs(val) < 1e-3 and val != 0):
        return f"{val:.{digits}e}"
    return f"{val:.{digits}g}"


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_No data available._\n"
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return "_No matching columns available._\n"
    shown = df[cols].head(max_rows).copy()
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in shown.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(fmt(v))
            else:
                vals.append(str(v).replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines) + "\n"


def evidence_level(summary: dict[str, Any], direct: pd.DataFrame, calibration_files: dict[str, bool], uncertainty: bool, drift: bool) -> tuple[str, list[str]]:
    missing = []
    dt_by_family = summary.get("dt_by_family", {}) if isinstance(summary, dict) else {}
    labels = summary.get("state_labels", {}) if isinstance(summary, dict) else {}
    if not dt_by_family:
        missing.append("physical timestep dt for parity/x_loop/z_loop")
    if not labels:
        missing.append("hidden-state to physical-state labels")
    if not calibration_files.get("calibration_metrics", False):
        missing.append("calibration-label readout metrics")
    if not drift:
        missing.append("drift diagnostics")
    if not uncertainty:
        missing.append("bootstrap/confidence intervals")
    if not calibration_files.get("raw_data_checks", False):
        missing.append("raw/prepared trace overlays or direct raw-data checks")

    if not dt_by_family and not labels:
        return "Level 1 - model-assisted characterization", missing
    if dt_by_family and labels and calibration_files.get("calibration_metrics", False) and uncertainty:
        if calibration_files.get("raw_data_checks", False) and drift:
            return "Level 3 - device-validated QCVV candidate", missing
        return "Level 2 - calibrated QCVV candidate", missing
    return "Level 1+ - partially calibrated model-assisted QCVV", missing


def best_model_rows(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty or "mean_confidence" not in pred:
        return pd.DataFrame()
    g = pred.copy()
    g["mean_confidence_num"] = pd.to_numeric(g["mean_confidence"], errors="coerce")
    g = g.dropna(subset=["mean_confidence_num"])
    if g.empty:
        return pd.DataFrame()
    idx = g.groupby(["bundle", "family"], dropna=False)["mean_confidence_num"].idxmax()
    return g.loc[idx].sort_values(["family", "bundle"])


def claim_language(evidence: str) -> tuple[list[str], list[str]]:
    allowed = [
        "model-assisted QCVV characterization",
        "effective model-time lifetimes",
        "CT-HMM-derived transition-rate estimates",
        "hidden-state prediction confidence and entropy",
        "pipeline-vs-individual model consistency",
        "dwell and drift diagnostics from exported predictions",
    ]
    not_allowed = [
        "certified physical lifetime in seconds",
        "certified parity readout fidelity",
        "true assignment error without calibration labels",
        "QND repeatability without repeated-readout or pre/post data",
        "measurement-induced transition probability without direct experiment alignment",
        "device-level validation without raw/prepared traces and run metadata",
    ]
    if evidence.startswith("Level 2") or evidence.startswith("Level 3"):
        allowed.extend([
            "calibrated readout fidelity, if qcvv_calibrate outputs are present",
            "physical lifetimes in seconds, if dt was applied to each family",
            "confidence intervals for model-derived summaries, if bootstrap outputs are present",
        ])
    if evidence.startswith("Level 3"):
        allowed.extend([
            "drift-aware device validation candidate",
            "raw-trace-supported state interpretation",
        ])
    return allowed, not_allowed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate official-style QCVV interpretation document.")
    parser.add_argument("--out-dir", default="qcvv_outputs", help="QCVV output directory.")
    parser.add_argument("--output", default=None, help="Optional output Markdown path. Default: out_dir/official_qcvv_interpretation.md")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    output = Path(args.output).resolve() if args.output else out_dir / "official_qcvv_interpretation.md"

    summary = load_json(out_dir / "qcvv_summary.json")
    cthmm = load_csv(out_dir / "cthmm_lifetimes.csv")
    pred = load_csv(out_dir / "prediction_metrics.csv")
    bundle = load_csv(out_dir / "bundle_comparison.csv")
    score = load_csv(out_dir / "qcvv_scorecard.csv")
    direct = load_csv(out_dir / "direct_data_needed.csv")
    drift_summary = load_csv(out_dir / "drift_summary.csv")
    boot_pred = load_csv(out_dir / "bootstrap_prediction_metrics.csv")
    boot_agree = load_csv(out_dir / "bootstrap_model_agreements.csv")
    cal_metrics_exists = (out_dir / "calibration_metrics.csv").exists() or any(out_dir.glob("**/calibration_metrics.csv"))
    raw_checks_exists = any(out_dir.glob("**/raw_data_qcvv_summary_*.json")) or any(out_dir.glob("**/aligned_raw_predictions_*.csv"))

    evidence, missing = evidence_level(
        summary,
        direct,
        {"calibration_metrics": cal_metrics_exists, "raw_data_checks": raw_checks_exists},
        uncertainty=not boot_pred.empty or not boot_agree.empty,
        drift=not drift_summary.empty,
    )
    allowed, not_allowed = claim_language(evidence)

    lines: list[str] = []
    lines.append("# Official-Style QCVV Interpretation\n")
    lines.append("## Executive conclusion\n")
    lines.append(f"Current evidence level: **{evidence}**.\n")
    lines.append("The project supports QCVV-style characterization from trained model artifacts. A stronger official claim requires physical calibration, raw or prepared trace records, uncertainty bounds, drift analysis, and predeclared thresholds.\n")

    lines.append("## Claim boundary\n")
    lines.append("### Claims currently supported\n")
    for item in allowed:
        lines.append(f"- {item}\n")
    lines.append("\n### Claims not yet supported unless the missing data are added\n")
    for item in not_allowed:
        lines.append(f"- {item}\n")

    lines.append("## Missing items for official/certification-style QCVV\n")
    if missing:
        for item in missing:
            lines.append(f"- {item}\n")
    else:
        lines.append("- No major missing items detected by the script. Review thresholds and experimental protocol manually.\n")
    if not direct.empty:
        lines.append("\n### Direct-data checklist from extraction\n")
        lines.append(md_table(direct, ["need", "why", "status", "how_to_use"], max_rows=20))

    lines.append("## CT-HMM lifetime interpretation\n")
    lines.append("CT-HMM outputs are the most interpretable lifetime estimates in the current project. They should be described as effective Markovian lifetimes unless sequence-aware dwell, drift, and non-Markovian checks support a stronger model.\n")
    lines.append(md_table(cthmm.sort_values(["family", "bundle"]) if not cthmm.empty else cthmm, [
        "bundle", "family", "gamma_01", "gamma_10", "tau_0_model_units", "tau_1_model_units", "tau_mean_model_units", "tau_mean_seconds", "train_log_likelihood_per_obs"
    ], max_rows=20))

    if not bundle.empty:
        lines.append("## Pipeline vs individual interpretation\n")
        lines.append("Pipeline should be treated as the primary QCVV workflow. Individual runs are ablation/baseline evidence. Ratios near 1 indicate stable model-derived quantities across workflows; large deviations indicate sensitivity to training organization or missing prediction exports.\n")
        key = bundle[bundle.get("metric", pd.Series(dtype=str)).isin(["tau_mean_model_units", "mean_confidence", "state_0_occupancy", "mean_entropy_bits"])] if "metric" in bundle else bundle
        lines.append(md_table(key.sort_values(["family", "model", "metric"]) if not key.empty else key, [
            "comparison_type", "family", "model", "metric", "pipeline_value", "individual_value", "absolute_delta_pipeline_minus_individual", "ratio_pipeline_over_individual"
        ], max_rows=60))

    lines.append("## Model role interpretation\n")
    role_rows = pd.DataFrame([
        {"model_family": "CT-HMM", "official_role": "interpretable characterization", "use_for": "transition rates and effective lifetimes", "caution": "Markovian assumption; needs dwell/drift checks"},
        {"model_family": "CNN-GRU independent", "official_role": "baseline decoder", "use_for": "ablation and learning-without-teacher comparison", "caution": "do not use if collapsed or low confidence"},
        {"model_family": "CNN-GRU teacher-assisted", "official_role": "deployable sequence decoder candidate", "use_for": "fast readout if calibrated", "caution": "teacher agreement is not physical fidelity"},
        {"model_family": "HSMM", "official_role": "duration/non-Markovian diagnostic", "use_for": "dwell-time checks", "caution": "not a substitute for raw-trace lifetime analysis"},
        {"model_family": "DMM", "official_role": "high-capacity sequence model", "use_for": "model-consistency and flexible inference", "caution": "watch for overconfidence and teacher imitation"},
    ])
    lines.append(md_table(role_rows, ["model_family", "official_role", "use_for", "caution"], max_rows=10))

    best = best_model_rows(pred)
    if not best.empty:
        lines.append("## Highest-confidence decoder candidates\n")
        lines.append("High confidence should be interpreted as internal model certainty. It becomes readout fidelity only after calibration-label analysis.\n")
        lines.append(md_table(best, ["bundle", "family", "stage", "model", "mean_confidence", "mean_entropy_bits", "state_0_occupancy", "state_1_occupancy"], max_rows=20))

    if not score.empty:
        lines.append("## Model-readiness scorecard\n")
        lines.append("These scores are artifact/model readiness indicators, not certification scores.\n")
        lines.append(md_table(score.sort_values(["family", "bundle"]) if not score.empty else score, [
            "bundle", "family", "qcvv_model_readiness_score_0_to_100", "has_cthmm_lifetime", "has_prediction_metrics", "has_model_agreement", "has_dwell_metrics", "has_pipeline_individual_comparison"
        ], max_rows=20))

    lines.append("## Uncertainty interpretation\n")
    if boot_pred.empty and boot_agree.empty:
        lines.append("No bootstrap uncertainty outputs were detected. Run `qcvv_uncertainty.py` before treating any point estimate as an official QCVV result.\n")
    else:
        lines.append("Bootstrap outputs were detected. They quantify resampling uncertainty of model summaries. They do not replace physical calibration uncertainty.\n")
        if not boot_pred.empty:
            lines.append(md_table(boot_pred, ["bundle", "family", "model", "metric", "bootstrap_median", "ci_low_2p5", "ci_high_97p5", "resample_unit"], max_rows=30))

    lines.append("## Drift / nonstationarity interpretation\n")
    if drift_summary.empty:
        lines.append("No drift summary was detected. Run `qcvv_drift.py` for official-style stability checks.\n")
    else:
        lines.append("Drift diagnostics were detected. Large relative ranges should be investigated before claiming stationary lifetimes or stable readout performance.\n")
        display = drift_summary.sort_values(["family", "bundle", "model", "metric"]) if not drift_summary.empty else drift_summary
        lines.append(md_table(display, ["bundle", "family", "model", "metric", "n_bins", "mean_across_bins", "range_max_minus_min", "relative_range_over_mean_abs"], max_rows=60))

    lines.append("## Acceptance criteria required before official claim\n")
    criteria = [
        "Physical dt is defined for parity, x_loop, and z_loop.",
        "Hidden states 0/1 are mapped to physical parity or loop states using calibration data.",
        "Readout fidelity/assignment error is computed against physical labels, not another model alone.",
        "Lifetime estimates include confidence intervals and specify the time unit.",
        "Dwell-time distributions and drift checks do not contradict the Markovian lifetime interpretation, or the report explicitly calls lifetimes effective Markovian rates.",
        "Repeated-readout or pre/post records support QND/repeatability and measurement-backaction claims.",
        "Predeclared thresholds are used for pass/fail/certification-style statements.",
    ]
    for item in criteria:
        lines.append(f"- {item}\n")

    lines.append("## Recommended final wording\n")
    lines.append("> This project implements a model-assisted QCVV workflow for parity readout and tetron X/Z loop lifetime characterization. CT-HMM provides interpretable transition-rate estimates; CNN-GRU and DMM provide learned sequence decoders; HSMM provides dwell-time diagnostics. Pipeline results are the primary workflow, while individual runs serve as independent baseline evidence. Current artifacts support effective model-time lifetimes and internal model-consistency checks. Fully official QCVV claims require physical timing metadata, state-label calibration, raw/prepared traces, repeated-readout data, uncertainty bounds, drift checks, and predeclared acceptance thresholds.\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote official QCVV interpretation: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
