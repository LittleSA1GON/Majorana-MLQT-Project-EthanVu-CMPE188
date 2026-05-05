import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

FAMILIES = ("parity", "x_loop", "z_loop")

# Paths are intentionally redundant so the tool works whether the user unzips
# a nested archive or places the artifacts directly under pipeline/individual.
BUNDLE_PATHS: dict[str, tuple[str, ...]] = {
    "pipeline": ("pipeline/pipeline", "pipeline"),
    "individual": ("individual/individual", "individual"),
}

STAGE_LABELS = {
    # Pipeline stages
    "01_cthmm": "cthmm_teacher",
    "02_cnn_gru_independent": "cnn_gru_independent",
    "03_cnn_gru_teacher_assisted": "cnn_gru_teacher_assisted",
    "04_hsmm_using_cnn_embeddings": "hsmm_duration",
    "05_dmm": "dmm",
    "06_export_predictions": "exported",
    # Individual stages
    "cthmm": "cthmm_teacher",
    "cnn_gru": "cnn_gru_independent",
    "hsmm": "hsmm_duration",
    "dmm": "dmm",
}

PREDICTION_NAMES = (
    "teacher_predictions.npz",
    "predictions_cnn_independent.npz",
    "predictions_cnn_teacher_assisted.npz",
    "hsmm_predictions.npz",
    "predictions_dmm.npz",
    "predictions.npz",
)


@dataclass
class ArtifactInfo:
    bundle: str
    family: str
    stage: str
    model: str
    run_dir: str
    prediction_npz: str | None = None
    qcvv_summary_json: str | None = None
    cthmm_model_json: str | None = None
    summary_json: str | None = None
    metrics_json: str | None = None
    metrics_csv: str | None = None
    config_json: str | None = None
    has_checkpoint: bool = False


@dataclass
class ExtractionWarning:
    code: str
    message: str
    artifact: str | None = None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_arg(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    return read_json(p)


def to_float(x: Any, default: float = math.nan) -> float:
    try:
        if x is None:
            return default
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else math.nan


def safe_median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.median(vals)) if vals else math.nan


def safe_ratio(num: float, den: float) -> float:
    if math.isfinite(num) and math.isfinite(den) and den != 0:
        return float(num / den)
    return math.nan


def newest_dir(stage_dir: Path) -> Path | None:
    if not stage_dir.exists():
        return None
    dirs = [p for p in stage_dir.iterdir() if p.is_dir()]
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def choose_existing_root(project_root: Path, rel_paths: Iterable[str]) -> Path | None:
    for rel in rel_paths:
        root = project_root / rel
        if not root.exists() or not root.is_dir():
            continue
        # A valid artifact root contains family folders.
        if any((root / fam).is_dir() for fam in FAMILIES):
            return root
    return None


def source_model_from_npz(path: Path | None, stage_name: str = "") -> str:
    if path is not None:
        try:
            d = np.load(path, allow_pickle=True)
            if "source_model" in d.files:
                arr = np.asarray(d["source_model"])
                if arr.size:
                    return str(arr.ravel()[0])
        except Exception:
            pass
        name = path.name
        if "cthmm" in name or name.startswith("teacher"):
            return "cthmm_teacher"
        if "teacher_assisted" in name:
            return "cnn_gru_teacher_assisted"
        if "independent" in name:
            return "cnn_gru_independent"
        if "hsmm" in name:
            return "hsmm_duration"
        if "dmm" in name:
            return "dmm"
    return STAGE_LABELS.get(stage_name, stage_name or "unknown")


def discover_artifacts(project_root: Path, bundles: Iterable[str]) -> tuple[list[ArtifactInfo], list[ExtractionWarning]]:
    artifacts: list[ArtifactInfo] = []
    warnings: list[ExtractionWarning] = []

    for bundle_name in bundles:
        rels = BUNDLE_PATHS.get(bundle_name)
        if not rels:
            warnings.append(ExtractionWarning("bundle", f"Unknown bundle requested: {bundle_name}", None))
            continue
        root = choose_existing_root(project_root, rels)
        if root is None:
            warnings.append(ExtractionWarning("bundle_missing", f"No artifact root found for bundle={bundle_name}; checked {list(rels)}", None))
            continue

        for family in FAMILIES:
            family_dir = root / family
            if not family_dir.is_dir():
                warnings.append(ExtractionWarning("family_missing", f"Missing {bundle_name}/{family} folder", str(family_dir)))
                continue
            for stage_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
                run = newest_dir(stage_dir)
                if run is None:
                    warnings.append(ExtractionWarning("empty_stage", f"No run directory found in {stage_dir}", str(stage_dir)))
                    continue

                pred = None
                for name in PREDICTION_NAMES:
                    candidate = run / name
                    if candidate.exists():
                        pred = candidate
                        break

                qcvv_summary = run / "qcvv_summary.json"
                cthmm_model = run / "cthmm_model.json"
                summary = run / "summary.json"
                metrics_json = run / "metrics.json"
                metrics_csv = run / "metrics.csv"
                config = run / "config.json"
                model = source_model_from_npz(pred, stage_dir.name)
                artifacts.append(
                    ArtifactInfo(
                        bundle=bundle_name,
                        family=family,
                        stage=stage_dir.name,
                        model=model,
                        run_dir=str(run.relative_to(project_root)),
                        prediction_npz=str(pred.relative_to(project_root)) if pred else None,
                        qcvv_summary_json=str(qcvv_summary.relative_to(project_root)) if qcvv_summary.exists() else None,
                        cthmm_model_json=str(cthmm_model.relative_to(project_root)) if cthmm_model.exists() else None,
                        summary_json=str(summary.relative_to(project_root)) if summary.exists() else None,
                        metrics_json=str(metrics_json.relative_to(project_root)) if metrics_json.exists() else None,
                        metrics_csv=str(metrics_csv.relative_to(project_root)) if metrics_csv.exists() else None,
                        config_json=str(config.relative_to(project_root)) if config.exists() else None,
                        has_checkpoint=(run / "best.pt").exists() or (run / "last.pt").exists(),
                    )
                )
    return artifacts, warnings


def get_rates_and_lifetimes(summary: dict[str, Any]) -> dict[str, float]:
    src = summary.get("rates_and_lifetimes") or {}
    rates = summary.get("rates") or {}
    lifetimes = summary.get("lifetimes") or {}
    out = {
        "gamma_01": to_float(src.get("gamma_01", rates.get("gamma_01"))),
        "gamma_10": to_float(src.get("gamma_10", rates.get("gamma_10"))),
        "tau_0": to_float(src.get("tau_0", lifetimes.get("tau_0"))),
        "tau_1": to_float(src.get("tau_1", lifetimes.get("tau_1"))),
        "tau_mean": to_float(src.get("tau_mean", lifetimes.get("tau_mean"))),
    }
    if not math.isfinite(out["tau_0"]) and math.isfinite(out["gamma_01"]) and out["gamma_01"] > 0:
        out["tau_0"] = 1.0 / out["gamma_01"]
    if not math.isfinite(out["tau_1"]) and math.isfinite(out["gamma_10"]) and out["gamma_10"] > 0:
        out["tau_1"] = 1.0 / out["gamma_10"]
    if not math.isfinite(out["tau_mean"]) and math.isfinite(out["tau_0"]) and math.isfinite(out["tau_1"]):
        out["tau_mean"] = 0.5 * (out["tau_0"] + out["tau_1"])
    return out


def extract_cthmm_rows(project_root: Path, artifacts: list[ArtifactInfo], dt_by_family: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for a in artifacts:
        summary_path = a.qcvv_summary_json or a.metrics_json
        if not summary_path:
            continue
        try:
            summary = read_json(project_root / summary_path)
        except Exception:
            continue
        rates = get_rates_and_lifetimes(summary)
        if not (math.isfinite(rates["gamma_01"]) or math.isfinite(rates["tau_0"])):
            continue
        dt = to_float(dt_by_family.get(a.family), math.nan)
        rows.append(
            {
                "bundle": a.bundle,
                "family": a.family,
                "stage": a.stage,
                "model": a.model,
                "run_dir": a.run_dir,
                "gamma_01": rates["gamma_01"],
                "gamma_10": rates["gamma_10"],
                "tau_0_model_units": rates["tau_0"],
                "tau_1_model_units": rates["tau_1"],
                "tau_mean_model_units": rates["tau_mean"],
                "tau_asymmetry_tau0_over_tau1": safe_ratio(rates["tau_0"], rates["tau_1"]),
                "physical_dt": dt,
                "tau_0_physical": rates["tau_0"] * dt if math.isfinite(dt) and math.isfinite(rates["tau_0"]) else math.nan,
                "tau_1_physical": rates["tau_1"] * dt if math.isfinite(dt) and math.isfinite(rates["tau_1"]) else math.nan,
                "tau_mean_physical": rates["tau_mean"] * dt if math.isfinite(dt) and math.isfinite(rates["tau_mean"]) else math.nan,
                "train_log_likelihood_per_obs": to_float(summary.get("train_log_likelihood_per_obs")),
                "source_file": summary_path,
            }
        )
    return rows


def entropy_from_probs(p: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -np.sum(p * np.log2(p), axis=1)


def summarize_prediction_npz(project_root: Path, a: ArtifactInfo, state_labels: dict[str, Any]) -> dict[str, Any] | None:
    if not a.prediction_npz:
        return None
    p = project_root / a.prediction_npz
    try:
        d = np.load(p, allow_pickle=True)
    except Exception as exc:
        return {"bundle": a.bundle, "family": a.family, "stage": a.stage, "model": a.model, "error": str(exc), "source_file": a.prediction_npz}
    if "hard_state" not in d.files:
        return None

    hard = np.asarray(d["hard_state"], dtype=np.int64).ravel()
    n = int(hard.size)
    if n == 0:
        return None
    sample_index = np.asarray(d["sample_index"], dtype=np.int64).ravel() if "sample_index" in d.files else np.arange(n)
    probs = np.asarray(d["state_probs"], dtype=np.float64) if "state_probs" in d.files else None
    conf = np.asarray(d["confidence"], dtype=np.float64).ravel() if "confidence" in d.files else None
    if conf is None and probs is not None and probs.ndim == 2:
        conf = probs.max(axis=1)

    num_states = int(max(int(hard.max()) + 1, probs.shape[1] if probs is not None and probs.ndim == 2 else 2))
    counts = np.bincount(np.clip(hard, 0, num_states - 1), minlength=num_states)
    occ = counts / max(n, 1)
    label_map = state_labels.get(a.family, {}) if isinstance(state_labels, dict) else {}

    row: dict[str, Any] = {
        "bundle": a.bundle,
        "family": a.family,
        "stage": a.stage,
        "model": a.model,
        "run_dir": a.run_dir,
        "source_file": a.prediction_npz,
        "n_predictions": n,
        "num_states": num_states,
        "sample_index_min": int(sample_index.min()) if sample_index.size else math.nan,
        "sample_index_max": int(sample_index.max()) if sample_index.size else math.nan,
        "has_sequence_id": "sequence_id" in d.files,
        "has_state_probs": probs is not None,
        "has_confidence": conf is not None,
        "mean_confidence": float(np.mean(conf)) if conf is not None and conf.size else math.nan,
        "median_confidence": float(np.median(conf)) if conf is not None and conf.size else math.nan,
        "p05_confidence": float(np.quantile(conf, 0.05)) if conf is not None and conf.size else math.nan,
        "p95_confidence": float(np.quantile(conf, 0.95)) if conf is not None and conf.size else math.nan,
    }
    if probs is not None and probs.ndim == 2 and probs.shape[0] == n:
        ent = entropy_from_probs(probs)
        row["mean_entropy_bits"] = float(np.mean(ent))
        row["median_entropy_bits"] = float(np.median(ent))
        for k in range(probs.shape[1]):
            row[f"mean_prob_state_{k}"] = float(np.mean(probs[:, k]))
    else:
        row["mean_entropy_bits"] = math.nan
        row["median_entropy_bits"] = math.nan

    for k in range(num_states):
        row[f"state_{k}_occupancy"] = float(occ[k])
        label = label_map.get(str(k), label_map.get(k, None)) if isinstance(label_map, dict) else None
        if label:
            row[f"state_{k}_label"] = str(label)

    # Prediction-window switches. These are not raw trace transitions unless
    # each prediction row represents one raw sample with correct sequence IDs.
    if "sequence_id" in d.files:
        seq = np.asarray(d["sequence_id"], dtype=np.int64).ravel()
        if seq.size == n:
            order = np.lexsort((sample_index, seq))
            hard_o = hard[order]
            seq_o = seq[order]
            same_seq = seq_o[1:] == seq_o[:-1]
            switches = np.logical_and(same_seq, hard_o[1:] != hard_o[:-1])
            comparisons = int(np.sum(same_seq))
            row["window_switch_count"] = int(np.sum(switches))
            row["window_switch_rate"] = float(np.sum(switches) / comparisons) if comparisons else math.nan
            row["n_sequences"] = int(np.unique(seq).size)
        else:
            row["window_switch_count"] = math.nan
            row["window_switch_rate"] = math.nan
            row["n_sequences"] = math.nan
    else:
        row["window_switch_count"] = math.nan
        row["window_switch_rate"] = math.nan
        row["n_sequences"] = math.nan
    return row


def prediction_data(project_root: Path, a: ArtifactInfo) -> dict[str, Any] | None:
    if not a.prediction_npz:
        return None
    try:
        d = np.load(project_root / a.prediction_npz, allow_pickle=True)
        if "hard_state" not in d.files:
            return None
        sample_index = np.asarray(d["sample_index"], dtype=np.int64).ravel() if "sample_index" in d.files else None
        hard_state = np.asarray(d["hard_state"], dtype=np.int64).ravel()
        out = {
            "artifact": a,
            "sample_index": sample_index if sample_index is not None else np.arange(len(hard_state), dtype=np.int64),
            "sequence_id": np.asarray(d["sequence_id"], dtype=np.int64).ravel() if "sequence_id" in d.files else None,
            "hard_state": hard_state,
            "state_probs": np.asarray(d["state_probs"], dtype=np.float64) if "state_probs" in d.files else None,
            "confidence": np.asarray(d["confidence"], dtype=np.float64).ravel() if "confidence" in d.files else None,
        }
        return out
    except Exception:
        return None


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    return float(0.5 * np.mean(np.sum(p * (np.log2(p) - np.log2(m)), axis=1)) + 0.5 * np.mean(np.sum(q * (np.log2(q) - np.log2(m)), axis=1)))


def align_pair(left: dict[str, Any], right: dict[str, Any], min_id_overlap: float) -> tuple[np.ndarray, np.ndarray, str, str | None]:
    li = np.asarray(left["sample_index"], dtype=np.int64)
    ri = np.asarray(right["sample_index"], dtype=np.int64)
    n_min = min(li.size, ri.size)
    if n_min == 0:
        return np.array([], dtype=int), np.array([], dtype=int), "none", "empty prediction array"

    common, lpos, rpos = np.intersect1d(li, ri, assume_unique=False, return_indices=True)
    overlap_fraction = common.size / max(n_min, 1)
    if overlap_fraction >= min_id_overlap:
        order = np.argsort(common)
        return lpos[order].astype(np.int64), rpos[order].astype(np.int64), "sample_index", None

    if li.size == ri.size:
        pos = np.arange(li.size, dtype=np.int64)
        return pos, pos, "position", f"low sample_index overlap ({overlap_fraction:.3f}); compared by array position because lengths match"

    return np.array([], dtype=int), np.array([], dtype=int), "none", f"low sample_index overlap ({overlap_fraction:.3f}) and unequal lengths ({li.size} vs {ri.size})"


def prediction_pair_metrics(left: dict[str, Any], right: dict[str, Any], min_id_overlap: float) -> tuple[dict[str, Any] | None, ExtractionWarning | None]:
    la: ArtifactInfo = left["artifact"]
    ra: ArtifactInfo = right["artifact"]
    lpos, rpos, mode, note = align_pair(left, right, min_id_overlap)
    warning = ExtractionWarning("alignment", note, f"{la.prediction_npz} vs {ra.prediction_npz}") if note else None
    if lpos.size == 0:
        return None, warning
    lh = left["hard_state"][lpos]
    rh = right["hard_state"][rpos]
    row: dict[str, Any] = {
        "left_bundle": la.bundle,
        "right_bundle": ra.bundle,
        "family": la.family,
        "left_model": la.model,
        "right_model": ra.model,
        "left_stage": la.stage,
        "right_stage": ra.stage,
        "n_aligned": int(lh.size),
        "alignment": mode,
        "hard_state_agreement": float(np.mean(lh == rh)) if lh.size else math.nan,
        "left_source_file": la.prediction_npz,
        "right_source_file": ra.prediction_npz,
    }
    lp = left.get("state_probs")
    rp = right.get("state_probs")
    if lp is not None and rp is not None and lp.shape[1] == rp.shape[1]:
        row["js_divergence_bits"] = js_divergence(lp[lpos], rp[rpos])
        row["mean_l1_probability_distance"] = float(np.mean(np.sum(np.abs(lp[lpos] - rp[rpos]), axis=1)))
    else:
        row["js_divergence_bits"] = math.nan
        row["mean_l1_probability_distance"] = math.nan
    return row, warning


def compare_predictions(project_root: Path, artifacts: list[ArtifactInfo], min_id_overlap: float) -> tuple[list[dict[str, Any]], list[ExtractionWarning], list[dict[str, Any]]]:
    within_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    warnings: list[ExtractionWarning] = []
    preds: list[dict[str, Any]] = []
    for a in artifacts:
        data = prediction_data(project_root, a)
        if data is not None:
            preds.append(data)

    by_bundle_family: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_family_model: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for p in preds:
        a: ArtifactInfo = p["artifact"]
        by_bundle_family.setdefault((a.bundle, a.family), []).append(p)
        by_family_model.setdefault((a.family, a.model), []).append(p)

    for (_bundle, _family), group in by_bundle_family.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                row, warn = prediction_pair_metrics(group[i], group[j], min_id_overlap)
                if warn:
                    warnings.append(warn)
                if row:
                    row["bundle"] = row["left_bundle"]
                    within_rows.append(row)

    for (_family, _model), group in by_family_model.items():
        pipelines = [p for p in group if p["artifact"].bundle == "pipeline"]
        individuals = [p for p in group if p["artifact"].bundle == "individual"]
        for lp in pipelines:
            for rp in individuals:
                row, warn = prediction_pair_metrics(lp, rp, min_id_overlap)
                if warn:
                    warnings.append(warn)
                if row:
                    cross_rows.append(row)

    return within_rows, warnings, cross_rows


def extract_dwell_rows(project_root: Path, artifacts: list[ArtifactInfo], dt_by_family: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for a in artifacts:
        if not a.prediction_npz:
            continue
        try:
            d = np.load(project_root / a.prediction_npz, allow_pickle=True)
        except Exception:
            continue
        if "hard_state" not in d.files or "sequence_id" not in d.files:
            continue
        hard = np.asarray(d["hard_state"], dtype=np.int64).ravel()
        seq = np.asarray(d["sequence_id"], dtype=np.int64).ravel()
        sample_index = np.asarray(d["sample_index"], dtype=np.int64).ravel() if "sample_index" in d.files else np.arange(hard.size)
        if hard.size == 0 or seq.size != hard.size:
            continue
        dt = to_float(dt_by_family.get(a.family), math.nan)
        order = np.lexsort((sample_index, seq))
        hard_o = hard[order]
        seq_o = seq[order]

        per_state_lengths: dict[int, list[int]] = {}
        current_seq = int(seq_o[0])
        current_state = int(hard_o[0])
        length = 1
        for s_value, h_value in zip(seq_o[1:], hard_o[1:]):
            s_int = int(s_value)
            h_int = int(h_value)
            if s_int == current_seq and h_int == current_state:
                length += 1
            else:
                per_state_lengths.setdefault(current_state, []).append(length)
                current_seq = s_int
                current_state = h_int
                length = 1
        per_state_lengths.setdefault(current_state, []).append(length)

        for state, lengths in sorted(per_state_lengths.items()):
            arr = np.asarray(lengths, dtype=np.float64)
            rows.append(
                {
                    "bundle": a.bundle,
                    "family": a.family,
                    "stage": a.stage,
                    "model": a.model,
                    "state": int(state),
                    "n_dwell_segments": int(arr.size),
                    "mean_dwell_windows": float(np.mean(arr)) if arr.size else math.nan,
                    "median_dwell_windows": float(np.median(arr)) if arr.size else math.nan,
                    "p90_dwell_windows": float(np.quantile(arr, 0.90)) if arr.size else math.nan,
                    "max_dwell_windows": float(np.max(arr)) if arr.size else math.nan,
                    "physical_dt": dt,
                    "mean_dwell_physical": float(np.mean(arr) * dt) if math.isfinite(dt) and arr.size else math.nan,
                    "median_dwell_physical": float(np.median(arr) * dt) if math.isfinite(dt) and arr.size else math.nan,
                    "source_file": a.prediction_npz,
                }
            )
    return rows


def extract_training_rows(project_root: Path, artifacts: list[ArtifactInfo]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for a in artifacts:
        row: dict[str, Any] = {
            "bundle": a.bundle,
            "family": a.family,
            "stage": a.stage,
            "model": a.model,
            "run_dir": a.run_dir,
            "has_checkpoint": a.has_checkpoint,
            "summary_json": a.summary_json,
            "metrics_json": a.metrics_json,
            "metrics_csv": a.metrics_csv,
        }
        if a.summary_json:
            try:
                s = read_json(project_root / a.summary_json)
                for key in ("train_loss", "val_loss", "test_loss", "best_selection_score"):
                    if key in s:
                        row[key] = to_float(s.get(key))
                tm = s.get("test_metrics", {}) if isinstance(s.get("test_metrics"), dict) else {}
                for key, value in tm.items():
                    if isinstance(value, (int, float, np.number)):
                        row[f"test_{key}"] = to_float(value)
                row["teacher_npz"] = s.get("teacher_npz")
                row["uses_teacher"] = bool(s.get("teacher_npz"))
            except Exception as exc:
                row["summary_error"] = str(exc)
        if a.metrics_json:
            try:
                m = read_json(project_root / a.metrics_json)
                for key in ("train_loss", "val_loss", "test_loss", "score", "accuracy"):
                    if key in m:
                        row[key] = to_float(m.get(key))
                if "fit_result" in m and isinstance(m["fit_result"], dict):
                    row["fit_n_iter"] = m["fit_result"].get("n_iter")
                    row["fit_converged"] = m["fit_result"].get("converged")
                if "split_info" in m:
                    row["split_info_json"] = json.dumps(m["split_info"], sort_keys=True)
            except Exception as exc:
                row["metrics_json_error"] = str(exc)
        if a.metrics_csv:
            try:
                if pd is not None:
                    df = pd.read_csv(project_root / a.metrics_csv)
                    row["metrics_csv_rows"] = int(len(df))
                    for col in df.columns:
                        if col.lower() in {"epoch", "step"}:
                            continue
                        vals = pd.to_numeric(df[col], errors="coerce")
                        if vals.notna().any():
                            row[f"last_{col}"] = float(vals.dropna().iloc[-1])
                            row[f"best_{col}"] = float(vals.min()) if "loss" in col.lower() else float(vals.max())
                else:
                    with (project_root / a.metrics_csv).open(newline="", encoding="utf-8") as f:
                        row["metrics_csv_rows"] = sum(1 for _ in csv.reader(f)) - 1
            except Exception as exc:
                row["metrics_csv_error"] = str(exc)
        if any(k not in {"bundle", "family", "stage", "model", "run_dir", "has_checkpoint", "summary_json", "metrics_json", "metrics_csv"} for k in row):
            rows.append(row)
    return rows


def build_bundle_comparison(cthmm_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]], training_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def paired_delta(items: list[dict[str, Any]], keys: list[str], metrics: list[str], group_type: str) -> None:
        groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
        for item in items:
            key = tuple(item.get(k) for k in keys)
            bundle = item.get("bundle")
            if bundle in {"pipeline", "individual"}:
                groups.setdefault(key, {})[str(bundle)] = item
        for key, pair in groups.items():
            if "pipeline" not in pair or "individual" not in pair:
                continue
            p = pair["pipeline"]
            i = pair["individual"]
            base = {"comparison_type": group_type}
            for k, v in zip(keys, key):
                base[k] = v
            for metric in metrics:
                pv = to_float(p.get(metric))
                iv = to_float(i.get(metric))
                if not (math.isfinite(pv) or math.isfinite(iv)):
                    continue
                row = dict(base)
                row.update(
                    {
                        "metric": metric,
                        "pipeline_value": pv,
                        "individual_value": iv,
                        "absolute_delta_pipeline_minus_individual": pv - iv if math.isfinite(pv) and math.isfinite(iv) else math.nan,
                        "ratio_pipeline_over_individual": safe_ratio(pv, iv),
                    }
                )
                rows.append(row)

    paired_delta(
        cthmm_rows,
        ["family", "model"],
        ["gamma_01", "gamma_10", "tau_0_model_units", "tau_1_model_units", "tau_mean_model_units", "tau_asymmetry_tau0_over_tau1", "train_log_likelihood_per_obs"],
        "cthmm_lifetimes",
    )
    paired_delta(
        prediction_rows,
        ["family", "model"],
        ["mean_confidence", "median_confidence", "mean_entropy_bits", "state_0_occupancy", "state_1_occupancy", "window_switch_rate", "n_predictions"],
        "prediction_metrics",
    )
    paired_delta(
        training_rows,
        ["family", "model"],
        ["test_loss", "best_selection_score", "last_train_loss", "last_val_loss", "best_val_loss", "last_loss", "best_loss"],
        "training_metrics",
    )
    return rows


def build_family_summary(cthmm_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]], agreement_rows: list[dict[str, Any]], cross_bundle_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in ("pipeline", "individual"):
        for family in FAMILIES:
            c = [r for r in cthmm_rows if r.get("bundle") == bundle and r.get("family") == family]
            p = [r for r in prediction_rows if r.get("bundle") == bundle and r.get("family") == family]
            a = [r for r in agreement_rows if r.get("bundle") == bundle and r.get("family") == family]
            row: dict[str, Any] = {
                "bundle": bundle,
                "family": family,
                "n_cthmm_lifetime_rows": len(c),
                "n_prediction_models": len(p),
                "n_within_bundle_agreements": len(a),
                "mean_prediction_confidence": safe_mean([to_float(x.get("mean_confidence")) for x in p]),
                "mean_prediction_entropy_bits": safe_mean([to_float(x.get("mean_entropy_bits")) for x in p]),
                "mean_within_bundle_agreement": safe_mean([to_float(x.get("hard_state_agreement")) for x in a]),
                "cthmm_tau_mean_model_units": safe_mean([to_float(x.get("tau_mean_model_units")) for x in c if x.get("model") == "cthmm_teacher"] or [to_float(x.get("tau_mean_model_units")) for x in c]),
            }
            rows.append(row)

    # Attach cross-bundle same-model prediction agreement where available.
    by_family = {fam: [] for fam in FAMILIES}
    for r in cross_bundle_rows:
        if r.get("family") in by_family:
            by_family[str(r.get("family"))].append(r)
    for row in rows:
        g = by_family.get(str(row["family"]), [])
        row["mean_cross_bundle_same_model_agreement"] = safe_mean([to_float(x.get("hard_state_agreement")) for x in g])
    return rows


def build_scorecard(family_summary: list[dict[str, Any]], prediction_rows: list[dict[str, Any]], agreement_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in family_summary:
        bundle = s["bundle"]
        family = s["family"]
        p = [r for r in prediction_rows if r.get("bundle") == bundle and r.get("family") == family]
        a = [r for r in agreement_rows if r.get("bundle") == bundle and r.get("family") == family]
        best_conf = max([to_float(r.get("mean_confidence"), math.nan) for r in p] or [math.nan])
        mean_agree = safe_mean([to_float(r.get("hard_state_agreement")) for r in a])
        n_predictions = sum(1 for r in p if r.get("n_predictions"))
        has_cthmm = int(s.get("n_cthmm_lifetime_rows", 0) > 0)
        has_multimodel = int(n_predictions >= 2)
        has_agreement = int(math.isfinite(mean_agree))
        has_confidence = int(math.isfinite(best_conf))
        # Conservative diagnostic score. This is NOT device fidelity.
        score = 0.0
        score += 25.0 * has_cthmm
        score += 25.0 * has_multimodel
        score += 25.0 * (mean_agree if math.isfinite(mean_agree) else 0.0)
        score += 25.0 * (best_conf if math.isfinite(best_conf) else 0.0)
        interpretation = "limited"
        if score >= 85:
            interpretation = "strong model-assisted characterization"
        elif score >= 65:
            interpretation = "usable model-assisted characterization"
        elif score >= 40:
            interpretation = "partial characterization"
        rows.append(
            {
                "bundle": bundle,
                "family": family,
                "qcvv_model_readiness_score_0_to_100": score,
                "interpretation": interpretation,
                "has_cthmm_lifetimes": bool(has_cthmm),
                "has_multiple_prediction_models": bool(has_multimodel),
                "mean_within_bundle_agreement": mean_agree,
                "best_model_confidence": best_conf,
                "n_prediction_models": n_predictions,
                "caution": "This score summarizes artifact completeness/model consistency only; it is not physical fidelity or certification.",
            }
        )
    return rows


def build_interpretation_notes(cthmm_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]], bundle_comparison: list[dict[str, Any]], scorecard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    # General notes.
    notes.append({"section": "scope", "finding": "The extraction covers both pipeline and individual artifact bundles by default.", "interpretation": "Use pipeline as the connected QCVV workflow and individual as independent baselines/ablation evidence."})
    notes.append({"section": "limits", "finding": "Model artifacts do not include physical calibration labels or raw timing metadata by themselves.", "interpretation": "Absolute readout fidelity, seconds-scale lifetimes, QND repeatability, and measurement backaction require direct data."})

    # Lifetime ratios: z/x within each bundle where available.
    for bundle in ("pipeline", "individual"):
        byfam = {r.get("family"): r for r in cthmm_rows if r.get("bundle") == bundle and r.get("model") == "cthmm_teacher"}
        x = byfam.get("x_loop")
        z = byfam.get("z_loop")
        if x and z:
            ratio = safe_ratio(to_float(z.get("tau_mean_model_units")), to_float(x.get("tau_mean_model_units")))
            notes.append({"section": "tetron_lifetime_ratio", "bundle": bundle, "finding": f"z_loop / x_loop CT-HMM tau_mean ratio = {ratio:.6g}" if math.isfinite(ratio) else "ratio unavailable", "interpretation": "Large ratios indicate longer effective Z-loop dwell/lifetime scale than X-loop in model units; convert with dt before reporting seconds."})

    # Flag collapsed low-confidence models.
    for r in prediction_rows:
        conf = to_float(r.get("mean_confidence"))
        occ0 = to_float(r.get("state_0_occupancy"))
        occ1 = to_float(r.get("state_1_occupancy"))
        if math.isfinite(conf) and conf < 0.6 and (occ0 > 0.95 or occ1 > 0.95):
            notes.append({"section": "model_quality", "bundle": r.get("bundle"), "family": r.get("family"), "model": r.get("model"), "finding": "Low confidence and near-single-state occupancy.", "interpretation": "Treat this model as a weak baseline, not a final decoder."})

    # Summarize scorecard.
    for r in scorecard:
        notes.append({"section": "scorecard", "bundle": r.get("bundle"), "family": r.get("family"), "finding": f"readiness score = {to_float(r.get('qcvv_model_readiness_score_0_to_100')):.2f}", "interpretation": r.get("interpretation")})
    return notes


def write_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pd is not None:
        pd.DataFrame(rows).to_csv(path, index=False)
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_bundles(args: argparse.Namespace) -> list[str]:
    if getattr(args, "pipeline_only", False):
        return ["pipeline"]
    if getattr(args, "individual_only", False):
        return ["individual"]
    if args.bundles == "pipeline":
        return ["pipeline"]
    if args.bundles == "individual":
        return ["individual"]
    return ["pipeline", "individual"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract comprehensive QCVV metrics from trained Majorana MLQT artifacts.")
    parser.add_argument("--project-root", default=".", help="Project root containing pipeline/ and individual/ folders.")
    parser.add_argument("--out-dir", default="qcvv_outputs", help="Directory for extracted CSV/JSON outputs.")
    parser.add_argument("--bundles", choices=["both", "pipeline", "individual"], default="both", help="Which artifact bundle(s) to scan. Default: both.")
    parser.add_argument("--include-individual", action="store_true", help="Legacy flag retained for compatibility. Both bundles are included by default now.")
    parser.add_argument("--pipeline-only", action="store_true", help="Scan only pipeline artifacts.")
    parser.add_argument("--individual-only", action="store_true", help="Scan only individual artifacts.")
    parser.add_argument("--dt-json", default=None, help="Optional JSON mapping family -> physical timestep multiplier.")
    parser.add_argument("--state-label-json", default=None, help="Optional JSON mapping family -> {'0': label, '1': label}.")
    parser.add_argument("--min-id-overlap", type=float, default=0.80, help="Minimum sample_index overlap for ID-based model comparison.")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    out_dir = (project_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir).resolve()
    dt_by_family = load_json_arg(args.dt_json)
    state_labels = load_json_arg(args.state_label_json)
    bundles = parse_bundles(args)

    warnings: list[ExtractionWarning] = []
    artifacts, discovery_warnings = discover_artifacts(project_root, bundles)
    warnings.extend(discovery_warnings)
    if not artifacts:
        checked = {b: BUNDLE_PATHS.get(b, ()) for b in bundles}
        raise FileNotFoundError(f"No QCVV artifacts found under {project_root}. Checked: {checked}")

    artifact_rows = [asdict(a) for a in artifacts]
    cthmm_rows = extract_cthmm_rows(project_root, artifacts, dt_by_family)
    prediction_rows = [row for a in artifacts if (row := summarize_prediction_npz(project_root, a, state_labels)) is not None]
    agreement_rows, agreement_warnings, cross_bundle_rows = compare_predictions(project_root, artifacts, float(args.min_id_overlap))
    warnings.extend(agreement_warnings)
    dwell_rows = extract_dwell_rows(project_root, artifacts, dt_by_family)
    training_rows = extract_training_rows(project_root, artifacts)
    bundle_comparison_rows = build_bundle_comparison(cthmm_rows, prediction_rows, training_rows)
    family_summary_rows = build_family_summary(cthmm_rows, prediction_rows, agreement_rows, cross_bundle_rows)
    scorecard_rows = build_scorecard(family_summary_rows, prediction_rows, agreement_rows)
    interpretation_rows = build_interpretation_notes(cthmm_rows, prediction_rows, bundle_comparison_rows, scorecard_rows)

    # Direct data checklist.
    direct_data_needed = [
        {
            "need": "physical timestep dt for each family",
            "why": "Converts tau_0/tau_1 and dwell-window statistics from model units into seconds.",
            "status": "provided" if dt_by_family else "missing",
            "how_to_use": "Pass --dt-json dt.json with keys parity, x_loop, z_loop.",
        },
        {
            "need": "calibration labels or known state preparations",
            "why": "Maps hidden state 0/1 to even/odd parity or physical X/Z loop states and enables true assignment error/readout fidelity.",
            "status": "provided" if state_labels else "missing",
            "how_to_use": "Pass --state-label-json for state names and use qcvv_calibrate.py for confusion matrices/fidelity.",
        },
        {
            "need": "raw or prepared I/Q traces with sample_dt and run/time metadata",
            "why": "Needed for physical drift analysis, raw trace dwell-time posteriors, repeated-readout/QND metrics, and model inference on new data.",
            "status": "not available in extracted model artifacts unless prepared bundles are accessible locally",
            "how_to_use": "Keep columns such as family, run_id, sequence_id, sample_index, t, I, Q, sample_dt.",
        },
        {
            "need": "repeated-readout or pre/post measurement experiments",
            "why": "Needed to quantify measurement backaction and QND repeatability rather than only decoder confidence.",
            "status": "not inferable from model artifacts alone",
            "how_to_use": "Prepare paired pre/post or repeated measurements and align them by run/sequence/time.",
        },
        {
            "need": "individual CNN-GRU/DMM prediction exports if you want prediction-level comparison for those baselines",
            "why": "The individual bundle has checkpoints and training metrics for some neural models, but no prediction NPZ for CNN-GRU/DMM in the current project files.",
            "status": "missing when prediction_npz is blank in artifact_inventory.csv",
            "how_to_use": "Run model inference/export on a shared held-out dataset, then rerun qcvv_extract.py.",
        },
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "artifact_inventory": out_dir / "artifact_inventory.csv",
        "cthmm_lifetimes": out_dir / "cthmm_lifetimes.csv",
        "prediction_metrics": out_dir / "prediction_metrics.csv",
        "model_agreements": out_dir / "model_agreements.csv",
        "cross_bundle_model_agreements": out_dir / "cross_bundle_model_agreements.csv",
        "dwell_metrics": out_dir / "dwell_metrics.csv",
        "training_metrics": out_dir / "training_metrics.csv",
        "bundle_comparison": out_dir / "bundle_comparison.csv",
        "family_summary": out_dir / "family_summary.csv",
        "qcvv_scorecard": out_dir / "qcvv_scorecard.csv",
        "interpretation_notes": out_dir / "interpretation_notes.csv",
        "extraction_warnings": out_dir / "extraction_warnings.csv",
        "direct_data_needed": out_dir / "direct_data_needed.csv",
    }
    write_table(artifact_rows, outputs["artifact_inventory"])
    write_table(cthmm_rows, outputs["cthmm_lifetimes"])
    write_table(prediction_rows, outputs["prediction_metrics"])
    write_table(agreement_rows, outputs["model_agreements"])
    write_table(cross_bundle_rows, outputs["cross_bundle_model_agreements"])
    write_table(dwell_rows, outputs["dwell_metrics"])
    write_table(training_rows, outputs["training_metrics"])
    write_table(bundle_comparison_rows, outputs["bundle_comparison"])
    write_table(family_summary_rows, outputs["family_summary"])
    write_table(scorecard_rows, outputs["qcvv_scorecard"])
    write_table(interpretation_rows, outputs["interpretation_notes"])
    write_table([asdict(w) for w in warnings], outputs["extraction_warnings"])
    write_table(direct_data_needed, outputs["direct_data_needed"])

    summary = {
        "project_root": str(project_root),
        "out_dir": str(out_dir),
        "bundles_scanned": bundles,
        "n_artifacts": len(artifacts),
        "n_cthmm_rows": len(cthmm_rows),
        "n_prediction_rows": len(prediction_rows),
        "n_agreement_rows": len(agreement_rows),
        "n_cross_bundle_agreement_rows": len(cross_bundle_rows),
        "n_dwell_rows": len(dwell_rows),
        "n_training_rows": len(training_rows),
        "n_bundle_comparison_rows": len(bundle_comparison_rows),
        "dt_by_family": dt_by_family,
        "state_labels": state_labels,
        "warnings": [asdict(w) for w in warnings],
        "direct_data_needed": direct_data_needed,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    with (out_dir / "qcvv_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote comprehensive QCVV extraction outputs to: {out_dir}")
    print(
        "Artifacts: {a} | CTHMM: {c} | predictions: {p} | agreements: {g} | cross-bundle: {x} | dwell: {d} | training: {t}".format(
            a=len(artifacts), c=len(cthmm_rows), p=len(prediction_rows), g=len(agreement_rows), x=len(cross_bundle_rows), d=len(dwell_rows), t=len(training_rows)
        )
    )
    if warnings:
        print(f"Warnings: {len(warnings)} (see extraction_warnings.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
