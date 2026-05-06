#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore


FAMILIES = ("parity", "x_loop", "z_loop")
BUNDLE_PATHS: dict[str, tuple[str, ...]] = {
    "pipeline": ("pipeline/pipeline", "pipeline"),
    "individual": ("individual/individual", "individual"),
}

STAGE_LABELS = {
    "01_cthmm": "cthmm_teacher",
    "02_cnn_gru_independent": "cnn_gru_independent",
    "03_cnn_gru_teacher_assisted": "cnn_gru_teacher_assisted",
    "04_hsmm_using_cnn_embeddings": "hsmm_duration",
    "05_dmm": "dmm",
    "06_export_predictions": "exported",
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

CHECKPOINT_NAMES = (
    "best.pt",
    "last.pt",
    "checkpoint.pt",
    "model.pt",
    "best.pth",
    "last.pth",
    "checkpoint.pth",
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


@dataclass
class PredictionData:
    artifact: ArtifactInfo
    sample_index: np.ndarray
    hard_state: np.ndarray
    sequence_id: np.ndarray | None
    state_probs: np.ndarray | None
    confidence: np.ndarray | None
    entropy_bits: np.ndarray | None


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def safe_mean(values: Iterable[Any]) -> float:
    vals = [to_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else math.nan


def safe_median(values: Iterable[Any]) -> float:
    vals = [to_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(statistics.median(vals)) if vals else math.nan


def safe_ratio(num: Any, den: Any) -> float:
    n = to_float(num)
    d = to_float(den)
    if math.isfinite(n) and math.isfinite(d) and d != 0:
        return float(n / d)
    return math.nan


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"value": data}


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def looks_like_project_root(path: Path) -> bool:
    return any((path / rel).is_dir() for rels in BUNDLE_PATHS.values() for rel in rels)


def resolve_project_root(raw: str) -> Path:
    """Resolve the repo root, with IDE-friendly upward search.

    In VS Code/PyCharm it is common to run the current file with the working
    directory set to src/qcvv/ instead of the repository root. This function
    accepts --project-root when provided, but also searches parent folders of
    the working directory and this script for individual/individual or
    pipeline/pipeline.
    """
    candidate = Path(raw).expanduser().resolve()
    search_starts = [candidate, Path.cwd().resolve()]
    try:
        search_starts.append(Path(__file__).resolve().parent)
    except NameError:  # pragma: no cover
        pass

    seen: set[Path] = set()
    for start in search_starts:
        for path in (start, *start.parents):
            if path in seen:
                continue
            seen.add(path)
            if looks_like_project_root(path):
                return path
    return candidate


def has_direct_artifacts(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = set(PREDICTION_NAMES) | {
        "qcvv_summary.json",
        "cthmm_model.json",
        "summary.json",
        "metrics.json",
        "metrics.csv",
        "config.json",
    } | set(CHECKPOINT_NAMES)
    return any((path / name).exists() for name in names)


def run_dirs_for_stage(stage_dir: Path, latest_only: bool) -> list[Path]:
    candidates: list[Path] = []
    if has_direct_artifacts(stage_dir):
        candidates.append(stage_dir)
    candidates.extend(p for p in stage_dir.iterdir() if p.is_dir())
    if not candidates:
        return []
    if latest_only:
        return [max(candidates, key=lambda p: p.stat().st_mtime)]
    return sorted(candidates, key=lambda p: p.name)


def prediction_files_for_run(run_dir: Path) -> list[Path | None]:
    files = [run_dir / name for name in PREDICTION_NAMES if (run_dir / name).exists()]
    return files if files else [None]


def newest_dir(stage_dir: Path) -> Path | None:
    if not stage_dir.is_dir():
        return None
    dirs = [p for p in stage_dir.iterdir() if p.is_dir()]
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def choose_existing_root(project_root: Path, bundle: str) -> Path | None:
    for rel in BUNDLE_PATHS.get(bundle, ()):
        candidate = project_root / rel
        if candidate.is_dir() and any((candidate / fam).is_dir() for fam in FAMILIES):
            return candidate
    return None


def first_existing(run_dir: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        p = run_dir / name
        if p.exists():
            return p
    return None


def source_model_from_stage_or_npz(prediction_npz: Path | None, stage_name: str) -> str:
    if prediction_npz is not None and prediction_npz.exists():
        try:
            with np.load(prediction_npz, allow_pickle=True) as d:
                if "source_model" in d.files:
                    arr = np.asarray(d["source_model"])
                    if arr.size:
                        return str(arr.ravel()[0])
        except Exception:
            pass
        name = prediction_npz.name.lower()
        if "teacher_assisted" in name:
            return "cnn_gru_teacher_assisted"
        if "independent" in name:
            return "cnn_gru_independent"
        if "teacher" in name or name.startswith("teacher_predictions"):
            return "cthmm_teacher"
        if "hsmm" in name:
            return "hsmm_duration"
        if "dmm" in name:
            return "dmm"
    return STAGE_LABELS.get(stage_name, stage_name or "unknown")


def discover_artifacts(
    project_root: Path,
    bundles: Sequence[str],
    families: Sequence[str],
    latest_only: bool,
) -> tuple[list[ArtifactInfo], list[ExtractionWarning]]:
    artifacts: list[ArtifactInfo] = []
    warnings: list[ExtractionWarning] = []

    for bundle in bundles:
        root = choose_existing_root(project_root, bundle)
        if root is None:
            warnings.append(
                ExtractionWarning(
                    "bundle_missing",
                    f"No artifact root found for bundle={bundle}; checked {BUNDLE_PATHS.get(bundle, ())}",
                    None,
                )
            )
            continue

        for family in families:
            family_dir = root / family
            if not family_dir.is_dir():
                warnings.append(
                    ExtractionWarning("family_missing", f"Missing {bundle}/{family} folder", str(family_dir))
                )
                continue

            stage_dirs = sorted(p for p in family_dir.iterdir() if p.is_dir())
            for stage_dir in stage_dirs:
                run_dirs = run_dirs_for_stage(stage_dir, latest_only)
                if not run_dirs:
                    warnings.append(ExtractionWarning("empty_stage", f"No run directory found in {stage_dir}", str(stage_dir)))
                    continue
                for run in run_dirs:
                    qcvv_summary = run / "qcvv_summary.json"
                    cthmm_model = run / "cthmm_model.json"
                    summary = run / "summary.json"
                    metrics_json = run / "metrics.json"
                    metrics_csv = run / "metrics.csv"
                    config = run / "config.json"
                    has_checkpoint = any((run / name).exists() for name in CHECKPOINT_NAMES)

                    # A normal training run usually contains one prediction NPZ; export
                    # runs may contain several model prediction NPZs in the same folder.
                    for prediction in prediction_files_for_run(run):
                        model = source_model_from_stage_or_npz(prediction, stage_dir.name)
                        artifacts.append(
                            ArtifactInfo(
                                bundle=bundle,
                                family=family,
                                stage=stage_dir.name,
                                model=model,
                                run_dir=relpath(run, project_root),
                                prediction_npz=relpath(prediction, project_root) if prediction else None,
                                qcvv_summary_json=relpath(qcvv_summary, project_root) if qcvv_summary.exists() else None,
                                cthmm_model_json=relpath(cthmm_model, project_root) if cthmm_model.exists() else None,
                                summary_json=relpath(summary, project_root) if summary.exists() else None,
                                metrics_json=relpath(metrics_json, project_root) if metrics_json.exists() else None,
                                metrics_csv=relpath(metrics_csv, project_root) if metrics_csv.exists() else None,
                                config_json=relpath(config, project_root) if config.exists() else None,
                                has_checkpoint=has_checkpoint,
                            )
                        )

    return artifacts, warnings


def npz_key(files: Sequence[str], candidates: Sequence[str]) -> str | None:
    exact = set(files)
    for c in candidates:
        if c in exact:
            return c
    lower = {f.lower(): f for f in files}
    for c in candidates:
        found = lower.get(c.lower())
        if found:
            return found
    return None


def softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim == 1:
        x = np.stack([-x, x], axis=1)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / np.sum(p, axis=1, keepdims=True)
    return -np.sum(p * np.log2(p), axis=1)


def load_prediction_data(project_root: Path, artifact: ArtifactInfo) -> tuple[PredictionData | None, ExtractionWarning | None]:
    if not artifact.prediction_npz:
        return None, None

    path = project_root / artifact.prediction_npz
    try:
        with np.load(path, allow_pickle=True) as d:
            files = list(d.files)
            state_key = npz_key(
                files,
                (
                    "hard_state",
                    "hard_states",
                    "pred_state",
                    "pred_states",
                    "state",
                    "states",
                    "teacher_state",
                    "teacher_states",
                    "y_pred",
                    "labels",
                ),
            )
            probs_key = npz_key(
                files,
                (
                    "state_probs",
                    "probs",
                    "probabilities",
                    "posterior_probs",
                    "posterior",
                    "teacher_probs",
                    "pred_probs",
                    "y_prob",
                    "softmax_probs",
                ),
            )
            logits_key = npz_key(files, ("logits", "pred_logits", "y_logits"))
            confidence_key = npz_key(files, ("confidence", "confidences", "max_prob"))
            sample_key = npz_key(files, ("sample_index", "sample_indices", "index", "indices"))
            sequence_key = npz_key(files, ("sequence_id", "sequence_ids", "seq_id", "run_id"))

            state_probs: np.ndarray | None = None
            if probs_key:
                state_probs = np.asarray(d[probs_key], dtype=np.float64)
            elif logits_key:
                state_probs = softmax(np.asarray(d[logits_key], dtype=np.float64))

            if state_probs is not None and state_probs.ndim == 1:
                state_probs = np.stack([1.0 - state_probs, state_probs], axis=1)

            if state_key:
                hard_state = np.asarray(d[state_key], dtype=np.int64).ravel()
            elif state_probs is not None:
                hard_state = np.argmax(state_probs, axis=1).astype(np.int64)
            else:
                return None, ExtractionWarning(
                    "prediction_unreadable",
                    f"No hard-state/probability/logit array found. NPZ keys={files}",
                    artifact.prediction_npz,
                )

            n = int(hard_state.size)
            sample_index = (
                np.asarray(d[sample_key], dtype=np.int64).ravel()
                if sample_key
                else np.arange(n, dtype=np.int64)
            )
            sequence_id = np.asarray(d[sequence_key], dtype=np.int64).ravel() if sequence_key else None
            confidence = np.asarray(d[confidence_key], dtype=np.float64).ravel() if confidence_key else None

        if state_probs is not None and state_probs.shape[0] != n:
            state_probs = None
        if confidence is not None and confidence.size != n:
            confidence = None
        if confidence is None and state_probs is not None:
            confidence = np.max(state_probs, axis=1)
        entropy = entropy_from_probs(state_probs) if state_probs is not None else None
        if sample_index.size != n:
            sample_index = np.arange(n, dtype=np.int64)
        if sequence_id is not None and sequence_id.size != n:
            sequence_id = None

        return (
            PredictionData(
                artifact=artifact,
                sample_index=sample_index,
                hard_state=hard_state,
                sequence_id=sequence_id,
                state_probs=state_probs,
                confidence=confidence,
                entropy_bits=entropy,
            ),
            None,
        )
    except Exception as exc:
        return None, ExtractionWarning("prediction_load_failed", str(exc), artifact.prediction_npz)


def prediction_metric_row(pred: PredictionData) -> dict[str, Any]:
    a = pred.artifact
    hard = pred.hard_state
    n = int(hard.size)
    probs = pred.state_probs
    conf = pred.confidence
    ent = pred.entropy_bits
    num_states = int(max(int(hard.max()) + 1 if n else 0, probs.shape[1] if probs is not None else 2))
    counts = np.bincount(np.clip(hard, 0, num_states - 1), minlength=num_states)
    occ = counts / max(n, 1)

    row: dict[str, Any] = {
        "bundle": a.bundle,
        "family": a.family,
        "stage": a.stage,
        "model": a.model,
        "run_dir": a.run_dir,
        "source_file": a.prediction_npz,
        "n_predictions": n,
        "num_states": num_states,
        "sample_index_min": int(np.min(pred.sample_index)) if pred.sample_index.size else math.nan,
        "sample_index_max": int(np.max(pred.sample_index)) if pred.sample_index.size else math.nan,
        "has_sequence_id": pred.sequence_id is not None,
        "has_state_probs": probs is not None,
        "has_confidence": conf is not None,
    }

    if conf is not None and conf.size:
        row.update(
            {
                "mean_confidence": float(np.mean(conf)),
                "median_confidence": float(np.median(conf)),
                "p05_confidence": float(np.quantile(conf, 0.05)),
                "p95_confidence": float(np.quantile(conf, 0.95)),
            }
        )
    else:
        row.update(
            {
                "mean_confidence": math.nan,
                "median_confidence": math.nan,
                "p05_confidence": math.nan,
                "p95_confidence": math.nan,
            }
        )

    if ent is not None and ent.size:
        row["mean_entropy_bits"] = float(np.mean(ent))
        row["median_entropy_bits"] = float(np.median(ent))
    else:
        row["mean_entropy_bits"] = math.nan
        row["median_entropy_bits"] = math.nan

    for k in range(num_states):
        row[f"state_{k}_occupancy"] = float(occ[k])
    if probs is not None:
        for k in range(probs.shape[1]):
            row[f"mean_prob_state_{k}"] = float(np.mean(probs[:, k]))

    if pred.sequence_id is not None:
        order = np.lexsort((pred.sample_index, pred.sequence_id))
        hard_o = hard[order]
        seq_o = pred.sequence_id[order]
        same_seq = seq_o[1:] == seq_o[:-1]
        switch_mask = np.logical_and(same_seq, hard_o[1:] != hard_o[:-1])
        denom = int(np.sum(same_seq))
        row["window_switch_count"] = int(np.sum(switch_mask))
        row["window_switch_rate"] = float(np.sum(switch_mask) / denom) if denom else math.nan
        row["n_sequences"] = int(np.unique(seq_o).size)
    else:
        switch_count = int(np.sum(hard[1:] != hard[:-1])) if n > 1 else 0
        row["window_switch_count"] = switch_count
        row["window_switch_rate"] = float(switch_count / max(n - 1, 1)) if n > 1 else math.nan
        row["n_sequences"] = math.nan

    return row


def extract_prediction_rows(
    project_root: Path,
    artifacts: list[ArtifactInfo],
) -> tuple[list[dict[str, Any]], list[ExtractionWarning], list[PredictionData]]:
    rows: list[dict[str, Any]] = []
    warnings: list[ExtractionWarning] = []
    predictions: list[PredictionData] = []

    for artifact in artifacts:
        pred, warning = load_prediction_data(project_root, artifact)
        if warning:
            warnings.append(warning)
        if pred is None:
            continue
        predictions.append(pred)
        rows.append(prediction_metric_row(pred))

    return rows, warnings, predictions


def get_nested_rates(summary: dict[str, Any]) -> dict[str, float]:
    src = summary.get("rates_and_lifetimes") if isinstance(summary.get("rates_and_lifetimes"), dict) else {}
    rates = summary.get("rates") if isinstance(summary.get("rates"), dict) else {}
    lifetimes = summary.get("lifetimes") if isinstance(summary.get("lifetimes"), dict) else {}
    out = {
        "gamma_01": to_float(src.get("gamma_01", rates.get("gamma_01", summary.get("gamma_01")))),
        "gamma_10": to_float(src.get("gamma_10", rates.get("gamma_10", summary.get("gamma_10")))),
        "tau_0": to_float(src.get("tau_0", lifetimes.get("tau_0", summary.get("tau_0")))),
        "tau_1": to_float(src.get("tau_1", lifetimes.get("tau_1", summary.get("tau_1")))),
        "tau_mean": to_float(src.get("tau_mean", lifetimes.get("tau_mean", summary.get("tau_mean")))),
    }
    if not math.isfinite(out["tau_0"]) and math.isfinite(out["gamma_01"]) and out["gamma_01"] > 0:
        out["tau_0"] = 1.0 / out["gamma_01"]
    if not math.isfinite(out["tau_1"]) and math.isfinite(out["gamma_10"]) and out["gamma_10"] > 0:
        out["tau_1"] = 1.0 / out["gamma_10"]
    if not math.isfinite(out["tau_mean"]) and math.isfinite(out["tau_0"]) and math.isfinite(out["tau_1"]):
        out["tau_mean"] = 0.5 * (out["tau_0"] + out["tau_1"])
    return out


def find_rate_matrix(obj: Any) -> np.ndarray | None:
    if isinstance(obj, dict):
        for key in ("Q", "rate_matrix", "generator_matrix", "transition_rate_matrix", "gamma"):
            if key in obj:
                try:
                    mat = np.asarray(obj[key], dtype=np.float64)
                    if mat.ndim == 2 and mat.shape[0] >= 2 and mat.shape[1] >= 2:
                        return mat
                except Exception:
                    pass
        for value in obj.values():
            found = find_rate_matrix(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_rate_matrix(value)
            if found is not None:
                return found
    return None


def extract_cthmm_rows(
    project_root: Path,
    artifacts: list[ArtifactInfo],
    dt_by_family: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for artifact in artifacts:
        candidate_files = [artifact.qcvv_summary_json, artifact.cthmm_model_json, artifact.metrics_json]
        source_path: str | None = None
        summary: dict[str, Any] | None = None
        for rel in candidate_files:
            if not rel:
                continue
            try:
                data = read_json(project_root / rel)
            except Exception:
                continue
            rates = get_nested_rates(data)
            mat = find_rate_matrix(data)
            if mat is not None:
                rates["gamma_01"] = float(mat[0, 1])
                rates["gamma_10"] = float(mat[1, 0])
                if rates["gamma_01"] > 0:
                    rates["tau_0"] = 1.0 / rates["gamma_01"]
                if rates["gamma_10"] > 0:
                    rates["tau_1"] = 1.0 / rates["gamma_10"]
                if math.isfinite(rates["tau_0"]) and math.isfinite(rates["tau_1"]):
                    rates["tau_mean"] = 0.5 * (rates["tau_0"] + rates["tau_1"])
            if any(math.isfinite(rates[k]) for k in ("gamma_01", "gamma_10", "tau_0", "tau_1")):
                summary = data
                source_path = rel
                break

        if summary is None or source_path is None:
            continue

        rates = get_nested_rates(summary)
        mat = find_rate_matrix(summary)
        if mat is not None:
            rates["gamma_01"] = float(mat[0, 1])
            rates["gamma_10"] = float(mat[1, 0])
            if rates["gamma_01"] > 0:
                rates["tau_0"] = 1.0 / rates["gamma_01"]
            if rates["gamma_10"] > 0:
                rates["tau_1"] = 1.0 / rates["gamma_10"]
            if math.isfinite(rates["tau_0"]) and math.isfinite(rates["tau_1"]):
                rates["tau_mean"] = 0.5 * (rates["tau_0"] + rates["tau_1"])

        dt = to_float(dt_by_family.get(artifact.family))
        rows.append(
            {
                "bundle": artifact.bundle,
                "family": artifact.family,
                "stage": artifact.stage,
                "model": artifact.model,
                "run_dir": artifact.run_dir,
                "gamma_01": rates["gamma_01"],
                "gamma_10": rates["gamma_10"],
                "tau_0_model_units": rates["tau_0"],
                "tau_1_model_units": rates["tau_1"],
                "tau_mean_model_units": rates["tau_mean"],
                "tau_asymmetry_tau0_over_tau1": safe_ratio(rates["tau_0"], rates["tau_1"]),
                "physical_dt": dt,
                "tau_0_physical": rates["tau_0"] * dt if math.isfinite(rates["tau_0"]) and math.isfinite(dt) else math.nan,
                "tau_1_physical": rates["tau_1"] * dt if math.isfinite(rates["tau_1"]) and math.isfinite(dt) else math.nan,
                "tau_mean_physical": rates["tau_mean"] * dt if math.isfinite(rates["tau_mean"]) and math.isfinite(dt) else math.nan,
                "train_log_likelihood_per_obs": to_float(summary.get("train_log_likelihood_per_obs")),
                "source_file": source_path,
            }
        )

    return rows


def dwell_rows_from_predictions(
    predictions: list[PredictionData],
    dt_by_family: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for pred in predictions:
        a = pred.artifact
        hard = pred.hard_state
        if hard.size == 0:
            continue
        dt = to_float(dt_by_family.get(a.family))

        if pred.sequence_id is not None:
            order = np.lexsort((pred.sample_index, pred.sequence_id))
            hard_o = hard[order]
            seq_o = pred.sequence_id[order]
        else:
            order = np.argsort(pred.sample_index)
            hard_o = hard[order]
            seq_o = np.zeros_like(hard_o)

        per_state: dict[int, list[int]] = {}
        current_seq = int(seq_o[0])
        current_state = int(hard_o[0])
        length = 1
        for s_value, h_value in zip(seq_o[1:], hard_o[1:]):
            s_int = int(s_value)
            h_int = int(h_value)
            if s_int == current_seq and h_int == current_state:
                length += 1
            else:
                per_state.setdefault(current_state, []).append(length)
                current_seq = s_int
                current_state = h_int
                length = 1
        per_state.setdefault(current_state, []).append(length)

        for state, lengths in sorted(per_state.items()):
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


def align_pair(left: PredictionData, right: PredictionData, min_id_overlap: float) -> tuple[np.ndarray, np.ndarray, str, str | None]:
    li = np.asarray(left.sample_index, dtype=np.int64)
    ri = np.asarray(right.sample_index, dtype=np.int64)
    n_min = min(li.size, ri.size)
    if n_min == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), "none", "empty prediction array"

    common, lpos, rpos = np.intersect1d(li, ri, assume_unique=False, return_indices=True)
    overlap = common.size / max(n_min, 1)
    if overlap >= min_id_overlap:
        order = np.argsort(common)
        return lpos[order].astype(np.int64), rpos[order].astype(np.int64), "sample_index", None
    if li.size == ri.size:
        pos = np.arange(li.size, dtype=np.int64)
        return pos, pos, "position", f"low sample_index overlap ({overlap:.3f}); used array position"
    return (
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        "none",
        f"low sample_index overlap ({overlap:.3f}) and unequal lengths ({li.size} vs {ri.size})",
    )


def js_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64) + 1e-12
    q = np.asarray(q, dtype=np.float64) + 1e-12
    p = p / np.sum(p, axis=1, keepdims=True)
    q = q / np.sum(q, axis=1, keepdims=True)
    m = 0.5 * (p + q)
    left = np.sum(p * (np.log2(p) - np.log2(m)), axis=1)
    right = np.sum(q * (np.log2(q) - np.log2(m)), axis=1)
    return float(np.mean(0.5 * left + 0.5 * right))


def prediction_pair_row(
    left: PredictionData,
    right: PredictionData,
    min_id_overlap: float,
) -> tuple[dict[str, Any] | None, ExtractionWarning | None]:
    la = left.artifact
    ra = right.artifact
    lpos, rpos, mode, note = align_pair(left, right, min_id_overlap)
    warning = ExtractionWarning("alignment", note, f"{la.prediction_npz} vs {ra.prediction_npz}") if note else None
    if lpos.size == 0:
        return None, warning

    lh = left.hard_state[lpos]
    rh = right.hard_state[rpos]
    row: dict[str, Any] = {
        "left_bundle": la.bundle,
        "right_bundle": ra.bundle,
        "bundle": la.bundle if la.bundle == ra.bundle else "cross_bundle",
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

    lp = left.state_probs
    rp = right.state_probs
    if lp is not None and rp is not None and lp.ndim == 2 and rp.ndim == 2 and lp.shape[1] == rp.shape[1]:
        row["js_divergence_bits"] = js_divergence_bits(lp[lpos], rp[rpos])
        row["mean_l1_probability_distance"] = float(np.mean(np.sum(np.abs(lp[lpos] - rp[rpos]), axis=1)))
    else:
        row["js_divergence_bits"] = math.nan
        row["mean_l1_probability_distance"] = math.nan

    return row, warning


def compare_predictions(
    predictions: list[PredictionData],
    min_id_overlap: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[ExtractionWarning]]:
    within: list[dict[str, Any]] = []
    cross: list[dict[str, Any]] = []
    warnings: list[ExtractionWarning] = []

    by_bundle_family: dict[tuple[str, str], list[PredictionData]] = {}
    by_family_model: dict[tuple[str, str], list[PredictionData]] = {}
    for pred in predictions:
        a = pred.artifact
        by_bundle_family.setdefault((a.bundle, a.family), []).append(pred)
        by_family_model.setdefault((a.family, a.model), []).append(pred)

    for group in by_bundle_family.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                row, warn = prediction_pair_row(group[i], group[j], min_id_overlap)
                if warn:
                    warnings.append(warn)
                if row:
                    within.append(row)

    for group in by_family_model.values():
        pipelines = [p for p in group if p.artifact.bundle == "pipeline"]
        individuals = [p for p in group if p.artifact.bundle == "individual"]
        for p in pipelines:
            for i in individuals:
                row, warn = prediction_pair_row(p, i, min_id_overlap)
                if warn:
                    warnings.append(warn)
                if row:
                    cross.append(row)

    return within, cross, warnings


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
                summary = read_json(project_root / a.summary_json)
                for key, value in summary.items():
                    if isinstance(value, (int, float, np.number)):
                        row[key] = to_float(value)
                test_metrics = summary.get("test_metrics")
                if isinstance(test_metrics, dict):
                    for key, value in test_metrics.items():
                        if isinstance(value, (int, float, np.number)):
                            row[f"test_{key}"] = to_float(value)
                row["teacher_npz"] = summary.get("teacher_npz")
                row["uses_teacher"] = bool(summary.get("teacher_npz"))
            except Exception as exc:
                row["summary_error"] = str(exc)

        if a.metrics_json:
            try:
                metrics = read_json(project_root / a.metrics_json)
                for key, value in metrics.items():
                    if isinstance(value, (int, float, np.number)):
                        row[key] = to_float(value)
                fit = metrics.get("fit_result")
                if isinstance(fit, dict):
                    row["fit_n_iter"] = fit.get("n_iter")
                    row["fit_converged"] = fit.get("converged")
            except Exception as exc:
                row["metrics_json_error"] = str(exc)

        if a.metrics_csv:
            try:
                if pd is not None:
                    df = pd.read_csv(project_root / a.metrics_csv)
                    row["metrics_csv_rows"] = int(len(df))
                    for col in df.columns:
                        if str(col).lower() in {"epoch", "step"}:
                            continue
                        vals = pd.to_numeric(df[col], errors="coerce").dropna()
                        if len(vals):
                            row[f"last_{col}"] = float(vals.iloc[-1])
                            row[f"best_{col}"] = float(vals.min()) if "loss" in str(col).lower() else float(vals.max())
                else:
                    with (project_root / a.metrics_csv).open(newline="", encoding="utf-8") as f:
                        row["metrics_csv_rows"] = max(sum(1 for _ in csv.reader(f)) - 1, 0)
            except Exception as exc:
                row["metrics_csv_error"] = str(exc)

        useful_keys = set(row) - {
            "bundle",
            "family",
            "stage",
            "model",
            "run_dir",
            "has_checkpoint",
            "summary_json",
            "metrics_json",
            "metrics_csv",
        }
        if useful_keys:
            rows.append(row)

    return rows


def extract_drift_rows(predictions: list[PredictionData], n_bins: int) -> list[dict[str, Any]]:
    if n_bins <= 1:
        return []
    rows: list[dict[str, Any]] = []

    for pred in predictions:
        a = pred.artifact
        n = pred.hard_state.size
        if n < n_bins:
            continue
        order = np.lexsort((pred.sample_index, pred.sequence_id)) if pred.sequence_id is not None else np.argsort(pred.sample_index)
        chunks = np.array_split(order, n_bins)
        num_states = int(max(int(pred.hard_state.max()) + 1 if n else 0, pred.state_probs.shape[1] if pred.state_probs is not None else 2))
        for bin_id, idx in enumerate(chunks):
            if idx.size == 0:
                continue
            hard = pred.hard_state[idx]
            row: dict[str, Any] = {
                "bundle": a.bundle,
                "family": a.family,
                "stage": a.stage,
                "model": a.model,
                "bin": int(bin_id),
                "n_bin_predictions": int(idx.size),
                "sample_index_min": int(np.min(pred.sample_index[idx])),
                "sample_index_max": int(np.max(pred.sample_index[idx])),
                "source_file": a.prediction_npz,
            }
            for state in range(num_states):
                row[f"state_{state}_occupancy"] = float(np.mean(hard == state))
            if pred.confidence is not None:
                row["mean_confidence"] = float(np.mean(pred.confidence[idx]))
            else:
                row["mean_confidence"] = math.nan
            if pred.entropy_bits is not None:
                row["mean_entropy_bits"] = float(np.mean(pred.entropy_bits[idx]))
            else:
                row["mean_entropy_bits"] = math.nan
            if idx.size > 1:
                row["switch_rate"] = float(np.mean(hard[1:] != hard[:-1]))
            else:
                row["switch_rate"] = math.nan
            rows.append(row)

    return rows


def paired_bundle_comparison(
    cthmm_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_pairs(items: list[dict[str, Any]], keys: list[str], metrics: list[str], comparison_type: str) -> None:
        grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
        for item in items:
            bundle = item.get("bundle")
            if bundle not in {"pipeline", "individual"}:
                continue
            key = tuple(item.get(k) for k in keys)
            grouped.setdefault(key, {})[str(bundle)] = item
        for key, pair in grouped.items():
            if "pipeline" not in pair or "individual" not in pair:
                continue
            p = pair["pipeline"]
            i = pair["individual"]
            base = {"comparison_type": comparison_type}
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

    add_pairs(
        cthmm_rows,
        ["family", "model"],
        ["gamma_01", "gamma_10", "tau_0_model_units", "tau_1_model_units", "tau_mean_model_units", "tau_asymmetry_tau0_over_tau1"],
        "cthmm_lifetimes",
    )
    add_pairs(
        prediction_rows,
        ["family", "model"],
        ["mean_confidence", "median_confidence", "mean_entropy_bits", "state_0_occupancy", "state_1_occupancy", "window_switch_rate", "n_predictions"],
        "prediction_metrics",
    )
    add_pairs(
        training_rows,
        ["family", "model"],
        ["test_loss", "best_selection_score", "last_train_loss", "last_val_loss", "best_val_loss", "last_loss", "best_loss"],
        "training_metrics",
    )
    return rows


def family_summary_rows(
    cthmm_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    agreement_rows: list[dict[str, Any]],
    cross_rows: list[dict[str, Any]],
    families: Sequence[str],
    bundles: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        for family in families:
            c = [r for r in cthmm_rows if r.get("bundle") == bundle and r.get("family") == family]
            p = [r for r in prediction_rows if r.get("bundle") == bundle and r.get("family") == family]
            a = [r for r in agreement_rows if r.get("bundle") == bundle and r.get("family") == family]
            g = [r for r in cross_rows if r.get("family") == family]
            rows.append(
                {
                    "bundle": bundle,
                    "family": family,
                    "n_cthmm_lifetime_rows": len(c),
                    "n_prediction_models": len(p),
                    "n_within_bundle_agreements": len(a),
                    "mean_prediction_confidence": safe_mean(r.get("mean_confidence") for r in p),
                    "mean_prediction_entropy_bits": safe_mean(r.get("mean_entropy_bits") for r in p),
                    "mean_within_bundle_agreement": safe_mean(r.get("hard_state_agreement") for r in a),
                    "mean_cross_bundle_same_model_agreement": safe_mean(r.get("hard_state_agreement") for r in g),
                    "cthmm_tau_mean_model_units": safe_mean(r.get("tau_mean_model_units") for r in c),
                }
            )
    return rows


def scorecard_rows(
    family_summary: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    agreement_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in family_summary:
        bundle = summary["bundle"]
        family = summary["family"]
        preds = [r for r in prediction_rows if r.get("bundle") == bundle and r.get("family") == family]
        ag = [r for r in agreement_rows if r.get("bundle") == bundle and r.get("family") == family]
        best_conf = max([to_float(r.get("mean_confidence")) for r in preds] or [math.nan])
        mean_agree = safe_mean(r.get("hard_state_agreement") for r in ag)
        has_cthmm = int(summary.get("n_cthmm_lifetime_rows", 0) > 0)
        has_multimodel = int(len(preds) >= 2)
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
                "qcvv_model_readiness_score_0_to_100": float(score),
                "interpretation": interpretation,
                "has_cthmm_lifetimes": bool(has_cthmm),
                "has_multiple_prediction_models": bool(has_multimodel),
                "mean_within_bundle_agreement": mean_agree,
                "best_model_confidence": best_conf,
                "n_prediction_models": len(preds),
                "caution": "This score summarizes artifact completeness/model consistency only; it is not physical fidelity or certification.",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_physical_dt(items: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected --physical-dt family=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in FAMILIES:
            raise ValueError(f"Unknown family in --physical-dt: {key!r}")
        out[key] = float(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract QCVV metrics directly from pipeline/ and individual/ trained runs.")
    parser.add_argument("--project-root", default=".", help="Repository root containing individual/ and pipeline/.")
    parser.add_argument("--out", default="qcvv_outputs_direct", help="Output directory for extracted CSV/JSON files.")
    parser.add_argument("--bundles", nargs="+", default=["pipeline", "individual"], choices=["pipeline", "individual"])
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    parser.add_argument("--all-runs", action="store_true", help="Extract all run folders instead of only the newest run per stage.")
    parser.add_argument("--min-id-overlap", type=float, default=0.50, help="Minimum sample_index overlap before alignment by sample_index.")
    parser.add_argument("--drift-bins", type=int, default=10, help="Number of ordered bins for drift metrics. Use 0 or 1 to disable.")
    parser.add_argument("--physical-dt", action="append", default=[], help="Optional family timestep, e.g. --physical-dt parity=1e-6")
    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    out_dir = Path(args.out).expanduser()
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dt_by_family = parse_physical_dt(args.physical_dt)

    artifacts, warnings = discover_artifacts(
        project_root=project_root,
        bundles=args.bundles,
        families=args.families,
        latest_only=not args.all_runs,
    )
    prediction_rows, prediction_warnings, predictions = extract_prediction_rows(project_root, artifacts)
    warnings.extend(prediction_warnings)
    cthmm_rows = extract_cthmm_rows(project_root, artifacts, dt_by_family)
    dwell_rows = dwell_rows_from_predictions(predictions, dt_by_family)
    within_agreements, cross_agreements, agreement_warnings = compare_predictions(predictions, args.min_id_overlap)
    warnings.extend(agreement_warnings)
    training_rows = extract_training_rows(project_root, artifacts)
    drift_rows = extract_drift_rows(predictions, args.drift_bins)
    bundle_comparison = paired_bundle_comparison(cthmm_rows, prediction_rows, training_rows)
    family_summary = family_summary_rows(cthmm_rows, prediction_rows, within_agreements, cross_agreements, args.families, args.bundles)
    scorecard = scorecard_rows(family_summary, prediction_rows, within_agreements)

    write_csv(out_dir / "artifact_inventory.csv", [asdict(a) for a in artifacts])
    write_csv(out_dir / "cthmm_lifetimes.csv", cthmm_rows)
    write_csv(out_dir / "prediction_metrics.csv", prediction_rows)
    write_csv(out_dir / "dwell_metrics.csv", dwell_rows)
    write_csv(out_dir / "model_agreements.csv", within_agreements)
    write_csv(out_dir / "cross_bundle_model_agreements.csv", cross_agreements)
    write_csv(out_dir / "training_metrics.csv", training_rows)
    write_csv(out_dir / "drift_metrics.csv", drift_rows)
    write_csv(out_dir / "bundle_comparison.csv", bundle_comparison)
    write_csv(out_dir / "family_summary.csv", family_summary)
    write_csv(out_dir / "qcvv_scorecard.csv", scorecard)
    write_csv(out_dir / "extraction_warnings.csv", [asdict(w) for w in warnings])

    summary = {
        "project_root": str(project_root),
        "out_dir": str(out_dir),
        "bundles": list(args.bundles),
        "families": list(args.families),
        "latest_only": not args.all_runs,
        "n_artifacts": len(artifacts),
        "n_prediction_rows": len(prediction_rows),
        "n_cthmm_rows": len(cthmm_rows),
        "n_dwell_rows": len(dwell_rows),
        "n_within_bundle_agreements": len(within_agreements),
        "n_cross_bundle_agreements": len(cross_agreements),
        "n_training_rows": len(training_rows),
        "n_drift_rows": len(drift_rows),
        "n_warnings": len(warnings),
        "outputs": {
            "artifact_inventory": "artifact_inventory.csv",
            "cthmm_lifetimes": "cthmm_lifetimes.csv",
            "prediction_metrics": "prediction_metrics.csv",
            "dwell_metrics": "dwell_metrics.csv",
            "model_agreements": "model_agreements.csv",
            "cross_bundle_model_agreements": "cross_bundle_model_agreements.csv",
            "training_metrics": "training_metrics.csv",
            "drift_metrics": "drift_metrics.csv",
            "bundle_comparison": "bundle_comparison.csv",
            "family_summary": "family_summary.csv",
            "scorecard": "qcvv_scorecard.csv",
            "warnings": "extraction_warnings.csv",
        },
    }
    with (out_dir / "qcvv_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
