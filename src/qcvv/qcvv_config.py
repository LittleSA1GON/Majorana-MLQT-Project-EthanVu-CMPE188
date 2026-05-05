import argparse
import json
import math
from pathlib import Path
from typing import Any

FAMILIES = ["parity", "x_loop", "z_loop"]


def default_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "description": "Physical metadata for parity/tetron QCVV. Replace null values before making physical claims.",
        "families": {
            "parity": {
                "dt_seconds": None,
                "dt_description": "Physical time represented by one model/sample step for parity predictions.",
                "state_labels": {"0": "unknown_state_0", "1": "unknown_state_1"},
                "physical_observable": "even_odd_parity",
            },
            "x_loop": {
                "dt_seconds": None,
                "dt_description": "Physical time represented by one model/sample step for X-loop predictions.",
                "state_labels": {"0": "unknown_state_0", "1": "unknown_state_1"},
                "physical_observable": "x_loop_state",
            },
            "z_loop": {
                "dt_seconds": None,
                "dt_description": "Physical time represented by one model/sample step for Z-loop predictions.",
                "state_labels": {"0": "unknown_state_0", "1": "unknown_state_1"},
                "physical_observable": "z_loop_state",
            },
        },
        "thresholds": {
            "minimum_readout_fidelity": None,
            "minimum_qnd_repeatability": None,
            "maximum_assignment_error": None,
            "maximum_drift_fraction": None,
            "minimum_cross_bundle_agreement": None,
        },
        "data_sources": {
            "raw_iq_traces": None,
            "calibration_labels": None,
            "repeated_readout_records": None,
            "run_metadata": None,
        },
        "claim_level_notes": {
            "level_1_model_assisted": "Allowed with trained artifacts only.",
            "level_2_calibrated": "Requires dt_seconds and calibration labels.",
            "level_3_device_validated": "Requires raw/prepared traces, run metadata, drift checks, repeated-readout or pre/post data.",
            "level_4_certification_style": "Requires predeclared thresholds and confidence intervals.",
        },
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def is_positive_number(x: Any) -> bool:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(val) and val > 0


def validate_config(config: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return (errors, warnings)."""
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    families = config.get("families")
    if not isinstance(families, dict):
        errors.append({"field": "families", "issue": "missing or not a dictionary"})
        return errors, warnings

    for family in FAMILIES:
        fam = families.get(family)
        if not isinstance(fam, dict):
            errors.append({"field": f"families.{family}", "issue": "missing family configuration"})
            continue
        dt = fam.get("dt_seconds")
        if dt is None:
            warnings.append({"field": f"families.{family}.dt_seconds", "issue": "missing; lifetimes remain in model units"})
        elif not is_positive_number(dt):
            errors.append({"field": f"families.{family}.dt_seconds", "issue": "must be a positive finite number or null"})

        labels = fam.get("state_labels")
        if not isinstance(labels, dict):
            errors.append({"field": f"families.{family}.state_labels", "issue": "missing or not a dictionary"})
        else:
            for state in ["0", "1"]:
                label = labels.get(state)
                if not isinstance(label, str) or not label.strip():
                    errors.append({"field": f"families.{family}.state_labels.{state}", "issue": "missing physical label"})
                elif label.startswith("unknown"):
                    warnings.append({"field": f"families.{family}.state_labels.{state}", "issue": "still unknown; hidden states are not physically mapped"})

    thresholds = config.get("thresholds", {})
    if isinstance(thresholds, dict):
        for key, value in thresholds.items():
            if value is not None:
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    errors.append({"field": f"thresholds.{key}", "issue": "threshold must be numeric or null"})
                    continue
                if not math.isfinite(val):
                    errors.append({"field": f"thresholds.{key}", "issue": "threshold must be finite"})
    else:
        errors.append({"field": "thresholds", "issue": "must be a dictionary"})

    return errors, warnings


def export_extract_inputs(config: dict[str, Any], out_dir: Path) -> tuple[Path, Path, Path]:
    families = config.get("families", {})
    dt_json: dict[str, float] = {}
    labels_json: dict[str, dict[str, str]] = {}
    for family, fam in families.items():
        if not isinstance(fam, dict):
            continue
        dt = fam.get("dt_seconds")
        if dt is not None and is_positive_number(dt):
            dt_json[family] = float(dt)
        labels = fam.get("state_labels")
        if isinstance(labels, dict):
            labels_json[family] = {str(k): str(v) for k, v in labels.items()}
    dt_path = out_dir / "dt.json"
    labels_path = out_dir / "state_labels.json"
    thresholds_path = out_dir / "qcvv_thresholds.json"
    write_json(dt_path, dt_json)
    write_json(labels_path, labels_json)
    write_json(thresholds_path, config.get("thresholds", {}))
    return dt_path, labels_path, thresholds_path


def write_validation_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("field,issue\n")
        for row in rows:
            field = row.get("field", "").replace('"', '""')
            issue = row.get("issue", "").replace('"', '""')
            f.write(f'"{field}","{issue}"\n')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create/validate/export QCVV physical metadata configuration.")
    parser.add_argument("--create-template", default=None, help="Write a template config JSON to this path.")
    parser.add_argument("--config", default=None, help="Existing config JSON to validate/export.")
    parser.add_argument("--export", default=None, help="Output directory for dt.json/state_labels.json/thresholds JSON.")
    args = parser.parse_args(argv)

    if args.create_template:
        path = Path(args.create_template).resolve()
        write_json(path, default_config())
        print(f"Wrote QCVV config template: {path}")

    if args.config:
        path = Path(args.config).resolve()
        config = load_json(path)
        errors, warnings = validate_config(config)
        print(f"Validated config: {path}")
        print(f"Errors: {len(errors)} | Warnings: {len(warnings)}")
        export_dir = Path(args.export).resolve() if args.export else path.parent
        write_validation_csv(export_dir / "qcvv_config_errors.csv", errors)
        write_validation_csv(export_dir / "qcvv_config_warnings.csv", warnings)
        if args.export:
            dt_path, labels_path, thresholds_path = export_extract_inputs(config, export_dir)
            print(f"Exported: {dt_path}")
            print(f"Exported: {labels_path}")
            print(f"Exported: {thresholds_path}")
        if errors:
            return 2

    if not args.create_template and not args.config:
        parser.error("Provide --create-template and/or --config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
