import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np

DEFAULT_PARITY_DIR = Path(
    r"E:\Software Engineering Stuff\Quantum\Majorana-MLQT-Project-EthanVu-CMPE188\manual-data\parity"
)
DEFAULT_OUTPUT_NAME = "parity_combined.h5"

EXPECTED_FILES: Dict[str, str] = {
    "mpr_A1": "mpr_A1_Cq.h5",
    "mpr_A2": "mpr_A2_Cq.h5",
    "mpr_B1": "mpr_B1_Cq.h5",
}

EXPECTED_TOP_LEVEL_NAMES = ["Cq", "iCq", "time"]
OPTIONAL_STANDARD_NAMES = [
    "B_perp",
    "Bperp",
    "V_qd_1_plunger_gate",
    "V_qd_3_plunger_gate",
    "V_lin_qd",
    "V_wire",
]


def _jsonable_attr_value(value):
    """Convert an HDF5 attribute value into something JSON-safe."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return repr(value)
    if isinstance(value, np.ndarray):
        try:
            return value.tolist()
        except Exception:
            return repr(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _copy_attrs(src_obj, dst_obj) -> None:
    for key, value in src_obj.attrs.items():
        try:
            dst_obj.attrs[key] = value
        except Exception:
            dst_obj.attrs[key] = repr(value)


def _copy_all_top_level_objects(src_h5: h5py.File, dst_group: h5py.Group) -> None:
    """Copy every top-level object from one HDF5 file into a destination group."""
    for name in src_h5.keys():
        src_h5.copy(name, dst_group, name=name)


def _collect_dataset_paths(h5obj: h5py.Group) -> List[str]:
    paths: List[str] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    h5obj.visititems(visitor)
    return sorted(paths)


def _collect_group_paths(h5obj: h5py.Group) -> List[str]:
    paths: List[str] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            paths.append(name)

    h5obj.visititems(visitor)
    return sorted(paths)


def _summarize_source_file(src_path: Path) -> Dict[str, object]:
    with h5py.File(src_path, "r") as src:
        top_level = sorted(list(src.keys()))
        datasets = _collect_dataset_paths(src)
        groups = _collect_group_paths(src)
        attrs = {k: _jsonable_attr_value(v) for k, v in src.attrs.items()}

        summary = {
            "file": str(src_path),
            "top_level_names": top_level,
            "dataset_paths": datasets,
            "group_paths": groups,
            "root_attrs": attrs,
            "has_expected_top_level": {name: (name in src) for name in EXPECTED_TOP_LEVEL_NAMES},
            "has_optional_standard_names": {name: (name in src) for name in OPTIONAL_STANDARD_NAMES},
        }
        return summary


def _write_run_group(dst_h5: h5py.File, run_name: str, src_path: Path) -> None:
    print(f"  - Copying {src_path.name} into group '{run_name}'")

    with h5py.File(src_path, "r") as src:
        run_group = dst_h5.create_group(run_name)
        run_group.attrs["family"] = "parity"
        run_group.attrs["run_name"] = run_name
        run_group.attrs["source_file"] = str(src_path)
        run_group.attrs["combined_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_group.attrs["copied_with"] = "h5py_only"

        root_attrs = {k: _jsonable_attr_value(v) for k, v in src.attrs.items()}
        run_group.attrs["source_root_attrs_json"] = json.dumps(root_attrs, sort_keys=True)

        _copy_all_top_level_objects(src, run_group)
        _copy_attrs(src, run_group)

        # Add a compact manifest for debugging and future loading.
        manifest = {
            "top_level_names": sorted(list(src.keys())),
            "dataset_paths": _collect_dataset_paths(src),
            "group_paths": _collect_group_paths(src),
            "has_expected_top_level": {name: (name in src) for name in EXPECTED_TOP_LEVEL_NAMES},
            "has_optional_standard_names": {name: (name in src) for name in OPTIONAL_STANDARD_NAMES},
        }
        run_group.attrs["manifest_json"] = json.dumps(manifest, sort_keys=True)


def _verify_combined_file(output_path: Path, expected_runs: Iterable[str]) -> None:
    with h5py.File(output_path, "r") as h5:
        if "runs" not in h5 or "sources" not in h5:
            raise RuntimeError("Missing root index datasets 'runs' or 'sources'.")

        for run_name in expected_runs:
            if run_name not in h5:
                raise RuntimeError(f"Missing run group '{run_name}' in combined file.")

            grp = h5[run_name]
            if not isinstance(grp, h5py.Group):
                raise RuntimeError(f"Object '{run_name}' is not a group.")

            found_any_top_level = len(list(grp.keys())) > 0
            if not found_any_top_level:
                raise RuntimeError(f"Run group '{run_name}' is empty.")

            missing_all_expected = all(name not in grp for name in EXPECTED_TOP_LEVEL_NAMES)
            if missing_all_expected:
                raise RuntimeError(
                    f"Run group '{run_name}' does not contain any of the expected top-level datasets: "
                    f"{', '.join(EXPECTED_TOP_LEVEL_NAMES)}"
                )


def _delete_originals(paths: Iterable[Path]) -> None:
    for path in paths:
        path.unlink()
        print(f"Deleted original: {path}")


def combine_parity_folder(parity_dir: Path, output_path: Path, keep_originals: bool) -> Path:
    if not parity_dir.exists():
        raise FileNotFoundError(f"Parity folder does not exist: {parity_dir}")
    if not parity_dir.is_dir():
        raise NotADirectoryError(f"Not a folder: {parity_dir}")

    source_paths = {run: parity_dir / filename for run, filename in EXPECTED_FILES.items()}
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing expected parity files:\n  - " + "\n  - ".join(missing))

    output_path = output_path.resolve()
    if output_path.exists() and output_path.name in EXPECTED_FILES.values():
        raise RuntimeError("Refusing to overwrite one of the source files.")

    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    run_names = list(source_paths.keys())
    source_list = [str(source_paths[run]) for run in run_names]

    print(f"Combining parity files from: {parity_dir}")
    print(f"Writing combined file to: {output_path}")

    # Preflight summaries so failures happen before deletion is even possible.
    preflight = {run: _summarize_source_file(path) for run, path in source_paths.items()}

    with h5py.File(tmp_output, "w") as h5:
        h5.attrs["family"] = "parity"
        h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["created_by"] = "combine_parity_manual_data_no_xarray.py"
        h5.attrs["description"] = (
            "Combined parity container created with h5py only. "
            "Each original file is copied into its own group. Originals are deleted only "
            "after successful verification."
        )
        h5.attrs["source_dir"] = str(parity_dir)
        h5.attrs["expected_runs_json"] = json.dumps(run_names)
        h5.attrs["preflight_json"] = json.dumps(preflight, sort_keys=True)

        str_dt = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset("runs", data=np.asarray(run_names, dtype=object), dtype=str_dt)
        h5.create_dataset("sources", data=np.asarray(source_list, dtype=object), dtype=str_dt)

        for run_name, source_path in source_paths.items():
            _write_run_group(h5, run_name, source_path)

    _verify_combined_file(tmp_output, run_names)
    os.replace(tmp_output, output_path)
    print(f"Verified combined file: {output_path}")

    if keep_originals:
        print("Keeping original files because --keep-originals was set.")
    else:
        _delete_originals(source_paths.values())

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine mpr_A1/A2/B1 parity H5 files into one parity_combined.h5 file using h5py only."
    )
    parser.add_argument(
        "--parity-dir",
        default=str(DEFAULT_PARITY_DIR),
        help="Folder containing mpr_A1_Cq.h5, mpr_A2_Cq.h5, and mpr_B1_Cq.h5",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output H5 path. Defaults to <parity-dir>/parity_combined.h5",
    )
    parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Do not delete the original three H5 files after successful combine.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parity_dir = Path(args.parity_dir)
    output_path = Path(args.output) if args.output else parity_dir / DEFAULT_OUTPUT_NAME

    try:
        final_path = combine_parity_folder(
            parity_dir=parity_dir,
            output_path=output_path,
            keep_originals=args.keep_originals,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    print(f"Combined file: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
