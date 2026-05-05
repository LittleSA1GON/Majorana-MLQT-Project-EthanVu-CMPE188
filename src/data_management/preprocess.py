import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


DEFAULT_ROOT = Path(
    r"E:\Software Engineering Stuff\Quantum\Majorana-MLQT-Project-EthanVu-CMPE188\manual-data"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "prepared_v2"


@dataclass(frozen=True)
class FamilySpec:
    family_name: str
    input_path: Path
    output_path: Path
    grouped: bool
    runs: tuple[str, ...]
    coord_names: tuple[str, str, str]
    axis1_alt_names: tuple[str, str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_to_str_list(arr: np.ndarray) -> list[str]:
    out: list[str] = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return out


def compute_window_starts(length: int, window_length: int, stride: int) -> list[int]:
    if length <= 0:
        return []
    if length <= window_length:
        return [0]
    starts = list(range(0, length - window_length + 1, stride))
    last = length - window_length
    if starts[-1] != last:
        starts.append(last)
    return starts


def compute_sample_dt(time_axis: np.ndarray) -> float:
    if time_axis.ndim != 1 or time_axis.size < 2:
        return 0.0
    diffs = np.diff(time_axis.astype(np.float64))
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def normalize_axis(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    if values.size == 0:
        return values
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


def write_utf8_1d(h5: h5py.File, name: str, values: Iterable[str]) -> None:
    dt = h5py.string_dtype(encoding="utf-8")
    vals = list(values)
    h5.create_dataset(name, data=np.asarray(vals, dtype=object), dtype=dt)


def expected_family_specs(root: Path, output_dir: Path) -> list[FamilySpec]:
    return [
        FamilySpec(
            family_name="parity",
            input_path=root / "parity" / "parity_combined.h5",
            output_path=output_dir / "prepared_parity.h5",
            grouped=True,
            runs=("mpr_A1", "mpr_A2", "mpr_B1"),
            coord_names=("B_perp", "V_wire", "V_lin_qd"),
            axis1_alt_names=("V_qd_1_plunger_gate", "V_qd_3_plunger_gate"),
        ),
        FamilySpec(
            family_name="x_loop",
            input_path=root / "x loop" / "xmpr_Cq.h5",
            output_path=output_dir / "prepared_x_loop.h5",
            grouped=False,
            runs=("xmpr",),
            coord_names=("Bperp", "VQD3", "VQD1"),
            axis1_alt_names=("", ""),
        ),
        FamilySpec(
            family_name="z_loop",
            input_path=root / "z loop" / "zmpr_Cq.h5",
            output_path=output_dir / "prepared_z_loop.h5",
            grouped=False,
            runs=("zmpr",),
            coord_names=("Bperp", "VQD4", "VQDL"),
            axis1_alt_names=("", ""),
        ),
    ]


def preflight_count(spec: FamilySpec, window_length: int, stride: int) -> tuple[int, list[dict]]:
    details: list[dict] = []
    total = 0
    with h5py.File(spec.input_path, "r") as f:
        run_names = spec.runs if spec.grouped else spec.runs
        for run_name in run_names:
            grp = f[run_name] if spec.grouped else f
            cq = grp["Cq"]
            if cq.ndim != 4:
                raise ValueError(f"{spec.family_name}:{run_name} Cq must be 4D, got {cq.shape}")
            n0, n1, n2, t = cq.shape
            starts = compute_window_starts(int(t), window_length, stride)
            n_operating = int(n0 * n1 * n2)
            n_samples = n_operating * len(starts)
            total += n_samples
            details.append(
                {
                    "run_name": run_name,
                    "cq_shape": tuple(int(x) for x in cq.shape),
                    "windows_per_operating_point": len(starts),
                    "operating_points": n_operating,
                    "total_samples": n_samples,
                    "window_starts": starts,
                }
            )
    return total, details


def create_output_datasets(
    out: h5py.File,
    total_samples: int,
    window_length: int,
    trace_dtype=np.float32,
) -> dict[str, h5py.Dataset]:
    ds: dict[str, h5py.Dataset] = {}
    ds["traces"] = out.create_dataset(
        "traces",
        shape=(total_samples, window_length, 2),
        dtype=trace_dtype,
        compression="gzip",
        shuffle=True,
        chunks=(min(2048, total_samples), window_length, 2),
    )
    ds["valid_length"] = out.create_dataset("valid_length", shape=(total_samples,), dtype=np.int32, compression="gzip", shuffle=True)
    ds["window_start"] = out.create_dataset("window_start", shape=(total_samples,), dtype=np.int32, compression="gzip", shuffle=True)
    ds["operating_index"] = out.create_dataset("operating_index", shape=(total_samples, 3), dtype=np.int32, compression="gzip", shuffle=True)
    ds["coord_values"] = out.create_dataset("coord_values", shape=(total_samples, 3), dtype=np.float32, compression="gzip", shuffle=True)
    ds["coord_normalized"] = out.create_dataset("coord_normalized", shape=(total_samples, 3), dtype=np.float32, compression="gzip", shuffle=True)
    ds["axis1_alt_values"] = out.create_dataset("axis1_alt_values", shape=(total_samples, 2), dtype=np.float32, compression="gzip", shuffle=True)
    ds["sample_dt"] = out.create_dataset("sample_dt", shape=(total_samples,), dtype=np.float32, compression="gzip", shuffle=True)
    ds["original_length"] = out.create_dataset("original_length", shape=(total_samples,), dtype=np.int32, compression="gzip", shuffle=True)

    str_dt = h5py.string_dtype(encoding="utf-8")
    ds["family_name"] = out.create_dataset("family_name", shape=(total_samples,), dtype=str_dt)
    ds["run_name"] = out.create_dataset("run_name", shape=(total_samples,), dtype=str_dt)
    ds["source_file"] = out.create_dataset("source_file", shape=(total_samples,), dtype=str_dt)
    ds["source_group"] = out.create_dataset("source_group", shape=(total_samples,), dtype=str_dt)

    out.create_dataset("coord_names", data=np.asarray(spec_coord_names_placeholder, dtype=object), dtype=str_dt)  # replaced later
    out.create_dataset("axis1_alt_names", data=np.asarray(spec_axis1_alt_names_placeholder, dtype=object), dtype=str_dt)  # replaced later
    out.create_dataset("trace_channels", data=np.asarray(["Cq", "iCq"], dtype=object), dtype=str_dt)
    return ds


# placeholders are replaced right after file creation
spec_coord_names_placeholder = ("", "", "")
spec_axis1_alt_names_placeholder = ("", "")


def fill_family(spec: FamilySpec, out: h5py.File, ds: dict[str, h5py.Dataset], window_length: int, stride: int) -> dict:
    cursor = 0
    family_summary: dict = {
        "family_name": spec.family_name,
        "runs": {},
    }

    # replace placeholder metadata datasets with actual names
    del out["coord_names"]
    del out["axis1_alt_names"]
    str_dt = h5py.string_dtype(encoding="utf-8")
    out.create_dataset("coord_names", data=np.asarray(spec.coord_names, dtype=object), dtype=str_dt)
    out.create_dataset("axis1_alt_names", data=np.asarray(spec.axis1_alt_names, dtype=object), dtype=str_dt)

    with h5py.File(spec.input_path, "r") as f:
        for run_name in spec.runs:
            grp = f[run_name] if spec.grouped else f
            source_group = run_name if spec.grouped else "/"

            cq = grp["Cq"]
            icq = grp["iCq"]
            time_axis = grp["time"][:].astype(np.float64)

            if cq.shape != icq.shape:
                raise ValueError(f"{spec.family_name}:{run_name} Cq and iCq shapes differ: {cq.shape} vs {icq.shape}")
            if cq.ndim != 4:
                raise ValueError(f"{spec.family_name}:{run_name} Cq must be 4D, got {cq.shape}")
            if time_axis.shape[0] != cq.shape[-1]:
                raise ValueError(
                    f"{spec.family_name}:{run_name} time length {time_axis.shape[0]} does not match Cq last dim {cq.shape[-1]}"
                )

            coord0 = grp[spec.coord_names[0]][:].astype(np.float32)
            coord1 = grp[spec.coord_names[1]][:].astype(np.float32)
            coord2 = grp[spec.coord_names[2]][:].astype(np.float32)

            if (coord0.shape[0], coord1.shape[0], coord2.shape[0]) != cq.shape[:3]:
                raise ValueError(
                    f"{spec.family_name}:{run_name} coord lengths {(coord0.shape[0], coord1.shape[0], coord2.shape[0])} "
                    f"do not match Cq spatial dims {cq.shape[:3]}"
                )

            coord0_n = normalize_axis(coord0)
            coord1_n = normalize_axis(coord1)
            coord2_n = normalize_axis(coord2)

            alt0 = np.full(coord1.shape, np.nan, dtype=np.float32)
            alt1 = np.full(coord1.shape, np.nan, dtype=np.float32)
            if spec.axis1_alt_names[0]:
                alt0 = grp[spec.axis1_alt_names[0]][:].astype(np.float32)
            if spec.axis1_alt_names[1]:
                alt1 = grp[spec.axis1_alt_names[1]][:].astype(np.float32)

            starts = compute_window_starts(int(cq.shape[-1]), window_length, stride)
            sample_dt = np.float32(compute_sample_dt(time_axis))
            original_length = int(cq.shape[-1])

            n0, n1, n2, t = cq.shape
            n_operating = int(n0 * n1 * n2)
            expected_samples = n_operating * len(starts)

            family_summary["runs"][run_name] = {
                "cq_shape": [int(x) for x in cq.shape],
                "sample_dt": float(sample_dt),
                "original_length": original_length,
                "window_starts": starts,
                "expected_samples": expected_samples,
            }

            print(
                f"  - Processing {run_name}: "
                f"Cq shape={tuple(int(x) for x in cq.shape)}, "
                f"windows per operating point={len(starts)}, total samples={expected_samples}"
            )

            for i0 in range(n0):
                cq_i0 = cq[i0]
                icq_i0 = icq[i0]
                c0 = coord0[i0]
                c0n = coord0_n[i0]
                for i1 in range(n1):
                    cq_i01 = cq_i0[i1]
                    icq_i01 = icq_i0[i1]
                    c1 = coord1[i1]
                    c1n = coord1_n[i1]
                    a0 = alt0[i1]
                    a1 = alt1[i1]
                    for i2 in range(n2):
                        trace_cq = np.asarray(cq_i01[i2], dtype=np.float32)
                        trace_icq = np.asarray(icq_i01[i2], dtype=np.float32)
                        c2 = coord2[i2]
                        c2n = coord2_n[i2]

                        for start in starts:
                            valid = int(min(window_length, max(0, original_length - start)))

                            block = np.zeros((window_length, 2), dtype=np.float32)
                            if valid > 0:
                                block[:valid, 0] = trace_cq[start:start + valid]
                                block[:valid, 1] = trace_icq[start:start + valid]

                            ds["traces"][cursor] = block
                            ds["valid_length"][cursor] = valid
                            ds["window_start"][cursor] = int(start)
                            ds["operating_index"][cursor] = (i0, i1, i2)
                            ds["coord_values"][cursor] = (c0, c1, c2)
                            ds["coord_normalized"][cursor] = (c0n, c1n, c2n)
                            ds["axis1_alt_values"][cursor] = (a0, a1)
                            ds["sample_dt"][cursor] = sample_dt
                            ds["original_length"][cursor] = original_length
                            ds["family_name"][cursor] = spec.family_name
                            ds["run_name"][cursor] = run_name
                            ds["source_file"][cursor] = str(spec.input_path)
                            ds["source_group"][cursor] = source_group
                            cursor += 1

    family_summary["total_written_samples"] = cursor
    return family_summary


def confirm_prepared_file(spec: FamilySpec, path: Path) -> dict:
    with h5py.File(path, "r") as f:
        traces = f["traces"]
        valid_length = f["valid_length"][:]
        window_start = f["window_start"][:]
        run_name = np.asarray(decode_to_str_list(f["run_name"][:]), dtype=object)
        family_name = np.asarray(decode_to_str_list(f["family_name"][:]), dtype=object)
        sample_dt = f["sample_dt"][:]
        original_length = f["original_length"][:]
        coord_names = decode_to_str_list(f["coord_names"][:])

        if traces.ndim != 3 or traces.shape[2] != 2:
            raise ValueError(f"{path.name}: traces must be [N, T, 2], got {traces.shape}")

        n, window_length, channels = traces.shape
        if valid_length.shape[0] != n:
            raise ValueError(f"{path.name}: valid_length length mismatch")
        if window_start.shape[0] != n or original_length.shape[0] != n or sample_dt.shape[0] != n:
            raise ValueError(f"{path.name}: one or more metadata lengths mismatch sample count")
        if np.any(valid_length < 0) or np.any(valid_length > window_length):
            raise ValueError(f"{path.name}: valid_length outside [0, window_length]")
        if np.any(original_length <= 0):
            raise ValueError(f"{path.name}: original_length must be positive")

        unique_runs = sorted(set(run_name.tolist()))
        per_run: dict[str, dict] = {}
        for run in unique_runs:
            mask = run_name == run
            run_orig = np.unique(original_length[mask])
            run_dt = np.unique(sample_dt[mask])

            if run_orig.size != 1:
                raise ValueError(f"{path.name}: run {run} has multiple original_length values: {run_orig}")
            if run_dt.size != 1:
                raise ValueError(f"{path.name}: run {run} has multiple sample_dt values: {run_dt}")

            max_expected = np.minimum(window_length, run_orig[0] - window_start[mask])
            if np.any(valid_length[mask] != max_expected):
                raise ValueError(f"{path.name}: run {run} has invalid valid_length values")

            # verify padded tail is zero where valid_length < window_length
            sample_indices = np.flatnonzero(mask)
            padded = sample_indices[valid_length[mask] < window_length]
            padded_checked = 0
            for idx in padded[:128]:
                v = int(valid_length[idx])
                tail = traces[idx, v:, :]
                if tail.size and not np.allclose(tail, 0.0):
                    raise ValueError(f"{path.name}: run {run} padded tail not zero for sample {idx}")
                padded_checked += 1

            per_run[run] = {
                "samples": int(mask.sum()),
                "original_length": int(run_orig[0]),
                "sample_dt": float(run_dt[0]),
                "valid_length_min": int(valid_length[mask].min()),
                "valid_length_max": int(valid_length[mask].max()),
                "padded_samples": int(np.sum(valid_length[mask] < window_length)),
                "checked_padded_samples": int(padded_checked),
            }

        unique_families = sorted(set(family_name.tolist()))
        if unique_families != [spec.family_name]:
            raise ValueError(f"{path.name}: family_name mismatch: {unique_families}")

        return {
            "status": "PASS",
            "file": str(path),
            "family_name": spec.family_name,
            "samples": int(n),
            "window_length": int(window_length),
            "channels": int(channels),
            "coord_names": coord_names,
            "unique_runs": unique_runs,
            "per_run": per_run,
        }


def process_one_family(spec: FamilySpec, window_length: int, stride: int, overwrite: bool) -> dict:
    if not spec.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {spec.input_path}")
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {spec.output_path}. Use --overwrite to replace it.")

    total_samples, details = preflight_count(spec, window_length, stride)

    print("=" * 100)
    print(f"Creating prepared file for family '{spec.family_name}'")
    print(f"Input:  {spec.input_path}")
    print(f"Output: {spec.output_path}")
    print(f"Window length: {window_length} | Stride: {stride}")
    print(f"Total samples to write: {total_samples}")

    if spec.output_path.exists():
        spec.output_path.unlink()

    with h5py.File(spec.output_path, "w") as out:
        out.attrs["created_utc"] = utc_now()
        out.attrs["description"] = (
            "Prepared, windowed CQ dataset. "
            "Each sample is one operating point plus one fixed-length time window. "
            "Required common datasets: traces, valid_length, window_start, operating_index, "
            "coord_values, coord_normalized, sample_dt, family, run_name, source_file, source_group. "
            "This v2 build writes run-aware original_length, sample_dt, and valid_length."
        )
        out.attrs["family"] = spec.family_name
        out.attrs["schema_version"] = "prepared_qcvv_v2"
        out.attrs["source_input_file"] = str(spec.input_path)
        out.attrs["stride"] = int(stride)
        out.attrs["window_length"] = int(window_length)
        out.attrs["preflight_json"] = json.dumps(details)

        datasets = create_output_datasets(out, total_samples, window_length)
        summary = fill_family(spec, out, datasets, window_length, stride)
        out.attrs["written_samples"] = int(summary["total_written_samples"])

    confirmation = confirm_prepared_file(spec, spec.output_path)

    print("  Format confirmation: PASS")
    print(f"    Samples: {confirmation['samples']:,}")
    print(f"    Window length: {confirmation['window_length']}")
    print(f"    Coord names: {', '.join(confirmation['coord_names'])}")
    print(f"    Unique runs: {', '.join(confirmation['unique_runs'])}")
    for run, info in confirmation["per_run"].items():
        print(
            f"    {run}: original_length={info['original_length']}, "
            f"sample_dt={info['sample_dt']:.12g}, "
            f"valid_length[min,max]=({info['valid_length_min']}, {info['valid_length_max']}), "
            f"padded_samples={info['padded_samples']}"
        )

    confirmation_path = spec.output_path.with_name(spec.output_path.stem + "_confirmation.json")
    confirmation_path.write_text(json.dumps(confirmation, indent=2), encoding="utf-8")
    print(f"Confirmation written to: {confirmation_path}")

    return confirmation


def main() -> int:
    parser = argparse.ArgumentParser(description="Corrected preprocessing for manual-data CQ files.")
    parser.add_argument("root", nargs="?", default=str(DEFAULT_ROOT), help="Root folder containing parity / x loop / z loop")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for prepared outputs")
    parser.add_argument("--window-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prepared outputs")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    if args.window_length <= 0 or args.stride <= 0:
        raise ValueError("window-length and stride must be positive integers")

    specs = expected_family_specs(root, output_dir)

    overall = {
        "created_utc": utc_now(),
        "root": str(root),
        "output_dir": str(output_dir),
        "window_length": int(args.window_length),
        "stride": int(args.stride),
        "families": {},
    }

    for spec in specs:
        conf = process_one_family(spec, args.window_length, args.stride, overwrite=args.overwrite)
        overall["families"][spec.family_name] = conf

    overall_path = output_dir / "prepared_overall_report.json"
    overall_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")

    print("=" * 100)
    print("ALL PREPARED FILES COMPLETED")
    print(f"Overall report written to: {overall_path}")
    for family, conf in overall["families"].items():
        print(
            f"  - {family}: {conf['file']} | samples={conf['samples']:,} | runs={', '.join(conf['unique_runs'])}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
