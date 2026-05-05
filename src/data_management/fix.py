import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


DEFAULT_INPUT = Path(
    r"E:\Software Engineering Stuff\Quantum\Majorana-MLQT-Project-EthanVu-CMPE188\manual-data\prepared\prepared_parity.h5"
)

RUN_LENGTHS = {
    "mpr_A1": 742,
    "mpr_A2": 562,
    "mpr_B1": 485,
}


def _decode_array(arr: np.ndarray) -> np.ndarray:
    """Decode object/bytes/string arrays to plain Python strings."""
    if arr.dtype.kind in ("S", "O", "U"):
        out = []
        for x in arr:
            if isinstance(x, bytes):
                out.append(x.decode("utf-8", errors="replace"))
            else:
                out.append(str(x))
        return np.asarray(out, dtype=object)
    return arr.astype(object)


def compute_fixed_lengths(run_names: np.ndarray, window_start: np.ndarray, window_length: int) -> tuple[np.ndarray, np.ndarray]:
    run_names = _decode_array(run_names)
    original_length = np.empty(run_names.shape[0], dtype=np.int32)

    missing = sorted({name for name in np.unique(run_names) if name not in RUN_LENGTHS})
    if missing:
        raise ValueError(f"Unknown run_name values found: {missing}")

    for run, length in RUN_LENGTHS.items():
        mask = run_names == run
        original_length[mask] = length

    valid_length = np.clip(original_length - window_start.astype(np.int64), 0, window_length).astype(np.int32)
    return original_length, valid_length


def copy_all(src: h5py.File, dst: h5py.File) -> None:
    # Copy root attrs
    for k, v in src.attrs.items():
        dst.attrs[k] = v
    # Copy all top-level items
    for name in src.keys():
        src.copy(name, dst, name=name)


def zero_padded_tail(traces_ds: h5py.Dataset, valid_length: np.ndarray, chunk_rows: int = 4096) -> int:
    """
    Zero out padded positions beyond valid_length for each sample.
    Operates in chunks to avoid excessive memory use.
    Returns count of samples that had padding.
    """
    n, window_length, channels = traces_ds.shape
    if channels <= 0:
        return 0

    padded_mask = valid_length < window_length
    padded_indices = np.flatnonzero(padded_mask)
    if padded_indices.size == 0:
        return 0

    for start in range(0, padded_indices.size, chunk_rows):
        block = padded_indices[start:start + chunk_rows]
        arr = traces_ds[block, :, :]  # shape (B, T, C)
        for i, sample_idx in enumerate(block):
            vl = int(valid_length[sample_idx])
            if vl < window_length:
                arr[i, vl:, :] = 0.0
        traces_ds[block, :, :] = arr
    return int(padded_indices.size)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix original_length/valid_length metadata in prepared_parity.h5.")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="Path to prepared_parity.h5")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the fixed file. Default: sibling prepared_parity_fixed.h5",
    )
    parser.add_argument(
        "--replace-original",
        action="store_true",
        help="After successfully writing the fixed file, replace the original input file.",
    )
    parser.add_argument(
        "--no-zero-pad-tail",
        action="store_true",
        help="Do not zero the padded tail beyond valid_length in traces.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_fixed.h5")
    if output_path.resolve() == input_path.resolve():
        raise ValueError("Output path must be different from input path. Use --replace-original to swap after success.")

    print(f"Reading: {input_path}")
    print(f"Writing fixed copy to: {output_path}")

    with h5py.File(input_path, "r") as src:
        required = ["run_name", "window_start", "traces", "original_length", "valid_length"]
        missing = [k for k in required if k not in src]
        if missing:
            raise KeyError(f"Missing required datasets in input file: {missing}")

        window_length_attr = int(src.attrs.get("window_length", src["traces"].shape[1]))
        run_names = src["run_name"][:]
        window_start = src["window_start"][:]

        fixed_original_length, fixed_valid_length = compute_fixed_lengths(run_names, window_start, window_length_attr)

        with h5py.File(output_path, "w") as dst:
            copy_all(src, dst)

            # Replace metadata datasets in the copied file
            del dst["original_length"]
            dst.create_dataset("original_length", data=fixed_original_length, compression="gzip", shuffle=True)

            del dst["valid_length"]
            dst.create_dataset("valid_length", data=fixed_valid_length, compression="gzip", shuffle=True)

            # Preserve/augment attrs for provenance
            dst.attrs["metadata_fixed_by"] = "fix_prepared_parity_metadata.py"
            dst.attrs["metadata_fix_note"] = (
                "original_length and valid_length repaired per sample using run_name and window_start."
            )

            if not args.no_zero_pad_tail:
                padded_count = zero_padded_tail(dst["traces"], fixed_valid_length)
                dst.attrs["padded_tail_zeroed_samples"] = padded_count
            else:
                dst.attrs["padded_tail_zeroed_samples"] = 0

    # Verification pass
    with h5py.File(output_path, "r") as f:
        orig = f["original_length"][:]
        valid = f["valid_length"][:]
        runs = _decode_array(f["run_name"][:])
        ws = f["window_start"][:]
        wl = int(f.attrs.get("window_length", f["traces"].shape[1]))

        exp_orig, exp_valid = compute_fixed_lengths(runs, ws, wl)

        if not np.array_equal(orig, exp_orig):
            raise RuntimeError("Verification failed: original_length mismatch after write.")
        if not np.array_equal(valid, exp_valid):
            raise RuntimeError("Verification failed: valid_length mismatch after write.")

        # Quick sanity summary
        print("Verification passed.")
        for run in sorted(np.unique(runs)):
            mask = runs == run
            print(
                f"  {run}: samples={int(mask.sum())}, "
                f"original_length={int(orig[mask][0])}, "
                f"valid_length[min,max]=({int(valid[mask].min())}, {int(valid[mask].max())})"
            )

    if args.replace_original:
        backup_path = input_path.with_name(input_path.stem + "_backup_before_fix.h5")
        if backup_path.exists():
            raise FileExistsError(f"Backup path already exists: {backup_path}")
        shutil.move(str(input_path), str(backup_path))
        shutil.move(str(output_path), str(input_path))
        print(f"Original replaced.")
        print(f"Backup saved to: {backup_path}")
        print(f"Fixed file now at: {input_path}")
    else:
        print(f"Fixed file saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
