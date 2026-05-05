import argparse
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]

DEFAULT_INPUT_DIR = REPO_ROOT / "manual-data" / "~ready"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "manual-data" / "~ready_torch"

SUPPORTED_NUMERIC_KINDS = {"b", "i", "u", "f"}


def decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_json_safe(value: Any) -> Any:
    value = decode_scalar(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.ndarray):
        return [make_json_safe(x) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [make_json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    return str(value)


def decode_string_array(arr: np.ndarray) -> list[str]:
    out: list[str] = []
    flat = arr.reshape(-1)
    for x in flat:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return out


def tensor_from_numpy(arr: np.ndarray) -> torch.Tensor:
    try:
        return torch.from_numpy(arr)
    except TypeError:
        if arr.dtype.kind == "u":
            if arr.dtype.itemsize <= 1:
                return torch.from_numpy(arr.astype(np.uint8, copy=False))
            if arr.dtype.itemsize <= 2:
                return torch.from_numpy(arr.astype(np.int32, copy=True))
            return torch.from_numpy(arr.astype(np.int64, copy=True))
        raise


def convert_one_h5(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.pt"

    bundle: dict[str, Any] = {
        "format": "prepared_pt_bundle_v1",
        "source_h5": str(input_path),
        "attrs": {},
        "manifest": {},
        "datasets": {},
    }

    with h5py.File(input_path, "r") as f:
        bundle["attrs"] = {str(k): make_json_safe(v) for k, v in f.attrs.items()}

        for name in sorted(f.keys()):
            ds = f[name]
            if not isinstance(ds, h5py.Dataset):
                continue

            arr = ds[...]
            source_dtype = str(arr.dtype)
            shape = list(arr.shape)

            if arr.dtype.kind in SUPPORTED_NUMERIC_KINDS:
                tensor = tensor_from_numpy(np.asarray(arr))
                bundle["datasets"][name] = tensor
                bundle["manifest"][name] = {
                    "kind": "tensor",
                    "shape": shape,
                    "source_dtype": source_dtype,
                    "torch_dtype": str(tensor.dtype),
                }
            else:
                strings = decode_string_array(np.asarray(arr))
                payload: Any = strings[0] if arr.ndim == 0 else strings
                bundle["datasets"][name] = payload
                bundle["manifest"][name] = {
                    "kind": "string_scalar" if arr.ndim == 0 else "string_list",
                    "shape": shape,
                    "source_dtype": source_dtype,
                }

    torch.save(bundle, output_path)
    return output_path


def discover_input_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted(input_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in: {input_dir}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert prepared QCVV H5 files into PyTorch-friendly .pt bundles."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Repo-local input directory containing prepared .h5 files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Repo-local output directory for converted .pt bundles.",
    )
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help="Optional specific .h5 filenames or paths. If omitted, converts all .h5 in --input-dir.",
    )
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    input_dir = Path(args.input_dir)
    if not args.input:
        return discover_input_files(input_dir)

    out: list[Path] = []
    for raw in args.input:
        p = Path(raw)
        if not p.is_absolute():
            p = input_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        out.append(p)
    return out


def main() -> None:
    args = parse_args()
    input_paths = resolve_inputs(args)
    output_dir = Path(args.output_dir)

    print("=" * 100, flush=True)
    print(f"Script: {CURRENT_FILE}", flush=True)
    print(f"Repo root: {REPO_ROOT}", flush=True)
    print(f"Input dir: {Path(args.input_dir)}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Files to convert: {len(input_paths)}", flush=True)
    print("=" * 100, flush=True)

    converted: list[Path] = []
    for input_path in input_paths:
        print(f"Converting: {input_path}", flush=True)
        out = convert_one_h5(input_path, output_dir)
        converted.append(out)
        print(f"Saved: {out}", flush=True)

    print("=" * 100, flush=True)
    print("DONE", flush=True)
    for p in converted:
        print(f"  - {p}", flush=True)


if __name__ == "__main__":
    main()
