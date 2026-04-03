import argparse
from pathlib import Path

import h5py


def list_h5_files(root_dir: Path):
    if not root_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {root_dir}")
    return sorted(root_dir.rglob('*.h5'))


def print_dataset_info(dataset, indent='  '):
    shape = dataset.shape
    dtype = dataset.dtype
    print(f"{indent}- dataset: {dataset.name} shape={shape} dtype={dtype}")


def print_group_info(group, indent=''):
    print(f"{indent}group: {group.name}")
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            print_dataset_info(item, indent + '  ')
        else:
            print_group_info(item, indent + '  ')


def inspect_h5_file(h5_path: Path, dataset_path: str | None = None, max_rows: int = 20):
    with h5py.File(h5_path, 'r') as f:
        print(f"\nInspecting: {h5_path}")
        if dataset_path:
            if dataset_path not in f:
                raise KeyError(f"{dataset_path} not found in {h5_path}")
            dataset = f[dataset_path]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"{dataset_path} is not a dataset")
            print_dataset_info(dataset)
            data = dataset[()]
            print(f"data sample ({min(max_rows, len(data))} rows):")
            print(data[:max_rows])
        else:
            print_group_info(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Inspect HDF5 files in the data folder')
    parser.add_argument('--data-root', type=Path, default=None,
                        help='Path to data folder (default: project_root/data)')
    parser.add_argument('--file', type=Path, default=None,
                        help='Specific .h5 file to inspect')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset inside HDF5 file (e.g., /ng1)')
    parser.add_argument('--max-rows', type=int, default=20,
                        help='Max rows to print from dataset preview')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    data_root = args.data_root or (project_root / 'data')

    if args.file:
        inspect_h5_file(args.file, args.dataset, args.max_rows)
    else:
        h5_files = list_h5_files(data_root)
        if not h5_files:
            print(f"No .h5 files found in {data_root}")
        else:
            print(f"Found {len(h5_files)} .h5 files under {data_root}")
            for path in h5_files:
                inspect_h5_file(path, args.dataset, args.max_rows)
