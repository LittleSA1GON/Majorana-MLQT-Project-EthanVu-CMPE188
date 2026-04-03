import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path


def dataset_to_dataframe(data: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(data)

    if arr.ndim == 0:
        return pd.DataFrame({'value': [arr.item()]})

    if arr.ndim == 1:
        return pd.DataFrame({'index': np.arange(arr.shape[0]), 'value': arr})

    if arr.ndim == 2:
        i, j = np.indices(arr.shape)
        return pd.DataFrame({'dim0': i.ravel(), 'dim1': j.ravel(), 'value': arr.ravel()})

    coords = np.stack(np.unravel_index(np.arange(arr.size), arr.shape), axis=1)
    columns = [f'dim{i}' for i in range(arr.ndim)]
    df = pd.DataFrame(coords, columns=columns)
    df['value'] = arr.ravel()
    return df


def convert_h5_to_csv(h5_path: Path, csv_path_base: Path) -> None:
    csv_path_base.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, 'r') as h5_file:
        datasets = []

        def _collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append((name, obj))

        h5_file.visititems(_collect)

        if not datasets:
            raise KeyError(f"No datasets found in {h5_path}")

        base = csv_path_base
        dataset_meta = []

        for name, dataset in datasets:
            ds_data = dataset[()]
            out_df = dataset_to_dataframe(ds_data)

            safe_name = name.strip('/').replace('/', '__') or 'root'
            out_file = base / f"{safe_name}.csv"
            out_df.to_csv(out_file, index=False)

            dataset_meta.append({
                'dataset_path': name,
                'output_csv': str(out_file),
                'shape': str(ds_data.shape),
                'dtype': str(ds_data.dtype),
            })

            print(f"Converted {h5_path}({name}) to {out_file}")

        meta_file = base / f"datasets.csv"
        pd.DataFrame(dataset_meta).to_csv(meta_file, index=False)
        print(f"Wrote dataset catalog: {meta_file}")


def convert_folder_data_to_parsed_data(root_dir: Path) -> None:
    src_root = root_dir / 'data'
    dst_root = root_dir / 'parsed_data'

    if not src_root.exists():
        raise FileNotFoundError(f"Source data directory not found: {src_root}")

    h5_files = list(src_root.rglob('*.h5'))
    if not h5_files:
        print(f"No .h5 files found in {src_root}")
        return

    for h5_path in h5_files:
        rel_path = h5_path.relative_to(src_root)
        csv_path_base = dst_root / rel_path.with_suffix('')
        convert_h5_to_csv(h5_path, csv_path_base)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert all *.h5 under data/ to one or more CSVs in parsed_data/')
    parser.add_argument('--root', type=Path, default=None, help='Project root containing data/ and parsed_data/')
    args = parser.parse_args()

    project_root = args.root or Path(__file__).resolve().parents[2]
    convert_folder_data_to_parsed_data(project_root)

