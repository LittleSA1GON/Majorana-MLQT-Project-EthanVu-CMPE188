"""
Comprehensive data mapper for Majorana MLQT project.

Reads parsed_data/ (CSV reconstructions of H5 files) and builds 8 task-specific
mapped datasets in mapped_data/ organized by ML objective.

Core principle: Each folder containing datasets.csv is ONE sample, not each CSV file.
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def reconstruct_array_from_csv(csv_path):
    """
    Reconstruct a numpy array from a flattened CSV.
    
    Formats:
    - Scalar: single column 'value'
    - 1D: columns 'index', 'value'
    - ND: columns 'dim0', 'dim1', ..., 'dimN', 'value'
    """
    df = pd.read_csv(csv_path)
    
    if df.shape[0] == 0:
        return np.array([])
    
    # Detect format
    if 'value' not in df.columns:
        return None
    
    if df.shape[1] == 1:  # Scalar
        return np.array(df['value'].iloc[0])
    
    # Extract dimension columns (all except 'value')
    dim_cols = [c for c in df.columns if c.startswith('dim') and c != 'value']
    index_col = 'index' if 'index' in df.columns and len(dim_cols) == 0 else None
    
    if index_col:
        # 1D array
        values = df['value'].values
        return values
    
    if dim_cols:
        # ND array
        dim_cols_sorted = sorted(dim_cols, key=lambda x: int(x[3:]))
        shape = tuple(df[c].max().astype(int) + 1 for c in dim_cols_sorted)
        arr = np.full(shape, np.nan)
        indices = tuple(df[c].values.astype(int) for c in dim_cols_sorted)
        arr[indices] = df['value'].values
        return arr
    
    return None


def get_csv_files(folder_path):
    """Get all CSV files (except datasets.csv) in a folder."""
    return {
        f.stem: f for f in folder_path.glob('*.csv')
        if f.name != 'datasets.csv'
    }


def load_sample_arrays(folder_path):
    """Load all arrays from a sample folder."""
    arrays = {}
    for name, csv_path in get_csv_files(folder_path).items():
        try:
            arr = reconstruct_array_from_csv(csv_path)
            if arr is not None:
                arrays[name] = arr
        except Exception as e:
            print(f"  Warning: Could not load {name} from {folder_path.name}: {e}")
    return arrays


def detect_family(path_str):
    """Detect data family from path."""
    path_lower = path_str.lower()
    
    if 'mpr_' in path_lower:
        return 'mpr'
    if 'cut_loop' in path_lower:
        return 'cut_loop'
    if 'trivial_' in path_lower:
        return 'trivial'
    if 'injector' in path_lower:
        return 'injector'
    if 'qpp' in path_lower:
        return 'qpp'
    if any(x in path_lower for x in ['qdmzm', 'qd1mzm', 'qd3mzm']):
        return 'qdmzm'
    if 'thermometry' in path_lower:
        return 'thermometry'
    if 'tgp2' in path_lower:
        return 'tgp2'
    if 'charge_noise' in path_lower:
        return 'charge_noise'
    if any(x in path_lower for x in ['qd1_bias', 'qd2_bias', 'qd3_bias', 'qd1_qd3']):
        return 'dot_tuneup'
    
    return 'unknown'


def find_all_samples(parsed_data_root):
    """Recursively find all folders containing datasets.csv."""
    samples = []
    for datasets_csv in parsed_data_root.rglob('datasets.csv'):
        folder = datasets_csv.parent
        samples.append(folder)
    return sorted(samples)


def extract_2d_features(arr_dict):
    """Extract summary features from 2D arrays."""
    features = {}
    for name, arr in arr_dict.items():
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            features[f'{name}_mean'] = float(np.nanmean(arr))
            features[f'{name}_std'] = float(np.nanstd(arr))
            features[f'{name}_min'] = float(np.nanmin(arr))
            features[f'{name}_max'] = float(np.nanmax(arr))
            features[f'{name}_median'] = float(np.nanmedian(arr))
            features[f'{name}_q25'] = float(np.nanpercentile(arr, 25))
            features[f'{name}_q75'] = float(np.nanpercentile(arr, 75))
            features[f'{name}_nan_frac'] = float(np.isnan(arr).sum() / arr.size)
            features[f'{name}_shape0'] = int(arr.shape[0])
            features[f'{name}_shape1'] = int(arr.shape[1])
    return features


def extract_1d_features(arr_dict):
    """Extract summary features from 1D arrays."""
    features = {}
    for name, arr in arr_dict.items():
        if isinstance(arr, np.ndarray) and arr.ndim == 1:
            features[f'{name}_mean'] = float(np.nanmean(arr))
            features[f'{name}_std'] = float(np.nanstd(arr))
            features[f'{name}_min'] = float(np.nanmin(arr))
            features[f'{name}_max'] = float(np.nanmax(arr))
            features[f'{name}_length'] = int(len(arr))
    return features


def build_map_classification(samples_by_family):
    """Build map classification dataset from mpr, cut_loop, trivial."""
    dataset = []
    tensors = {}
    labels = []
    
    sample_id = 0
    for family in ['mpr', 'cut_loop', 'trivial']:
        if family not in samples_by_family:
            continue
        
        for folder in samples_by_family[family]:
            arrays = load_sample_arrays(folder)
            
            features = extract_2d_features(arrays)
            features['sample_id'] = sample_id
            features['source_folder'] = folder.name
            features['family'] = family
            dataset.append(features)
            
            tensors[sample_id] = arrays
            labels.append(family)
            sample_id += 1
    
    return pd.DataFrame(dataset), tensors, np.array(labels)


def build_sequence_readout(samples_by_family):
    """Build sequence readout dataset from qpp, injector, time-bearing folders."""
    sequences = {}
    features_list = []
    
    sample_id = 0
    for family in ['qpp', 'injector', 'mpr', 'cut_loop', 'trivial']:
        if family not in samples_by_family:
            continue
        
        for folder in samples_by_family[family]:
            arrays = load_sample_arrays(folder)
            
            # Look for time-bearing sequences
            has_time = 'time' in arrays
            has_cq = 'Cq' in arrays
            
            if has_time or has_cq:
                record = {
                    'sample_id': sample_id,
                    'source_folder': folder.name,
                    'family': family,
                    'has_time': has_time,
                    'has_cq': has_cq,
                }
                
                # Extract 1D features
                record.update(extract_1d_features(arrays))
                
                # Store context scalars
                for key in ['B_perp', 'bias', 'V_lin_qd', 'V_wire']:
                    if key in arrays and isinstance(arrays[key], np.ndarray) and arrays[key].ndim <= 1:
                        if arrays[key].ndim == 0:
                            record[f'{key}_scalar'] = float(arrays[key])
                        else:
                            record[f'{key}_mean'] = float(np.nanmean(arrays[key]))
                
                features_list.append(record)
                sequences[sample_id] = arrays
                sample_id += 1
    
    return pd.DataFrame(features_list), sequences


def build_qdmzm_alignment(samples_by_family, simulated_samples):
    """Build QDMZM alignment dataset linking measured to simulated."""
    dataset = []
    tensors = {}
    
    sample_id = 0
    if 'qdmzm' in samples_by_family:
        for folder in samples_by_family['qdmzm']:
            arrays = load_sample_arrays(folder)
            
            features = extract_2d_features(arrays)
            features['sample_id'] = sample_id
            features['source_folder'] = folder.name
            features['source_type'] = 'measured'
            
            dataset.append(features)
            tensors[sample_id] = arrays
            sample_id += 1
    
    # Add simulated FigS11 data as reference
    if 'simulated' in simulated_samples:
        figs11 = [s for s in simulated_samples['simulated'] if 'FigS11' in s.name]
        for folder in figs11:
            arrays = load_sample_arrays(folder)
            
            features = extract_2d_features(arrays)
            features['sample_id'] = sample_id
            features['source_folder'] = folder.name
            features['source_type'] = 'simulated'
            
            dataset.append(features)
            tensors[sample_id] = arrays
            sample_id += 1
    
    return pd.DataFrame(dataset), tensors


def build_thermometry(samples_by_family, simulated_samples):
    """Build thermometry regression dataset."""
    dataset = []
    tensors = {}
    targets = []
    
    sample_id = 0
    if 'thermometry' in samples_by_family:
        for folder in samples_by_family['thermometry']:
            arrays = load_sample_arrays(folder)
            
            features = extract_2d_features(arrays)
            record = {
                'sample_id': sample_id,
                'source_folder': folder.name,
                'source_type': 'measured',
            }
            record.update(features)
            dataset.append(record)
            
            # Extract temperature target
            if 'T_MC' in arrays:
                target_val = float(np.nanmean(arrays['T_MC']))
            else:
                target_val = np.nan
            targets.append({'sample_id': sample_id, 'T_MC': target_val})
            
            tensors[sample_id] = arrays
            sample_id += 1
    
    # Add simulated thermometry reference
    if 'simulated' in simulated_samples:
        thermo_sim = [s for s in simulated_samples['simulated'] if 'thermometry' in s.name.lower()]
        for folder in thermo_sim:
            arrays = load_sample_arrays(folder)
            
            features = extract_2d_features(arrays)
            record = {
                'sample_id': sample_id,
                'source_folder': folder.name,
                'source_type': 'simulated',
            }
            record.update(features)
            dataset.append(record)
            
            if 'temperature' in arrays:
                target_val = float(np.nanmean(arrays['temperature']))
            else:
                target_val = np.nan
            targets.append({'sample_id': sample_id, 'temperature': target_val})
            
            tensors[sample_id] = arrays
            sample_id += 1
    
    return pd.DataFrame(dataset), tensors, pd.DataFrame(targets)


def build_tuneup(samples_by_family):
    """Build tuneup dataset from dot_tuneup_A1."""
    dataset = []
    tensors = {}
    
    sample_id = 0
    if 'dot_tuneup' in samples_by_family:
        for folder in samples_by_family['dot_tuneup']:
            arrays = load_sample_arrays(folder)
            
            features = {}
            features['sample_id'] = sample_id
            features['source_folder'] = folder.name
            features.update(extract_2d_features(arrays))
            features.update(extract_1d_features(arrays))
            
            dataset.append(features)
            tensors[sample_id] = arrays
            sample_id += 1
    
    return pd.DataFrame(dataset), tensors


def build_tgp2_transport(samples_by_family):
    """Build TGP2 transport/anomaly dataset."""
    dataset = []
    tensors = {}
    
    sample_id = 0
    if 'tgp2' in samples_by_family:
        for folder in samples_by_family['tgp2']:
            arrays = load_sample_arrays(folder)
            
            features = {}
            features['sample_id'] = sample_id
            features['source_folder'] = folder.name
            
            # Parse device/side info
            name_lower = folder.name.lower()
            if 'left' in name_lower:
                features['side'] = 'left'
                features['g_prefix'] = 'g_ll' if 'll' in name_lower else 'g_rl'
            elif 'right' in name_lower:
                features['side'] = 'right'
                features['g_prefix'] = 'g_lr' if 'lr' in name_lower else 'g_rr'
            
            features.update(extract_2d_features(arrays))
            features.update(extract_1d_features(arrays))
            
            dataset.append(features)
            tensors[sample_id] = arrays
            sample_id += 1
    
    return pd.DataFrame(dataset), tensors


def build_charge_noise(samples_by_family):
    """Build charge noise time-series dataset."""
    dataset = []
    sequences = {}
    
    sample_id = 0
    if 'charge_noise' in samples_by_family:
        for folder in samples_by_family['charge_noise']:
            arrays = load_sample_arrays(folder)
            
            features = {
                'sample_id': sample_id,
                'source_folder': folder.name,
            }
            features.update(extract_1d_features(arrays))
            features.update(extract_2d_features(arrays))
            
            dataset.append(features)
            sequences[sample_id] = arrays
            sample_id += 1
    
    return pd.DataFrame(dataset), sequences


def build_simulation_pretraining(simulated_samples):
    """Build simulation pretraining dataset."""
    dataset = []
    tensors = {}
    targets = []
    
    sample_id = 0
    if 'simulated' in simulated_samples:
        for folder in simulated_samples['simulated']:
            arrays = load_sample_arrays(folder)
            
            features = {
                'sample_id': sample_id,
                'source_folder': folder.name,
            }
            features.update(extract_2d_features(arrays))
            features.update(extract_1d_features(arrays))
            
            dataset.append(features)
            
            # Extract parameters as targets
            param_record = {'sample_id': sample_id}
            for key in ['parity', 'phi', 'ng1', 'ng2', 'ng3', 'tm1', 'tm2', 'E_M', 'temperature']:
                if key in arrays:
                    arr = arrays[key]
                    if isinstance(arr, (int, float)):
                        param_record[key] = arr
                    else:
                        param_record[key] = float(np.nanmean(arr))
            targets.append(param_record)
            
            tensors[sample_id] = arrays
            sample_id += 1
    
    return pd.DataFrame(dataset), tensors, pd.DataFrame(targets)


def write_dataset(dataset_name, task_type, features_df, tensors_dict, labels_arr=None, 
                  targets_df=None, sequences_dict=None, output_dir=None):
    """Write a complete mapped dataset to disk."""
    if output_dir is None:
        output_dir = Path.cwd() / 'mapped_data'
    
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Write features
    features_df.to_csv(dataset_dir / f'{dataset_name}_features.csv', index=False)
    
    # Write labels or targets
    if labels_arr is not None:
        labels_df = pd.DataFrame({
            'sample_id': range(len(labels_arr)),
            'label': labels_arr,
        })
        labels_df.to_csv(dataset_dir / 'labels.csv', index=False)
    
    if targets_df is not None:
        targets_df.to_csv(dataset_dir / 'targets.csv', index=False)
    
    # Write tensors or sequences
    if tensors_dict:
        with open(dataset_dir / 'tensors.pkl', 'wb') as f:
            pickle.dump(tensors_dict, f)
    
    if sequences_dict:
        with open(dataset_dir / 'sequences.pkl', 'wb') as f:
            pickle.dump(sequences_dict, f)
    
    # Write manifest
    manifest = []
    for idx, row in features_df.iterrows():
        record = {
            'sample_id': idx,
            'dataset_name': dataset_name,
            'source_folder': row.get('source_folder', 'unknown'),
            'family': row.get('family', 'unknown'),
        }
        manifest.append(record)
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(dataset_dir / 'sample_manifest.csv', index=False)
    
    # Write mapping metadata
    mapping = {
        'dataset_name': dataset_name,
        'task_type': task_type,
        'num_samples': len(features_df),
        'feature_columns': list(features_df.columns),
    }
    if labels_arr is not None:
        mapping['label_names'] = sorted(np.unique(labels_arr).tolist())
    if targets_df is not None:
        mapping['target_columns'] = [c for c in targets_df.columns if c != 'sample_id']
    
    with open(dataset_dir / 'mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"✓ {dataset_name}: {len(features_df)} samples")


def main():
    """Main pipeline: read parsed_data/, build all task datasets."""
    parsed_data_root = Path.cwd() / 'parsed_data'
    output_dir = Path.cwd() / 'mapped_data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("MAJORANA MLQT DATA MAPPER")
    print("="*70)
    
    # Find all samples
    print("\n[1/3] Discovering samples...")
    all_samples = find_all_samples(parsed_data_root)
    print(f"Found {len(all_samples)} sample folders")
    
    # Classify samples by family
    print("\n[2/3] Classifying samples...")
    samples_by_family = defaultdict(list)
    samples_by_source_and_family = defaultdict(lambda: defaultdict(list))
    
    for folder in all_samples:
        rel_path = folder.relative_to(parsed_data_root)
        source_type = 'simulated' if 'simulated' in rel_path.parts else 'raw' if 'raw_data' in rel_path.parts else 'converted'
        family = detect_family(str(rel_path))
        samples_by_family[family].append(folder)
        samples_by_source_and_family[source_type][family].append(folder)
    
    print(f"Classified into {len(samples_by_family)} families")
    for family, folders in sorted(samples_by_family.items()):
        if family != 'unknown':
            print(f"  {family}: {len(folders)}")
    
    # Organize by source for simulated lookup
    simulated_samples = {
        'simulated': samples_by_source_and_family['simulated']['simulated'] 
        if 'simulated' in samples_by_source_and_family else []
    }
    
    # Build datasets
    print("\n[3/3] Building task-specific datasets...")
    
    # 1. Map classification
    if 'mpr' in samples_by_family or 'cut_loop' in samples_by_family or 'trivial' in samples_by_family:
        features, tensors, labels = build_map_classification(samples_by_family)
        write_dataset('map_classification', 'multi-class classification', features, tensors, 
                      labels_arr=labels, output_dir=output_dir)
    
    # 2. Sequence readout
    if any(f in samples_by_family for f in ['qpp', 'injector', 'mpr', 'cut_loop', 'trivial']):
        features, sequences = build_sequence_readout(samples_by_family)
        if len(features) > 0:
            write_dataset('sequence_readout', 'sequence analysis', features, {}, 
                          sequences_dict=sequences, output_dir=output_dir)
    
    # 3. QDMZM alignment
    if 'qdmzm' in samples_by_family or simulated_samples['simulated']:
        features, tensors = build_qdmzm_alignment(samples_by_family, simulated_samples)
        if len(features) > 0:
            write_dataset('qdmzm_alignment', 'sim-to-real alignment', features, tensors, 
                          output_dir=output_dir)
    
    # 4. Thermometry
    if 'thermometry' in samples_by_family or simulated_samples['simulated']:
        features, tensors, targets = build_thermometry(samples_by_family, simulated_samples)
        if len(features) > 0:
            write_dataset('thermometry', 'regression (temperature)', features, tensors, 
                          targets_df=targets, output_dir=output_dir)
    
    # 5. Tuneup
    if 'dot_tuneup' in samples_by_family:
        features, tensors = build_tuneup(samples_by_family)
        write_dataset('tuneup', 'scan scoring / clustering', features, tensors, 
                      output_dir=output_dir)
    
    # 6. TGP2 transport
    if 'tgp2' in samples_by_family:
        features, tensors = build_tgp2_transport(samples_by_family)
        write_dataset('tgp2_transport', 'anomaly detection / clustering', features, tensors, 
                      output_dir=output_dir)
    
    # 7. Charge noise
    if 'charge_noise' in samples_by_family:
        features, sequences = build_charge_noise(samples_by_family)
        write_dataset('charge_noise', 'time-series forecasting', features, {}, 
                      sequences_dict=sequences, output_dir=output_dir)
    
    # 8. Simulation pretraining
    if simulated_samples['simulated']:
        features, tensors, targets = build_simulation_pretraining(simulated_samples)
        write_dataset('simulation_pretraining', 'self-supervised / parameter regression', 
                      features, tensors, targets_df=targets, output_dir=output_dir)
    
    print("\n" + "="*70)
    print("MAPPING COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated datasets:")
    for subdir in sorted(output_dir.iterdir()):
        if subdir.is_dir():
            num_samples = len(pd.read_csv(subdir / f'{subdir.name}_features.csv')) if (subdir / f'{subdir.name}_features.csv').exists() else 0
            print(f"  ✓ {subdir.name} ({num_samples} samples)")


if __name__ == '__main__':
    main()