"""
Export Merged Dataset to CSV (Comma-Separated, No Header)

This script:
1. Loads the merged_dataset.h5 file
2. Extracts merged samples data with cluster labels and probabilities
3. Exports to CSV with specific format for metric scripts

CSV Format:
- Separator: Comma (,)
- Header: None
- Columns: merged_id, cluster_id, start_time, end_time, probability, long_wav_name

Usage:
    python3 Stg3g_export_merged_csv.py \
        --merged_dataset_h5 /path/to/merged_dataset.h5 \
        --output_csv /path/to/output.csv

Notes:
- Uses LP labels if available (Stage 4), otherwise uses initial cluster labels
- Probability is the averaged cluster probability from constituent samples
- Long wav name is extracted from merged_wav_path (removes last 3 parts)
- Output sorted by long_wav_name, then start_time
"""

import argparse
import sys
from pathlib import Path
import h5py
import pandas as pd
import numpy as np


def valid_path(path):
    """Validate that a path exists"""
    path = Path(path)
    if path.exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"Invalid path: {path}")


def extract_long_wav_name(wav_path):
    """
    Extract long wav base name from merged_wav_path.

    Format: {long_wav_name}_{cluster_label}_{start_time}_{end_time}.wav
    Returns: {long_wav_name}.wav

    Example:
        Input:  G-C1L1P-Apr27-E-Irma_q2_03-08-377_C5_206.40_207.40.wav
        Output: G-C1L1P-Apr27-E-Irma_q2_03-08-377.wav
    """
    filename_stem = Path(wav_path).stem
    parts = filename_stem.split('_')

    # Long wav name is everything before last 3 parts (cluster, start, end)
    if len(parts) >= 3:
        long_wav_name = '_'.join(parts[:-3]) + '.wav'
    else:
        long_wav_name = filename_stem + '.wav'

    return long_wav_name


def load_merged_dataset(merged_h5_path):
    """
    Load merged sample data from HDF5 file.

    Args:
        merged_h5_path: Path to merged_dataset.h5 file

    Returns:
        DataFrame with columns: merged_id, cluster_id, start_time, end_time,
                                probability, long_wav_name
    """
    print(f"\n{'='*80}")
    print(f"LOADING MERGED DATASET")
    print(f"{'='*80}")
    print(f"HDF5 file: {merged_h5_path}\n")

    with h5py.File(merged_h5_path, 'r') as hf:
        # Load merged samples data
        merged_ids = [uid.decode() if isinstance(uid, bytes) else uid
                     for uid in hf['merged_samples']['merged_unique_ids'][:]]

        merged_wav_paths = [wp.decode() if isinstance(wp, bytes) else wp
                           for wp in hf['merged_samples']['merged_wav_paths'][:]]

        start_times = hf['merged_samples']['start_times'][:]
        end_times = hf['merged_samples']['end_times'][:]

        # Load cluster probabilities (averaged from constituent samples)
        cluster_probs = hf['merged_samples']['merged_cluster_probs_avgd'][:]

        # Check if LP labels exist (from Stage 4), otherwise use initial cluster labels
        if 'lp_labels' in hf['merged_samples']:
            cluster_labels = hf['merged_samples']['lp_labels'][:]
            label_source = "LP (Label Propagation) labels from Stage 4"
        else:
            cluster_labels = hf['merged_samples']['merged_cluster_labels_avgd'][:]
            label_source = "initial cluster labels from Stage 3"

        print(f"✓ Using {label_source}")

    print(f"✓ Loaded {len(merged_ids)} merged samples")

    # Extract long wav names from merged_wav_paths
    long_wav_names = [extract_long_wav_name(wp) for wp in merged_wav_paths]

    # Create DataFrame
    df = pd.DataFrame({
        'merged_id': merged_ids,
        'cluster_id': cluster_labels,
        'start_time': start_times,
        'end_time': end_times,
        'probability': cluster_probs,
        'long_wav_name': long_wav_names
    })

    # Sort by long_wav_name, then start_time
    df = df.sort_values(['long_wav_name', 'start_time']).reset_index(drop=True)

    print(f"\n✓ Dataset statistics:")
    print(f"  - Unique long wav files: {df['long_wav_name'].nunique()}")
    print(f"  - Total merged samples: {len(df)}")
    print(f"  - Cluster distribution:")
    for cluster_id, count in df['cluster_id'].value_counts().sort_index().items():
        avg_prob = df[df['cluster_id'] == cluster_id]['probability'].mean()
        print(f"    Cluster {cluster_id}: {count} samples (avg prob: {avg_prob:.3f})")

    return df


def export_to_csv(df, output_csv_path):
    """
    Export DataFrame to CSV file (comma-separated, no header).

    Args:
        df: DataFrame with columns: merged_id, cluster_id, start_time, end_time,
                                    probability, long_wav_name
        output_csv_path: Output CSV file path
    """
    print(f"\n{'='*80}")
    print(f"EXPORTING TO CSV")
    print(f"{'='*80}")
    print(f"Output file: {output_csv_path}")
    print(f"Format: Comma-separated, no header\n")

    # Create output directory if it doesn't exist
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Select columns in the required order
    df_export = df[['merged_id', 'cluster_id', 'start_time', 'end_time',
                    'probability', 'long_wav_name']]

    # Export to CSV: comma-separated, no header
    df_export.to_csv(output_csv_path, sep=',', index=False, header=False,
                     float_format='%.6f')

    print(f"✓ Exported {len(df_export)} rows")
    print(f"  - Columns (in order): merged_id, cluster_id, start_time, end_time, probability, long_wav_name")
    print(f"  - Format: Comma-separated")
    print(f"  - Header: None")

    print(f"\nFirst 5 rows preview:")
    for i, row in df_export.head().iterrows():
        print(f"  {row['merged_id']},{row['cluster_id']},{row['start_time']:.6f},"
              f"{row['end_time']:.6f},{row['probability']:.6f},{row['long_wav_name']}")

    print(f"\nLast 5 rows preview:")
    for i, row in df_export.tail().iterrows():
        print(f"  {row['merged_id']},{row['cluster_id']},{row['start_time']:.6f},"
              f"{row['end_time']:.6f},{row['probability']:.6f},{row['long_wav_name']}")


# ============================================================================
# DEFAULT EXAMPLE VARIABLES (for standalone testing)
# ============================================================================

root_ex = Path.home().joinpath('Dropbox', 'DATASETS_AUDIO', 'Unsupervised_Pipeline')
dataset_name_ex = 'TestAO-Irma'

# Stage 3 merged dataset (input)
stg3_folder_ex = root_ex / dataset_name_ex / 'STG_3' / f'STG3_EXP010-SHAS-DV-hdb'
merged_dataset_h5_ex = stg3_folder_ex / 'merged_dataset.h5'

# Output CSV
output_csv_ex = stg3_folder_ex / 'merged_predictions.csv'

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(
    description='Export merged dataset to CSV (comma-separated, no header) for metric scripts',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
    # Use default paths (standalone)
    python3 Stg3g_export_merged_csv.py

    # Specify custom paths
    python3 Stg3g_export_merged_csv.py \\
        --merged_dataset_h5 /path/to/merged_dataset.h5 \\
        --output_csv /path/to/output.csv

Output Format:
    CSV with columns: merged_id, cluster_id, start_time, end_time, probability, long_wav_name
    - Comma-separated
    - No header row
    - Sorted by long_wav_name, then start_time
    """
)

parser.add_argument(
    '--merged_dataset_h5',
    type=str,
    default=str(merged_dataset_h5_ex),
    help=f'Path to merged_dataset.h5 file (default: {merged_dataset_h5_ex})'
)

parser.add_argument(
    '--output_csv',
    type=str,
    default=str(output_csv_ex),
    help=f'Output CSV file path (default: {output_csv_ex})'
)

args = parser.parse_args()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    try:
        # Convert to Path objects
        merged_h5_path = Path(args.merged_dataset_h5)
        output_csv_path = Path(args.output_csv)

        # Validate input file exists
        if not merged_h5_path.exists():
            raise FileNotFoundError(f"Merged dataset not found: {merged_h5_path}")

        # Load merged dataset
        df_merged = load_merged_dataset(merged_h5_path)

        # Export to CSV
        export_to_csv(df_merged, output_csv_path)

        print(f"\n{'='*80}")
        print(f"EXPORT COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"\nOutput file: {output_csv_path}")
        print(f"Total rows: {len(df_merged)}")
        print()

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\n{'='*80}")
        print(f"ERROR: File not found")
        print(f"{'='*80}")
        print(f"{str(e)}\n")
        sys.exit(1)

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: Export failed")
        print(f"{'='*80}")
        print(f"{str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
