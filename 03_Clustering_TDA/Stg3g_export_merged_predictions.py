"""
Export Merged Predictions to CSV

This script:
1. Loads the merged_dataset.h5 file
2. Extracts the latest predicted labels, start_time, end_time, and long_wav_filename
3. Exports the data to a tab-separated CSV file
4. Intended to be called from STG3A_META_HDB.sh bash script

The exported CSV contains:
- long_wav_filename: Original long wav file name (extracted from merged_wav_path)
- start_time: Start time in seconds
- end_time: End time in seconds
- predicted_label: Latest predicted cluster label
- merged_unique_id: Unique identifier for the merged sample
- duration: Duration in seconds
- n_constituents: Number of constituent samples
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


def load_merged_predictions(merged_h5_path):
    """
    Load merged sample predictions from HDF5 file.

    Args:
        merged_h5_path: Path to merged_dataset.h5 file

    Returns:
        DataFrame with columns: long_wav_filename, start_time, end_time,
                                predicted_label, merged_unique_id, duration, n_constituents
    """
    print(f"\n{'='*80}")
    print(f"LOADING MERGED PREDICTIONS")
    print(f"{'='*80}")
    print(f"HDF5 file: {merged_h5_path}\n")

    with h5py.File(merged_h5_path, 'r') as hf:
        # Load merged samples data
        merged_unique_ids = [uid.decode() if isinstance(uid, bytes) else uid
                            for uid in hf['merged_samples']['merged_unique_ids'][:]]

        merged_wav_paths = [wp.decode() if isinstance(wp, bytes) else wp
                           for wp in hf['merged_samples']['merged_wav_paths'][:]]

        start_times = hf['merged_samples']['start_times'][:]
        end_times = hf['merged_samples']['end_times'][:]
        durations = hf['merged_samples']['durations'][:]
        n_constituents = hf['merged_samples']['n_constituents'][:]

        # Check if LP labels exist (from stage 4), otherwise use initial cluster labels
        if 'lp_labels' in hf['merged_samples']:
            predicted_labels = hf['merged_samples']['lp_labels'][:]
            print("✓ Using LP (Label Propagation) labels from Stage 4")
        else:
            predicted_labels = hf['merged_samples']['merged_cluster_labels_avgd'][:]
            print("✓ Using initial cluster labels from Stage 3")

    print(f"✓ Loaded {len(merged_unique_ids)} merged samples")

    # Extract long_wav_filename from merged_wav_path
    # Format: {long_wav_filename}_{predicted_label}_{start_time}_{stop_time}.wav
    long_wav_filenames = []
    for wav_path in merged_wav_paths:
        # Get filename without extension
        filename_stem = Path(wav_path).stem
        # Split by underscore and take the first part
        parts = filename_stem.split('_')
        # The long_wav_filename is everything before the last 3 parts (label, start, end)
        long_wav_filename = '_'.join(parts[:-3]) + '.wav'
        long_wav_filenames.append(long_wav_filename)

    # Create DataFrame
    df = pd.DataFrame({
        'long_wav_filename': long_wav_filenames,
        'start_time': start_times,
        'end_time': end_times,
        'predicted_label': predicted_labels,
        'merged_unique_id': merged_unique_ids,
        'duration': durations,
        'n_constituents': n_constituents
    })

    # Sort by long_wav_filename, then start_time
    df = df.sort_values(['long_wav_filename', 'start_time']).reset_index(drop=True)

    print(f"\n✓ Extracted predictions:")
    print(f"  - Unique files: {df['long_wav_filename'].nunique()}")
    print(f"  - Total segments: {len(df)}")
    print(f"  - Label distribution:")
    for label, count in df['predicted_label'].value_counts().sort_index().items():
        print(f"    Label {label}: {count} segments")

    return df


def export_predictions_to_csv(df, output_csv_path):
    """
    Export predictions DataFrame to tab-separated CSV file.

    Args:
        df: DataFrame with prediction data
        output_csv_path: Output CSV file path
    """
    print(f"\n{'='*80}")
    print(f"EXPORTING PREDICTIONS TO CSV")
    print(f"{'='*80}")
    print(f"Output file: {output_csv_path}\n")

    # Create output directory if it doesn't exist
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Export to tab-separated CSV
    df.to_csv(output_csv_path, sep='\t', index=False, float_format='%.3f')

    print(f"✓ Exported {len(df)} predictions to CSV")
    print(f"  - Columns: {', '.join(df.columns)}")
    print(f"  - Format: Tab-separated")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(
    description='Export merged predictions from HDF5 to CSV file'
)

parser.add_argument(
    '--merged_dataset_h5',
    type=valid_path,
    required=True,
    help='Path to merged_dataset.h5 file'
)

parser.add_argument(
    '--output_csv',
    type=str,
    required=True,
    help='Output CSV file path for predictions'
)

args = parser.parse_args()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

try:
    # Load predictions from HDF5
    df_predictions = load_merged_predictions(args.merged_dataset_h5)

    # Export to CSV
    output_csv_path = Path(args.output_csv)
    export_predictions_to_csv(df_predictions, output_csv_path)

    print(f"\n{'='*80}")
    print(f"EXPORT COMPLETED SUCCESSFULLY")
    print(f"{'='*80}\n")
    sys.exit(0)

except Exception as e:
    print(f"\n{'='*80}")
    print(f"ERROR: Export failed")
    print(f"{'='*80}")
    print(f"{str(e)}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)
