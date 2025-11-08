import argparse
import os
import pandas as pd
import h5py
import numpy as np
from pathlib import Path


def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def load_merged_sample_temporal_info(merged_h5_path, merged_unique_ids):
    """
    Load temporal information (start_time, end_time) for merged samples from HDF5.

    Parameters:
    -----------
    merged_h5_path : Path
        Path to merged HDF5 dataset
    merged_unique_ids : list
        List of merged unique IDs to retrieve temporal info for

    Returns:
    --------
    dict : Dictionary mapping merged_unique_id to (start_time, end_time, wav_path)
    """
    temporal_info = {}

    with h5py.File(merged_h5_path, 'r') as hf:
        # Load all merged sample data
        all_unique_ids = [uid.decode() if isinstance(uid, bytes) else uid
                         for uid in hf['merged_samples']['merged_unique_ids'][:]]
        all_wav_paths = [wp.decode() if isinstance(wp, bytes) else wp
                        for wp in hf['merged_samples']['merged_wav_paths'][:]]
        all_start_times = hf['merged_samples']['start_times'][:]
        all_end_times = hf['merged_samples']['end_times'][:]

        # Create mapping for requested IDs
        for uid in merged_unique_ids:
            if uid in all_unique_ids:
                idx = all_unique_ids.index(uid)
                temporal_info[uid] = {
                    'start_time': float(all_start_times[idx]),
                    'end_time': float(all_end_times[idx]),
                    'wav_path': all_wav_paths[idx]
                }
            else:
                print(f"Warning: Merged unique ID '{uid}' not found in HDF5")

    return temporal_info


# ============================================================================
# ARGUMENT PARSING
# ============================================================================
base_path_ex = Path.home().joinpath('Dropbox', 'DATASETS_AUDIO', 'Unsupervised_Pipeline', 'TestAO-Irma')
stg3_folder_ex = base_path_ex.joinpath('STG_3', 'STG3_EXP011-SHAS-DV-hdb')
stg3_al_folder_ex = stg3_folder_ex / 'active_learning'
stg3_al_csv_ex = stg3_al_folder_ex / 'active_learning_samples.csv'
merged_h5_ex = stg3_folder_ex / 'merged_dataset.h5'

stg4_folder_ex = base_path_ex.joinpath('STG_4', 'STG4_EXP011-SHAS-DV-hdb')
stg4_al_folder_ex = stg4_folder_ex / 'AL_input'

parser = argparse.ArgumentParser(
    description='Stage 4a: Generate Active Learning input files per long WAV from merged samples'
)
parser.add_argument(
    '--stg3_al_input',
    type=valid_path,
    default=stg3_al_csv_ex,
    help='Stage 3f Active Learning CSV path (with merged sample selections)'
)
parser.add_argument(
    '--merged_dataset_h5',
    default=merged_h5_ex,
    help='Merged samples HDF5 dataset path'
)
parser.add_argument(
    '--stg4_al_folder',
    default=stg4_al_folder_ex,
    help='Stage 4 AL input folder path'
)

args = parser.parse_args()

stg3_al_csv = args.stg3_al_input
merged_h5_path = Path(args.merged_dataset_h5)
stg4_al_folder = Path(args.stg4_al_folder)

# ============================================================================
# LOAD ACTIVE LEARNING CSV
# ============================================================================
# CSV columns: cluster_id, sample_index, merged_unique_id, wav_path,
#              selection_reason, hdbscan_prob, gt_label, suggested_label
if not stg3_al_csv.exists():
    raise FileNotFoundError(f"Active Learning CSV not found: {stg3_al_csv}")

print(f"\n{'='*80}")
print(f"GENERATING ACTIVE LEARNING INPUT FILES")
print(f"{'='*80}")
print(f"Reading AL input from: {stg3_al_csv}")

df_al = pd.read_csv(stg3_al_csv)
print(f"  ✓ {len(df_al)} samples selected for Active Learning")
print(f"  - Unique cluster IDs: {df_al['cluster_id'].nunique()}")
print(f"  - Selection strategy breakdown:")
for strategy, count in df_al['selection_reason'].value_counts().items():
    print(f"    • {strategy}: {count}")

# ============================================================================
# LOAD TEMPORAL INFO FROM HDF5
# ============================================================================
print(f"\nLoading temporal information from HDF5: {merged_h5_path}")
if not merged_h5_path.exists():
    raise FileNotFoundError(f"Merged HDF5 dataset not found: {merged_h5_path}")

merged_unique_ids = df_al['merged_unique_id'].tolist()
temporal_info = load_merged_sample_temporal_info(merged_h5_path, merged_unique_ids)

print(f"  ✓ Loaded temporal info for {len(temporal_info)} merged samples")

# ============================================================================
# ENRICH DATAFRAME WITH TEMPORAL INFO
# ============================================================================
df_al['start_time'] = df_al['merged_unique_id'].map(
    lambda uid: temporal_info[uid]['start_time'] if uid in temporal_info else None
)
df_al['end_time'] = df_al['merged_unique_id'].map(
    lambda uid: temporal_info[uid]['end_time'] if uid in temporal_info else None
)

# Extract long WAV name from wav_path (format: longwav_label_start_end.wav)
df_al['long_wav'] = df_al['wav_path'].apply(
    lambda p: Path(p).stem.rsplit('_', 3)[0]
)

# Check for any missing temporal info
missing_count = df_al['start_time'].isna().sum()
if missing_count > 0:
    print(f"\n  ⚠ Warning: {missing_count} samples missing temporal info")

# ============================================================================
# GENERATE PER-LONG-WAV AL INPUT FILES
# ============================================================================
print(f"\nGenerating AL input files per long WAV...")

grouped = df_al.groupby('long_wav')
files_created = 0

for long_wav, group in grouped:
    output_csv = stg4_al_folder / f"{long_wav}_ALinput.csv"

    # Output format: sample_index, cluster_id, start_time, end_time
    # This format is compatible with downstream AL processing
    output_data = group[['sample_index', 'cluster_id', 'start_time', 'end_time']].copy()

    # Remove any rows with missing temporal info
    output_data = output_data.dropna()

    # Round start_time and end_time to 2 decimal places for consistency
    output_data['start_time'] = output_data['start_time'].round(2)
    output_data['end_time'] = output_data['end_time'].round(2)

    # Save as CSV (no header, tab-separated or comma-separated based on downstream needs)
    output_data.to_csv(output_csv, index=False, header=False)

    files_created += 1
    print(f"  ✓ Created: {output_csv} ({len(output_data)} samples)")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print(f"ACTIVE LEARNING INPUT GENERATION COMPLETED")
print(f"{'='*80}")
print(f"Total AL input files created: {files_created}")
print(f"Output directory: {stg4_al_folder}")
print(f"\nFiles created:")
for long_wav in grouped.groups.keys():
    print(f"  - {long_wav}_ALinput.csv")
print(f"\nNext step: Use these files for manual labeling in Active Learning loop")
