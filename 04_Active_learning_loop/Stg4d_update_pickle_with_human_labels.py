"""
Stage 4d: Update merged HDF5 dataset with human labels from Active Learning webapp.

This script reads human labels from CSV files, matches them to merged samples,
and updates the HDF5 dataset with the human-provided speaker labels.

Handles:
- Time corrections (when humans adjusted start/end times)
- Split samples (when humans split a sample into multiple speakers)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py
import sys
import re
from collections import defaultdict

# ============================================================================
# HARDCODED CONFIGURATION - EDIT THESE PATHS
# ============================================================================
DATASET_NAME = 'TestAO-Irma'
STG3_METHOD = 'STG3_EXP010-SHAS-DV-hdb'
LP_METHOD = 'LP1'
OVERLAP_THRESHOLD = 90.0  # Minimum overlap percentage for matching

# Construct paths
BASE_PATH = Path.home().joinpath('Dropbox', 'DATASETS_AUDIO', 'Unsupervised_Pipeline', DATASET_NAME)
STG3_FOLDER = BASE_PATH.joinpath('STG_3', STG3_METHOD)
STG4_FOLDER = BASE_PATH.joinpath('STG_4', f'STG4_{LP_METHOD}')

# Input files
MERGED_H5_PATH = STG3_FOLDER / 'merged_dataset.h5'
WEBAPP_RESULTS_FOLDER = STG4_FOLDER / 'webapp_results'

# Output folder
OUTPUT_FOLDER = STG4_FOLDER / 'updated_h5_data'
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Output HDF5 path
OUTPUT_H5_PATH = OUTPUT_FOLDER / 'merged_dataset_with_labels.h5'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_base_name_from_csv_filename(csv_filename):
    """Extract base_name_long_wav from CSV filename with timestamp format."""
    name_without_ext = csv_filename.replace('.csv', '')
    timestamp_pattern = r'-\d{4}-\d{2}-\d{2}T[\d_]+\.?\d*Z?$'
    base_name = re.sub(timestamp_pattern, '', name_without_ext)

    if base_name == name_without_ext:
        return name_without_ext

    return base_name


def calculate_overlap_percentage(start1, end1, start2, end2):
    """Calculate percentage of overlap between two time intervals."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_start >= overlap_end:
        return 0.0

    overlap_duration = overlap_end - overlap_start
    original_duration = end1 - start1

    if original_duration == 0:
        return 0.0

    return (overlap_duration / original_duration) * 100.0


def read_human_labels_csv(csv_folder_path):
    """
    Read all human-labeled CSV files from webapp_results folder.

    Returns:
    --------
    human_labels_df : pandas.DataFrame
        DataFrame with columns: SampleIndex, SpeakerLP, StartTime, EndTime, ClusterPred
    split_samples : dict
        Dictionary mapping original SampleIndex to list of split entries
    """
    csv_folder = Path(csv_folder_path)

    if not csv_folder.exists() or not csv_folder.is_dir():
        print(f"Error: Folder '{csv_folder}' does not exist")
        sys.exit(1)

    csv_files = list(csv_folder.glob("*.csv"))

    if not csv_files:
        print(f"Error: No CSV files found in '{csv_folder}'")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files")

    all_dataframes = []

    for csv_file in csv_files:
        base_name = extract_base_name_from_csv_filename(csv_file.name)
        print(f"  Reading: {csv_file.name}")
        print(f"    Base name: {base_name}")

        try:
            df = pd.read_csv(csv_file, sep='\t', encoding='utf-8')
            df['source_csv'] = csv_file.name
            df['base_name'] = base_name
            all_dataframes.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            sys.exit(1)

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Total human-labeled samples: {len(combined_df)}")

    # Detect split samples
    split_samples = defaultdict(list)
    for idx, row in combined_df.iterrows():
        sample_idx = int(row['SampleIndex'])
        split_samples[sample_idx].append(idx)
    split_samples = dict(split_samples)

    # Add split information
    combined_df['OriginalIndex'] = combined_df['SampleIndex']
    combined_df['SplitOrder'] = 0

    for original_idx, splits in split_samples.items():
        if len(splits) > 1:
            print(f"\n  Split detected: SampleIndex {original_idx} -> {len(splits)} parts")
            for split_order, row_idx in enumerate(splits):
                combined_df.loc[row_idx, 'SplitOrder'] = split_order
                print(f"    Part {split_order}: {combined_df.loc[row_idx, 'StartTime']:.2f}s - {combined_df.loc[row_idx, 'EndTime']:.2f}s (Speaker: {combined_df.loc[row_idx, 'SpeakerLP']})")

    return combined_df, split_samples


def load_merged_h5_metadata(merged_h5_path):
    """
    Load merged sample metadata from HDF5.

    Returns:
    --------
    dict with merged sample data and mappings
    """
    print(f"\nLoading merged HDF5 metadata: {merged_h5_path}")

    with h5py.File(merged_h5_path, 'r') as hf:
        n_merged = len(hf['merged_samples']['merged_unique_ids'])

        data = {
            'merged_unique_ids': [uid.decode() if isinstance(uid, bytes) else uid
                                 for uid in hf['merged_samples']['merged_unique_ids'][:]],
            'merged_wav_paths': [wp.decode() if isinstance(wp, bytes) else wp
                                for wp in hf['merged_samples']['merged_wav_paths'][:]],
            'start_times': hf['merged_samples']['start_times'][:],
            'end_times': hf['merged_samples']['end_times'][:]
        }

        print(f"✓ Loaded {n_merged} merged samples")

    # Create index mapping (sample_index is position in arrays)
    data['idx_to_metadata'] = {}
    for idx in range(n_merged):
        data['idx_to_metadata'][idx] = {
            'merged_unique_id': data['merged_unique_ids'][idx],
            'wav_path': data['merged_wav_paths'][idx],
            'start_time': float(data['start_times'][idx]),
            'end_time': float(data['end_times'][idx]),
        }

    return data


def match_human_labels_to_merged_data(human_labels_df, merged_metadata, split_samples,
                                       overlap_threshold=90.0):
    """
    Match human labels to merged samples based on time overlap.

    Returns:
    --------
    matched_updates : list
        List of dicts with update information for existing samples
    new_samples : list
        List of dicts for new samples to be created from splits
    sample_metadata : dict
        Additional metadata for matched samples
    """
    matched_updates = []
    new_samples = []
    sample_metadata = {}

    print(f"\nMatching human labels (overlap threshold: {overlap_threshold}%)...")

    for _, row in human_labels_df.iterrows():
        original_idx = int(row['OriginalIndex'])
        speaker_lp = row['SpeakerLP']
        human_start = float(row['StartTime'])
        human_end = float(row['EndTime'])
        cluster_pred = int(row['ClusterPred'])
        split_order = int(row['SplitOrder'])
        is_split = original_idx in split_samples and len(split_samples[original_idx]) > 1

        # Get metadata for this sample
        if original_idx not in merged_metadata['idx_to_metadata']:
            print(f"  Warning: SampleIndex {original_idx} not found in merged data")
            continue

        metadata = merged_metadata['idx_to_metadata'][original_idx]
        merged_start = metadata['start_time']
        merged_end = metadata['end_time']

        # Calculate overlap
        overlap = calculate_overlap_percentage(merged_start, merged_end, human_start, human_end)

        # Check if times were modified
        time_modified = (abs(human_start - merged_start) > 0.01 or abs(human_end - merged_end) > 0.01)

        # Handle splits vs updates
        if is_split and split_order > 0:
            # New split sample (not the first part)
            new_sample_info = {
                'original_idx': original_idx,
                'split_order': split_order,
                'speaker_lp': speaker_lp,
                'start_time': human_start,
                'end_time': human_end,
                'cluster_pred': cluster_pred,
                'overlap_percent': overlap,
                'merged_unique_id': metadata['merged_unique_id']
            }
            new_samples.append(new_sample_info)
            print(f"  New split: SampleIndex {original_idx} Part {split_order} -> {speaker_lp} ({human_start:.2f}s-{human_end:.2f}s)")
        else:
            # Update existing sample (or first part of split)
            if overlap >= overlap_threshold or time_modified:
                update_info = {
                    'sample_idx': original_idx,
                    'speaker_lp': speaker_lp,
                    'start_time': human_start,
                    'end_time': human_end,
                    'cluster_pred': cluster_pred,
                    'overlap_percent': overlap,
                    'time_modified': time_modified,
                    'is_split_first': is_split
                }
                matched_updates.append(update_info)

                status_msg = f"  Matched: [{original_idx}] -> {speaker_lp} (overlap: {overlap:.1f}%"
                if time_modified:
                    status_msg += f", time: {merged_start:.2f}-{merged_end:.2f} → {human_start:.2f}-{human_end:.2f}"
                if is_split:
                    status_msg += ", split Part 0"
                status_msg += ")"
                print(status_msg)

                sample_metadata[original_idx] = {
                    'speaker_lp': speaker_lp,
                    'human_start': human_start,
                    'human_end': human_end,
                    'merged_start': merged_start,
                    'merged_end': merged_end,
                    'overlap_percent': overlap,
                    'time_modified': time_modified
                }
            else:
                print(f"  Warning: [{original_idx}] overlap too low ({overlap:.1f}% < {overlap_threshold}%)")

    print(f"\nTotal updates: {len(matched_updates)}")
    print(f"Total new splits: {len(new_samples)}")
    return matched_updates, new_samples, sample_metadata


def update_h5_with_human_labels(merged_h5_path, matched_updates, new_samples, output_h5_path):
    """
    Update HDF5 dataset with human labels.

    Creates a new HDF5 file with human_labels group containing speaker labels.
    """
    print(f"\nUpdating HDF5 with human labels...")
    print(f"  Input: {merged_h5_path}")
    print(f"  Output: {output_h5_path}")

    # Copy original HDF5 to new file
    import shutil
    shutil.copy2(merged_h5_path, output_h5_path)

    # Open and update
    with h5py.File(output_h5_path, 'a') as hf:
        n_samples = len(hf['merged_samples']['merged_unique_ids'])

        # Create speaker_lp array (empty string for unlabeled)
        speaker_lp = np.array([''] * n_samples, dtype='S50')

        # Update with human labels
        updated_count = 0
        for update in matched_updates:
            sample_idx = update['sample_idx']
            if 0 <= sample_idx < n_samples:
                speaker_lp[sample_idx] = update['speaker_lp'].encode('utf-8')
                updated_count += 1

        print(f"  ✓ Updated {updated_count} samples with labels")

        # Create human_labels group
        if 'human_labels' in hf:
            del hf['human_labels']

        hl_group = hf.create_group('human_labels')

        # Store speaker labels
        hl_group.create_dataset(
            'speaker_lp',
            data=speaker_lp,
            dtype='S50',
            compression='gzip',
            compression_opts=4
        )

        # Store metadata
        hl_group.attrs['n_labeled'] = np.sum(speaker_lp != b'')
        hl_group.attrs['n_unlabeled'] = np.sum(speaker_lp == b'')
        hl_group.attrs['n_total'] = n_samples
        hl_group.attrs['source'] = 'STG4D_UPDATE_WITH_HUMAN_LABELS'

        # Handle split samples (store as new samples in a separate group)
        if new_samples:
            print(f"\n  Creating {len(new_samples)} new split samples...")

            if 'human_splits' in hf:
                del hf['human_splits']

            split_group = hf.create_group('human_splits')

            # Prepare split sample data
            split_original_indices = []
            split_orders = []
            split_speaker_lps = []
            split_start_times = []
            split_end_times = []
            split_cluster_preds = []
            split_unique_ids = []

            for new_sample in new_samples:
                split_original_indices.append(new_sample['original_idx'])
                split_orders.append(new_sample['split_order'])
                split_speaker_lps.append(new_sample['speaker_lp'].encode('utf-8'))
                split_start_times.append(new_sample['start_time'])
                split_end_times.append(new_sample['end_time'])
                split_cluster_preds.append(new_sample['cluster_pred'])

                # Generate new unique ID for split
                orig_id = new_sample['merged_unique_id']
                new_id = f"{orig_id}_split{new_sample['split_order']}"
                split_unique_ids.append(new_id.encode('utf-8'))

                print(f"    Split [{new_sample['original_idx']}] Part {new_sample['split_order']}: {new_sample['speaker_lp']} ({new_sample['start_time']:.2f}s-{new_sample['end_time']:.2f}s)")

            # Store split data
            split_group.create_dataset('original_indices', data=np.array(split_original_indices, dtype='int32'))
            split_group.create_dataset('split_orders', data=np.array(split_orders, dtype='int32'))
            split_group.create_dataset('speaker_lps', data=np.array(split_speaker_lps, dtype='S50'))
            split_group.create_dataset('start_times', data=np.array(split_start_times, dtype='float32'))
            split_group.create_dataset('end_times', data=np.array(split_end_times, dtype='float32'))
            split_group.create_dataset('cluster_preds', data=np.array(split_cluster_preds, dtype='int32'))
            split_group.create_dataset('unique_ids', data=np.array(split_unique_ids, dtype='S50'))

            split_group.attrs['n_splits'] = len(new_samples)

            print(f"  ✓ Created {len(new_samples)} split samples in /human_splits/")

        print(f"\n  HDF5 structure updated:")
        print(f"    /human_labels/speaker_lp - Speaker labels for merged samples")
        if new_samples:
            print(f"    /human_splits/ - Split sample information ({len(new_samples)} splits)")

    # Summary
    labeled_count = np.sum(speaker_lp != b'')
    unlabeled_count = np.sum(speaker_lp == b'')

    print(f"\n  Summary:")
    print(f"    Total samples: {n_samples}")
    print(f"    Labeled: {labeled_count}")
    print(f"    Unlabeled: {unlabeled_count}")
    if new_samples:
        print(f"    New splits: {len(new_samples)}")

    return {
        'original_samples': n_samples,
        'labeled_samples': int(labeled_count),
        'unlabeled_samples': int(unlabeled_count),
        'new_splits': len(new_samples)
    }


def save_matching_report(sample_metadata, matched_updates, new_samples, output_folder):
    """Save detailed report of matching process."""
    report_path = output_folder / 'human_labels_matching_report.csv'

    print(f"\nSaving matching report: {report_path}")

    report_data = []

    # Existing sample updates
    for update in matched_updates:
        sample_idx = update['sample_idx']
        metadata = sample_metadata.get(sample_idx, {})
        report_data.append({
            'SampleIndex': sample_idx,
            'Type': 'Update' + (' (Split Part 0)' if update.get('is_split_first') else ''),
            'SpeakerLP': update['speaker_lp'],
            'HumanStartTime': update['start_time'],
            'HumanEndTime': update['end_time'],
            'MergedStartTime': metadata.get('merged_start', update['start_time']),
            'MergedEndTime': metadata.get('merged_end', update['end_time']),
            'OverlapPercent': update['overlap_percent'],
            'TimeModified': 'Yes' if update['time_modified'] else 'No',
            'ClusterPred': update['cluster_pred']
        })

    # New split samples
    for new_sample in new_samples:
        report_data.append({
            'SampleIndex': f"{new_sample['original_idx']} (split {new_sample['split_order']})",
            'Type': f'New Split Part {new_sample["split_order"]}',
            'SpeakerLP': new_sample['speaker_lp'],
            'HumanStartTime': new_sample['start_time'],
            'HumanEndTime': new_sample['end_time'],
            'MergedStartTime': '-',
            'MergedEndTime': '-',
            'OverlapPercent': new_sample['overlap_percent'],
            'TimeModified': 'Yes',
            'ClusterPred': new_sample['cluster_pred']
        })

    report_df = pd.DataFrame(report_data)
    report_df.to_csv(report_path, index=False)

    print(f"  ✓ Report saved with {len(report_data)} entries")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("="*80)
print("STAGE 4D: UPDATE HDF5 WITH HUMAN LABELS")
print("="*80)
print(f"Dataset: {DATASET_NAME}")
print(f"Stage 3 Method: {STG3_METHOD}")
print(f"LP Method: {LP_METHOD}")
print(f"Overlap Threshold: {OVERLAP_THRESHOLD}%")
print(f"\nInput Files:")
print(f"  Merged HDF5: {MERGED_H5_PATH}")
print(f"  Webapp CSV Folder: {WEBAPP_RESULTS_FOLDER}")
print(f"\nOutput Files:")
print(f"  Updated HDF5: {OUTPUT_H5_PATH}")
print(f"  Report Folder: {OUTPUT_FOLDER}")
print("="*80)

# Validate inputs
if not MERGED_H5_PATH.exists():
    print(f"\nError: Merged HDF5 not found: {MERGED_H5_PATH}")
    sys.exit(1)

if not WEBAPP_RESULTS_FOLDER.exists():
    print(f"\nError: Webapp results folder not found: {WEBAPP_RESULTS_FOLDER}")
    sys.exit(1)

# Step 1: Read human labels from CSV files
print("\n" + "="*80)
print("STEP 1: Reading Human Labels from CSV")
print("="*80)
human_labels_df, split_samples = read_human_labels_csv(WEBAPP_RESULTS_FOLDER)

# Step 2: Load merged HDF5 metadata
print("\n" + "="*80)
print("STEP 2: Loading Merged HDF5 Metadata")
print("="*80)
merged_metadata = load_merged_h5_metadata(MERGED_H5_PATH)

# Step 3: Match human labels to merged data
print("\n" + "="*80)
print("STEP 3: Matching Human Labels to Merged Data")
print("="*80)
matched_updates, new_samples, sample_metadata = match_human_labels_to_merged_data(
    human_labels_df,
    merged_metadata,
    split_samples,
    overlap_threshold=OVERLAP_THRESHOLD
)

# Step 4: Update HDF5 with human labels
print("\n" + "="*80)
print("STEP 4: Updating HDF5 with Human Labels")
print("="*80)
update_summary = update_h5_with_human_labels(
    MERGED_H5_PATH,
    matched_updates,
    new_samples,
    OUTPUT_H5_PATH
)

# Step 5: Save matching report
print("\n" + "="*80)
print("STEP 5: Saving Matching Report")
print("="*80)
save_matching_report(sample_metadata, matched_updates, new_samples, OUTPUT_FOLDER)

# Final summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Human-labeled samples from CSV: {len(human_labels_df)}")
print(f"Original merged samples: {update_summary['original_samples']}")
print(f"Successfully labeled: {update_summary['labeled_samples']}")
print(f"Unlabeled: {update_summary['unlabeled_samples']}")
print(f"New split samples: {update_summary['new_splits']}")
print(f"\nOutput HDF5: {OUTPUT_H5_PATH}")
print(f"Matching report: {OUTPUT_FOLDER / 'human_labels_matching_report.csv'}")
print("="*80)
print("\nProcess completed successfully!")
