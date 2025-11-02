from pathlib import Path
import pickle
import json
import argparse
import numpy as np
import pandas as pd
import csv
import sys
import re
from collections import defaultdict

# Fix for numpy version compatibility when loading pickle files
# Handle differences between numpy 1.x and 2.x
try:
    # Try to import numpy._core (numpy 2.x)
    import numpy._core as numpy_core
    import numpy._core.multiarray as numpy_multiarray
    # Create backward compatibility mappings
    sys.modules['numpy.core'] = numpy_core
    sys.modules['numpy.core.multiarray'] = numpy_multiarray
    sys.modules['numpy.core._multiarray_umath'] = numpy_multiarray
except (ImportError, AttributeError):
    # numpy 1.x or different structure
    try:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        sys.modules['numpy._core'] = numpy_core
        sys.modules['numpy._core.multiarray'] = numpy_multiarray
    except ImportError:
        pass  # Will handle during unpickling

class NumpyCompatUnpickler(pickle.Unpickler):
    """Custom unpickler to handle numpy version compatibility"""
    def find_class(self, module, name):
        # Remap old numpy module paths to new ones
        if 'numpy.core' in module or 'numpy._core' in module:
            # Try both paths
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                # Try alternative module path
                if 'numpy.core' in module:
                    module_alt = module.replace('numpy.core', 'numpy._core')
                elif 'numpy._core' in module:
                    module_alt = module.replace('numpy._core', 'numpy.core')
                else:
                    module_alt = module

                try:
                    return super().find_class(module_alt, name)
                except ModuleNotFoundError:
                    # Last resort: try numpy directly
                    module_simple = 'numpy'
                    return super().find_class(module_simple, name)
        return super().find_class(module, name)


def extract_base_name_from_csv_filename(csv_filename):
    """
    Extract the base_name_long_wav from CSV filename with timestamp format.

    Format: {base_name_long_wav}_{timestamp}.csv
    Example: G-C1L1P-Apr27-E-Irma_q2_03-08-377-2025-11-01T18_47_23.330Z.csv
    Base name: G-C1L1P-Apr27-E-Irma_q2_03-08-377

    Parameters:
    -----------
    csv_filename : str
        Name of the CSV file

    Returns:
    --------
    base_name : str
        Extracted base name without timestamp
    """
    # Remove .csv extension
    name_without_ext = csv_filename.replace('.csv', '')

    # Pattern to match timestamp: YYYY-MM-DDTHH_MM_SS.SSSZ or similar
    # Look for date pattern: YYYY-MM-DD or similar followed by T
    timestamp_pattern = r'-\d{4}-\d{2}-\d{2}T[\d_]+\.?\d*Z?$'

    # Try to remove timestamp
    base_name = re.sub(timestamp_pattern, '', name_without_ext)

    # If no timestamp found, return the original name without extension
    if base_name == name_without_ext:
        # Maybe it's a simpler format like "tts4_easy_labels"
        return name_without_ext

    return base_name


def calculate_overlap_percentage(start1, end1, start2, end2):
    """
    Calculate the percentage of overlap between two time intervals.

    Parameters:
    -----------
    start1, end1 : float
        First interval boundaries
    start2, end2 : float
        Second interval boundaries

    Returns:
    --------
    overlap_percent : float
        Percentage of overlap (0-100) relative to the first interval
    """
    # Calculate the overlap
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    # If no overlap, return 0
    if overlap_start >= overlap_end:
        return 0.0

    overlap_duration = overlap_end - overlap_start
    original_duration = end1 - start1

    # Avoid division by zero
    if original_duration == 0:
        return 0.0

    overlap_percent = (overlap_duration / original_duration) * 100.0
    return overlap_percent


def read_human_labels_csv(csv_folder_path):
    """
    Read all human-labeled CSV files from the webapp_results folder.
    Handles CSV files with timestamp format in their names.
    Detects split samples (same SampleIndex appearing multiple times).

    Parameters:
    -----------
    csv_folder_path : Path
        Path to the folder containing CSV files with human labels

    Returns:
    --------
    human_labels_df : pandas.DataFrame
        DataFrame with columns: SampleIndex, SpeakerLP, StartTime, EndTime, ClusterPred, OriginalIndex, SplitOrder
    split_samples : dict
        Dictionary mapping original SampleIndex to list of split entries
    """
    csv_folder = Path(csv_folder_path)

    if not csv_folder.exists() or not csv_folder.is_dir():
        print(f"Error: Folder '{csv_folder}' does not exist or is not a directory.")
        sys.exit(1)

    # Find all CSV files in the folder
    csv_files = list(csv_folder.glob("*.csv"))

    if not csv_files:
        print(f"Error: No CSV files found in '{csv_folder}'")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files in '{csv_folder}'")

    # Read and concatenate all CSV files
    all_dataframes = []

    for csv_file in csv_files:
        base_name = extract_base_name_from_csv_filename(csv_file.name)
        print(f"  Reading: {csv_file.name}")
        print(f"    Extracted base name: {base_name}")

        try:
            df = pd.read_csv(csv_file, sep='\t', encoding='utf-8')
            # Add source file information
            df['source_csv'] = csv_file.name
            df['base_name'] = base_name
            all_dataframes.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            sys.exit(1)

    # Concatenate all dataframes
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Total human-labeled samples: {len(combined_df)}")

        # Detect split samples (same SampleIndex appearing multiple times)
        split_samples = detect_split_samples(combined_df)

        # Add split information to dataframe
        combined_df['OriginalIndex'] = combined_df['SampleIndex']
        combined_df['SplitOrder'] = 0

        # Mark split samples
        for original_idx, splits in split_samples.items():
            if len(splits) > 1:
                print(f"\n  Split detected: SampleIndex {original_idx} -> {len(splits)} parts")
                for split_order, row_idx in enumerate(splits):
                    combined_df.loc[row_idx, 'SplitOrder'] = split_order
                    print(f"    Part {split_order}: {combined_df.loc[row_idx, 'StartTime']:.2f}s - {combined_df.loc[row_idx, 'EndTime']:.2f}s (Speaker: {combined_df.loc[row_idx, 'SpeakerLP']})")

        return combined_df, split_samples
    else:
        print("Error: No data loaded from CSV files")
        sys.exit(1)


def detect_split_samples(df):
    """
    Detect samples that have been split (same SampleIndex appearing multiple times).

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with human labels

    Returns:
    --------
    split_samples : dict
        Dictionary mapping SampleIndex to list of row indices where it appears
    """
    split_samples = defaultdict(list)

    for idx, row in df.iterrows():
        sample_idx = int(row['SampleIndex'])
        split_samples[sample_idx].append(idx)

    return dict(split_samples)


def load_merged_files_mapping(json_path):
    """
    Load the merged_files_mapping.json file.

    Parameters:
    -----------
    json_path : Path
        Path to the merged_files_mapping.json file

    Returns:
    --------
    mapping : dict
        Dictionary mapping merged file names to their metadata
    idx_to_filename : dict
        Dictionary mapping merged_idx to filename
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    # Create reverse mapping from merged_idx to filename
    idx_to_filename = {}
    for filename, metadata in mapping.items():
        idx_to_filename[metadata['merged_idx']] = filename

    return mapping, idx_to_filename


def match_human_labels_to_merged_data(human_labels_df, merged_files_mapping,
                                       idx_to_filename, split_samples,
                                       overlap_threshold=90.0):
    """
    Match human labels to merged data samples based on time overlap.
    Handles split samples and time corrections.

    Parameters:
    -----------
    human_labels_df : pandas.DataFrame
        DataFrame with human labels
    merged_files_mapping : dict
        Mapping from merged file names to their metadata
    idx_to_filename : dict
        Mapping from merged_idx to filename
    split_samples : dict
        Dictionary mapping original SampleIndex to list of split entries
    overlap_threshold : float
        Minimum overlap percentage to consider a match (default: 90.0)

    Returns:
    --------
    matched_updates : list
        List of dicts with update information for existing samples
    new_samples : list
        List of dicts with information for new samples to be created
    sample_metadata : dict
        Dictionary with additional metadata for each matched sample
    """
    matched_updates = []
    new_samples = []
    sample_metadata = {}

    print(f"\nMatching human labels to merged data (overlap threshold: {overlap_threshold}%)...")

    for _, row in human_labels_df.iterrows():
        original_idx = int(row['OriginalIndex'])
        sample_idx = int(row['SampleIndex'])
        speaker_lp = row['SpeakerLP']
        human_start = float(row['StartTime'])
        human_end = float(row['EndTime'])
        cluster_pred = int(row['ClusterPred'])
        split_order = int(row['SplitOrder'])
        is_split = original_idx in split_samples and len(split_samples[original_idx]) > 1

        # Get the filename from the mapping
        if original_idx not in idx_to_filename:
            print(f"  Warning: SampleIndex {original_idx} not found in merged data mapping")
            continue

        filename = idx_to_filename[original_idx]
        metadata = merged_files_mapping[filename]

        # Get original merged data times
        merged_start = metadata['start_time']
        merged_end = metadata['stop_time']

        # Calculate overlap
        overlap = calculate_overlap_percentage(merged_start, merged_end,
                                                human_start, human_end)

        # Check if times were modified
        time_modified = (abs(human_start - merged_start) > 0.01 or
                        abs(human_end - merged_end) > 0.01)

        # Handle based on whether it's a split or simple update
        if is_split and split_order > 0:
            # This is a new split sample (not the first one)
            new_sample_info = {
                'original_idx': original_idx,
                'split_order': split_order,
                'speaker_lp': speaker_lp,
                'start_time': human_start,
                'end_time': human_end,
                'cluster_pred': cluster_pred,
                'overlap_percent': overlap,
                'original_filename': filename,
                'original_metadata': metadata
            }
            new_samples.append(new_sample_info)
            print(f"  New split sample: SampleIndex {original_idx} Part {split_order} -> {speaker_lp} ({human_start:.2f}s - {human_end:.2f}s)")

        else:
            # This is either a non-split sample or the first part of a split
            if overlap >= overlap_threshold or time_modified:
                update_info = {
                    'sample_idx': original_idx,
                    'speaker_lp': speaker_lp,
                    'start_time': human_start,
                    'end_time': human_end,
                    'cluster_pred': cluster_pred,
                    'overlap_percent': overlap,
                    'time_modified': time_modified,
                    'is_split_first': is_split,
                    'filename': filename
                }
                matched_updates.append(update_info)

                status_msg = f"  Matched: SampleIndex {original_idx} -> {speaker_lp} (overlap: {overlap:.1f}%"
                if time_modified:
                    status_msg += f", time adjusted: {merged_start:.2f}-{merged_end:.2f} -> {human_start:.2f}-{human_end:.2f}"
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
                    'cluster_pred': cluster_pred,
                    'filename': filename,
                    'time_modified': time_modified,
                    'is_split': is_split
                }
            else:
                print(f"  Warning: SampleIndex {original_idx} overlap too low ({overlap:.1f}% < {overlap_threshold}%)")

    print(f"\nTotal updates for existing samples: {len(matched_updates)}")
    print(f"Total new split samples to create: {len(new_samples)}")
    return matched_updates, new_samples, sample_metadata


def create_new_split_samples(new_samples, original_data_arrays, merged_files_mapping,
                              clustering_data_dict=None, path_to_index=None,
                              recalculate_features=False):
    """
    Create new data entries for split samples.

    Parameters:
    -----------
    new_samples : list
        List of dicts with new sample information
    original_data_arrays : dict
        Dictionary containing all the original pickle data arrays
    merged_files_mapping : dict
        Mapping from merged file names to their metadata
    clustering_data_dict : dict, optional
        Dictionary with constituent files data for feature recalculation
    path_to_index : dict, optional
        Mapping from file stem to index for feature recalculation
    recalculate_features : bool
        Whether to recalculate features based on new time ranges

    Returns:
    --------
    new_data_dict : dict
        Dictionary with arrays of new sample data for each field
    """
    if not new_samples:
        return None

    print(f"\nCreating {len(new_samples)} new split samples...")
    if recalculate_features and clustering_data_dict is not None:
        print("  (with feature recalculation)")

    new_data = {
        'x_data': [],
        'paths': [],
        'hdb_data': [],
        'tsne_2d': [],
        'y_labels': [],
        'sample_labels': [],
        'sample_probs': [],
        'sample_outliers': [],
        'speaker_lp': []
    }

    for idx, new_sample in enumerate(new_samples):
        original_idx = new_sample['original_idx']
        split_order = new_sample['split_order']
        original_filename = new_sample['original_filename']

        print(f"\n  Split sample {idx+1}/{len(new_samples)} from original idx [{original_idx}]:")
        print(f"    Original file: {original_filename}")
        print(f"    Split part: {split_order}")
        print(f"    Time range: {new_sample['start_time']:.2f}s - {new_sample['end_time']:.2f}s")
        print(f"    Speaker: {new_sample['speaker_lp']}")

        # Try to recalculate features if enabled
        use_recalc = False
        if recalculate_features and clustering_data_dict is not None and path_to_index is not None:
            if original_filename in merged_files_mapping:
                constituent_files = merged_files_mapping[original_filename]['constituent_files']
                target_start = new_sample['start_time']
                target_end = new_sample['end_time']

                print(f"    Recalculating features from {len(constituent_files)} constituent files...")

                # Recalculate features
                recalc_features = recalculate_features_for_time_range(
                    constituent_files, target_start, target_end,
                    clustering_data_dict, path_to_index
                )

                if recalc_features is not None:
                    new_data['x_data'].append(recalc_features['x_data'])
                    new_data['hdb_data'].append(recalc_features['hdb_data'])
                    new_data['tsne_2d'].append(recalc_features['tsne_2d'])
                    use_recalc = True
                    print(f"    ✓ Features recalculated from {recalc_features['num_files']} filtered constituent files")
                    print(f"      X_data shape: {recalc_features['x_data'].shape}")
                    print(f"      t-SNE coords: [{recalc_features['tsne_2d'][0]:.3f}, {recalc_features['tsne_2d'][1]:.3f}]")
                else:
                    print(f"    ✗ Feature recalculation failed, using original features")

        # Fallback to copying data from original sample if recalc failed or disabled
        if not use_recalc:
            new_data['x_data'].append(original_data_arrays['x_data'][original_idx].copy())
            new_data['hdb_data'].append(original_data_arrays['hdb_data'][original_idx].copy())
            new_data['tsne_2d'].append(original_data_arrays['tsne_2d'][original_idx].copy())
            if recalculate_features:
                print(f"    → Using original sample features (no recalculation)")

        # Copy other metadata
        new_data['y_labels'].append(original_data_arrays['y_labels'][original_idx])
        new_data['sample_labels'].append(new_sample['cluster_pred'])
        new_data['sample_probs'].append(original_data_arrays['sample_probs'][original_idx])
        new_data['sample_outliers'].append(original_data_arrays['sample_outliers'][original_idx])
        new_data['speaker_lp'].append(new_sample['speaker_lp'])

        # Create new path name
        original_path = original_data_arrays['paths'][original_idx]
        # Extract components from original path
        # Format: {base}_{label}_{start}_{end}
        parts = original_path.split('_')
        if len(parts) >= 4:
            base = '_'.join(parts[:-3])
            new_path = f"{base}_{new_sample['cluster_pred']}_{new_sample['start_time']}_{new_sample['end_time']}_split{split_order}"
        else:
            new_path = f"{original_path}_split{split_order}"

        new_data['paths'].append(new_path)
        print(f"    New path: {new_path}")

    # Convert lists to numpy arrays
    new_data['x_data'] = np.array(new_data['x_data'])
    new_data['hdb_data'] = np.array(new_data['hdb_data'])
    new_data['tsne_2d'] = np.array(new_data['tsne_2d'])
    new_data['y_labels'] = np.array(new_data['y_labels'])
    new_data['sample_labels'] = np.array(new_data['sample_labels'])
    new_data['sample_probs'] = np.array(new_data['sample_probs'])
    new_data['sample_outliers'] = np.array(new_data['sample_outliers'])
    new_data['speaker_lp'] = np.array(new_data['speaker_lp'], dtype=object)

    return new_data


def load_clustering_data(clustering_pickle_path, feats_pickle_path):
    """
    Load the original clustering_data.pickle and features.pickle files.

    Parameters:
    -----------
    clustering_pickle_path : Path
        Path to clustering_data.pickle
    feats_pickle_path : Path
        Path to features.pickle (D-vectors)

    Returns:
    --------
    clustering_data_dict : dict
        Dictionary containing all clustering data arrays
    path_to_index : dict
        Mapping from file stem to index in clustering data
    """
    print(f"\nLoading clustering data: {clustering_pickle_path}")

    try:
        with open(clustering_pickle_path, "rb") as file:
            clustering_data = NumpyCompatUnpickler(file).load()
    except Exception as e:
        print(f"Error loading clustering pickle: {e}")
        with open(clustering_pickle_path, "rb") as file:
            clustering_data = pickle.load(file)

    # Unpack clustering data (7 elements)
    Mixed_X_paths, hdb_data_input, x_tsne_2d, Mixed_y_labels, \
    samples_label, samples_prob, samples_outliers = clustering_data

    print(f"Loaded {len(Mixed_X_paths)} constituent samples from clustering data")

    # Load features pickle
    print(f"Loading features data: {feats_pickle_path}")
    try:
        with open(feats_pickle_path, "rb") as file:
            feats_data = NumpyCompatUnpickler(file).load()
    except Exception as e:
        print(f"Error loading features pickle: {e}")
        with open(feats_pickle_path, "rb") as file:
            feats_data = pickle.load(file)

    x_data, x_paths, _ = feats_data
    print(f"Loaded {len(x_paths)} samples from features data")

    # Create path to index mapping
    path_to_index = {Path(path).stem: idx for idx, path in enumerate(Mixed_X_paths)}

    clustering_data_dict = {
        'paths': Mixed_X_paths,
        'hdb_data': hdb_data_input,
        'tsne_2d': x_tsne_2d,
        'y_labels': Mixed_y_labels,
        'sample_labels': samples_label,
        'sample_probs': samples_prob,
        'sample_outliers': samples_outliers,
        'x_data': x_data
    }

    return clustering_data_dict, path_to_index


def extract_time_from_filename(filename):
    """
    Extract start and end times from constituent file name.

    Format: {base}_{label}_{start}_{end}_{prob}.wav
    Example: D0_78_0.70_1.70_1.00.wav -> (0.70, 1.70)

    Parameters:
    -----------
    filename : str
        Constituent filename

    Returns:
    --------
    start_time : float
        Start time in seconds
    end_time : float
        End time in seconds
    """
    stem = filename.replace('.wav', '')
    parts = stem.split('_')

    if len(parts) >= 5:
        start_time = float(parts[-3])
        end_time = float(parts[-2])
        return start_time, end_time
    else:
        return None, None


def filter_constituents_by_time_range(constituent_files, target_start, target_end,
                                       overlap_threshold=0.5):
    """
    Filter constituent files that fall within the target time range.

    Parameters:
    -----------
    constituent_files : list
        List of constituent filenames
    target_start : float
        Target start time
    target_end : float
        Target end time
    overlap_threshold : float
        Minimum overlap ratio to include file (default: 0.5 = 50%)

    Returns:
    --------
    filtered_files : list
        List of filenames that overlap with target range
    """
    filtered_files = []

    for filename in constituent_files:
        file_start, file_end = extract_time_from_filename(filename)

        if file_start is None or file_end is None:
            continue

        # Calculate overlap
        overlap_start = max(file_start, target_start)
        overlap_end = min(file_end, target_end)

        if overlap_start < overlap_end:
            # Calculate overlap ratio relative to the file duration
            file_duration = file_end - file_start
            overlap_duration = overlap_end - overlap_start
            overlap_ratio = overlap_duration / file_duration if file_duration > 0 else 0

            if overlap_ratio >= overlap_threshold:
                filtered_files.append(filename)

    return filtered_files


def recalculate_features_for_time_range(constituent_files, target_start, target_end,
                                         clustering_data_dict, path_to_index,
                                         overlap_threshold=0.5):
    """
    Recalculate averaged features for a specific time range using constituent files.

    Parameters:
    -----------
    constituent_files : list
        List of all constituent filenames for the original merged sample
    target_start : float
        New start time
    target_end : float
        New end time
    clustering_data_dict : dict
        Dictionary with all clustering data
    path_to_index : dict
        Mapping from file stem to index
    overlap_threshold : float
        Minimum overlap ratio to include file

    Returns:
    --------
    recalc_features : dict
        Dictionary with recalculated x_data and tsne_2d (or None if no files found)
    """
    # Filter constituent files within the time range
    filtered_files = filter_constituents_by_time_range(
        constituent_files, target_start, target_end, overlap_threshold
    )

    if not filtered_files:
        print(f"    Warning: No constituent files found in range {target_start:.2f}-{target_end:.2f}s")
        return None

    # Gather features from filtered files
    x_data_list = []
    tsne_2d_list = []
    hdb_data_list = []

    for filename in filtered_files:
        # Extract file stem (remove probability part)
        file_stem = '_'.join(filename.replace('.wav', '').split('_')[:-1])

        if file_stem in path_to_index:
            idx = path_to_index[file_stem]
            x_data_list.append(clustering_data_dict['x_data'][idx])
            tsne_2d_list.append(clustering_data_dict['tsne_2d'][idx])
            hdb_data_list.append(clustering_data_dict['hdb_data'][idx])

    if not x_data_list:
        print(f"    Warning: No matching features found for files in range {target_start:.2f}-{target_end:.2f}s")
        return None

    # Calculate averages
    recalc_features = {
        'x_data': np.mean(x_data_list, axis=0),
        'tsne_2d': np.mean(tsne_2d_list, axis=0),
        'hdb_data': np.mean(hdb_data_list, axis=0),
        'num_files': len(x_data_list)
    }

    return recalc_features


def update_pickle_with_human_labels(merged_pickle_path, matched_updates, new_samples,
                                     output_pickle_path=None,
                                     recalculate_features=False,
                                     clustering_pickle_path=None,
                                     feats_pickle_path=None,
                                     merged_files_mapping=None):
    """
    Update the merged clustering data pickle file with human labels.
    Handles time corrections and creates new samples for splits.

    Parameters:
    -----------
    merged_pickle_path : Path
        Path to the merged_clustering_data.pickle file
    matched_updates : list
        List of dicts with update information for existing samples
    new_samples : list
        List of dicts with information for new samples to be created
    output_pickle_path : Path, optional
        Path for output pickle file. If None, will append '_with_labels' to original name

    Returns:
    --------
    updated_pickle_path : Path
        Path to the updated pickle file
    update_summary : dict
        Summary of changes made
    """
    # Load the original pickle file
    print(f"\nLoading pickle file: {merged_pickle_path}")
    try:
        with open(merged_pickle_path, "rb") as file:
            merged_clustering_data = NumpyCompatUnpickler(file).load()
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        print("Trying standard pickle.load()...")
        with open(merged_pickle_path, "rb") as file:
            merged_clustering_data = pickle.load(file)

    # Unpack the data
    merged_x_data, \
    merged_paths, \
    merged_hdb_data, \
    merged_tsne_2d, \
    merged_y_labels, \
    merged_sample_labels, \
    merged_sample_probs, \
    merged_sample_outliers = merged_clustering_data

    n_samples = len(merged_paths)
    print(f"Total samples in original pickle: {n_samples}")

    # Create a new column for SpeakerLP (initialize with None)
    speaker_lp_column = np.array([None] * n_samples, dtype=object)

    # Track changes
    updated_count = 0
    time_corrected_count = 0

    # Update existing samples with human labels
    for update in matched_updates:
        sample_idx = update['sample_idx']
        if 0 <= sample_idx < n_samples:
            speaker_lp_column[sample_idx] = update['speaker_lp']

            # Note: We store the corrected times in metadata but don't modify
            # the paths array to maintain consistency with the actual WAV files
            # The corrected times are stored in the sample_metadata for reference

            updated_count += 1
            if update['time_modified']:
                time_corrected_count += 1
        else:
            print(f"  Warning: SampleIndex {sample_idx} out of range [0, {n_samples-1}]")

    print(f"Updated {updated_count} existing samples with human labels")
    print(f"Time corrections applied: {time_corrected_count}")

    # Recalculate features if requested and time corrections were made
    features_recalc_count = 0
    if recalculate_features and time_corrected_count > 0:
        if clustering_pickle_path is None or feats_pickle_path is None or merged_files_mapping is None:
            print("\n  Warning: Cannot recalculate features - missing clustering data paths or mapping")
        else:
            print(f"\n  Recalculating features for {time_corrected_count} time-corrected samples...")

            # Load clustering data
            clustering_data_dict, path_to_index = load_clustering_data(
                clustering_pickle_path, feats_pickle_path
            )

            # Recalculate features for time-corrected samples
            for update in matched_updates:
                if update['time_modified']:
                    sample_idx = update['sample_idx']
                    filename = update['filename']

                    if filename in merged_files_mapping:
                        constituent_files = merged_files_mapping[filename]['constituent_files']
                        original_start = merged_files_mapping[filename]['start_time']
                        original_end = merged_files_mapping[filename]['stop_time']
                        target_start = update['start_time']
                        target_end = update['end_time']

                        print(f"\n    Sample [{sample_idx}]: {filename}")
                        print(f"      Original time range: {original_start:.2f}s - {original_end:.2f}s")
                        print(f"      Corrected time range: {target_start:.2f}s - {target_end:.2f}s")
                        print(f"      Time shift: Δstart={target_start-original_start:+.2f}s, Δend={target_end-original_end:+.2f}s")

                        # Recalculate features
                        recalc_features = recalculate_features_for_time_range(
                            constituent_files, target_start, target_end,
                            clustering_data_dict, path_to_index
                        )

                        if recalc_features is not None:
                            # Update the arrays
                            merged_x_data[sample_idx] = recalc_features['x_data']
                            merged_tsne_2d[sample_idx] = recalc_features['tsne_2d']
                            merged_hdb_data[sample_idx] = recalc_features['hdb_data']
                            features_recalc_count += 1
                            print(f"      ✓ Features recalculated from {recalc_features['num_files']} constituent files")
                            print(f"        X_data shape: {recalc_features['x_data'].shape}")
                            print(f"        t-SNE coords: [{recalc_features['tsne_2d'][0]:.3f}, {recalc_features['tsne_2d'][1]:.3f}]")
                        else:
                            print(f"      ✗ Feature recalculation failed (no constituent files found)")

            print(f"\n  ✓ Successfully recalculated features for {features_recalc_count}/{time_corrected_count} samples")

    # Create new samples for splits
    original_data_arrays = {
        'x_data': merged_x_data,
        'paths': merged_paths,
        'hdb_data': merged_hdb_data,
        'tsne_2d': merged_tsne_2d,
        'y_labels': merged_y_labels,
        'sample_labels': merged_sample_labels,
        'sample_probs': merged_sample_probs,
        'sample_outliers': merged_sample_outliers
    }

    # Load merged_files_mapping to pass to create_new_split_samples
    if merged_files_mapping is None:
        json_path = merged_pickle_path.parent / 'merged_files_mapping.json'
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                merged_files_mapping = json.load(f)
        else:
            merged_files_mapping = {}

    # Pass clustering data to create_new_split_samples if feature recalculation is enabled
    clustering_data_for_splits = None
    path_to_index_for_splits = None
    if recalculate_features and len(new_samples) > 0:
        if clustering_pickle_path is not None and feats_pickle_path is not None:
            # Reuse or load clustering data
            if 'clustering_data_dict' not in locals():
                clustering_data_for_splits, path_to_index_for_splits = load_clustering_data(
                    clustering_pickle_path, feats_pickle_path
                )
            else:
                clustering_data_for_splits = clustering_data_dict
                path_to_index_for_splits = path_to_index

    new_data = create_new_split_samples(
        new_samples, original_data_arrays, merged_files_mapping,
        clustering_data_dict=clustering_data_for_splits,
        path_to_index=path_to_index_for_splits,
        recalculate_features=recalculate_features
    )

    # Combine original and new data
    if new_data is not None:
        # Record the starting index for new samples
        new_sample_start_idx = len(merged_paths)

        # Append new samples to existing arrays
        merged_x_data = np.vstack([merged_x_data, new_data['x_data']])
        merged_paths = list(merged_paths) + list(new_data['paths'])
        merged_hdb_data = np.vstack([merged_hdb_data, new_data['hdb_data']])
        merged_tsne_2d = np.vstack([merged_tsne_2d, new_data['tsne_2d']])
        merged_y_labels = np.concatenate([merged_y_labels, new_data['y_labels']])
        merged_sample_labels = np.concatenate([merged_sample_labels, new_data['sample_labels']])
        merged_sample_probs = np.concatenate([merged_sample_probs, new_data['sample_probs']])
        merged_sample_outliers = np.concatenate([merged_sample_outliers, new_data['sample_outliers']])
        speaker_lp_column = np.concatenate([speaker_lp_column, new_data['speaker_lp']])

        print(f"\n  Added {len(new_data['paths'])} new split samples")
        print(f"  New sample index mapping:")
        for i, new_sample in enumerate(new_samples):
            original_idx = new_sample['original_idx']
            new_idx = new_sample_start_idx + i
            split_order = new_sample['split_order']
            speaker = new_sample['speaker_lp']
            time_range = f"{new_sample['start_time']:.2f}s-{new_sample['end_time']:.2f}s"
            print(f"    Original idx [{original_idx}] → New idx [{new_idx}] (Split part {split_order}): {speaker} ({time_range})")

        print(f"  Total samples after split additions: {len(merged_paths)}")

    print(f"Samples without human labels: {sum(1 for x in speaker_lp_column if x is None)}")

    # Create updated clustering data (add speaker_lp as 9th element)
    print("\n" + "="*80)
    print("CREATING UPDATED CLUSTERING DATA STRUCTURE")
    print("="*80)
    print("Building 9-element pickle structure:")
    print(f"  [0] merged_X_data:          shape {merged_x_data.shape}")
    print(f"  [1] merged_paths:           {len(merged_paths)} items")
    print(f"  [2] merged_hdb_data:        shape {merged_hdb_data.shape}")
    print(f"  [3] merged_tsne_2d:         shape {merged_tsne_2d.shape}")
    print(f"  [4] merged_y_labels:        shape {merged_y_labels.shape}")
    print(f"  [5] merged_sample_labels:   shape {merged_sample_labels.shape}")
    print(f"  [6] merged_sample_probs:    shape {merged_sample_probs.shape}")
    print(f"  [7] merged_sample_outliers: shape {merged_sample_outliers.shape}")
    print(f"  [8] speaker_lp_column:      shape {speaker_lp_column.shape} (NEW!)")

    if recalculate_features and features_recalc_count > 0:
        print(f"\n✓ Features recalculated for {features_recalc_count} samples")
        print("  - X_data (D-vectors): Updated with averaged constituent features")
        print("  - t-SNE coordinates: Updated with averaged projections")
        print("  - HDBSCAN embeddings: Updated with averaged embeddings")

    updated_clustering_data = [
        merged_x_data,
        merged_paths,
        merged_hdb_data,
        merged_tsne_2d,
        merged_y_labels,
        merged_sample_labels,
        merged_sample_probs,
        merged_sample_outliers,
        speaker_lp_column  # New column with human labels
    ]

    # Determine output path
    if output_pickle_path is None:
        output_pickle_path = merged_pickle_path.parent / f"{merged_pickle_path.stem}_with_labels.pickle"

    # Save the updated pickle file
    print("\n" + "="*80)
    print("SAVING UPDATED PICKLE FILE")
    print("="*80)
    print(f"Output path: {output_pickle_path}")
    print(f"File size: {len(updated_clustering_data)} elements")
    print(f"Total samples: {len(merged_paths)}")
    with open(output_pickle_path, 'wb') as file:
        pickle.dump(updated_clustering_data, file)

    file_size_mb = output_pickle_path.stat().st_size / (1024 * 1024)
    print(f"✓ Pickle file saved successfully ({file_size_mb:.2f} MB)")

    update_summary = {
        'original_samples': n_samples,
        'updated_samples': updated_count,
        'time_corrected': time_corrected_count,
        'features_recalculated': features_recalc_count if recalculate_features else 0,
        'new_split_samples': len(new_samples) if new_samples else 0,
        'final_samples': len(merged_paths),
        'labeled_samples': sum(1 for x in speaker_lp_column if x is not None),
        'unlabeled_samples': sum(1 for x in speaker_lp_column if x is None)
    }

    print("="*80)
    print("Update completed successfully!")
    return output_pickle_path, update_summary


def save_metadata_report(sample_metadata, matched_updates, new_samples, output_folder):
    """
    Save a detailed report of the matching and update process.

    Parameters:
    -----------
    sample_metadata : dict
        Dictionary with metadata for each matched sample
    matched_updates : list
        List of update information
    new_samples : list
        List of new sample information
    output_folder : Path
        Output folder for the report
    """
    report_path = output_folder / 'human_labels_matching_report.csv'

    print(f"\nSaving matching report to: {report_path}")

    # Convert to DataFrame for easy CSV export
    report_data = []

    # Add existing sample updates
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
            'ClusterPred': update['cluster_pred'],
            'Filename': update['filename']
        })

    # Add new split samples
    for new_sample in new_samples:
        report_data.append({
            'SampleIndex': f"{new_sample['original_idx']} (new split {new_sample['split_order']})",
            'Type': f'New Split Part {new_sample["split_order"]}',
            'SpeakerLP': new_sample['speaker_lp'],
            'HumanStartTime': new_sample['start_time'],
            'HumanEndTime': new_sample['end_time'],
            'MergedStartTime': new_sample['original_metadata']['start_time'],
            'MergedEndTime': new_sample['original_metadata']['stop_time'],
            'OverlapPercent': new_sample['overlap_percent'],
            'TimeModified': 'Yes',
            'ClusterPred': new_sample['cluster_pred'],
            'Filename': new_sample['original_filename']
        })

    report_df = pd.DataFrame(report_data)
    if len(report_df) > 0:
        # Don't sort if SampleIndex contains mixed types (int and str with new splits)
        # Just save in the order they were processed
        pass
    report_df.to_csv(report_path, index=False)

    print(f"Report saved with {len(report_data)} entries")


def main():
    """Main execution function."""

    # Example paths (can be modified via command line arguments)
    parser = argparse.ArgumentParser(
        description='Update merged_clustering_data.pickle with human labels from webapp_results CSV files'
    )

    parser.add_argument('--dataset_name', type=str, default='TestAO-Irma',
                        help='Dataset name (default: TestAO-Irma)')
    parser.add_argument('--lp_method', type=str, default='LP1',
                        help='Label propagation method name (default: LP1)')
    parser.add_argument('--stg3_method', type=str, default='STG3_EXP011-SHAS-DV-hdb',
                        help='Stage 3 method name (default: STG3_EXP011-SHAS-DV-hdb)')
    parser.add_argument('--overlap_threshold', type=float, default=90.0,
                        help='Minimum overlap percentage for matching (default: 90.0)')
    parser.add_argument('--output_suffix', type=str, default='_with_labels',
                        help='Suffix for output pickle file (default: _with_labels)')
    parser.add_argument('--recalculate_features', action='store_true',
                        help='Recalculate X_data and tsne_2d based on corrected time ranges')
    parser.add_argument('--stg2_method', type=str, default='STG2_EXP010-SHAS-DV',
                        help='Stage 2 method name for features pickle (default: STG2_EXP010-SHAS-DV)')

    args = parser.parse_args()

    # Construct paths
    base_path = Path.home().joinpath('Dropbox', 'DATASETS_AUDIO', 'Unsupervised_Pipeline', args.dataset_name)

    # Input paths
    stg3_folder = base_path.joinpath('STG_3', args.stg3_method)
    merged_pickle_path = stg3_folder / 'merged_clustering_data.pickle'
    merged_mapping_json_path = stg3_folder / 'merged_files_mapping.json'
    clustering_pickle_path = stg3_folder / 'clustering_data.pickle'

    # Features pickle path
    stg2_folder = base_path.joinpath('STG_2', args.stg2_method)
    # Extract method abbreviation (e.g., SHAS_DV from STG2_EXP010-SHAS-DV)
    stg2_abbrev = '_'.join(args.stg2_method.split('-')[1:]) if '-' in args.stg2_method else args.stg2_method.split('_', 1)[1]
    feats_pickle_path = stg2_folder / f'{args.dataset_name}_{stg2_abbrev}_featsEN.pickle'

    stg4_folder = base_path.joinpath('STG_4', f'STG4_{args.lp_method}')
    webapp_results_folder = stg4_folder / 'webapp_results'

    # Output path
    output_folder = stg4_folder / 'updated_pickle_data'
    output_folder.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("="*80)
    print("UPDATE PICKLE WITH HUMAN LABELS")
    print("="*80)
    print(f"Dataset: {args.dataset_name}")
    print(f"Stage 3 Method: {args.stg3_method}")
    print(f"Stage 2 Method: {args.stg2_method}")
    print(f"Label Propagation Method: {args.lp_method}")
    print(f"Overlap Threshold: {args.overlap_threshold}%")
    print(f"Recalculate Features: {args.recalculate_features}")
    print(f"\nInput Files:")
    print(f"  Pickle: {merged_pickle_path}")
    print(f"  Mapping: {merged_mapping_json_path}")
    print(f"  CSV Folder: {webapp_results_folder}")
    if args.recalculate_features:
        print(f"  Clustering Pickle: {clustering_pickle_path}")
        print(f"  Features Pickle: {feats_pickle_path}")
    print(f"\nOutput Folder:")
    print(f"  {output_folder}")
    print("="*80)

    # Validate input files exist
    if not merged_pickle_path.exists():
        print(f"\nError: Pickle file not found: {merged_pickle_path}")
        sys.exit(1)

    if not merged_mapping_json_path.exists():
        print(f"\nError: Mapping JSON file not found: {merged_mapping_json_path}")
        sys.exit(1)

    if not webapp_results_folder.exists():
        print(f"\nError: Webapp results folder not found: {webapp_results_folder}")
        sys.exit(1)

    if args.recalculate_features:
        if not clustering_pickle_path.exists():
            print(f"\nError: Clustering pickle file not found: {clustering_pickle_path}")
            sys.exit(1)

        if not feats_pickle_path.exists():
            print(f"\nError: Features pickle file not found: {feats_pickle_path}")
            sys.exit(1)

    # Step 1: Read human labels from CSV files
    print("\n" + "="*80)
    print("STEP 1: Reading Human Labels from CSV Files")
    print("="*80)
    human_labels_df, split_samples = read_human_labels_csv(webapp_results_folder)

    # Step 2: Load merged files mapping
    print("\n" + "="*80)
    print("STEP 2: Loading Merged Files Mapping")
    print("="*80)
    merged_files_mapping, idx_to_filename = load_merged_files_mapping(merged_mapping_json_path)
    print(f"Loaded mapping for {len(merged_files_mapping)} merged files")

    # Step 3: Match human labels to merged data
    print("\n" + "="*80)
    print("STEP 3: Matching Human Labels to Merged Data")
    print("="*80)
    matched_updates, new_samples, sample_metadata = match_human_labels_to_merged_data(
        human_labels_df,
        merged_files_mapping,
        idx_to_filename,
        split_samples,
        overlap_threshold=args.overlap_threshold
    )

    # Step 4: Update pickle file with human labels
    print("\n" + "="*80)
    print("STEP 4: Updating Pickle File with Human Labels")
    if args.recalculate_features:
        print("  (with feature recalculation enabled)")
    print("="*80)
    output_pickle_path = output_folder / f"merged_clustering_data{args.output_suffix}.pickle"
    updated_pickle_path, update_summary = update_pickle_with_human_labels(
        merged_pickle_path,
        matched_updates,
        new_samples,
        output_pickle_path,
        recalculate_features=args.recalculate_features,
        clustering_pickle_path=clustering_pickle_path if args.recalculate_features else None,
        feats_pickle_path=feats_pickle_path if args.recalculate_features else None,
        merged_files_mapping=merged_files_mapping
    )

    # Step 5: Save metadata report
    print("\n" + "="*80)
    print("STEP 5: Saving Matching Report")
    print("="*80)
    save_metadata_report(sample_metadata, matched_updates, new_samples, output_folder)

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total human-labeled samples in CSV: {len(human_labels_df)}")
    print(f"Original samples in pickle: {update_summary['original_samples']}")
    print(f"Successfully updated samples: {update_summary['updated_samples']}")
    print(f"Time corrections applied: {update_summary['time_corrected']}")
    if args.recalculate_features:
        print(f"Features recalculated: {update_summary['features_recalculated']}")
    print(f"New split samples created: {update_summary['new_split_samples']}")
    print(f"Final total samples: {update_summary['final_samples']}")
    print(f"  - Labeled: {update_summary['labeled_samples']}")
    print(f"  - Unlabeled: {update_summary['unlabeled_samples']}")
    print(f"\nUpdated pickle file: {updated_pickle_path}")
    print(f"Matching report: {output_folder / 'human_labels_matching_report.csv'}")
    print("="*80)
    print("\nProcess completed successfully!")


if __name__ == "__main__":
    main()
