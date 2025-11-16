from pathlib import Path
import pickle
import json
import os
import argparse
import shutil
import numpy as np
import h5py
import soundfile as sf

from merge_utils import ffmpeg_split_audio, create_folder_if_missing


def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def generate_merged_unique_id(cluster_label, merged_index, total_merged_samples):
    """
    Generate unique merged sample ID in format: M{label}_{index}

    Parameters:
    -----------
    cluster_label : int
        Cluster label assigned to this merged sample
    merged_index : int
        Sequential index within this cluster label group
    total_merged_samples : int
        Total number of merged samples (for determining padding)

    Returns:
    --------
    unique_id : str
        Unique identifier (e.g., "M0_0001", "M1_0042")
    """
    # Determine padding based on total samples
    if total_merged_samples <= 100:
        padding = 2
    elif total_merged_samples <= 1000:
        padding = 3
    else:
        padding = 4  # Handles up to 9999

    # Format: M{label}_{index with padding}
    unique_id = f"M{cluster_label}_{merged_index:0{padding}d}"

    return unique_id


def load_original_hdf5_data(hdf5_path):
    """
    Load original clustering HDF5 dataset.

    Returns:
    --------
    dict with all original sample data and ID→index mapping
    """
    print(f"\nLoading original HDF5 dataset: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as hf:
        data = {
            'unique_ids': [uid.decode() if isinstance(uid, bytes) else uid
                          for uid in hf['samples']['unique_ids'][:]],
            'wav_paths': [wp.decode() if isinstance(wp, bytes) else wp
                         for wp in hf['samples']['wav_paths'][:]],
            'enhanced_features': hf['samples']['enhanced_features'][:],
            'gt_labels': hf['samples']['gt_labels'][:],
            'umap_features': hf['clustering']['umap_features'][:],
            'tsne_2d': hf['clustering']['tsne_2d'][:],
            'cluster_labels': hf['clustering']['cluster_labels'][:],
            'cluster_probs': hf['clustering']['cluster_probs'][:],
            'outlier_scores': hf['clustering']['outlier_scores'][:],
        }

        # Load audio if available
        if 'audio' in hf:
            data['audio_waveforms'] = [hf['audio']['waveforms'][i][:]
                                      for i in range(len(data['unique_ids']))]
            data['sample_rates'] = hf['audio']['sample_rates'][:]
        else:
            data['audio_waveforms'] = [None] * len(data['unique_ids'])
            data['sample_rates'] = [None] * len(data['unique_ids'])

        print(f"✓ Loaded {len(data['unique_ids'])} original samples")

    # Create mapping: wav_path_stem → index
    data['path_to_index'] = {Path(wp).stem: idx for idx, wp in enumerate(data['wav_paths'])}

    # Create mapping: unique_id → index
    data['id_to_index'] = {uid: idx for idx, uid in enumerate(data['unique_ids'])}

    return data


def create_merged_hdf5_dataset(
    merged_samples_data,
    original_hdf5_data,
    output_path
):
    """
    Create HDF5 dataset for merged samples.

    Parameters:
    -----------
    merged_samples_data : list of dict
        List of merged sample information
    original_hdf5_data : dict
        Original HDF5 data for reference
    output_path : Path
        Output HDF5 file path
    """
    n_merged = len(merged_samples_data)

    print("\n" + "="*80)
    print("CREATING MERGED SAMPLES HDF5 DATASET")
    print("="*80)
    print(f"Number of merged samples: {n_merged}")

    # Prepare data arrays
    merged_unique_ids = []
    merged_wav_paths = []
    merged_cluster_labels = []
    merged_gt_labels = []
    merged_cluster_probs = []
    merged_start_times = []
    merged_end_times = []
    merged_durations = []
    merged_n_constituents = []
    constituent_ids_list = []
    constituent_indices_list = []
    merged_audio_list = []
    merged_sample_rates = []

    for merged_sample in merged_samples_data:
        merged_unique_ids.append(merged_sample['merged_unique_id'])
        merged_wav_paths.append(merged_sample['merged_wav_path'])
        merged_cluster_labels.append(merged_sample['cluster_label'])
        merged_gt_labels.append(merged_sample['avg_gt_label'])
        merged_cluster_probs.append(merged_sample['avg_cluster_prob'])
        merged_start_times.append(merged_sample['start_time'])
        merged_end_times.append(merged_sample['end_time'])
        merged_durations.append(merged_sample['duration'])
        merged_n_constituents.append(merged_sample['n_constituents'])

        # Constituent IDs and indices
        constituent_ids_list.append(merged_sample['constituent_ids'])
        constituent_indices_list.append(merged_sample['constituent_indices'])

        # Audio
        merged_audio_list.append(merged_sample['merged_audio'])
        merged_sample_rates.append(merged_sample['sample_rate'])

    # Convert to numpy arrays
    merged_unique_ids = np.array(merged_unique_ids, dtype='S25')
    merged_cluster_labels = np.array(merged_cluster_labels, dtype='int32')
    merged_gt_labels = np.array(merged_gt_labels, dtype='int32')
    merged_cluster_probs = np.array(merged_cluster_probs, dtype='float32')
    merged_start_times = np.array(merged_start_times, dtype='float32')
    merged_end_times = np.array(merged_end_times, dtype='float32')
    merged_durations = np.array(merged_durations, dtype='float32')
    merged_n_constituents = np.array(merged_n_constituents, dtype='int32')
    merged_sample_rates = np.array(merged_sample_rates, dtype='int32')

    print(f"✓ Prepared merged sample data")

    # Create HDF5 file
    print(f"\nCreating merged HDF5 file: {output_path}")

    with h5py.File(output_path, 'w') as hf:
        # =====================================================================
        # MERGED_SAMPLES GROUP - Core merged sample information
        # =====================================================================
        merged_group = hf.create_group('merged_samples')

        # Store merged unique IDs
        merged_group.create_dataset(
            'merged_unique_ids',
            data=merged_unique_ids,
            dtype='S25',
            compression='gzip',
            compression_opts=4
        )

        # Store merged wav paths (variable-length strings)
        dt = h5py.string_dtype(encoding='utf-8')
        merged_wav_paths_encoded = np.array([str(p) for p in merged_wav_paths], dtype=object)
        merged_group.create_dataset(
            'merged_wav_paths',
            data=merged_wav_paths_encoded,
            dtype=dt,
            compression='gzip',
            compression_opts=4
        )

        # Store cluster labels
        merged_group.create_dataset(
            'merged_cluster_labels_avgd',
            data=merged_cluster_labels,
            dtype='int32',
            compression='gzip',
            compression_opts=4
        )

        # Store GT labels (averaged/most frequent)
        merged_group.create_dataset(
            'gt_labels',
            data=merged_gt_labels,
            dtype='int32',
            compression='gzip',
            compression_opts=4
        )

        # Store cluster probabilities (averaged)
        merged_group.create_dataset(
            'merged_cluster_probs_avgd',
            data=merged_cluster_probs,
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )

        # Store temporal information
        merged_group.create_dataset(
            'start_times',
            data=merged_start_times,
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )

        merged_group.create_dataset(
            'end_times',
            data=merged_end_times,
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )

        merged_group.create_dataset(
            'durations',
            data=merged_durations,
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )

        # Store constituent information
        merged_group.create_dataset(
            'n_constituents',
            data=merged_n_constituents,
            dtype='int32',
            compression='gzip',
            compression_opts=4
        )

        # Store constituent IDs as variable-length arrays
        dt_vlen_str = h5py.string_dtype(encoding='utf-8')
        constituent_ids_dataset = merged_group.create_dataset(
            'constituent_ids',
            (n_merged,),
            dtype=h5py.vlen_dtype(dt_vlen_str),
            compression='gzip',
            compression_opts=4
        )

        for i, id_list in enumerate(constituent_ids_list):
            constituent_ids_dataset[i] = np.array(id_list, dtype=object)

        # Store constituent indices as variable-length arrays
        constituent_indices_dataset = merged_group.create_dataset(
            'constituent_indices',
            (n_merged,),
            dtype=h5py.vlen_dtype(np.dtype('int32')),
            compression='gzip',
            compression_opts=4
        )

        for i, idx_list in enumerate(constituent_indices_list):
            constituent_indices_dataset[i] = np.array(idx_list, dtype='int32')

        # Add metadata
        merged_group.attrs['n_merged_samples'] = n_merged
        merged_group.attrs['creation_date'] = str(np.datetime64('now'))
        merged_group.attrs['source'] = 'STG3D_MERGE_WAVS'
        merged_group.attrs['description'] = 'Merged samples with constituent mappings'

        # =====================================================================
        # MERGED_AUDIO GROUP - Merged audio waveforms
        # =====================================================================
        audio_group = hf.create_group('merged_audio')

        # Store audio as variable-length arrays
        dt_audio = h5py.vlen_dtype(np.dtype('float32'))
        audio_dataset = audio_group.create_dataset(
            'waveforms',
            (n_merged,),
            dtype=dt_audio,
            compression='gzip',
            compression_opts=4
        )

        for i, audio in enumerate(merged_audio_list):
            if audio is not None and len(audio) > 0:
                audio_dataset[i] = audio.astype(np.float32)
            else:
                audio_dataset[i] = np.array([], dtype=np.float32)

        # Store sample rates
        audio_group.create_dataset(
            'sample_rates',
            data=merged_sample_rates,
            compression='gzip',
            compression_opts=4
        )

        audio_group.attrs['n_loaded'] = len([a for a in merged_audio_list if a is not None and len(a) > 0])
        audio_group.attrs['n_failed'] = len([a for a in merged_audio_list if a is None or len(a) == 0])

        print("\n✓ Merged HDF5 Dataset Structure:")
        print("  /merged_samples/")
        print("    - merged_unique_ids: Merged sample IDs (M prefix)")
        print("    - merged_wav_paths: Merged wav file paths")
        print("    - cluster_labels: HDBSCAN cluster assignments")
        print("    - gt_labels: Ground truth labels (averaged/most frequent)")
        print("    - cluster_probs: Cluster probabilities (averaged)")
        print("    - start_times, end_times, durations: Temporal info")
        print("    - n_constituents: Number of constituent samples")
        print("    - constituent_ids: List of constituent sample IDs")
        print("    - constituent_indices: List of constituent sample indices")
        print("  /merged_audio/")
        print("    - waveforms: Merged audio waveform data")
        print("    - sample_rates: Audio sample rates")

    # Verify file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Merged HDF5 file created successfully")
    print(f"  File size: {file_size_mb:.2f} MB")

    # Display statistics
    print("\n" + "="*80)
    print("MERGED SAMPLE STATISTICS")
    print("="*80)
    print(f"Cluster Labels:")
    for label in sorted(np.unique(merged_cluster_labels)):
        count = np.sum(merged_cluster_labels == label)
        if label == -1:
            print(f"  Noise: {count} merged samples")
        else:
            print(f"  Cluster {label}: {count} merged samples")

    print(f"\nConstituent counts:")
    print(f"  Min: {np.min(merged_n_constituents)}")
    print(f"  Max: {np.max(merged_n_constituents)}")
    print(f"  Mean: {np.mean(merged_n_constituents):.2f}")
    print(f"  Median: {np.median(merged_n_constituents):.0f}")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================
base_path_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO','Unsupervised_Pipeline','TestAO-Irma')
stg3_folder_ex = base_path_ex.joinpath('STG_3','STG3_EXP011-SHAS-DV-hdb')
stg3_pred_folders_ex = stg3_folder_ex.joinpath('HDBSCAN_pred_output')
stg3_merged_wavs_ex = stg3_folder_ex.joinpath('merged_wavs')
stg3_separated_wavs_ex = stg3_folder_ex.joinpath('separated_merged_wavs')
stg3_outliers_ex = stg3_folder_ex.joinpath('outliers_wavs')
stg1_long_wavs_ex = base_path_ex.joinpath('input_wavs')

clustering_h5_ex = base_path_ex.joinpath('Testset_stage3','clustering_dataset.h5')
merged_h5_ex = stg3_folder_ex / 'merged_dataset.h5'

seg_ln_ex = '1.0'
step_size_ex = '0.3'
gap_size_ex = '0.4'
consc_th_ex = 1

Exp_name_ex = 'TestAO-Irma'

parser = argparse.ArgumentParser(
    description='Stage 3d: Merge consecutive audio segments and create merged HDF5 dataset'
)
parser.add_argument('--stg1_long_wavs', type=valid_path, default=stg1_long_wavs_ex,
                   help='Input initial WAVs folder path')
parser.add_argument('--stg3_pred_folders', type=valid_path, default=stg3_pred_folders_ex,
                   help='Input prediction with folders per label')
parser.add_argument('--stg3_separated_wavs', type=valid_path, default=stg3_separated_wavs_ex,
                   help='Output separated per Long wav folder path')
parser.add_argument('--stg3_merged_wavs', type=valid_path, default=stg3_merged_wavs_ex,
                   help='Output merged wavs folder path')
parser.add_argument('--stg3_outliers', type=valid_path, default=stg3_outliers_ex,
                   help='Output outliers wavs folder path')
parser.add_argument('--data_clusters_h5', default=clustering_h5_ex,
                   help='Input path for original clustering HDF5 dataset')
parser.add_argument('--merged_dataset_h5', default=merged_h5_ex,
                   help='Output path for merged samples HDF5 dataset')

parser.add_argument('--ln', type=float, default=seg_ln_ex, help='Stg2 chunks length in seconds')
parser.add_argument('--st', type=float, default=step_size_ex, help='Stg2 chunks step_size in seconds')
parser.add_argument('--gap', type=float, default=gap_size_ex, help='Stg3 gap threshold in seconds')
parser.add_argument('--consc_th', type=int, default=consc_th_ex, help='Stg3 consecutive chunks threshold')
parser.add_argument('--exp_name', default=Exp_name_ex, help='Experiment name')

args = parser.parse_args()

stg3_pred_folders = args.stg3_pred_folders
output_merged_audio = args.stg3_merged_wavs
output_separated_wavs = args.stg3_separated_wavs
output_wav_folder_outliers = args.stg3_outliers
original_wav_files = args.stg1_long_wavs
clustering_h5_path = Path(args.data_clusters_h5)
merged_h5_path = Path(args.merged_dataset_h5)

chunk_duration = float(args.ln)
minimum_chunk_duration = chunk_duration - 0.1
step_length = float(args.st)
gap_duration = float(args.gap)
consecutive_threshold = int(args.consc_th)

exp_name = args.exp_name

# Create output folders if they don't exist
create_folder_if_missing(output_merged_audio)
create_folder_if_missing(output_separated_wavs)
create_folder_if_missing(output_wav_folder_outliers)

# ============================================================================
# LOAD ORIGINAL HDF5 DATA
# ============================================================================
original_data = load_original_hdf5_data(clustering_h5_path)

# Extract frequently used variables
path_to_index = original_data['path_to_index']
id_to_index = original_data['id_to_index']
unique_ids = original_data['unique_ids']
cluster_labels = original_data['cluster_labels']
cluster_probs = original_data['cluster_probs']
gt_labels = original_data['gt_labels']
enhanced_features = original_data['enhanced_features']
umap_features = original_data['umap_features']

# ============================================================================
# PROCESS MERGED SEGMENTS
# ============================================================================
merged_samples_data = []
merged_idx_global = 0
label_merged_counters = {}

counts_segments = []
verbose = True

label_subfolders = [f for f in stg3_pred_folders.iterdir() if f.is_dir()]

for current_pred_label_path in label_subfolders:
    current_predicted_label = current_pred_label_path.name

    print(f'\nProcessing label: {current_predicted_label}')

    # 1) Copy chunk wavs to separated folders
    all_stg2_wav_files = list(current_pred_label_path.glob('*.wav'))
    base_names = [('_'.join(Path(f).name.split('_')[:-4])) for f in all_stg2_wav_files]
    base_names_list = list(set(base_names))

    if verbose:
        print(f'\tFound {len(all_stg2_wav_files)} .wav files in {current_pred_label_path}')
        print(f'\tUnique base names (long audio files): {base_names_list}')

    # Create sub-directories for each unique base name
    for base_name in base_names_list:
        sub_directory = output_separated_wavs.joinpath(current_predicted_label, base_name)
        create_folder_if_missing(sub_directory)

    # Copy files to corresponding sub-directory
    for idx, wav_file in enumerate(all_stg2_wav_files):
        dst_folder = output_separated_wavs.joinpath(current_predicted_label, base_names[idx])
        dst_file = dst_folder.joinpath(wav_file.name)
        shutil.copy(str(wav_file), str(dst_file))

    # 2) Create dict of successive files (per long audio)
    for sub_folder in base_names_list:
        current_sub_directory = output_separated_wavs.joinpath(current_predicted_label, sub_folder)
        sub_wav_files = list(current_sub_directory.glob('*.wav'))

        time_file_tuples = []
        for wav_file in sub_wav_files:
            segments = wav_file.stem.split('_')
            start_time = segments[-3]
            stop_time = segments[-2]
            prob = segments[-1]
            time_file_tuples.append((start_time, stop_time, prob, wav_file))

        time_file_tuples.sort(key=lambda x: float(x[0]))

        merged_segments = []
        if not time_file_tuples:
            print(f'\t!!No wav files found in {current_sub_directory}, skipping...')
            continue

        current_start, current_stop, current_prob, first_file = time_file_tuples[0]
        current_start = float(current_start)
        current_stop = float(current_stop)
        current_files = [first_file]
        current_count = 1

        for start, stop, prob, wav_file in time_file_tuples[1:]:
            start = float(start)
            stop = float(stop)
            if start - current_stop <= gap_duration:
                current_stop = stop
                current_files.append(wav_file)
                current_count += 1
            else:
                merged_segments.append((current_start, current_stop, current_files))
                counts_segments.append(current_count)
                current_start, current_stop = start, stop
                current_files = [wav_file]
                current_count = 1

        # Add the last segment
        if not merged_segments or merged_segments[-1][0:2] != (current_start, current_stop):
            merged_segments.append((current_start, current_stop, current_files))
            counts_segments.append(current_count)

        print(f'\n\nMerged segments:')
        for i, (start, stop, files) in enumerate(merged_segments):
            print(f'  Segment {i}: {start:.2f}s - {stop:.2f}s ({len(files)} files)')

        # 3) For each merged segment, create merged file and compute metadata
        for idx_seg, current_merged_data in enumerate(merged_segments):
            start_time, stop_time, constituent_files = current_merged_data

            print(f'\tConstituent files in segment {idx_seg}: {len(constituent_files)}')

            if counts_segments[len(counts_segments) - len(merged_segments) + idx_seg] < consecutive_threshold:
                continue

            # Create output filename
            output_filename = f"{sub_folder}_{current_predicted_label}_{start_time}_{stop_time}.wav"

            if current_predicted_label == '-1':
                current_merged_wav_path = output_wav_folder_outliers.joinpath(output_filename)
            else:
                current_merged_wav_path = output_merged_audio.joinpath(output_filename)

            # Extract audio from original long wav file
            current_original_wav_filename = f'{sub_folder}.wav'
            original_wav_path = original_wav_files.joinpath(current_original_wav_filename)

            # Create merged audio file
            ffmpeg_split_audio(original_wav_path,
                             current_merged_wav_path,
                             start_time_csv=str(start_time),
                             stop_time_csv=str(stop_time))

            # Load merged audio for HDF5 storage
            try:
                merged_audio, sample_rate = sf.read(str(current_merged_wav_path))
            except Exception as e:
                print(f"Warning: Could not load merged audio from {current_merged_wav_path}: {e}")
                merged_audio = None
                sample_rate = None

            # Find constituent sample IDs and indices
            constituent_ids = []
            constituent_indices = []
            constituent_metadata = {
                'gt_labels': [],
                'cluster_labels': [],
                'cluster_probs': [],
                'enhanced_features': [],
                'umap_features': [],
            }

            for const_file in constituent_files:
                file_path_str = (const_file.stem).split('_')[:-1]
                file_path_str = '_'.join(file_path_str)

                if file_path_str in path_to_index:
                    idx = path_to_index[file_path_str]
                    constituent_indices.append(idx)
                    constituent_ids.append(unique_ids[idx])

                    constituent_metadata['gt_labels'].append(gt_labels[idx])
                    constituent_metadata['cluster_labels'].append(cluster_labels[idx])
                    constituent_metadata['cluster_probs'].append(cluster_probs[idx])
                    constituent_metadata['enhanced_features'].append(enhanced_features[idx])
                    constituent_metadata['umap_features'].append(umap_features[idx])
                else:
                    print(f'Warning: File path {file_path_str} not found in clustering data.')

            if not constituent_indices:
                print(f'Warning: No constituent files found for merged sample. Skipping.')
                continue

            # Compute averaged/aggregated metadata
            avg_cluster_prob = float(np.mean(constituent_metadata['cluster_probs']))

            # Most frequent GT label
            unique_gt_labels, counts = np.unique(constituent_metadata['gt_labels'], return_counts=True)
            most_frequent_gt_label = int(unique_gt_labels[np.argmax(counts)])

            # Generate merged unique ID
            cluster_label_int = int(most_frequent_gt_label)

            # Initialize counter for this label if not exists
            if cluster_label_int not in label_merged_counters:
                label_merged_counters[cluster_label_int] = 0

            # Note: We'll determine total_merged_samples after processing all
            # For now, use a large number (10000) for padding
            merged_unique_id = generate_merged_unique_id(
                cluster_label_int,
                label_merged_counters[cluster_label_int],
                10000
            )
            label_merged_counters[cluster_label_int] += 1

            # Store merged sample data
            merged_sample_info = {
                'merged_unique_id': merged_unique_id,
                'merged_wav_path': output_filename,
                'cluster_label': current_predicted_label,
                'avg_gt_label': most_frequent_gt_label,
                'avg_cluster_prob': avg_cluster_prob,
                'start_time': float(start_time),
                'end_time': float(stop_time),
                'duration': float(stop_time - start_time),
                'n_constituents': len(constituent_ids),
                'constituent_ids': constituent_ids,
                'constituent_indices': constituent_indices,
                'merged_audio': merged_audio,
                'sample_rate': sample_rate if sample_rate is not None else 0,
            }

            merged_samples_data.append(merged_sample_info)

            if verbose:
                print(f'    - merged: {merged_unique_id} \t n_constituents: {len(constituent_ids)}')

print(f'\n\n*** Summary ***')
print(f'Stats of concatenated files: {len(counts_segments)} segments processed')
print(f'Total merged samples created: {len(merged_samples_data)}')

# ============================================================================
# CREATE MERGED HDF5 DATASET
# ============================================================================
if merged_samples_data:
    create_merged_hdf5_dataset(
        merged_samples_data,
        original_data,
        merged_h5_path
    )
else:
    print("Warning: No merged samples were created. Skipping HDF5 creation.")

# ============================================================================
# SAVE LEGACY FILES FOR COMPATIBILITY
# ============================================================================
# Save counts segments
counts_pickle_path = output_merged_audio.parent / 'counts_segments.pickle'
with open(str(counts_pickle_path), 'wb') as file:
    pickle.dump(counts_segments, file)
print(f'\nSaved segment counts to: {counts_pickle_path}')

# Save merged files mapping to JSON (for backward compatibility)
merged_files_mapping = {}
for merged_sample in merged_samples_data:
    merged_files_mapping[merged_sample['merged_wav_path']] = {
        'merged_unique_id': merged_sample['merged_unique_id'],
        'cluster_label': merged_sample['cluster_label'],
        'avg_gt_label': merged_sample['avg_gt_label'],
        'avg_cluster_prob': merged_sample['avg_cluster_prob'],
        'start_time': merged_sample['start_time'],
        'end_time': merged_sample['end_time'],
        'n_constituents': merged_sample['n_constituents'],
        'constituent_ids': merged_sample['constituent_ids'],
    }

merged_mapping_json_path = output_merged_audio.parent / 'merged_files_mapping.json'
with open(str(merged_mapping_json_path), 'w', encoding='utf-8') as json_file:
    json.dump(merged_files_mapping, json_file, indent=2, ensure_ascii=False)

print(f'Saved merged files mapping to: {merged_mapping_json_path}')
print(f'Total mappings saved: {len(merged_files_mapping)}')
print('\n' + "="*80)
print('MERGING PROCESS COMPLETED SUCCESSFULLY!')
print("="*80)
