from pathlib import Path
import pickle
import json
import os
import argparse
import shutil
import numpy as np

from clustering_utils import plot_clustering_dual
from merge_utils import ffmpeg_split_audio, create_folder_if_missing\
    , active_learning_sample_selection, format_active_learning_results

def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

base_path_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO','Unsupervised_Pipeline','TTS4_easy')
stg3_pred_folders_ex = base_path_ex.joinpath('STG_3','STG3_EXP010-SHAS-DV-hdb','HDBSCAN_pred_output')
stg3_merged_wavs_ex = base_path_ex.joinpath('STG_3','STG3_EXP010-SHAS-DV-hdb','merged_wavs')
stg3_separated_wavs_ex = base_path_ex.joinpath('STG_3','STG3_EXP010-SHAS-DV-hdb','separated_merged_wavs')
stg3_outliers_ex = base_path_ex.joinpath('STG_3','STG3_EXP010-SHAS-DV-hdb','outliers_wavs')
stg1_long_wavs_ex = base_path_ex.joinpath('input_wavs')
clusters_data_pickle_ex = stg3_merged_wavs_ex.parent / 'clustering_data.pickle'

feats_pickle_ex = base_path_ex.joinpath('STG_2','STG2_EXP010-SHAS-DV','TTS4_easy_SHAS_DV_featsEN.pickle')
al_input_csv_ex = stg3_merged_wavs_ex.parent / 'AL_input_merged.csv'

seg_ln_ex = '1.0'
step_size_ex = '0.3'
gap_size_ex = '0.4'
consc_th_ex = 1

Exp_name_ex = 'TTS1'

parser = argparse.ArgumentParser()
parser.add_argument('--stg1_long_wavs', type=valid_path, default=stg1_long_wavs_ex, help='Input initial WAVs folder path')
parser.add_argument('--stg3_pred_folders', type=valid_path, default=stg3_pred_folders_ex, help='Input prediction with folders per label')
parser.add_argument('--stg3_separated_wavs', type=valid_path, default=stg3_separated_wavs_ex, help='Output separated per Long wav folder path')
parser.add_argument('--stg3_merged_wavs', type=valid_path, default=stg3_merged_wavs_ex, help='Output merged wavs folder path')
parser.add_argument('--stg3_outliers', type=valid_path, default=stg3_outliers_ex, help='Output outliers wavs folder path')
parser.add_argument('--data_clusters_pickle', default=clusters_data_pickle_ex, help='Input path for clustering labels')
parser.add_argument('--input_feats_pickle', default=feats_pickle_ex, help='Path to the folder to store the D-vectors features')

parser.add_argument('--ln', type=float, default=seg_ln_ex, help='Stg2 chunks length ihn seconds')
parser.add_argument('--st', type=float, default=step_size_ex, help='Stg2 chunks step_size in seconds')
parser.add_argument('--gap', type=float, default=gap_size_ex, help='Stg2 chunks gap in seconds')
parser.add_argument('--consc_th', type=int, default=consc_th_ex, help='Stg3 consecutive chunks threshold')
parser.add_argument('--exp_name', default=Exp_name_ex, help='string with the experiment name')
parser.add_argument('--AL_input_csv', default=al_input_csv_ex, help='Path to the folder to store predictions for Active Learning')
args = parser.parse_args()

stg3_pred_folders = args.stg3_pred_folders 
output_merged_audio = args.stg3_merged_wavs
output_separated_wavs = args.stg3_separated_wavs
output_wav_folder_outliers = args.stg3_outliers
original_wav_files = args.stg1_long_wavs
data_clusters_pickle = Path(args.data_clusters_pickle)

chunk_duration = float(args.ln)
minimum_chunk_duration = chunk_duration - 0.1 # seconds
step_length = float(args.st) 
gap_duration = float(args.gap) 
consecutive_threshold = int(args.consc_th)
al_input_csv = Path(args.AL_input_csv)

exp_name = args.exp_name

feats_pickle_path = Path(args.input_feats_pickle)

# Create output folders if they don't exist
create_folder_if_missing(output_merged_audio)
create_folder_if_missing(output_separated_wavs)
create_folder_if_missing(output_wav_folder_outliers)

with open(f'{data_clusters_pickle}', "rb") as file:
    clustering_data = pickle.load(file)

Mixed_X_paths, hdb_data_input, x_tsne_2d, Mixed_y_labels, samples_label, samples_prob, samples_outliers = clustering_data

with open(f'{feats_pickle_path}', "rb") as file:
    x_data, x_paths, _ = pickle.load(file)

output_folder_al = al_input_csv.parent

# Convert Mixed_X_paths wav names to a dictionary for faster lookup
path_to_index = {Path(path).stem: idx for idx, path in enumerate(Mixed_X_paths)}

# Verify the x_paths stems are equal to Mixed_X_paths stems
x_paths_stems = [Path(path).stem for path in x_paths]
mixed_x_paths_stems = [Path(path).stem for path in Mixed_X_paths]
assert set(x_paths_stems) == set(mixed_x_paths_stems), "Mismatch between x_paths and Mixed_X_paths"

# Lists to store data for merged samples
merged_paths = []
merged_hdb_data = []
merged_tsne_2d = []
merged_x_data = []
merged_y_labels = []
merged_sample_labels = []
merged_sample_probs = []
merged_sample_outliers = []
merged_files_mapping = {}
merged_idx = 0

counts_segments = []
verbose = True

label_subfolders = [f for f in stg3_pred_folders.iterdir() if f.is_dir()]

for current_pred_label_path in label_subfolders:
    current_predicted_label = current_pred_label_path.name 
    
    print(f'\nProcessing label: {current_predicted_label}')

    ############################### 1) Copying chunk wavs -> separated folders 
    # List all .wav files in the directory
    all_stg2_wav_files = list(current_pred_label_path.glob('*.wav'))

    # Get the base name of each file, excluding the last substring divided by a dash
    base_names = [('_'.join(Path(f).name.split('_')[:-4])) for f in all_stg2_wav_files]

    base_names_list = list(set(base_names))

    if verbose:
        print(f'\tFound {len(all_stg2_wav_files)} .wav files in {current_pred_label_path}')
        print(f'\tUnique base names (long audio files): {base_names_list}')

    # Create sub-directories for each unique base name
    for base_name in base_names_list:
        sub_directory = output_separated_wavs.joinpath(current_predicted_label, base_name)
        create_folder_if_missing(sub_directory)

    # Iterate over each .wav file and copy it to the corresponding sub-directory
    for idx, wav_file in enumerate(all_stg2_wav_files):
        dst_folder = output_separated_wavs.joinpath(current_predicted_label, base_names[idx])
        dst_file = dst_folder.joinpath(wav_file.name)
        shutil.copy(str(wav_file), str(dst_file))


    ############################### 2) Create DICT successive files (per long audio) 
    for sub_folder in base_names_list:
        # print(f'\n{current_predicted_label}\tInside subfolder - {sub_folder}')
        current_sub_directory = output_separated_wavs.joinpath(current_predicted_label, sub_folder)

        # List all .wav files in the sub-directory
        sub_wav_files = list(current_sub_directory.glob('*.wav'))

        # Initialize an empty list to store the tuples with file info
        time_file_tuples = []

        # Iterate over each .wav file
        for wav_file in sub_wav_files:
            # Split the filename by underscores
            segments = wav_file.stem.split('_')

            # Get the timing and probability information
            # Format: {original_long_wav}_{startTime}_{stopTime}_{prob}.wav
            start_time = segments[-3]
            stop_time = segments[-2]
            prob = segments[-1]

            # Append the tuple to the list
            time_file_tuples.append((start_time, stop_time, prob, wav_file))

        # Sort the time_file_tuples list by start_time
        time_file_tuples.sort(key=lambda x: float(x[0]))

        merged_segments = []
        if not time_file_tuples:
            print(f'\t!!No wav files found in {current_sub_directory}, skipping...')
            continue

        # # For debugging: print the time_file_tuples
        # if verbose:
        #     print(f'Time and file tuples for {current_sub_directory}:')
        #     for t in time_file_tuples:
        #         print(f'  Start: {t[0]}, Stop: {t[1]}, Prob: {t[2]}, File: {t[3].name}')
            
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

        # # Print the merged segments info
        # print(f'\n\nMerged segments:')
        # for i, (start, stop, files) in enumerate(merged_segments):
        #     print(f'  Segment {i}: {start:.2f}s - {stop:.2f}s ({len(files)} files)')

        ############################### 3) FOR EACH MERGED SEGMENT -> Create merged file and compute averaged metadata

        for idx_seg, current_merged_data in enumerate(merged_segments):
            start_time, stop_time, constituent_files = current_merged_data

            # print(f'\tConstitutent files in segment {idx_seg}: {len(constituent_files)}')
            
            if counts_segments[len(counts_segments) - len(merged_segments) + idx_seg] < consecutive_threshold:
                continue

            # Create the output filename
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

            # print(f'\t\tCreated merged wav: {output_filename}')

            # Store the mapping of merged file to constituent files
            constituent_filenames = [const_file.name for const_file in constituent_files]


            # Compute averaged metadata for this merged sample
            constituent_indices = []
            constituent_metadata = {
                'hdb_data': [],
                'tsne_2d': [],
                'xdata': [],
                'y_labels': [],
                'sample_labels': [],
                'sample_probs': [],
                'sample_outliers': []
            }

            # Find indices of constituent files in the original clustering data
            for const_file in constituent_files:
                file_path_str = (const_file.stem).split('_')[:-1]  # Remove the last part (probability)
                file_path_str = '_'.join(file_path_str)  # Join back to form the
                # print(f'Looking for {file_path_str} in dict')
                if file_path_str in path_to_index:
                    # print(f'>>>>>> Found file path in clustering data.')
                    idx = path_to_index[file_path_str]
                    constituent_indices.append(idx)
                    
                    constituent_metadata['hdb_data'].append(hdb_data_input[idx])
                    constituent_metadata['tsne_2d'].append(x_tsne_2d[idx])
                    constituent_metadata['xdata'].append(x_data[idx])
                    constituent_metadata['y_labels'].append(Mixed_y_labels[idx])
                    constituent_metadata['sample_labels'].append(samples_label[idx])
                    constituent_metadata['sample_probs'].append(samples_prob[idx])
                    constituent_metadata['sample_outliers'].append(samples_outliers[idx])


                    tmp_umap_data = np.array(constituent_metadata['hdb_data'])
                    # print(f'hdb data shape: {tmp_umap_data.shape}')
                else:
                    print(f'>>>>>> Warning: File path {file_path_str} not found in clustering data.')
            
            # print(f'Found {len(constituent_indices)} constituent files in clustering data.')

            most_frequent_y_label = 9999  # Default value in case no labels found

            if constituent_indices:

                # Compute averages for numerical data
                avg_hdb_data = np.mean(constituent_metadata['hdb_data'], axis=0)
                avg_tsne_2d = np.mean(constituent_metadata['tsne_2d'], axis=0)
                avg_x_data = np.mean(constituent_metadata['xdata'], axis=0)
                avg_sample_prob = np.mean(constituent_metadata['sample_probs'])
                avg_sample_outlier = np.mean(constituent_metadata['sample_outliers'])
                
                # For categorical data, use the most frequent value or first occurrence
                # For sample label, use the current predicted label (converted to int if possible)
                try:
                    merged_sample_label = int(current_predicted_label)
                except ValueError:
                    merged_sample_label = current_predicted_label
                
                # For ground truth label, use the most frequent one
                unique_y_labels, counts = np.unique(constituent_metadata['y_labels'], return_counts=True)
                most_frequent_y_label = unique_y_labels[np.argmax(counts)]

                # Store the merged sample data
                merged_paths.append(current_merged_wav_path.stem)
                merged_hdb_data.append(avg_hdb_data)
                merged_tsne_2d.append(avg_tsne_2d)
                merged_x_data.append(avg_x_data)
                merged_y_labels.append(most_frequent_y_label)
                merged_sample_labels.append(merged_sample_label)
                merged_sample_probs.append(avg_sample_prob)
                merged_sample_outliers.append(avg_sample_outlier)

                # For debugging print all merged sample data
                if verbose:
                    print(f'    - merged: {current_merged_wav_path.stem} \t n_files: {len(constituent_files)}')

            merged_files_mapping[output_filename] = {
                'merged_idx': merged_idx ,
                'merged_file_path': str(current_merged_wav_path),
                'original_long_wav': current_original_wav_filename,
                'start_time': float(start_time),
                'stop_time': float(stop_time),
                'prob': round(float(avg_sample_prob), 3) if constituent_indices else None,
                'predicted_label': int(current_predicted_label),
                'avg_GT_label': int(most_frequent_y_label),
                'num_constituent_files': len(constituent_files),
                'constituent_files': constituent_filenames
            }
            merged_idx += 1


print(f'\n\n*** Summary ***')
print(f'Total merged samples created: {len(merged_paths)}')
print(f'Stats of concatenated files: {len(counts_segments)} segments processed')

# Convert lists to numpy arrays
merged_hdb_data = np.array(merged_hdb_data)
merged_tsne_2d = np.array(merged_tsne_2d)
merged_x_data = np.array(merged_x_data)
merged_y_labels = np.array(merged_y_labels)
merged_sample_labels = np.array(merged_sample_labels)
merged_sample_probs = np.array(merged_sample_probs)
merged_sample_outliers = np.array(merged_sample_outliers)

# Plot sample chunks and merged samples comparison

current_run_id = f'{exp_name}_merged_samples'
plot_clustering_dual(x_tsne_2d, merged_y_labels,
                        merged_sample_labels, merged_sample_probs,
                        current_run_id, output_merged_audio.parent,
                        'store')

# Create the merged clustering data
merged_clustering_data = [
    merged_x_data,
    merged_paths, 
    merged_hdb_data, 
    merged_tsne_2d, 
    merged_y_labels, 
    merged_sample_labels, 
    merged_sample_probs, 
    merged_sample_outliers
]

print(f'Shape of merged_hdb_data: {merged_hdb_data.shape}')
print(f'Shape of merged_x_data: {merged_x_data.shape}')

# Save the merged clustering data
merged_data_pickle_path = output_merged_audio.parent / 'merged_clustering_data.pickle'
with open(str(merged_data_pickle_path), 'wb') as file:
    pickle.dump(merged_clustering_data, file)

print(f'Saved merged clustering data to: {merged_data_pickle_path}')

# Save counts segments for compatibility
counts_pickle_path = output_merged_audio.parent / 'counts_segments.pickle'
with open(str(counts_pickle_path), 'wb') as file:
    pickle.dump(counts_segments, file)

print(f'Saved segment counts to: {counts_pickle_path}')
# Save the merged files mapping to JSON
merged_mapping_json_path = output_merged_audio.parent / 'merged_files_mapping.json'
with open(str(merged_mapping_json_path), 'w', encoding='utf-8') as json_file:
    json.dump(merged_files_mapping, json_file, indent=2, ensure_ascii=False)

print(f'Saved merged files mapping to: {merged_mapping_json_path}')
print(f'Total mappings saved: {len(merged_files_mapping)}')
print('Merging process completed successfully!')

# Active Learning Sample Selection
selected_samples, selection_reasons = active_learning_sample_selection(
    merged_sample_labels, merged_sample_probs, merged_hdb_data, output_folder_al, n_samples_per_cluster=3, plot_flag=True
)

# Format and save results for manual labeling
active_learning_df = format_active_learning_results(
    selected_samples, selection_reasons, merged_paths, merged_y_labels, merged_sample_probs, 
    al_input_csv, exp_name
)


