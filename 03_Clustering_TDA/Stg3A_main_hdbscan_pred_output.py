from __future__ import print_function
import os
import warnings
import numpy as np
import hdbscan
from pathlib import Path
import sys
import warnings
import argparse
import re
import pickle
import matplotlib.pyplot as plt
import h5py
import soundfile as sf

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import umap

warnings.filterwarnings('ignore', category=FutureWarning)

from clustering_utils import gen_tsne, check_number_clusters, plot_clustering_dual, organize_samples_by_label,\
    membership_curve, n_clusters_curve


warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

min_cluster_size = 25 
pca_elem = 0
hdb_mode = 'eom'
min_samples = 5

def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def generate_unique_id(pickle_label, sample_index, total_samples):
    """
    Generate unique sample ID in format: C{label}_{index}

    Parameters:
    -----------
    pickle_label : int
        Label assigned to this sample
    sample_index : int
        Sequential index within this label group
    total_samples : int
        Total number of samples (for determining padding)

    Returns:
    --------
    unique_id : str
        Unique identifier (e.g., "C0_0001", "C1_0042")
    """
    # Determine padding based on total samples (max 10000)
    if total_samples <= 100:
        padding = 2
    elif total_samples <= 1000:
        padding = 3
    else:
        padding = 4  # Handles up to 9999

    # Format: C{label}_{index with padding}
    unique_id = f"C{pickle_label}_{sample_index:0{padding}d}"

    return unique_id


def load_wav_audio(wav_path):
    """
    Load audio waveform from wav file.

    Parameters:
    -----------
    wav_path : str or Path
        Path to wav file

    Returns:
    --------
    audio_data : np.ndarray or None
        Audio waveform, None if loading fails
    sample_rate : int or None
        Sample rate, None if loading fails
    """
    try:
        audio_data, sample_rate = sf.read(str(wav_path))
        return audio_data, sample_rate
    except Exception as e:
        print(f"Warning: Could not load audio from {wav_path}: {e}")
        return None, None


def create_hdf5_dataset_with_clustering(
    enhanced_features, wavs_paths, pickle_labels,
    hdb_data, tsne_2d, cluster_labels, cluster_probs, outlier_scores,
    output_path, wavs_folder_path=None, load_audio=True
):
    """
    Create HDF5 dataset with unique IDs, initial features, and clustering results.

    Parameters:
    -----------
    enhanced_features : np.ndarray
        Original D-vector features (n_samples, n_features)
    wavs_paths : list
        List of wav file paths or filenames
    pickle_labels : np.ndarray
        Original pickle labels (ground truth)
    hdb_data : np.ndarray
        HDBSCAN input features (UMAP reduced) (n_samples, n_components)
    tsne_2d : np.ndarray
        t-SNE 2D coordinates (n_samples, 2)
    cluster_labels : np.ndarray
        HDBSCAN cluster labels (n_samples,)
    cluster_probs : np.ndarray
        HDBSCAN cluster probabilities (n_samples,)
    outlier_scores : np.ndarray
        HDBSCAN outlier scores (n_samples,)
    output_path : Path
        Output HDF5 file path
    wavs_folder_path : Path or None
        Base folder path containing wav files. If provided, will be combined with
        filenames from wavs_paths to construct full paths. (default: None)
    load_audio : bool
        Whether to load and store audio waveforms (default: True)
    """
    n_samples = len(wavs_paths)
    n_features = enhanced_features.shape[1]
    n_hdb_features = hdb_data.shape[1]

    # Validation: ensure all arrays have the same length
    assert enhanced_features.shape[0] == n_samples, \
        f"Enhanced features has {enhanced_features.shape[0]} samples, expected {n_samples}"
    assert len(pickle_labels) == n_samples, \
        f"Pickle labels has {len(pickle_labels)} samples, expected {n_samples}"
    assert hdb_data.shape[0] == n_samples, \
        f"HDB data has {hdb_data.shape[0]} samples, expected {n_samples}"
    assert tsne_2d.shape[0] == n_samples, \
        f"t-SNE data has {tsne_2d.shape[0]} samples, expected {n_samples}"
    assert len(cluster_labels) == n_samples, \
        f"Cluster labels has {len(cluster_labels)} samples, expected {n_samples}"
    assert len(cluster_probs) == n_samples, \
        f"Cluster probs has {len(cluster_probs)} samples, expected {n_samples}"
    assert len(outlier_scores) == n_samples, \
        f"Outlier scores has {len(outlier_scores)} samples, expected {n_samples}"

    print("\n" + "="*80)
    print("CREATING HDF5 DATASET WITH CLUSTERING RESULTS")
    print("="*80)
    print(f"Number of samples: {n_samples}")
    print(f"Original feature dimensions: {n_features}")
    print(f"UMAP reduced dimensions: {n_hdb_features}")
    print(f"Number of clusters: {len(np.unique(cluster_labels[cluster_labels >= 0]))}")
    print(f"Load audio waveforms: {load_audio}")

    # Generate unique IDs for all samples
    print("\nGenerating unique sample IDs...")

    # Count samples per label for sequential indexing
    unique_labels = np.unique(pickle_labels)
    label_counters = {label: 0 for label in unique_labels}

    unique_ids = []
    for i in range(n_samples):
        label = pickle_labels[i]
        label_idx = label_counters[label]
        label_counters[label] += 1

        # Generate unique ID
        unique_id = generate_unique_id(label, label_idx, n_samples)
        unique_ids.append(unique_id)

    # Convert to numpy array for storage
    unique_ids = np.array(unique_ids, dtype='S25')  # Fixed-length string (25 chars max)

    print(f"✓ Generated {len(unique_ids)} unique IDs")
    print(f"  ID format examples: {[uid.decode() for uid in unique_ids[:3]]}")

    # Load audio waveforms if requested
    audio_data_list = []
    sample_rates = []

    if load_audio:
        print("\nLoading audio waveforms...")

        # Construct full paths if wavs_folder_path is provided
        if wavs_folder_path is not None:
            print(f"  Using base folder: {wavs_folder_path}")
            full_wav_paths = [Path(wavs_folder_path) / Path(wav).name for wav in wavs_paths]
        else:
            # Assume wavs_paths already contains full paths
            full_wav_paths = [Path(wav) for wav in wavs_paths]

        for i, wav_path in enumerate(full_wav_paths):
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{n_samples} audio files...")

            audio, sr = load_wav_audio(wav_path)
            audio_data_list.append(audio if audio is not None else np.array([]))
            sample_rates.append(sr if sr is not None else 0)

        print(f"✓ Loaded {len([a for a in audio_data_list if len(a) > 0])}/{n_samples} audio files successfully")

    # Create HDF5 file
    print(f"\nCreating HDF5 file: {output_path}")

    with h5py.File(output_path, 'w') as hf:
        # =====================================================================
        # SAMPLES GROUP - Core sample information
        # =====================================================================
        main_group = hf.create_group('samples')

        # Store unique IDs (primary key)
        main_group.create_dataset(
            'unique_ids',
            data=unique_ids,
            dtype='S25',
            compression='gzip',
            compression_opts=4
        )

        # Store enhanced features (original D-vectors)
        main_group.create_dataset(
            'enhanced_features',
            data=enhanced_features,
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )

        # Store wav paths (variable-length strings)
        dt = h5py.string_dtype(encoding='utf-8')

        # Extract filename strings from Paths
        wavs_filenames = [os.path.basename(p) for p in wavs_paths]
        wavs_filenames_encoded = np.array([str(f) for f in wavs_filenames], dtype=object)

        main_group.create_dataset(
            'wav_paths',
            data=wavs_filenames_encoded,
            dtype=dt,
            compression='gzip',
            compression_opts=4
        )

        # Store pickle labels (ground truth)
        main_group.create_dataset(
            'gt_labels',
            data=pickle_labels,
            dtype='int32',
            compression='gzip',
            compression_opts=4
        )

        # Add metadata
        main_group.attrs['n_samples'] = n_samples
        main_group.attrs['n_features'] = n_features
        main_group.attrs['creation_date'] = str(np.datetime64('now'))
        main_group.attrs['source'] = 'STG3A_HDBSCAN_PRED_OUTPUT'
        main_group.attrs['description'] = 'Dataset with enhanced features and HDBSCAN clustering'

        # =====================================================================
        # CLUSTERING GROUP - HDBSCAN results
        # =====================================================================
        clustering_group = hf.create_group('clustering')

        # Store UMAP reduced features (HDBSCAN input)
        clustering_group.create_dataset(
            'umap_features',
            data=hdb_data,
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )

        # Store t-SNE 2D coordinates
        clustering_group.create_dataset(
            'tsne_2d',
            data=tsne_2d,
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )

        # Store cluster labels
        clustering_group.create_dataset(
            'cluster_labels',
            data=cluster_labels,
            dtype='int32',
            compression='gzip',
            compression_opts=4
        )

        # Store cluster probabilities
        clustering_group.create_dataset(
            'cluster_probs',
            data=cluster_probs,
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )

        # Store outlier scores
        clustering_group.create_dataset(
            'outlier_scores',
            data=outlier_scores,
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )

        # Add clustering metadata
        clustering_group.attrs['n_umap_components'] = n_hdb_features
        clustering_group.attrs['n_clusters'] = len(np.unique(cluster_labels[cluster_labels >= 0]))
        clustering_group.attrs['n_noise'] = np.sum(cluster_labels == -1)
        clustering_group.attrs['algorithm'] = 'HDBSCAN'

        # =====================================================================
        # AUDIO GROUP - Waveform data (if loaded)
        # =====================================================================
        if load_audio and len(audio_data_list) > 0:
            audio_group = hf.create_group('audio')

            # Store audio as variable-length arrays
            dt_audio = h5py.vlen_dtype(np.dtype('float32'))
            audio_dataset = audio_group.create_dataset(
                'waveforms',
                (n_samples,),
                dtype=dt_audio,
                compression='gzip',
                compression_opts=4
            )

            for i, audio in enumerate(audio_data_list):
                if len(audio) > 0:
                    audio_dataset[i] = audio.astype(np.float32)
                else:
                    audio_dataset[i] = np.array([], dtype=np.float32)

            # Store sample rates
            audio_group.create_dataset(
                'sample_rates',
                data=np.array(sample_rates, dtype='int32'),
                compression='gzip',
                compression_opts=4
            )

            audio_group.attrs['n_loaded'] = len([a for a in audio_data_list if len(a) > 0])
            audio_group.attrs['n_failed'] = len([a for a in audio_data_list if len(a) == 0])

        # =====================================================================
        # LABELS GROUP - For future human labels and label propagation
        # =====================================================================
        hf.create_group('labels')

        # =====================================================================
        # METADATA GROUP - For additional metadata
        # =====================================================================
        hf.create_group('metadata')

        print("\n✓ HDF5 Dataset Structure:")
        print("  /samples/")
        print("    - unique_ids: Sample unique identifiers")
        print("    - enhanced_features: Original D-vector features")
        print("    - wav_paths: Original wav file paths")
        print("    - gt_labels: Ground truth labels (pickle labels)")
        print("  /clustering/")
        print("    - umap_features: UMAP reduced features (HDBSCAN input)")
        print("    - tsne_2d: t-SNE 2D coordinates")
        print("    - cluster_labels: HDBSCAN cluster assignments")
        print("    - cluster_probs: HDBSCAN cluster probabilities")
        print("    - outlier_scores: HDBSCAN outlier scores")
        if load_audio:
            print("  /audio/")
            print("    - waveforms: Audio waveform data")
            print("    - sample_rates: Audio sample rates")
        print("  /labels/ (placeholder for human labels & LP)")
        print("  /metadata/ (placeholder for additional data)")

    # Verify file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ HDF5 file created successfully")
    print(f"  File size: {file_size_mb:.2f} MB")

    # Display sample statistics
    print("\n" + "="*80)
    print("SAMPLE STATISTICS")
    print("="*80)
    print(f"Ground Truth Labels:")
    for label in sorted(np.unique(pickle_labels)):
        count = np.sum(pickle_labels == label)
        print(f"  Label {label}: {count} samples")

    print(f"\nClustering Results:")
    for cluster_id in sorted(np.unique(cluster_labels)):
        count = np.sum(cluster_labels == cluster_id)
        if cluster_id == -1:
            print(f"  Noise: {count} samples")
        else:
            print(f"  Cluster {cluster_id}: {count} samples")

base_path_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO','Unsupervised_Pipeline','TestAO-Irma')
clusters_f5_ex = base_path_ex.joinpath('Testset_stage3','clustering_dataset.h5')
feats_pickle_ex = base_path_ex.joinpath('STG_2','STG2A_ENHANCED_FEATURES','enhanced_features.pickle')
main_dir_ex = base_path_ex / 'STG_1' / 'STG1_SHAS'
filtered_chunks_wavs_ex = main_dir_ex / "wav_chunks_filtered"

output_folder_path_ex = base_path_ex.joinpath('STG_3','STG3A_HDBSCAN_PRED') 
# Create the output directory if it doesn't exist

run_params_ex = f"pca{pca_elem}_mcs{min_cluster_size}_ms{min_samples}_{hdb_mode}"
Exp_name_ex = 'TestAO-Irma'
clusters_data_pickle_ex = output_folder_path_ex / 'clusters_data.pickle'
reduced_features_pickle_ex = output_folder_path_ex / 'reduced_features.pickle'
pred_label_pickle_ex = output_folder_path_ex / 'pred_label.pickle'


parser = argparse.ArgumentParser()
parser.add_argument('--input_feats_pickle', default=feats_pickle_ex, help='Path to the folder to store the D-vectors features')
parser.add_argument('--stg1_filtered_chunks_wavs', type=valid_path, default=filtered_chunks_wavs_ex, help='Stg1 VAD csvs folder path')
parser.add_argument('--output_pred_folder', type=valid_path, default=output_folder_path_ex, help='Path to the folder to store the predictions')
parser.add_argument('--run_params', default=run_params_ex, help='string with the run params for HDBSCAN')
parser.add_argument('--exp_name', default=Exp_name_ex, help='string with the experiment name')
parser.add_argument('--data_clusters_h5', default=clusters_f5_ex, help='Path to the HDF5 clustering dataset file')

args = parser.parse_args()

output_folder_path = Path(args.output_pred_folder)
feats_pickle_path = Path(args.input_feats_pickle)
wavs_folder_path = Path(args.stg1_filtered_chunks_wavs)
hdf5_output_path = Path(args.data_clusters_h5)

run_params = args.run_params
Exp_name = args.exp_name

print(f'run_params: {run_params}')

pattern = r"pca(\d+)_mcs(\d+)_ms(\d+)_(\w+)"
match = re.match(pattern, run_params)

if match:
    pca_elem = int(match.group(1))
    min_cluster_size = int(match.group(2))
    min_samples = int(match.group(3))
    hdb_mode = match.group(4)
else:
    sys.exit("Invalid run_name format")

# Print the extracted values
print(f"pca_elem: {pca_elem}")
print(f"min_cluster_size: {min_cluster_size}")
print(f"min_samples: {min_samples}")
print(f"hdb_mode: {hdb_mode}")

plot_mode = 'store' # 'show' or 'show_store'

with open(f'{feats_pickle_path}', "rb") as file:
    X_data_and_labels = pickle.load(file)
Mixed_X_data, Mixed_X_paths, Mixed_y_labels = X_data_and_labels

current_run_id = f'{Exp_name}_{run_params}'

n_components = 20
data_standardized = StandardScaler().fit_transform(Mixed_X_data)

# comp20, neigh5, manhattan
# comp20, neigh10, cosine 
best_score_hdb = -1
hdb_data_input = None
hdb_selected = None

for repetition_idx in range(4):
    # Apply UMAP
    umap_reducer = umap.UMAP(
        n_neighbors=10,  # Adjust based on dataset size
        min_dist=0.1,    # Controls compactness of clusters
        n_components=n_components,  # Reduced dimensionality
        metric='cosine'  # Good default for many feature types
        # random_state=42
    )
    hdb_data_input_candidate = umap_reducer.fit_transform(data_standardized)

    ## try cluster_selection_method = 'leaf' | default = 'eom'
    hdb_candidate = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,\
                            min_samples=min_samples,\
                        cluster_selection_method = hdb_mode).fit(hdb_data_input_candidate)

    n_clusters, membership_percentage = check_number_clusters(hdb_candidate.probabilities_, hdb_candidate.labels_, verbose = True)

    if n_clusters < 3:
        print(f'Skipping repetition {repetition_idx} due to insufficient clusters: {n_clusters}')
        continue

    current_hdb_score = n_clusters_curve(n_clusters)/2 + membership_curve(membership_percentage)/2

    print(f'{repetition_idx} Current HDB score: {current_hdb_score}')

    if current_hdb_score > best_score_hdb:
        best_score_hdb = current_hdb_score
        hdb_data_input = hdb_data_input_candidate
        hdb_selected = hdb_candidate

if hdb_selected is None:
    sys.exit("No valid HDBSCAN model found.")

samples_outliers = hdb_selected.outlier_scores_
samples_prob = hdb_selected.probabilities_
samples_label = hdb_selected.labels_

df_mixed = gen_tsne(hdb_data_input, Mixed_y_labels)
x_tsne_2d = np.array(list(zip(df_mixed['tsne-2d-one'], df_mixed['tsne-2d-two'])))

# Create HDF5 dataset with all clustering results and audio waveforms
create_hdf5_dataset_with_clustering(
    enhanced_features=Mixed_X_data,
    wavs_paths=Mixed_X_paths,
    pickle_labels=Mixed_y_labels,
    hdb_data=hdb_data_input,
    tsne_2d=x_tsne_2d,
    cluster_labels=samples_label,
    cluster_probs=samples_prob,
    outlier_scores=samples_outliers,
    output_path=hdf5_output_path,
    wavs_folder_path=wavs_folder_path,  # Base folder containing wav files
    load_audio=True  # Set to False to skip audio loading for faster processing
)

plot_clustering_dual(x_tsne_2d, Mixed_y_labels,
                        samples_label, samples_prob,
                        current_run_id, output_folder_path,
                        plot_mode)

# # Use PCA to plot 2D from UMAP features
# pca = PCA(n_components=2)
# hdb_data_input_2d = pca.fit_transform(hdb_data_input)

# # Plot and store the hdb_data_input_2d
# plt.figure(figsize=(10, 8))
# plt.scatter(hdb_data_input_2d[:, 0], hdb_data_input_2d[:, 1], s=5)
# plt.title("HDBSCAN Clustering (2D Projection)")
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.grid()
# plt.savefig(f"{output_folder_path}/{current_run_id}_umap_2d.png")
# plt.close()

organize_samples_by_label(Mixed_X_paths, samples_label, samples_prob, output_folder_path)