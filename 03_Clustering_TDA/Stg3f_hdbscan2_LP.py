from __future__ import print_function
import os
import warnings
import numpy as np
import hdbscan
from pathlib import Path
import sys
import argparse
import h5py
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import umap

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

from clustering_utils import gen_tsne, check_number_clusters, plot_clustering_dual, \
    membership_curve, n_clusters_curve, plot_clustering
from merge_utils import active_learning_sample_selection
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def determine_min_cluster_size(n_samples):
    """
    Determine min_cluster_size for HDBSCAN based on number of merged samples.

    Parameters:
    -----------
    n_samples : int
        Number of merged samples

    Returns:
    --------
    int : min_cluster_size value
    """
    if n_samples > 1000:
        return 25
    elif n_samples >= 500:
        return 12
    else:
        return 5


def load_merged_hdf5_data(merged_h5_path):
    """
    Load merged samples HDF5 data including recalculated features.

    Returns:
    --------
    dict with merged sample data
    """
    print(f"\nLoading merged HDF5 dataset: {merged_h5_path}")

    with h5py.File(merged_h5_path, 'r') as hf:
        n_merged = len(hf['merged_samples']['merged_unique_ids'])

        data = {
            'merged_unique_ids': [uid.decode() if isinstance(uid, bytes) else uid
                                 for uid in hf['merged_samples']['merged_unique_ids'][:]],
            'merged_wav_paths': [wp.decode() if isinstance(wp, bytes) else wp
                                for wp in hf['merged_samples']['merged_wav_paths'][:]],
            'merged_cluster_labels_avgd': hf['merged_samples']['merged_cluster_labels_avgd'][:],
            'gt_labels': hf['merged_samples']['gt_labels'][:],
            'n_constituents': hf['merged_samples']['n_constituents'][:],
            'merged_cluster_probs_avgd': hf['merged_samples']['merged_cluster_probs_avgd'][:],
        }

        # Load recalculated D-vector features
        if 'recalculated_features' in hf and 'dvectors' in hf['recalculated_features']:
            data['dvectors'] = hf['recalculated_features']['dvectors'][:]
            print(f"✓ Loaded recalculated D-vectors: {data['dvectors'].shape}")
        else:
            raise ValueError("Recalculated features not found in HDF5. Run Stage 3e first.")

        print(f"✓ Loaded {len(data['merged_unique_ids'])} merged samples")
        print(f"  - Cluster distribution: {dict(zip(*np.unique(data['merged_cluster_labels_avgd'], return_counts=True)))}")

    return data


def load_original_clustering_data(clustering_h5_path):
    """
    Load original Stage 3A clustering data (t-SNE, labels, probs) for chunk-level samples.

    Returns:
    --------
    dict with original clustering data: tsne_2d, labels, probs
    """
    print(f"\nLoading original Stage 3A clustering data: {clustering_h5_path}")

    with h5py.File(clustering_h5_path, 'r') as hf:
        data = {
            'tsne_2d': hf['clustering']['tsne_2d'][:],
            'cluster_labels': hf['clustering']['cluster_labels'][:],
            'cluster_probs': hf['clustering']['cluster_probs'][:]
        }

        print(f"✓ Loaded Stage 3A clustering data")
        print(f"  - t-SNE 2D shape: {data['tsne_2d'].shape}")
        print(f"  - Cluster labels: {len(data['cluster_labels'])} samples")
        print(f"  - Cluster distribution: {dict(zip(*np.unique(data['cluster_labels'], return_counts=True)))}")

    return data


def update_merged_hdf5_with_clustering(
    merged_h5_path,
    umap_features,
    tsne_2d,
    cluster_labels_new,
    cluster_probs_new,
    outlier_scores_new
):
    """
    Update merged HDF5 dataset with new clustering results from Stage 3f.

    Parameters:
    -----------
    merged_h5_path : Path
        Path to merged HDF5 file
    umap_features : np.ndarray
        UMAP-reduced features (n_samples, n_components)
    tsne_2d : np.ndarray
        t-SNE 2D coordinates (n_samples, 2)
    cluster_labels_new : np.ndarray
        New HDBSCAN cluster labels (n_samples,)
    cluster_probs_new : np.ndarray
        New HDBSCAN cluster probabilities (n_samples,)
    outlier_scores_new : np.ndarray
        New HDBSCAN outlier scores (n_samples,)
    """
    print(f"\nUpdating merged HDF5 with Stage 3f clustering results: {merged_h5_path}")

    with h5py.File(merged_h5_path, 'a') as hf:
        # Create or update /clustering_stage3f/ group
        if 'clustering_stage3f' in hf:
            del hf['clustering_stage3f']

        clustering_group = hf.create_group('clustering_stage3f')

        # Store UMAP features
        clustering_group.create_dataset(
            'umap_features',
            data=umap_features,
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
            data=cluster_labels_new,
            dtype='int32',
            compression='gzip',
            compression_opts=4
        )

        # Store cluster probabilities
        clustering_group.create_dataset(
            'cluster_probs',
            data=cluster_probs_new,
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )

        # Store outlier scores
        clustering_group.create_dataset(
            'outlier_scores',
            data=outlier_scores_new,
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )

        # Add metadata
        clustering_group.attrs['n_samples'] = len(cluster_labels_new)
        clustering_group.attrs['n_umap_components'] = umap_features.shape[1]
        clustering_group.attrs['n_clusters'] = len(np.unique(cluster_labels_new[cluster_labels_new != -1]))
        clustering_group.attrs['source'] = 'STG3F_HDBSCAN2_LP'
        clustering_group.attrs['description'] = 'Second-stage HDBSCAN clustering on merged samples'

        print(f"✓ Updated HDF5 with Stage 3f clustering")
        print(f"  - UMAP features: {umap_features.shape}")
        print(f"  - t-SNE 2D: {tsne_2d.shape}")
        print(f"  - Cluster labels: {cluster_labels_new.shape}")
        print(f"  - Number of clusters: {clustering_group.attrs['n_clusters']}")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================
base_path_ex = Path.home().joinpath('Dropbox', 'DATASETS_AUDIO', 'Unsupervised_Pipeline', 'TestAO-Irma')
stg3_folder_ex = base_path_ex.joinpath('STG_3', 'STG3_EXP011-SHAS-DV-hdb')
merged_h5_ex = stg3_folder_ex / 'merged_dataset.h5'
clustering_h5_ex = stg3_folder_ex / 'clustering_dataset.h5'

stg3_al_folder_ex = stg3_folder_ex / 'active_learning'
al_input_csv_ex = stg3_al_folder_ex / 'active_learning_samples.csv'

exp_name_ex = 'TestAO-Irma_merged'
hdb_mode_ex = 'eom'
min_samples_ex = 5
n_umap_components_ex = 20

parser = argparse.ArgumentParser(
    description='Stage 3f: Second-stage HDBSCAN clustering on merged samples with Active Learning'
)
parser.add_argument(
    '--merged_dataset_h5',
    default=merged_h5_ex,
    help='Input path for merged samples HDF5 dataset (with recalculated features)'
)
parser.add_argument(
    '--clustering_dataset_h5',
    default=clustering_h5_ex,
    help='Input path for Stage 3A clustering HDF5 dataset (contains original chunk-level t-SNE and labels)'
)
parser.add_argument(
    '--output_folder_al',
    type=valid_path,
    default=stg3_al_folder_ex,
    help='Output folder for Active Learning results'
)
parser.add_argument(
    '--al_input_csv',
    default=al_input_csv_ex,
    help='Output CSV file for Active Learning selected samples'
)
parser.add_argument(
    '--exp_name',
    default=exp_name_ex,
    help='Experiment name for plots and outputs'
)
parser.add_argument(
    '--hdb_mode',
    default=hdb_mode_ex,
    help='HDBSCAN cluster selection method (eom or leaf)'
)
parser.add_argument(
    '--min_samples',
    type=int,
    default=min_samples_ex,
    help='HDBSCAN min_samples parameter'
)
parser.add_argument(
    '--n_umap_components',
    type=int,
    default=n_umap_components_ex,
    help='Number of UMAP components for dimensionality reduction'
)

args = parser.parse_args()
merged_h5_path = Path(args.merged_dataset_h5)
clustering_h5_path = Path(args.clustering_dataset_h5)
output_folder_al = args.output_folder_al
al_input_csv = Path(args.al_input_csv)
exp_name = args.exp_name
hdb_mode = args.hdb_mode
min_samples = args.min_samples
n_umap_components = args.n_umap_components

# Verify paths exist
if not merged_h5_path.exists():
    sys.exit(f"Error: Merged HDF5 file not found: {merged_h5_path}")

if not clustering_h5_path.exists():
    sys.exit(f"Error: Clustering HDF5 file not found: {clustering_h5_path}")

# ============================================================================
# MAIN WORKFLOW
# ============================================================================
print("=" * 80)
print("STAGE 3F: SECOND-STAGE HDBSCAN CLUSTERING ON MERGED SAMPLES")
print("=" * 80)

# Step 1: Load merged HDF5 data (including recalculated features)
merged_data = load_merged_hdf5_data(merged_h5_path)

# Step 1b: Load original Stage 3A clustering data (chunk-level t-SNE and labels)
stg3a_data = load_original_clustering_data(clustering_h5_path)
x_tsne_2d_Stg3A = stg3a_data['tsne_2d']
labels_Stg3A = stg3a_data['cluster_labels']
probs_Stg3A = stg3a_data['cluster_probs']

# Extract data
merged_dvectors = merged_data['dvectors']
merged_unique_ids = merged_data['merged_unique_ids']
merged_wav_paths = merged_data['merged_wav_paths']
merged_gt_labels = merged_data['gt_labels']
merged_cluster_labels_avgd = merged_data['merged_cluster_labels_avgd']  # Original HDBSCAN labels from Stage 3A
merged_cluster_probs_avgd = merged_data['merged_cluster_probs_avgd']  # Original probabilities
n_merged = len(merged_unique_ids)

print(f"\n{'='*80}")
print(f"MERGED SAMPLES OVERVIEW")
print(f"{'='*80}")
print(f"Number of merged samples: {n_merged}")
print(f"D-vector features shape: {merged_dvectors.shape}")

# Step 2: Determine min_cluster_size based on number of merged samples
min_cluster_size_merged = determine_min_cluster_size(n_merged)
print(f"\nDetermined min_cluster_size for merged samples: {min_cluster_size_merged}")
print(f"  - Rule: >1000 → 25, 500-1000 → 12, <500 → 5")

# Step 3: Standardize features
print(f"\nStandardizing features...")
data_standardized = StandardScaler().fit_transform(merged_dvectors)

# Step 4: Apply UMAP + HDBSCAN with multiple repetitions to find best clustering
print(f"\n{'='*80}")
print(f"UMAP + HDBSCAN CLUSTERING (Multiple Repetitions)")
print(f"{'='*80}")

best_score_hdb = -1
best_umap_features = None
best_hdb_model = None
best_repetition = -1

n_repetitions = 4

for repetition_idx in range(n_repetitions):
    print(f"\n--- Repetition {repetition_idx + 1}/{n_repetitions} ---")

    # Apply UMAP
    umap_reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.1,
        n_components=n_umap_components,
        metric='cosine'
        # random_state is not set for randomness across repetitions
    )
    umap_features_candidate = umap_reducer.fit_transform(data_standardized)

    # Apply HDBSCAN
    hdb_candidate = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size_merged,
        min_samples=min_samples,
        cluster_selection_method=hdb_mode
    ).fit(umap_features_candidate)

    # Check clustering quality
    n_clusters, membership_percentage = check_number_clusters(
        hdb_candidate.probabilities_,
        hdb_candidate.labels_,
        verbose=True
    )

    if n_clusters < 3:
        print(f'  ⚠ Skipping repetition {repetition_idx} - insufficient clusters: {n_clusters}')
        continue

    # Calculate combined score
    current_hdb_score = n_clusters_curve(n_clusters) / 2 + membership_curve(membership_percentage) / 2

    print(f'  Current HDB score: {current_hdb_score:.3f}')

    if current_hdb_score > best_score_hdb:
        best_score_hdb = current_hdb_score
        best_umap_features = umap_features_candidate
        best_hdb_model = hdb_candidate
        best_repetition = repetition_idx
        print(f'  ✓ New best score!')

if best_hdb_model is None:
    sys.exit("Error: No valid HDBSCAN model found after all repetitions.")

print(f"\n{'='*80}")
print(f"BEST CLUSTERING RESULTS")
print(f"{'='*80}")
print(f"Best repetition: {best_repetition + 1}")
print(f"Best score: {best_score_hdb:.3f}")

# Extract best clustering results
merged_samples_outliers = best_hdb_model.outlier_scores_
merged_samples_prob = best_hdb_model.probabilities_
merged_samples_label = best_hdb_model.labels_

# Step 5: Generate t-SNE 2D visualization
print(f"\nGenerating t-SNE 2D visualization...")
df_tsne = gen_tsne(best_umap_features, merged_gt_labels)
x_tsne_2d = np.array(list(zip(df_tsne['tsne-2d-one'], df_tsne['tsne-2d-two'])))

# Step 6: Plot clustering results
print(f"\nPlotting clustering results...")

# Plot 1: GT labels vs New Stage 3f HDBSCAN predictions
current_run_id = f'{exp_name}_stg3f_vs_GT'
plot_clustering_dual(
    x_tsne_2d,
    merged_gt_labels,
    merged_samples_label,
    merged_samples_prob,
    current_run_id,
    output_folder_al,
    'store'
)
print(f"  ✓ Saved: {current_run_id} (Ground Truth vs Stage 3f HDBSCAN)")

# Plot 2: Original Stage 3A clusters vs New Stage 3f HDBSCAN predictions
combined_fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_clustering(x_tsne_2d_Stg3A, labels=labels_Stg3A,
                probabilities=probs_Stg3A,
                remove_outliers=True,
                pre_title='Chunks',
                ax=axes[0])

plot_clustering(x_tsne_2d, labels=merged_samples_label,
                probabilities=merged_samples_prob,
                pre_title='Merged',
                remove_outliers=True, ax=axes[1])

current_fig_path = output_folder_al.joinpath(f'{exp_name}_Samples_chunks_merged.png') 

combined_fig.suptitle(f'{exp_name}', fontsize=14)
plt.tight_layout()
combined_fig.savefig(current_fig_path, dpi=300)

print(f"Saved: {current_fig_path} (Stage 3A clusters vs Stage 3f HDBSCAN)")

# Step 7: Store clustering results in HDF5
update_merged_hdf5_with_clustering(
    merged_h5_path,
    best_umap_features,
    x_tsne_2d,
    merged_samples_label,
    merged_samples_prob,
    merged_samples_outliers
)

# Step 8: Active Learning Sample Selection
print(f"\n{'='*80}")
print(f"ACTIVE LEARNING SAMPLE SELECTION")
print(f"{'='*80}")

selected_samples, selection_reasons = active_learning_sample_selection(
    merged_samples_label,
    merged_samples_prob,
    best_umap_features,
    output_folder_al,
    x_tsne_2d,
    n_samples_per_cluster=3,
    plot_flag=True
)

# Step 9: Format and save Active Learning results
print(f"\nFormatting Active Learning results for manual labeling...")

# Create custom formatting with merged unique IDs
summary_data = []

for cluster_id, sample_indices in selected_samples.items():
    reasons = selection_reasons[cluster_id]

    for idx, reason in zip(sample_indices, reasons):
        summary_data.append({
            'cluster_id': cluster_id,
            'sample_index': idx,
            'merged_unique_id': merged_unique_ids[idx],
            'wav_path': merged_wav_paths[idx],
            'selection_reason': reason,
            'hdbscan_prob': merged_samples_prob[idx],
            'gt_label': merged_gt_labels[idx],
            'suggested_label': f'Speaker_C{cluster_id}'  # Placeholder for manual labeling
        })

active_learning_df = pd.DataFrame(summary_data)

# Save to CSV for manual labeling
active_learning_df.to_csv(al_input_csv, index=False)

print(f"\nActive Learning Results:")
print(f"  Total samples selected: {len(summary_data)}")
print(f"  Samples saved to: {al_input_csv}")
print(f"  Please manually label the 'suggested_label' column with actual speaker IDs")

# Print summary by strategy
strategy_counts = active_learning_df['selection_reason'].value_counts()
print(f"  Selection strategy breakdown: {dict(strategy_counts)}")

print(f"\n{'='*80}")
print(f"STAGE 3F COMPLETED SUCCESSFULLY!")
print(f"{'='*80}")
print(f"Merged HDF5 updated with Stage 3f clustering: {merged_h5_path}")
print(f"  - UMAP features stored in: /clustering_stage3f/umap_features")
print(f"  - Cluster labels stored in: /clustering_stage3f/cluster_labels")
print(f"  - t-SNE 2D stored in: /clustering_stage3f/tsne_2d")
print(f"\nClustering comparison plots:")
print(f"  - GT vs Stage 3f: {exp_name}_stg3f_vs_GT")
print(f"  - Stage 3A vs Stage 3f: {exp_name}_stg3f_vs_stg3a")
print(f"\nActive Learning results:")
print(f"  - Selected samples CSV: {al_input_csv}")
print(f"  - Total samples selected: {len(active_learning_df)}")
print(f"  - AL plot: tsne_active_learning_selected_samples.png")
print(f"  - All outputs saved in: {output_folder_al}")
