"""
Smart Sample Selection for Manual Labeling using Active Learning

This script uses machine learning (SVM) and active learning strategies to
intelligently select which samples should be manually labeled for distance-based
Label Propagation.

Strategy combines:
1. Uncertainty Sampling: Select samples the model is most uncertain about
2. Diversity Sampling: Ensure selected samples cover different regions of the embedding space
3. Cluster-Aware Sampling: Similar to selecting N samples per cluster, but without explicit clustering

Target: ~20 samples out of 300 (similar to 2 per cluster × 10 clusters)

Methods:
- Method 1: SVM Uncertainty + K-Means Diversity (Recommended)
- Method 2: Pure Uncertainty Sampling
- Method 3: K-Means Representatives (fastest)

Dimensionality Reduction:
- PCA: Linear reduction, fast, good for general use
- UMAP: Non-linear reduction, preserves local structure, better for complex embeddings

Output:
    Creates one CSV file per long wav file in the output folder
    CSV format: sample_id, file_path, uncertainty_score, gt_label (tab-separated with header)
    Filename format: {long_wav_basename}_samples.csv

Usage (with arguments):
    python3 stg4c_smart_sample_selection.py \
        --dvectors_pickle /path/to/dvectors.pickle \
        --output_folder /path/to/output_folder \
        --n_samples 20 \
        --method hybrid \
        --reduction_method pca

Usage (standalone with default example variables):
    python3 stg4c_smart_sample_selection.py

    Default paths are configured for TestAO-Irma dataset.
    Edit the example variables section to customize for your dataset.

Example output:
    output_folder/
    ├── G-C1L1P-Apr27-E-Irma_q2_03-08-377_samples.csv
    ├── G-C1L1P-Apr27-E-Irma_q2_03-08-378_samples.csv
    └── G-C1L1P-Apr27-E-Irma_q2_03-08-379_samples.csv
"""

import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from umap import UMAP
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def load_dvectors_pickle(pickle_path):
    """
    Load D-vectors pickle from Stage 2

    Returns:
        dict with keys: 'unique_ids', 'dvectors', 'file_paths'
    """
    print(f"Loading D-vectors pickle: {pickle_path}")

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # Stage 2 pickle format: [X_train, X_train_paths, y_train]
    dvectors = data[0]  # numpy array of D-vectors
    file_paths = data[1]  # list of file paths
    gt_labels = data[2]  # numpy array of GT labels (for evaluation only)

    # Extract unique_ids from file paths
    unique_ids = [Path(fp).stem for fp in file_paths]

    print(f"  Loaded {len(unique_ids)} samples")
    print(f"  D-vector shape: {dvectors.shape}")
    print(f"  D-vector dimension: {dvectors.shape[1]}")

    return {
        'unique_ids': unique_ids,
        'dvectors': dvectors,
        'file_paths': file_paths,
        'gt_labels': gt_labels  # Keep for analysis, but won't use in selection
    }


def reduce_dimensionality(X, n_components=50, variance_threshold=0.95, method='pca'):
    """
    Reduce dimensionality with PCA or UMAP

    Parameters:
    -----------
    X : np.ndarray
        Input data (n_samples, n_features)
    n_components : int
        Target number of components
    variance_threshold : float
        For PCA: variance to preserve (default: 0.95)
    method : str
        'pca' or 'umap' (default: 'pca')

    Returns:
    --------
    X_reduced : np.ndarray
        Reduced data
    reducer : object
        Fitted PCA or UMAP object
    """
    if X.shape[1] <= n_components:
        print(f"  Skipping reduction: already low-dimensional ({X.shape[1]} dims)")
        return X, None

    if method == 'pca':
        pca = PCA(n_components=min(n_components, X.shape[0], X.shape[1]))
        X_reduced = pca.fit_transform(X)

        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_needed = np.searchsorted(cumsum_variance, variance_threshold) + 1

        print(f"  PCA: {X.shape[1]} dims → {n_components_needed} dims ({variance_threshold*100:.1f}% variance)")

        return X_reduced[:, :n_components_needed], pca

    elif method == 'umap':
        print(f"  UMAP: {X.shape[1]} dims → {n_components} dims")
        umap_reducer = UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42,
            verbose=False
        )
        X_reduced = umap_reducer.fit_transform(X)
        print(f"  ✓ UMAP reduction completed")
        return X_reduced, umap_reducer

    else:
        raise ValueError(f"Unknown reduction method: {method}")


def method_hybrid_uncertainty_diversity(X, unique_ids, n_samples, n_clusters_estimate=10, reduction_method='pca'):
    """
    Method 1: Hybrid - SVM Uncertainty + K-Means Diversity (RECOMMENDED)

    This is the most sophisticated approach:
    1. Use K-Means to partition space into regions (similar to clusters)
    2. Train SVM on pseudo-labels from K-Means
    3. Select most uncertain samples from each region
    4. Ensures both uncertainty AND diversity

    Similar to: "Select 2 uncertain samples per cluster"
    """
    print(f"\nMethod: HYBRID (SVM Uncertainty + K-Means Diversity)")
    print(f"  Target: {n_samples} samples")
    print(f"  Estimated clusters: {n_clusters_estimate}")
    print(f"  Reduction method: {reduction_method.upper()}")

    # Step 1: Reduce dimensionality for speed
    print("\n  Step 1: Dimensionality reduction")
    n_components = 20 if reduction_method == 'umap' else 50
    X_reduced, reducer = reduce_dimensionality(X, n_components=n_components, method=reduction_method)

    # Step 2: K-Means to partition space
    print(f"\n  Step 2: K-Means clustering (k={n_clusters_estimate})")
    kmeans = KMeans(n_clusters=n_clusters_estimate, random_state=42, n_init=10)
    pseudo_labels = kmeans.fit_predict(X_reduced)

    cluster_counts = np.bincount(pseudo_labels)
    print(f"    Cluster sizes: {cluster_counts}")

    # Step 3: Train SVM on K-Means pseudo-labels
    print(f"\n  Step 3: Train SVM on pseudo-labels")
    svm = SVC(kernel='rbf', probability=True, random_state=42, gamma='auto')
    svm.fit(X_reduced, pseudo_labels)

    # Step 4: Get prediction probabilities (uncertainty)
    print(f"\n  Step 4: Compute uncertainty scores")
    probs = svm.predict_proba(X_reduced)  # Shape: (n_samples, n_clusters)

    # Uncertainty = 1 - max_probability (lower confidence = higher uncertainty)
    max_probs = np.max(probs, axis=1)
    uncertainty_scores = 1 - max_probs

    print(f"    Uncertainty range: [{uncertainty_scores.min():.3f}, {uncertainty_scores.max():.3f}]")
    print(f"    Mean uncertainty: {uncertainty_scores.mean():.3f}")

    # Step 5: Select most uncertain samples from each cluster
    print(f"\n  Step 5: Select uncertain samples from each cluster")
    samples_per_cluster = max(1, n_samples // n_clusters_estimate)
    selected_indices = []

    for cluster_id in range(n_clusters_estimate):
        cluster_mask = pseudo_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            continue

        # Get uncertainty scores for this cluster
        cluster_uncertainties = uncertainty_scores[cluster_indices]

        # Select top uncertain samples
        n_select = min(samples_per_cluster, len(cluster_indices))
        top_uncertain_local = np.argsort(cluster_uncertainties)[-n_select:]
        top_uncertain_global = cluster_indices[top_uncertain_local]

        selected_indices.extend(top_uncertain_global)

        print(f"    Cluster {cluster_id}: {len(cluster_indices)} samples, selected {n_select}")

    # If we haven't reached n_samples, add more from top uncertain overall
    if len(selected_indices) < n_samples:
        remaining = n_samples - len(selected_indices)
        all_indices = np.argsort(uncertainty_scores)[-n_samples:]
        for idx in all_indices:
            if idx not in selected_indices:
                selected_indices.append(idx)
                remaining -= 1
                if remaining == 0:
                    break

    # If we have too many, trim to exact n_samples
    selected_indices = selected_indices[:n_samples]

    print(f"\n  Final selection: {len(selected_indices)} samples")

    return selected_indices, uncertainty_scores


def method_pure_uncertainty(X, unique_ids, n_samples, n_clusters_estimate=10, reduction_method='pca'):
    """
    Method 2: Pure Uncertainty Sampling

    Train SVM with random pseudo-labels, select most uncertain samples
    Faster but may not ensure diversity
    """
    print(f"\nMethod: PURE UNCERTAINTY SAMPLING")
    print(f"  Target: {n_samples} samples")
    print(f"  Reduction method: {reduction_method.upper()}")

    # Step 1: Reduce dimensionality
    print("\n  Step 1: Dimensionality reduction")
    n_components = 20 if reduction_method == 'umap' else 50
    X_reduced, reducer = reduce_dimensionality(X, n_components=n_components, method=reduction_method)

    # Step 2: Random pseudo-labels for initial SVM
    print(f"\n  Step 2: Generate random pseudo-labels")
    pseudo_labels = np.random.randint(0, n_clusters_estimate, size=len(X))

    # Step 3: Train SVM
    print(f"\n  Step 3: Train SVM")
    svm = SVC(kernel='rbf', probability=True, random_state=42, gamma='auto')
    svm.fit(X_reduced, pseudo_labels)

    # Step 4: Compute uncertainty
    print(f"\n  Step 4: Compute uncertainty scores")
    probs = svm.predict_proba(X_reduced)
    uncertainty_scores = 1 - np.max(probs, axis=1)

    # Step 5: Select top uncertain
    print(f"\n  Step 5: Select top {n_samples} uncertain samples")
    selected_indices = np.argsort(uncertainty_scores)[-n_samples:]

    return selected_indices, uncertainty_scores


def method_kmeans_representatives(X, unique_ids, n_samples, n_clusters_estimate=10, reduction_method='pca'):
    """
    Method 3: K-Means Representatives (FASTEST)

    Use K-Means, select samples closest to each cluster center
    Very fast, ensures diversity, but doesn't consider uncertainty
    """
    print(f"\nMethod: K-MEANS REPRESENTATIVES")
    print(f"  Target: {n_samples} samples")
    print(f"  Estimated clusters: {n_clusters_estimate}")
    print(f"  Reduction method: {reduction_method.upper()}")

    # Step 1: Reduce dimensionality
    print("\n  Step 1: Dimensionality reduction")
    n_components = 20 if reduction_method == 'umap' else 50
    X_reduced, reducer = reduce_dimensionality(X, n_components=n_components, method=reduction_method)

    # Step 2: K-Means
    print(f"\n  Step 2: K-Means clustering")
    kmeans = KMeans(n_clusters=n_clusters_estimate, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_reduced)
    centers = kmeans.cluster_centers_

    # Step 3: Select closest to centers + some diversity
    print(f"\n  Step 3: Select representatives")
    samples_per_cluster = max(1, n_samples // n_clusters_estimate)
    selected_indices = []

    for cluster_id in range(n_clusters_estimate):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            continue

        # Compute distances to center
        cluster_samples = X_reduced[cluster_indices]
        center = centers[cluster_id]
        distances = euclidean_distances(cluster_samples, center.reshape(1, -1)).flatten()

        # Select closest samples (most representative)
        n_select = min(samples_per_cluster, len(cluster_indices))
        closest_local = np.argsort(distances)[:n_select]
        closest_global = cluster_indices[closest_local]

        selected_indices.extend(closest_global)

        print(f"    Cluster {cluster_id}: {len(cluster_indices)} samples, selected {n_select}")

    selected_indices = selected_indices[:n_samples]

    # Dummy uncertainty scores (distances to centers)
    uncertainty_scores = np.zeros(len(X))
    for idx in selected_indices:
        cluster_id = labels[idx]
        center = centers[cluster_id]
        uncertainty_scores[idx] = euclidean_distances(
            X_reduced[idx].reshape(1, -1),
            center.reshape(1, -1)
        )[0, 0]

    return selected_indices, uncertainty_scores


def extract_long_wav_basename(sample_id):
    """
    Extract long wav base name from sample ID

    Example:
        Input: G-C1L1P-Apr27-E-Irma_q2_03-08-377_Herminio10P_206.40_207.40
        Output: G-C1L1P-Apr27-E-Irma_q2_03-08-377

    Pattern: Remove last 3 parts (speaker_name, start_time, end_time)
    """
    parts = sample_id.split('_')
    if len(parts) >= 3:
        # Keep all parts except last 3
        long_wav_base = '_'.join(parts[:-3])
    else:
        # Fallback: use the whole sample_id
        long_wav_base = sample_id
    return long_wav_base


def save_results(selected_indices, unique_ids, file_paths, dvectors, uncertainty_scores,
                 gt_labels, output_folder):
    """
    Save selected samples to CSV files (one per long wav)

    CSV format (no header, comma-separated): filename, cluster_id, start_time, end_time
    Creates one CSV per long wav base name
    """
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    # Group samples by long wav base name
    samples_by_longwav = defaultdict(list)

    for idx in selected_indices:
        sample_id = unique_ids[idx]
        long_wav_base = extract_long_wav_basename(sample_id)

        # Parse start_time and end_time from sample_id
        # Format: ..._{speaker}_{start_time}_{end_time}
        parts = sample_id.split('_')
        try:
            start_time = float(parts[-2])
            end_time = float(parts[-1])
        except (ValueError, IndexError):
            print(f"Warning: Could not parse times from {sample_id}, using 0.0")
            start_time = 0.0
            end_time = 0.0

        samples_by_longwav[long_wav_base].append({
            'filename': sample_id,
            'cluster_id': 0,  # Constant 0 as requested
            'start_time': start_time,
            'end_time': end_time,
            'uncertainty_score': uncertainty_scores[idx],  # For sorting, not saved to CSV
            'gt_label': gt_labels[idx] if gt_labels is not None else 'N/A'  # For display, not saved to CSV
        })

    # Save one CSV per long wav
    print(f"\n{'='*80}")
    print(f"SAVING RESULTS")
    print(f"{'='*80}")
    print(f"Output folder: {output_folder_path}")
    print(f"Number of long wav files: {len(samples_by_longwav)}")
    print()

    csv_files_created = []
    for long_wav_base, samples_list in sorted(samples_by_longwav.items()):
        # Create CSV for this long wav
        csv_filename = f"{long_wav_base}_samples.csv"
        csv_path = output_folder_path / csv_filename

        # Create dataframe with all data
        df = pd.DataFrame(samples_list)

        # Sort by uncertainty score (highest first) before saving
        df = df.sort_values('uncertainty_score', ascending=False)

        # Select only the 4 columns for CSV output: filename, cluster_id, start_time, end_time
        df_output = df[['filename', 'cluster_id', 'start_time', 'end_time']]

        # Save without header, comma-separated
        df_output.to_csv(csv_path, index=False, header=False, sep=',')

        csv_files_created.append(csv_path)
        print(f"  ✓ {csv_filename}: {len(samples_list)} samples")

    # Print summary
    print(f"\n{'='*80}")
    print(f"SAMPLE SELECTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples in dataset: {len(unique_ids)}")
    print(f"Selected samples: {len(selected_indices)}")
    print(f"Selection ratio: {len(selected_indices)/len(unique_ids)*100:.1f}%")
    print(f"CSV files created: {len(csv_files_created)}")

    # Print distribution across long wavs
    print(f"\nSamples per long wav:")
    for long_wav_base, samples_list in sorted(samples_by_longwav.items()):
        print(f"  {long_wav_base}: {len(samples_list)} samples")

    # Create combined summary dataframe for display
    all_samples_data = []
    for idx in selected_indices:
        all_samples_data.append({
            'sample_id': unique_ids[idx],
            'long_wav_base': extract_long_wav_basename(unique_ids[idx]),
            'uncertainty_score': uncertainty_scores[idx],
            'gt_label': gt_labels[idx] if gt_labels is not None else 'N/A'
        })

    summary_df = pd.DataFrame(all_samples_data)
    summary_df = summary_df.sort_values('uncertainty_score', ascending=False)

    print(f"\nTop 10 samples to label (highest uncertainty):")
    print(summary_df[['sample_id', 'long_wav_base', 'uncertainty_score', 'gt_label']].head(10).to_string(index=False))

    if gt_labels is not None:
        print(f"\nSpeaker distribution in selected samples (GT for reference):")
        speaker_counts = summary_df['gt_label'].value_counts()
        for speaker, count in speaker_counts.items():
            print(f"  {speaker}: {count} samples")

    return csv_files_created


# ============================================================================
# DEFAULT EXAMPLE VARIABLES (for standalone testing)
# ============================================================================
root_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO','Unsupervised_Pipeline')
dataset_name_ex = 'TestAO-IrmaAlt'
exp_name_ex = 'EXP010'
vad_name_ex = 'SHAS'
feat_name_ex = 'DV'

# Stage 2 D-vectors pickle (input)
stg2_folder_ex = root_ex / dataset_name_ex / 'STG_2' / f'STG2_EXP010-SHAS-DV'
dvectors_pickle_ex = stg2_folder_ex / f'TestAO-IrmaAlt_SHAS_DV_feats.pickle'

# Stage 4 output folder (CSVs will be saved here)
stg4_folder_ex = root_ex / dataset_name_ex / 'STG_4' / 'STG4_LP1'
output_folder_ex = stg4_folder_ex

# Selection parameters
n_samples_ex = 20
method_ex = 'hybrid'  # Options: 'hybrid', 'uncertainty', 'kmeans'
n_clusters_ex = 10
seed_ex = 42
reduction_method_ex = 'pca'  # Options: 'pca', 'umap'

# Create output folder if it doesn't exist
output_folder_ex.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(
    description='Smart sample selection for manual labeling using active learning',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__
)

parser.add_argument(
    '--dvectors_pickle',
    type=str,
    default=dvectors_pickle_ex,
    help='Path to Stage 2 D-vectors pickle file'
)

parser.add_argument(
    '--output_folder',
    type=str,
    default=output_folder_ex,
    help='Path to output folder where CSV files will be saved (one CSV per long wav)'
)

parser.add_argument(
    '--n_samples',
    type=int,
    default=n_samples_ex,
    help='Number of samples to select for manual labeling (default: 20)'
)

parser.add_argument(
    '--method',
    type=str,
    choices=['hybrid', 'uncertainty', 'kmeans'],
    default=method_ex,
    help='Selection method: hybrid (recommended), uncertainty, kmeans (default: hybrid)'
)

parser.add_argument(
    '--n_clusters',
    type=int,
    default=n_clusters_ex,
    help='Estimated number of clusters/speakers (default: 10)'
)

parser.add_argument(
    '--seed',
    type=int,
    default=seed_ex,
    help='Random seed for reproducibility (default: 42)'
)

parser.add_argument(
    '--reduction_method',
    type=str,
    choices=['pca', 'umap'],
    default=reduction_method_ex,
    help='Dimensionality reduction method: pca (fast, linear) or umap (slow, non-linear) (default: pca)'
)

args = parser.parse_args()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("="*80)
print("SMART SAMPLE SELECTION FOR MANUAL LABELING")
print("="*80)

# Set random seed
np.random.seed(args.seed)
print(f"\nRandom seed: {args.seed}")

# Load data
data = load_dvectors_pickle(args.dvectors_pickle)

X = data['dvectors']
unique_ids = data['unique_ids']
file_paths = data['file_paths']
gt_labels = data['gt_labels']

# Normalize features
print(f"\nNormalizing D-vectors...")
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Select samples based on method
if args.method == 'hybrid':
    selected_indices, uncertainty_scores = method_hybrid_uncertainty_diversity(
        X_normalized, unique_ids, args.n_samples, args.n_clusters, args.reduction_method
    )
elif args.method == 'uncertainty':
    selected_indices, uncertainty_scores = method_pure_uncertainty(
        X_normalized, unique_ids, args.n_samples, args.n_clusters, args.reduction_method
    )
elif args.method == 'kmeans':
    selected_indices, uncertainty_scores = method_kmeans_representatives(
        X_normalized, unique_ids, args.n_samples, args.n_clusters, args.reduction_method
    )

# Save results
csv_files = save_results(
    selected_indices, unique_ids, file_paths, X, uncertainty_scores,
    gt_labels, args.output_folder
)

print(f"\n{'='*80}")
print(f"NEXT STEPS")
print(f"{'='*80}")
print(f"CSV files were created (one per long wav file):")
print(f"  Location: {args.output_folder}")
print(f"  Files: {len(csv_files)} CSV files")
print()
print(f"Next steps:")
print(f"  1. Review the CSV files in: {args.output_folder}")
print(f"  2. Each CSV contains samples from one long wav file")
print(f"  3. Columns: sample_id, file_path, uncertainty_score, gt_label")
print(f"  4. Listen to the audio files and manually label speakers")
print(f"  5. Use the webapp or create human_labels.json for LP")
print()
