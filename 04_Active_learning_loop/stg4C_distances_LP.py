"""
Stage 4c: Distance-Based Label Propagation using Stage 2 D-vectors

This script performs label propagation directly from Stage 2 D-vectors pickle file,
bypassing all clustering stages. Uses pairwise distances (euclidean, manhattan, or cosine)
to build the affinity graph for propagation.

Key features:
1. Loads D-vectors directly from Stage 2 pickle (TestAO-Irma_SHAS_DV_feats.pickle)
2. Requires manual labels provided via command line or interactive prompt
3. Computes pairwise distances between samples using specified metric
4. Builds affinity matrix from distances using RBF kernel or inverse distance
5. Performs label propagation from manual labels to unlabeled samples
6. Generates visualization and results CSV

This approach is:
- Simpler: No clustering stages needed
- Faster: Direct from D-vectors to LP
- More interpretable: Distance-based affinity is intuitive
- Flexible: Can experiment with different distance metrics and affinity conversions
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import sys
import json
from umap import UMAP


from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from collections import Counter

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def valid_path(path):
    """Validate that a path exists"""
    path = Path(path)
    if path.exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"Invalid path: {path}")


def compute_umap_reduction(X_data, n_components=20, n_neighbors=15, min_dist=0.1, 
                           metric='cosine', random_state=42):
    """
    Apply UMAP dimensionality reduction.
    
    Parameters:
    -----------
    X_data : np.ndarray
        Input data (n_samples, n_features)
    n_components : int
        Target dimensionality (default: 20)
    n_neighbors : int
        UMAP n_neighbors parameter
    min_dist : float
        UMAP min_dist parameter
    metric : str
        Distance metric for UMAP
    random_state : int
        Random seed
        
    Returns:
    --------
    np.ndarray : UMAP-reduced data (n_samples, n_components)
    """
    print(f"\nApplying UMAP dimensionality reduction...")
    print(f"  Input shape: {X_data.shape}")
    print(f"  Target dimensions: {n_components}")
    print(f"  n_neighbors: {n_neighbors}")
    print(f"  min_dist: {min_dist}")
    print(f"  metric: {metric}")
    
    umap_reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False
    )
    
    X_umap = umap_reducer.fit_transform(X_data)
    
    print(f"  ✓ UMAP completed")
    print(f"  Output shape: {X_umap.shape}")
    
    return X_umap


def load_dvectors_pickle(pickle_path):
    """
    Load D-vectors from Stage 2 pickle file.

    Returns:
    --------
    dict with:
        - dvectors: np.ndarray (n_samples, n_features)
        - file_paths: list of file paths
        - gt_labels: np.ndarray of ground truth labels
        - unique_ids: list of unique identifiers
    """
    print(f"\nLoading D-vectors pickle: {pickle_path}")

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # Pickle contains: [X_train, X_train_paths, y_train]
    dvectors = data[0]
    file_paths = data[1]
    gt_labels = data[2]

    # Create unique IDs from file paths
    unique_ids = [Path(fp).stem for fp in file_paths]

    print(f"  ✓ Loaded {len(unique_ids)} samples")
    print(f"  D-vector shape: {dvectors.shape}")
    print(f"  GT labels: {np.unique(gt_labels)}")

    return {
        'dvectors': dvectors,
        'file_paths': file_paths,
        'gt_labels': gt_labels,
        'unique_ids': unique_ids
    }


def load_human_labels_json(json_path, unique_ids):
    """
    Load human labels from JSON file.

    JSON format:
    {
        "sample_id_1": "Speaker_A",
        "sample_id_2": "Speaker_B",
        ...
    }

    Returns:
    --------
    dict mapping sample index to speaker label
    """
    print(f"\nLoading human labels from: {json_path}")

    with open(json_path, 'r') as f:
        labels_dict = json.load(f)

    # Map sample IDs to indices
    id_to_idx = {uid: idx for idx, uid in enumerate(unique_ids)}
    human_labels = {}

    for sample_id, speaker_label in labels_dict.items():
        if sample_id in id_to_idx:
            idx = id_to_idx[sample_id]
            human_labels[idx] = speaker_label

    print(f"  ✓ Loaded {len(human_labels)} manual labels")
    print(f"  Speaker IDs: {sorted(set(human_labels.values()))}")

    return human_labels


def compute_distance_matrix(X_data, metric='euclidean'):
    """Compute pairwise distance matrix using specified metric."""
    print(f"\nComputing pairwise {metric} distances...")
    n_samples = X_data.shape[0]

    if metric == 'euclidean':
        D = euclidean_distances(X_data)
    elif metric == 'manhattan':
        D = manhattan_distances(X_data)
    elif metric == 'cosine':
        D = cosine_distances(X_data)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    print(f"  ✓ Distance matrix computed: {D.shape}")
    print(f"    Mean distance: {np.mean(D):.4f}")
    print(f"    Median distance: {np.median(D):.4f}")
    print(f"    Min distance (non-zero): {np.min(D[D > 0]):.4f}")
    print(f"    Max distance: {np.max(D):.4f}")

    return D


def distances_to_affinity(D, sigma=None, method='rbf', knn_sparsify=None):
    """Convert distance matrix to affinity matrix."""
    print(f"\nConverting distances to affinities...")
    print(f"  Method: {method}")

    n_samples = D.shape[0]

    # Determine sigma if not provided
    if sigma is None:
        sigma = np.median(D[D > 0])
        print(f"  Auto sigma (median distance): {sigma:.4f}")
    else:
        print(f"  User-provided sigma: {sigma:.4f}")

    # Convert distances to affinities
    if method == 'rbf':
        A = np.exp(-(D**2) / (2 * sigma**2))
    elif method == 'inverse':
        A = 1.0 / (1.0 + D)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Set diagonal to zero
    np.fill_diagonal(A, 0.0)

    # Sparsify by keeping only k nearest neighbors
    if knn_sparsify is not None:
        print(f"  Sparsifying: keeping {knn_sparsify} nearest neighbors per sample")
        A_sparse = np.zeros_like(A)
        for i in range(n_samples):
            k_nearest_indices = np.argsort(D[i])[1:knn_sparsify+1]
            A_sparse[i, k_nearest_indices] = A[i, k_nearest_indices]
        A = A_sparse
        A = 0.5 * (A + A.T)

    # Add small self-connections
    np.fill_diagonal(A, 0.1)

    n_edges = np.sum(A > 1e-8)
    sparsity = n_edges / (n_samples ** 2)
    print(f"  ✓ Affinity matrix created")
    print(f"    Non-zero elements: {n_edges}")
    print(f"    Sparsity: {sparsity*100:.2f}%")

    return A, sigma


def add_mst_edges_dense(A, D, sigma):
    """Add MST edges to affinity matrix for connectivity."""
    print(f"\nAdding MST for connectivity...")

    D_sparse = csr_matrix(D)
    mst = minimum_spanning_tree(D_sparse, overwrite=False)
    mst_coo = mst.tocoo()

    mst_edges_added = 0
    for i, j, weight in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        mst_affinity = np.exp(-weight / sigma)
        if A[i, j] < mst_affinity:
            A[i, j] = mst_affinity
            A[j, i] = mst_affinity
            mst_edges_added += 1

    print(f"  ✓ Added/strengthened {mst_edges_added} MST edges")
    return A


def run_label_propagation_simple(W, Y, human_labels, id_to_idx, alpha, max_iter, tol, anchor_strength):
    """
    Run simple label propagation without HDBSCAN priors.

    Only uses manual labels for anchoring.
    """
    print(f"\nRunning distance-based label propagation...")
    print(f"  Alpha: {alpha}")
    print(f"  Anchor strength: {anchor_strength}")
    print(f"  Manual labels: {len(human_labels)}")

    n_samples = Y.shape[0]
    n_classes = Y.shape[1]
    F = Y.copy()

    idx_to_id = {v: k for k, v in id_to_idx.items()}

    # Track metrics
    convergence_history = []
    confidence_history = []

    for it in range(max_iter):
        F_prev = F.copy()

        # Propagation step
        F_new = alpha * W.dot(F)

        # Strong anchoring for manual labels
        for idx, spk in human_labels.items():
            speaker_idx = id_to_idx[spk]
            F_new[idx] = (1 - anchor_strength) * F_new[idx]
            F_new[idx, speaker_idx] += anchor_strength

        # Normalize rows
        row_sums = F_new.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        F_new = F_new / row_sums

        # Calculate convergence
        delta = np.abs(F_new - F_prev).sum()
        convergence_history.append(delta)

        # Calculate average confidence
        confidence_scores = np.max(F_new, axis=1)
        avg_confidence = np.mean(confidence_scores)
        confidence_history.append(avg_confidence)

        F = F_new

        if (it + 1) % 10 == 0:
            print(f"  Iteration {it+1}/{max_iter}, delta={delta:.6f}, conf={avg_confidence:.4f}")

        if delta < tol:
            print(f"  ✓ Converged at iteration {it+1}")
            break

    metrics = {
        'convergence_history': convergence_history,
        'confidence_history': confidence_history
    }

    return F, metrics


def compute_tsne(X_data, n_components=2, perplexity=30, random_state=42):
    """Compute t-SNE for visualization."""
    print(f"\nComputing t-SNE for visualization...")
    print(f"  Perplexity: {perplexity}")

    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=random_state, n_jobs=-1)
    tsne_2d = tsne.fit_transform(X_data)

    print(f"  ✓ t-SNE completed")
    return tsne_2d


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(
    description='Distance-based Label Propagation from Stage 2 D-vectors'
)

parser.add_argument(
    '--dvectors_pickle',
    type=valid_path,
    required=True,
    help='Path to Stage 2 D-vectors pickle file'
)

parser.add_argument(
    '--human_labels_json',
    type=valid_path,
    required=True,
    help='Path to JSON file with manual labels (sample_id -> speaker_label)'
)

parser.add_argument(
    '--output_folder',
    type=str,
    required=True,
    help='Output folder for results'
)

parser.add_argument(
    '--run_id',
    type=str,
    default='RUN_DIST',
    help='Run identifier for output files'
)

parser.add_argument(
    '--distance_metric',
    type=str,
    default='cosine',
    choices=['euclidean', 'manhattan', 'cosine'],
    help='Distance metric (default: cosine)'
)

parser.add_argument(
    '--affinity_method',
    type=str,
    default='rbf',
    choices=['rbf', 'inverse'],
    help='Affinity conversion method (default: rbf)'
)

parser.add_argument(
    '--sigma',
    type=float,
    default=None,
    help='Bandwidth for affinity (default: auto - median distance)'
)

parser.add_argument(
    '--knn_sparsify',
    type=int,
    default=None,
    help='Sparsify by keeping k nearest neighbors (default: None - dense)'
)

parser.add_argument(
    '--add_mst',
    action='store_true',
    help='Add MST edges for connectivity'
)

parser.add_argument(
    '--alpha',
    type=float,
    default=0.5,
    help='Propagation strength (default: 0.5)'
)

parser.add_argument(
    '--max_iter',
    type=int,
    default=100,
    help='Maximum iterations (default: 100)'
)

parser.add_argument(
    '--tol',
    type=float,
    default=1e-6,
    help='Convergence tolerance (default: 1e-6)'
)

parser.add_argument(
    '--anchor_strength',
    type=float,
    default=0.9,
    help='Anchoring strength for manual labels (default: 0.9)'
)

parser.add_argument(
    '--tsne_perplexity',
    type=int,
    default=30,
    help='t-SNE perplexity (default: 30)'
)

parser.add_argument(
    '--umap_n_components',
    type=int,
    default=20,
    help='UMAP dimensions before t-SNE (default: 20)'
)

parser.add_argument(
    '--umap_n_neighbors',
    type=int,
    default=15,
    help='UMAP n_neighbors parameter (default: 15)'
)

parser.add_argument(
    '--umap_min_dist',
    type=float,
    default=0.1,
    help='UMAP min_dist parameter (default: 0.1)'
)

args = parser.parse_args()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("="*80)
print("STAGE 4C: DISTANCE-BASED LABEL PROPAGATION (FROM STAGE 2)")
print("="*80)
print(f"D-vectors pickle: {args.dvectors_pickle}")
print(f"Human labels JSON: {args.human_labels_json}")
print(f"Output folder: {args.output_folder}")
print(f"Distance metric: {args.distance_metric}")
print(f"Affinity method: {args.affinity_method}")
print("="*80)

# Create output folder
output_folder = Path(args.output_folder)
output_folder.mkdir(parents=True, exist_ok=True)

# Load D-vectors
data = load_dvectors_pickle(args.dvectors_pickle)
X_data = data['dvectors']
unique_ids = data['unique_ids']
gt_labels = data['gt_labels']
n_samples = len(unique_ids)

# Load human labels
human_labels = load_human_labels_json(args.human_labels_json, unique_ids)

if len(human_labels) == 0:
    print("\nError: No human labels found in JSON file.")
    sys.exit(1)

# Compute distance matrix
D = compute_distance_matrix(X_data, metric=args.distance_metric)

# Convert to affinity matrix
A, sigma_used = distances_to_affinity(
    D, sigma=args.sigma, method=args.affinity_method, knn_sparsify=args.knn_sparsify
)

# Add MST if requested
if args.add_mst:
    A = add_mst_edges_dense(A, D, sigma_used)

# Normalize affinity matrix
print(f"\nNormalizing affinity matrix...")
row_sums = A.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
W = A / row_sums
print(f"  ✓ Affinity matrix normalized")

# Initialize label distributions
print(f"\n{'='*80}")
print("INITIALIZING LABEL DISTRIBUTIONS")
print("="*80)

speaker_ids = sorted(set(human_labels.values()))
id_to_idx = {spk: idx for idx, spk in enumerate(speaker_ids)}
idx_to_id = {v: k for k, v in id_to_idx.items()}
n_classes = len(speaker_ids)

print(f"  Speaker classes: {n_classes}")
print(f"  Speakers: {speaker_ids}")
print(f"  Manually labeled: {len(human_labels)}")
print(f"  Unlabeled: {n_samples - len(human_labels)}")

# Initialize label matrix - uniform distribution for unlabeled
Y = np.ones((n_samples, n_classes)) / n_classes

# Strong priors for manual labels
for idx, spk in human_labels.items():
    Y[idx] = 0
    Y[idx, id_to_idx[spk]] = 1.0

# Run label propagation
print(f"\n{'='*80}")
print("RUNNING LABEL PROPAGATION")
print("="*80)

F, metrics = run_label_propagation_simple(
    W, Y, human_labels, id_to_idx, args.alpha, args.max_iter, args.tol, args.anchor_strength
)

# Get final predictions
y_pred_idx = np.argmax(F, axis=1)
y_pred = np.array([idx_to_id[i] for i in y_pred_idx])
confidence_scores = np.max(F, axis=1)

# Map GT labels to speaker names
print(f"\n{'='*80}")
print("MAPPING GROUND TRUTH")
print("="*80)

gt_label_to_speaker = {}
for sample_idx, speaker in human_labels.items():
    numeric_gt = gt_labels[sample_idx]
    if numeric_gt not in gt_label_to_speaker:
        gt_label_to_speaker[numeric_gt] = speaker

print(f"  GT to speaker mapping: {gt_label_to_speaker}")

gt_labels_mapped = np.array([gt_label_to_speaker.get(gt, 'UNKNOWN') for gt in gt_labels])

# Results
print(f"\n{'='*80}")
print("RESULTS")
print("="*80)
print(f"  Average confidence: {np.mean(confidence_scores):.4f}")
print(f"  Min confidence: {np.min(confidence_scores):.4f}")
print(f"  Max confidence: {np.max(confidence_scores):.4f}")

pred_counts = Counter(y_pred)
print(f"\n  Label distribution:")
for spk, count in sorted(pred_counts.items()):
    print(f"    {spk}: {count} samples")

# Compute t-SNE for visualization
print(f"\n{'='*80}")
print("PREPARING DATA FOR VISUALIZATION")
print("="*80)

# Apply UMAP reduction before t-SNE
X_umap = compute_umap_reduction(
    X_data,
    n_components=args.umap_n_components,
    n_neighbors=args.umap_n_neighbors,
    min_dist=args.umap_min_dist,
    metric=args.distance_metric,
    random_state=42
)

# Compute t-SNE on UMAP-reduced data
tsne_2d = compute_tsne(X_umap, perplexity=min(args.tsne_perplexity, n_samples-1))

# Save results
results_df = pd.DataFrame({
    'unique_id': unique_ids,
    'file_path': data['file_paths'],
    'gt_label': gt_labels,
    'gt_speaker': gt_labels_mapped,
    'lp_label': y_pred,
    'lp_confidence': confidence_scores,
    'human_label': [human_labels.get(i, None) for i in range(n_samples)],
    'tsne_x': tsne_2d[:, 0],
    'tsne_y': tsne_2d[:, 1]
})

results_csv_path = output_folder / f"{args.run_id}_lp_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\n✓ Results saved to: {results_csv_path}")

# Visualization
print(f"\n{'='*80}")
print("GENERATING VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plot 1: Ground Truth
ax = axes[0]
unique_gt = np.unique(gt_labels_mapped)
palette_gt = sns.color_palette('tab10', n_colors=len(unique_gt))
gt_to_color = {label: palette_gt[i] for i, label in enumerate(unique_gt)}
colors_gt = [gt_to_color[label] for label in gt_labels_mapped]
ax.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=colors_gt, s=50, alpha=0.7)
handles = [mlines.Line2D([], [], color=gt_to_color[label], marker='o', linestyle='None',
                         markersize=8, label=label) for label in unique_gt]
ax.legend(handles=handles, title='Ground Truth', loc='best')
ax.set_title('Ground Truth Labels')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')

# Plot 2: LP Results
ax = axes[1]
palette_lp = sns.color_palette('tab10', n_colors=len(speaker_ids))
lp_to_color = {spk: palette_lp[i] for i, spk in enumerate(speaker_ids)}

# Unlabeled samples with confidence-based transparency
unlabeled_indices = [i for i in range(n_samples) if i not in human_labels]
for i in unlabeled_indices:
    conf = confidence_scores[i]
    ax.scatter(tsne_2d[i, 0], tsne_2d[i, 1],
              c=[lp_to_color[y_pred[i]]], s=50, alpha=0.3 + 0.6*conf, edgecolors='none')

# Manual labels with stars
labeled_indices = list(human_labels.keys())
if labeled_indices:
    ax.scatter(tsne_2d[labeled_indices, 0], tsne_2d[labeled_indices, 1],
              c=[lp_to_color[y_pred[i]] for i in labeled_indices],
              s=200, alpha=1.0, marker='*', edgecolors='black', linewidths=2, zorder=10)

handles = [mlines.Line2D([], [], color=lp_to_color[spk], marker='o', linestyle='None',
                         markersize=8, label=f"{spk} (n={pred_counts[spk]})") for spk in speaker_ids]
handles.append(mlines.Line2D([], [], color='gray', marker='*', linestyle='None',
                            markersize=15, markeredgecolor='black', markeredgewidth=2,
                            label='Manual Labels'))
ax.legend(handles=handles, title='Label Propagation', loc='best')
ax.set_title(f'Label Propagation Results ({len(human_labels)} manual labels)')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')

plt.tight_layout()
plot_path = output_folder / f"{args.run_id}_visualization.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {plot_path}")
plt.close()

# Accuracy
gt_accuracy = accuracy_score(gt_labels_mapped, y_pred)
print(f"\n{'='*80}")
print(f"ACCURACY vs GROUND TRUTH: {gt_accuracy*100:.2f}%")
print(f"{'='*80}")

print(f"\nSummary:")
print(f"  - Total samples: {n_samples}")
print(f"  - Manual labels: {len(human_labels)}")
print(f"  - Distance metric: {args.distance_metric}")
print(f"  - Affinity method: {args.affinity_method}")
print(f"  - Sigma: {sigma_used:.4f}")
print(f"  - UMAP dimensions: {args.umap_n_components}")
print(f"  - Accuracy vs GT: {gt_accuracy*100:.2f}%")
print(f"  - Avg confidence: {np.mean(confidence_scores):.4f}")
print(f"  - Converged in {len(metrics['convergence_history'])} iterations")
print("="*80)
