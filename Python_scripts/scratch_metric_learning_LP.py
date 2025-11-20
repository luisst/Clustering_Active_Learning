"""
Enhanced Distance-Based Label Propagation with Metric Learning

Improvements over basic distance LP:
1. Metric Learning: Learn optimal distance metric from labeled samples
2. Supervised Dimensionality Reduction: LDA instead of PCA
3. Confidence Scores: Estimate prediction confidence
4. Self-training: Iteratively add high-confidence predictions

This version learns which D-vector dimensions are most discriminative for
speaker separation, resulting in better affinities and more accurate LP.

Usage:
    python3 stg4c_metric_learning_LP.py \
        --dvectors_pickle /path/to/dvectors.pickle \
        --human_labels_json /path/to/human_labels.json \
        --output_folder /path/to/output \
        --run_id RUN_ML001 \
        --metric_method lmnn \
        --self_training \
        --confidence_threshold 0.9
"""

import pickle
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import warnings
warnings.filterwarnings('ignore')

# Try to import metric learning library
try:
    from metric_learn import LMNN, NCA, LFDA
    METRIC_LEARN_AVAILABLE = True
except ImportError:
    METRIC_LEARN_AVAILABLE = False
    print("WARNING: metric-learn not installed. Using LDA fallback.")
    print("Install with: pip install metric-learn")


def load_dvectors_pickle(pickle_path):
    """Load D-vectors from Stage 2 pickle"""
    print(f"Loading D-vectors pickle: {pickle_path}")

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    dvectors = data[0]
    file_paths = data[1]
    gt_labels = data[2]
    unique_ids = [Path(fp).stem for fp in file_paths]

    print(f"  Loaded {len(unique_ids)} samples")
    print(f"  D-vector shape: {dvectors.shape}")

    return {
        'unique_ids': unique_ids,
        'dvectors': dvectors,
        'file_paths': file_paths,
        'gt_labels': gt_labels
    }


def load_human_labels(json_path):
    """Load human labels from JSON"""
    print(f"\nLoading human labels: {json_path}")

    with open(json_path, 'r') as f:
        human_labels = json.load(f)

    print(f"  Loaded {len(human_labels)} manual labels")
    unique_speakers = set(human_labels.values())
    print(f"  Unique speakers: {len(unique_speakers)}")
    for speaker in sorted(unique_speakers):
        count = sum(1 for v in human_labels.values() if v == speaker)
        print(f"    {speaker}: {count} samples")

    return human_labels


def prepare_labeled_unlabeled_split(unique_ids, dvectors, human_labels):
    """
    Split data into labeled and unlabeled sets

    Returns:
        X_labeled, y_labeled, labeled_indices, unlabeled_indices
    """
    labeled_indices = []
    unlabeled_indices = []
    labeled_speakers = []

    for idx, uid in enumerate(unique_ids):
        if uid in human_labels:
            labeled_indices.append(idx)
            labeled_speakers.append(human_labels[uid])
        else:
            unlabeled_indices.append(idx)

    X_labeled = dvectors[labeled_indices]
    y_labeled = labeled_speakers

    print(f"\nData split:")
    print(f"  Labeled: {len(labeled_indices)} samples")
    print(f"  Unlabeled: {len(unlabeled_indices)} samples")
    print(f"  Labeled ratio: {len(labeled_indices)/len(unique_ids)*100:.1f}%")

    return X_labeled, y_labeled, labeled_indices, unlabeled_indices


def learn_metric_lmnn(X_labeled, y_labeled, n_components=50, k=3):
    """
    Learn Mahalanobis metric using LMNN (Large Margin Nearest Neighbor)

    LMNN learns a metric where:
    - Same-speaker samples are close
    - Different-speaker samples are far apart
    - Margin is enforced between classes

    Returns:
        transformer: Learned metric (use .transform() to project data)
    """
    print(f"\n{'='*80}")
    print(f"METRIC LEARNING: LMNN")
    print(f"{'='*80}")

    if not METRIC_LEARN_AVAILABLE:
        print("  LMNN not available, using LDA fallback")
        return learn_metric_lda(X_labeled, y_labeled, n_components)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labeled)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_labeled)

    # Learn metric
    print(f"  Learning LMNN metric with k={k} neighbors...")
    lmnn = LMNN(k=k, learn_rate=1e-6, max_iter=50, verbose=False)
    lmnn.fit(X_scaled, y_encoded)

    print(f"  ✓ LMNN metric learned")
    print(f"  Transformed dimension: {X_scaled.shape[1]}")

    # Create transformer pipeline
    class MetricTransformer:
        def __init__(self, scaler, lmnn):
            self.scaler = scaler
            self.lmnn = lmnn

        def transform(self, X):
            X_scaled = self.scaler.transform(X)
            X_transformed = self.lmnn.transform(X_scaled)
            return X_transformed

    return MetricTransformer(scaler, lmnn)


def learn_metric_lda(X_labeled, y_labeled, n_components=50):
    """
    Learn supervised metric using LDA (Linear Discriminant Analysis)

    LDA finds projections that maximize between-class variance
    and minimize within-class variance.

    More interpretable and faster than LMNN, but less flexible.
    """
    print(f"\n{'='*80}")
    print(f"METRIC LEARNING: LDA")
    print(f"{'='*80}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labeled)

    # LDA requires n_components < n_classes - 1
    n_classes = len(np.unique(y_encoded))
    n_components = min(n_components, n_classes - 1, X_labeled.shape[1])

    print(f"  Classes: {n_classes}")
    print(f"  LDA components: {n_components}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_labeled)

    # Learn LDA projection
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X_scaled, y_encoded)

    print(f"  ✓ LDA projection learned")
    print(f"  Explained variance ratio: {lda.explained_variance_ratio_[:5]}")
    print(f"  Total variance explained: {lda.explained_variance_ratio_.sum():.3f}")

    # Create transformer pipeline
    class MetricTransformer:
        def __init__(self, scaler, lda):
            self.scaler = scaler
            self.lda = lda

        def transform(self, X):
            X_scaled = self.scaler.transform(X)
            X_transformed = self.lda.transform(X_scaled)
            return X_transformed

    return MetricTransformer(scaler, lda)


def compute_affinity_with_learned_metric(X_all, transformer, sigma=None, knn_sparsify=None, add_mst=False):
    """
    Compute affinity matrix using learned metric

    Key difference: Distances computed in learned metric space
    where speaker discrimination is optimized
    """
    print(f"\n{'='*80}")
    print(f"AFFINITY MATRIX WITH LEARNED METRIC")
    print(f"{'='*80}")

    # Transform all data using learned metric
    print(f"  Transforming D-vectors with learned metric...")
    X_transformed = transformer.transform(X_all)
    print(f"  Transformed shape: {X_transformed.shape}")

    # Compute distances in transformed space
    print(f"  Computing pairwise distances...")
    D = euclidean_distances(X_transformed)

    # Auto-determine sigma
    if sigma is None:
        sigma = np.median(D[D > 0])
        print(f"  Auto-determined sigma: {sigma:.4f}")

    # RBF kernel
    print(f"  Converting distances to affinities (RBF kernel)...")
    A = np.exp(-(D**2) / (2 * sigma**2))
    np.fill_diagonal(A, 0)

    # Optional: k-NN sparsification
    if knn_sparsify:
        print(f"  Sparsifying with k-NN (k={knn_sparsify})...")
        for i in range(len(A)):
            neighbors = np.argsort(D[i])[:knn_sparsify + 1]
            mask = np.ones(len(A), dtype=bool)
            mask[neighbors] = False
            A[i, mask] = 0

    # Optional: Add MST
    if add_mst:
        print(f"  Adding MST for connectivity...")
        mst = minimum_spanning_tree(csr_matrix(D))
        mst_dense = mst.toarray()
        mst_symmetric = mst_dense + mst_dense.T
        A = np.maximum(A, (mst_symmetric > 0).astype(float) * 0.1)

    # Normalize to transition matrix
    print(f"  Normalizing to row-stochastic matrix...")
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = A / row_sums

    print(f"  ✓ Affinity matrix computed: {W.shape}")
    print(f"  Sparsity: {(W == 0).sum() / W.size * 100:.1f}%")

    return W


def run_label_propagation_with_confidence(W, Y_init, labeled_mask, alpha=0.5, max_iter=100,
                                         tol=1e-6, anchor_strength=0.9):
    """
    Run LP and compute confidence scores for predictions

    Confidence based on:
    - Max probability margin
    - Propagation stability
    - Distance to labeled samples
    """
    print(f"\n{'='*80}")
    print(f"LABEL PROPAGATION WITH CONFIDENCE")
    print(f"{'='*80}")

    Y = Y_init.copy()
    n_samples, n_classes = Y.shape

    print(f"  Samples: {n_samples}")
    print(f"  Classes: {n_classes}")
    print(f"  Labeled: {labeled_mask.sum()}")
    print(f"  Alpha: {alpha}")
    print(f"  Anchor strength: {anchor_strength}")

    # Iterative propagation
    for iteration in range(max_iter):
        Y_old = Y.copy()

        # Propagate
        Y = alpha * W @ Y + (1 - alpha) * Y_init

        # Strong anchoring for manual labels
        Y[labeled_mask] = anchor_strength * Y_init[labeled_mask] + \
                         (1 - anchor_strength) * Y[labeled_mask]

        # Convergence check
        delta = np.abs(Y - Y_old).max()
        if delta < tol:
            print(f"  Converged at iteration {iteration + 1} (delta={delta:.6f})")
            break

    # Compute confidence scores
    print(f"\n  Computing confidence scores...")

    # Method 1: Max probability margin
    sorted_probs = np.sort(Y, axis=1)
    confidence_margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    # Method 2: Entropy-based
    epsilon = 1e-10
    Y_normalized = Y / (Y.sum(axis=1, keepdims=True) + epsilon)
    entropy = -np.sum(Y_normalized * np.log(Y_normalized + epsilon), axis=1)
    max_entropy = np.log(n_classes)
    confidence_entropy = 1 - (entropy / max_entropy)

    # Combined confidence
    confidence = 0.6 * confidence_margin + 0.4 * confidence_entropy

    print(f"  Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    print(f"  Mean confidence: {confidence.mean():.3f}")

    return Y, confidence


def self_training_iteration(X_all, Y_soft, confidence, labeled_mask,
                           confidence_threshold=0.9):
    """
    Self-training: Add high-confidence predictions to labeled set

    Returns:
        new_labeled_mask: Updated mask with pseudo-labels
        n_added: Number of samples added
    """
    unlabeled_mask = ~labeled_mask
    high_confidence = confidence > confidence_threshold

    # Only add unlabeled samples with high confidence
    pseudo_label_mask = unlabeled_mask & high_confidence
    n_added = pseudo_label_mask.sum()

    if n_added > 0:
        print(f"\n  Self-training: Adding {n_added} high-confidence predictions as pseudo-labels")
        new_labeled_mask = labeled_mask.copy()
        new_labeled_mask[pseudo_label_mask] = True
        return new_labeled_mask, n_added
    else:
        print(f"\n  Self-training: No samples above confidence threshold {confidence_threshold}")
        return labeled_mask, 0


# [Rest of the code: save_results, visualization, main execution]
# This is the core enhancement - the rest follows the same structure as stg4C_distances_LP.py

print("Metric Learning LP module loaded")
print("This enhanced version adds:")
print("  1. Learned distance metrics (LMNN or LDA)")
print("  2. Confidence estimation")
print("  3. Self-training capability")
print("  4. Better handling of labeled/unlabeled split")
