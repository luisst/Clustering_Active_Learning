from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import hdbscan
import umap
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pprint
import mplcursors

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import normalize

import matplotlib.lines as mlines
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.exceptions import UndefinedMetricWarning
from scipy.stats import entropy
from scipy import sparse

def mst_hdbscan_to_sparse_matrix(mst_object):
    """
    Convert MST HDBSCAN object to scipy sparse matrix.
    
    Parameters:
    -----------
    mst_object : object
        MST object with _data and _mst attributes

    Returns:
    --------
    scipy.sparse matrix
        Sparse matrix representation of the MST
    """
    
    # Extract data from the MST object
    data = mst_object._data  # Shape: (n_samples, n_components)
    mst_edges = mst_object._mst  # Shape: (n_samples - 1, 3)
    
    n_samples = data.shape[0]
    
    # Extract edge information
    start_nodes = mst_edges[:, 0].astype(int)
    end_nodes = mst_edges[:, 1].astype(int)
    weights = mst_edges[:, 2]

    row_indices = start_nodes
    col_indices = end_nodes
    edge_weights = weights
    
    # Create the sparse matrix
    sparse_matrix = sparse.coo_matrix(
        (edge_weights, (row_indices, col_indices)),
        shape=(n_samples, n_samples)
    )
    
    return sparse_matrix

def active_learning_sample_selection(hdb_labels, hdb_probs, X_data, umap_data, n_samples_per_cluster=3, plot_flag=False):
    """
    Select samples for manual labeling using Active Learning strategies.
    
    Parameters:
    -----------
    hdb_labels : array
        HDBSCAN cluster labels
    hdb_probs : array  
        HDBSCAN membership probabilities
    X_data : array
        Original feature data
    umap_data : array
        UMAP-reduced data for visualization
    n_samples_per_cluster : int
        Number of samples to select per cluster (max 3)
        
    Returns:
    --------
    dict : Dictionary mapping cluster_id to list of selected sample indices
    dict : Dictionary with selection reasons for logging
    """
    
    from sklearn.metrics.pairwise import euclidean_distances
    import numpy as np
    
    selected_samples = {}
    selection_reasons = {}
    
    # Get unique clusters (excluding noise -1)
    unique_clusters = np.unique(hdb_labels)
    unique_clusters = unique_clusters[unique_clusters != -1]
    
    print(f"Active Learning: Selecting up to {n_samples_per_cluster} samples from {len(unique_clusters)} clusters")
    
    for cluster_id in unique_clusters:
        cluster_mask = hdb_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_probs = hdb_probs[cluster_mask]
        cluster_features = X_data[cluster_mask]
        cluster_umap = umap_data[cluster_mask]
        
        if len(cluster_indices) < n_samples_per_cluster:
            # If cluster has fewer samples than requested, select all
            selected_samples[cluster_id] = cluster_indices.tolist()
            selection_reasons[cluster_id] = ['small_cluster'] * len(cluster_indices)
            continue
        
        selected_for_cluster = []
        reasons_for_cluster = []
        
        # Strategy 1: Centroid-like sample (most representative)
        # Use UMAP space for centroid calculation (better for visualization)
        centroid = np.mean(cluster_umap, axis=0)
        distances_to_centroid = euclidean_distances(cluster_umap, [centroid]).flatten()
        centroid_idx = cluster_indices[np.argmin(distances_to_centroid)]
        selected_for_cluster.append(centroid_idx)
        reasons_for_cluster.append('centroid')
        
        # Strategy 2: Uncertain sample (low probability of belonging to cluster)
        if len(cluster_indices) > 1:
            # Find sample with lowest probability in this cluster
            uncertain_local_idx = np.argmin(cluster_probs)
            uncertain_idx = cluster_indices[uncertain_local_idx]
            
            # Avoid selecting the same sample twice
            if uncertain_idx != centroid_idx:
                selected_for_cluster.append(uncertain_idx)
                reasons_for_cluster.append('uncertain')
        
        # Strategy 3: Boundary sample (diverse/informative)
        if len(cluster_indices) > 2 and len(selected_for_cluster) < n_samples_per_cluster:
            # Find sample that is farthest from already selected samples
            # This helps capture cluster diversity and potential boundary cases
            
            selected_umap = umap_data[selected_for_cluster]
            
            max_min_distance = -1
            boundary_idx = None
            
            for i, candidate_idx in enumerate(cluster_indices):
                if candidate_idx in selected_for_cluster:
                    continue
                    
                candidate_umap = umap_data[candidate_idx].reshape(1, -1)
                
                # Calculate minimum distance to already selected samples
                distances_to_selected = euclidean_distances(candidate_umap, selected_umap).flatten()
                min_distance_to_selected = np.min(distances_to_selected)
                
                # Select the sample that maximizes the minimum distance (diversity)
                if min_distance_to_selected > max_min_distance:
                    max_min_distance = min_distance_to_selected
                    boundary_idx = candidate_idx
            
            if boundary_idx is not None:
                selected_for_cluster.append(boundary_idx)
                reasons_for_cluster.append('boundary_diverse')
        
        # Alternative Strategy 3: Edge/Boundary sample (if we have enough samples)
        # This finds samples at the edge of the cluster in feature space
        if len(selected_for_cluster) < n_samples_per_cluster and len(cluster_indices) > 5:
            # Calculate distances from each sample to all other samples in cluster
            intra_cluster_distances = euclidean_distances(cluster_umap)
            
            # Find sample with highest average distance to other cluster members
            # (likely to be on the boundary)
            avg_distances = np.mean(intra_cluster_distances, axis=1)
            
            # Exclude already selected samples
            available_indices = []
            available_distances = []
            
            for i, cluster_idx in enumerate(cluster_indices):
                if cluster_idx not in selected_for_cluster:
                    available_indices.append(cluster_idx)
                    available_distances.append(avg_distances[i])
            
            if available_distances:
                edge_local_idx = np.argmax(available_distances)
                edge_idx = available_indices[edge_local_idx]
                selected_for_cluster.append(edge_idx)
                reasons_for_cluster.append('edge_boundary')
        
        selected_samples[cluster_id] = selected_for_cluster[:n_samples_per_cluster]
        selection_reasons[cluster_id] = reasons_for_cluster[:n_samples_per_cluster]
        
        print(f"  Cluster {cluster_id}: Selected {len(selected_for_cluster)} samples "
              f"from {len(cluster_indices)} total ({reasons_for_cluster})")
    if plot_flag:
        # Plot UMAP with selected samples highlighted
        plt.figure(figsize=(10, 8))
        plt.scatter(umap_data[:, 0], umap_data[:, 1], c='lightgray', s=20, label='All samples')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
        color_map = {cid: colors[i % len(colors)] for i, cid in enumerate(unique_clusters)}
        
        for cluster_id, sample_indices in selected_samples.items():
            cluster_umap = umap_data[sample_indices]
            plt.scatter(cluster_umap[:, 0], cluster_umap[:, 1], 
                        c=color_map[cluster_id], s=25, label=f'Cluster {cluster_id}')
        
        plt.title('UMAP Projection with Selected Active Learning Samples')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return selected_samples, selection_reasons


def format_active_learning_results(selected_samples, selection_reasons, wav_stems, hdb_labels, hdb_probs, output_folder_path, run_id):
    """
    Format and save active learning results for manual labeling.
    
    Parameters:
    -----------
    selected_samples : dict
        Dictionary mapping cluster_id to list of selected sample indices
    selection_reasons : dict  
        Dictionary with selection reasons
    wav_stems : list
        List of wav file stems
    hdb_labels : array
        HDBSCAN labels
    hdb_probs : array
        HDBSCAN probabilities
    output_folder_path : Path
        Output folder path
    run_id : str
        Run identifier
    """
    
    # Create a summary DataFrame
    summary_data = []
    
    for cluster_id, sample_indices in selected_samples.items():
        reasons = selection_reasons[cluster_id]
        
        for idx, reason in zip(sample_indices, reasons):
            summary_data.append({
                'cluster_id': cluster_id,
                'sample_index': idx,
                'wav_stem': wav_stems[idx],
                'selection_reason': reason,
                'hdbscan_prob': hdb_probs[idx],
                'suggested_label': f'Speaker_C{cluster_id}'  # Placeholder for manual labeling
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV for manual labeling
    active_learning_path = output_folder_path / f"{run_id}_active_learning_samples.csv"
    summary_df.to_csv(active_learning_path, index=False)
    
    print(f"\nActive Learning Results:")
    print(f"  Total samples selected: {len(summary_data)}")
    print(f"  Samples saved to: {active_learning_path}")
    print(f"  Please manually label the 'suggested_label' column with actual speaker IDs")
    
    # Print summary by strategy
    strategy_counts = summary_df['selection_reason'].value_counts()
    print(f"  Selection strategy breakdown: {dict(strategy_counts)}")
    
    return summary_df


if __name__ == "__main__":

    pickle_file = Path(r"C:\Users\luis2\Dropbox\DATASETS_AUDIO\Unsupervised_Pipeline\MiniClusters\STG_2\STG2_EXP010-SHAS-DV\MiniClusters_SHAS_DV_feats.pickle")
    run_id = "distx"

    output_folder_path = pickle_file.parent / "hdb_lp4"  # Update with the actual path
    log_path = output_folder_path / f"{run_id}_log.txt"
    output_folder_path.mkdir(parents=True, exist_ok=True)

    umap_pickle_path = output_folder_path / f"{run_id}_umap_data.pickle"

    with open(pickle_file, "rb") as file:
        X_data_and_labels = pickle.load(file)
    X_data, X_paths, y_labels = X_data_and_labels

    # Extract wav file stems from paths
    wav_stems = [Path(path).stem for path in X_paths]

    # # If umap pickle path exists, then do not compute umap again
    # if umap_pickle_path.exists():
    #     with open(umap_pickle_path, "rb") as f:
    #         umap_data = pickle.load(f)
    # else:

    data_standardized = StandardScaler().fit_transform(X_data)

    # Apply UMAP
    umap_reducer = umap.UMAP(
        n_neighbors=10,  # Adjust based on dataset size
        min_dist=0.1,    # Controls compactness of clusters
        n_components=20,  # Reduced dimensionality
        metric='cosine'  # Good default for many feature types
        # random_state=42  # For reproducibility
    )
    # umap_reducer = umap.UMAP(
    #     n_neighbors=5,  # Adjust based on dataset size
    #     min_dist=0.1,    # Controls compactness of clusters
    #     n_components=20,  # Reduced dimensionality
    #     metric='manhattan'  # Good default for many feature types
    #     # random_state=42  # For reproducibility
    # )
    umap_data = umap_reducer.fit_transform(data_standardized)

    # Store umap_data in a pickle
    with open(umap_pickle_path, "wb") as f:
        pickle.dump(umap_data, f)

    ### --------------------------------------------------------------------------------------

    hdb = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5, approx_min_span_tree=True,\
                    gen_min_span_tree=True, prediction_data=True,\
                    cluster_selection_method='eom').fit(umap_data)

    hdb_labels = hdb.labels_
    hdb_probs = hdb.probabilities_
    n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)  # Exclude noise label (-1)
    n_samples = len(hdb_labels)
    n_assigned = len(hdb_labels[hdb_labels != -1])
    percentage_assigned = (n_assigned / n_samples) * 100

    print(f'Number of clusters: {n_clusters}')
    print(f'Percentage of assigned samples: {percentage_assigned:.2f}%')


    # Active Learning Sample Selection
    selected_samples, selection_reasons = active_learning_sample_selection(
        hdb_labels, hdb_probs, X_data, umap_data, n_samples_per_cluster=3, plot_flag=True
    )

    # Format and save results for manual labeling
    active_learning_df = format_active_learning_results(
        selected_samples, selection_reasons, wav_stems, hdb_labels, hdb_probs, 
        output_folder_path, run_id
    )


    ### --------------------------------------------------------------------------------------
    # k=5
    # metric='cosine'
    # add_mst=True
    # alpha=0.5
    # max_iter=100

    # tol=1e-6
    # human_labels = {0: 'Ari', 40: 'Ed', 75: 'Eve', 120: 'Jad', 170: 'Lan',
    #                 25: 'Ari', 55: 'Ed', 96: 'Eve', 140: 'Jad'}  # Example: index to speaker ID

    # # X: (N, D) numpy array
    # nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1).fit(X_data)
    # distances, indices = nbrs.kneighbors(X_data)  # distances[:,0] is zero (self)
    # # drop self
    # distances = distances[:, 1:]  # (N, k)
    # indices = indices[:, 1:]      # (N, k)

    # n_samples = X_data.shape[0]
    # # compute local scale sigma_i as distance to k-th neighbor
    # sigma = distances[:, -1]

    # # Avoid division by zero
    # sigma = np.maximum(sigma, 1e-8)

    # rows = []
    # cols = []
    # data = []

    # for i in range(n_samples):
    #     for j_idx, d in zip(indices[i], distances[i]):
    #         # Skip self-connections
    #         if i == j_idx:
    #             continue

    #         # compute weight using local scaling
    #         wij = np.exp(- (d**2) / (sigma[i] * sigma[j_idx] + 1e-12))
    #         rows.append(i)
    #         cols.append(j_idx)
    #         data.append(wij)

    # A = coo_matrix((data, (rows, cols)), shape=(n_samples, n_samples)).tocsr()

    # # plt.figure(figsize=(16, 8))
    # # plt.spy(A, markersize=2, aspect='auto')
    # # plt.title('Sparsity Pattern of the CSR Matrix')
    # # plt.xlabel('Column Index')
    # # plt.ylabel('Row Index')
    # # plt.grid(True)

    # # dense_matrix = A.toarray()

    # # plt.figure(figsize=(16, 8))
    # # sns.heatmap(dense_matrix, cmap='viridis')
    # # plt.title('Seaborn Heatmap of the CSR Matrix')
    # # plt.xlabel('Column Index')
    # # plt.ylabel('Row Index')
    # # plt.show()

    # # symmetrize by intersection (mutual): keep only edges present both ways (A & A.T)
    # A_mutual = A.multiply(A.transpose())
    # # make it symmetric
    # A = 0.5 * (A_mutual + A_mutual.transpose())

    # # Add small self-connections before normalization
    # A.setdiag(0.1)  # Small self-weight



    # # ensure connectivity: add MST edges computed on Euclidean distances
    # if add_mst:
    #     # Get MST from HDBSCAN
    #     mst = hdb.minimum_spanning_tree_
        
    #     # Extract edge information from HDBSCAN MST
    #     # MST is stored as a sparse matrix with edge weights
    #     mst_coo = mst_hdbscan_to_sparse_matrix(mst)
        
    #     # Add MST edges to our adjacency matrix
    #     for i, j, weight in zip(mst_coo.row, mst_coo.col, mst_coo.data):
    #         # Convert MST distance to similarity weight
    #         # Use a small weight to ensure connectivity without dominating local structure
    #         mst_weight = np.exp(-weight / np.mean(sigma))  # normalize by average local scale
            
    #         # Add edge in both directions (symmetric)
    #         A[i, j] = max(A[i, j], mst_weight)
    #         A[j, i] = max(A[j, i], mst_weight)
        
    #     print(f"Added {len(mst_coo.data)} MST edges for connectivity")

    # # row-normalize if needed later for propagation
    # A_final = A.tocsr()

    # # Option 1: Using sklearn's normalize function (recommended)
    # W = normalize(A_final, norm='l1', axis=1)

    # # --- Step 3: Initialize label distributions with HDBSCAN weak priors ---
    # speaker_ids = list(set(human_labels.values()))
    # id_to_idx = {spk: idx for idx, spk in enumerate(speaker_ids)}
    # n_classes = len(speaker_ids)

    # # Create cluster-to-speaker mapping from manual labels
    # cluster_to_speaker = {}
    # for idx, spk in human_labels.items():
    #     hdb_cluster = hdb_labels[idx]
    #     if hdb_cluster != -1:  # Not noise
    #         if hdb_cluster in cluster_to_speaker:
    #             # If cluster already mapped, verify consistency
    #             if cluster_to_speaker[hdb_cluster] != spk:
    #                 print(f"Warning: Cluster {hdb_cluster} has conflicting labels: {cluster_to_speaker[hdb_cluster]} vs {spk}")
    #         cluster_to_speaker[hdb_cluster] = spk

    # print(f"Cluster-to-speaker mapping: {cluster_to_speaker}")

    # # Initialize label matrix
    # Y = np.zeros((n_samples, n_classes))

    # # Set strong priors for manually labeled samples
    # for idx, spk in human_labels.items():
    #     Y[idx, id_to_idx[spk]] = 1.0  # Strong prior (will be anchored during propagation)

    # # Set weak priors for HDBSCAN predictions
    # weak_prior_strength = 0.5  # Adjust this value (0.1-0.7 range)
    # min_confidence_threshold = 0.7  # Only use HDBSCAN predictions above this confidence

    # for i in range(n_samples):
    #     # Skip if already manually labeled
    #     if i in human_labels:
    #         continue
        
    #     hdb_cluster = hdb_labels[i]
    #     hdb_confidence = hdb_probs[i]
        
    #     # Only add weak prior if:
    #     # 1. Not noise (-1)
    #     # 2. Cluster is mapped to a speaker
    #     # 3. HDBSCAN confidence is above threshold
    #     if (hdb_cluster != -1 and 
    #         hdb_cluster in cluster_to_speaker and 
    #         hdb_confidence >= min_confidence_threshold):
            
    #         speaker = cluster_to_speaker[hdb_cluster]
    #         speaker_idx = id_to_idx[speaker]
            
    #         # Set weak prior proportional to HDBSCAN confidence
    #         prior_strength = weak_prior_strength * hdb_confidence
    #         Y[i, speaker_idx] = prior_strength
            
    #         # Distribute remaining probability uniformly among other classes
    #         remaining_prob = 1.0 - prior_strength
    #         uniform_prob = remaining_prob / (n_classes - 1)
    #         for j in range(n_classes):
    #             if j != speaker_idx:
    #                 Y[i, j] = uniform_prob

    # print(f"Label propagation setup:")
    # print(f"  Number of manually labeled samples: {len(human_labels)}")
    # print(f"  Number of HDBSCAN weak priors: {np.sum((hdb_labels != -1) & (hdb_probs >= min_confidence_threshold) & (~np.isin(np.arange(n_samples), list(human_labels.keys()))))}")
    # print(f"  Number of speaker classes: {n_classes}")
    # print(f"  Speaker IDs: {speaker_ids}")
    # print(f"  Weak prior strength: {weak_prior_strength}")
    # print(f"  Min confidence threshold: {min_confidence_threshold}")

    # # Propagated distributions start from Y
    # F = Y.copy()

    # # --- Step 4: Modified iterative propagation with different anchor strengths ---
    # manual_anchor_strength = 0.2  # Strong anchoring for manual labels (1 - alpha for manual)
    # weak_anchor_strength = 0.05   # Weak anchoring for HDBSCAN priors

    # for it in range(max_iter):
    #     F_new = alpha * W.dot(F)  # Smoothing term
        
    #     # Add strong anchoring for manually labeled samples
    #     for idx, spk in human_labels.items():
    #         speaker_idx = id_to_idx[spk]
    #         F_new[idx] = (1 - manual_anchor_strength) * F_new[idx]
    #         F_new[idx, speaker_idx] += manual_anchor_strength
        
    #     # Add weak anchoring for HDBSCAN priors
    #     for i in range(n_samples):
    #         if i in human_labels:
    #             continue
                
    #         hdb_cluster = hdb_labels[i]
    #         hdb_confidence = hdb_probs[i]
            
    #         if (hdb_cluster != -1 and 
    #             hdb_cluster in cluster_to_speaker and 
    #             hdb_confidence >= min_confidence_threshold):
                
    #             speaker = cluster_to_speaker[hdb_cluster]
    #             speaker_idx = id_to_idx[speaker]
                
    #             # Apply weak anchoring
    #             anchor_weight = weak_anchor_strength * hdb_confidence
    #             F_new[i] = (1 - anchor_weight) * F_new[i]
    #             F_new[i, speaker_idx] += anchor_weight
        
    #     # Normalize rows to maintain probability distributions
    #     row_sums = F_new.sum(axis=1, keepdims=True)
    #     row_sums[row_sums == 0] = 1  # Avoid division by zero
    #     F_new = F_new / row_sums
        
    #     # Convergence check
    #     delta = np.abs(F_new - F).sum()
    #     F = F_new
    #     if delta < tol:
    #         print(f"Converged at iteration {it}")
    #         break
    # else:
    #     print(f"Reached maximum iterations ({max_iter})")


    # # --- Step 5: Assign final labels ---
    # y_pred = np.argmax(F, axis=1)
    # idx_to_id = {v: k for k, v in id_to_idx.items()}
    # y_pred = np.array([idx_to_id[i] for i in y_pred])
    
    # # Print some statistics
    # confidence_scores = np.max(F, axis=1)
    # print(f"Propagation results:")
    # print(f"  Average confidence: {np.mean(confidence_scores):.4f}")
    # print(f"  Min confidence: {np.min(confidence_scores):.4f}")
    # print(f"  Max confidence: {np.max(confidence_scores):.4f}")

    # # Use the human labels to map the y_labels numbers to speaker IDs
    # y_labels_dict = {} 
    # for sample_idx, lbl in enumerate(y_labels):
    #     if sample_idx in human_labels:
    #         y_labels_dict[lbl] = human_labels[sample_idx]

    # y_labels_mapped = np.array([y_labels_dict[lbl] for lbl in y_labels])


    # # --- Step 6: Compare with Ground Truth Labels ---
    # print(f"\nGround Truth Comparison:")
    
    # # Calculate accuracy against ground truth
    # gt_accuracy = accuracy_score(y_labels_mapped, y_pred)
    # print(f"  Overall accuracy vs GT: {gt_accuracy*100:.2f}%")
    
    # # Get unique labels from both predictions and ground truth
    # unique_gt_labels = np.unique(y_labels_mapped)
    # unique_pred_labels = np.unique(y_pred)
    # all_labels = np.unique(np.concatenate([unique_gt_labels, unique_pred_labels]))

    # print(f"  Unique labels: {sorted(all_labels)}")
    # print(f"  Unique GT labels: {sorted(unique_gt_labels)}")
    # print(f"  Unique predicted labels: {sorted(unique_pred_labels)}")
    
    # # Classification report
    # try:
    #     class_report = classification_report(y_labels_mapped, y_pred, labels=all_labels, zero_division=0)
    #     print(f"  Classification Report:\n{class_report}")
    # except Exception as e:
    #     print(f"  Could not generate classification report: {e}")
    
    # # Confusion matrix
    # try:
    #     conf_matrix = confusion_matrix(y_labels_mapped, y_pred, labels=all_labels)
    #     print(f"  Confusion Matrix:")
    #     print(f"  GT\\Pred: {all_labels}")
    #     for i, gt_label in enumerate(all_labels):
    #         print(f"  {gt_label}: {conf_matrix[i]}")
    # except Exception as e:
    #     print(f"  Could not generate confusion matrix: {e}")
    
    # # Per-class accuracy
    # class_accuracies = {}
    # for label in unique_gt_labels:
    #     mask = y_labels_mapped == label
    #     if np.sum(mask) > 0:
    #         class_acc = accuracy_score(y_labels_mapped[mask], y_pred[mask])
    #         class_accuracies[label] = class_acc
    #         print(f"  Class '{label}' accuracy: {class_acc*100:.2f}% ({np.sum(mask)} samples)")
    
    # # Analyze label distribution changes
    # from collections import Counter
    # gt_counts = Counter(y_labels_mapped)
    # original_counts = Counter([human_labels.get(i, 'unlabeled') for i in range(n_samples)])
    # propagated_counts = Counter(y_pred)
    # print(f"GT label distribution: {gt_counts}")
    # print(f"Label distribution before propagation: {original_counts}")
    # print(f"Label distribution after propagation: {propagated_counts}")
    # # Save results to a CSV
    # results_df = pd.DataFrame({
    #     'wav_stem': wav_stems,
    #     'gt_label': y_labels_mapped,
    #     'hdbscan_label': hdb_labels,
    #     'hdbscan_prob': hdb_probs,
    #     'lp_label': y_pred,
    #     'lp_confidence': confidence_scores
    # })
    # results_df.to_csv(output_folder_path / f"{run_id}_hdb_lp_results.csv", index=False)
    # print(f"Results saved to {output_folder_path / f'{run_id}_hdb_lp_results.csv'}")
    # # Save log
    # with open(log_path, "w") as log_file:
    #     log_file.write(f"Number of clusters: {n_clusters}\n")
    #     log_file.write(f"Percentage of assigned samples: {percentage_assigned:.2f}%\n")
    #     log_file.write(f"Label propagation setup:\n")
    #     log_file.write(f"  Number of labeled samples: {len(human_labels)}\n")
    #     log_file.write(f"  Number of speaker classes: {n_classes}\n")
    #     log_file.write(f"  Speaker IDs: {speaker_ids}\n")
    #     log_file.write(f"Propagation results:\n")
    #     log_file.write(f"  Average confidence: {np.mean(confidence_scores):.4f}\n")
    #     log_file.write(f"  Min confidence: {np.min(confidence_scores):.4f}\n")
    #     log_file.write(f"  Max confidence: {np.max(confidence_scores):.4f}\n")
    #     log_file.write(f"Ground Truth Comparison:\n")
    #     log_file.write(f"  Overall accuracy vs GT: {gt_accuracy*100:.2f}%\n")
    #     log_file.write(f"  Unique GT labels: {sorted(unique_gt_labels.tolist())}\n")
    #     log_file.write(f"  Unique predicted labels: {sorted(unique_pred_labels.tolist())}\n")
    #     for label, acc in class_accuracies.items():
    #         gt_count = np.sum(y_labels_mapped == label)
    #         log_file.write(f"  Class '{label}' accuracy: {acc*100:.2f}% ({gt_count} samples)\n")
    #     log_file.write(f"GT label distribution: {dict(gt_counts)}\n")
    #     log_file.write(f"Label distribution before propagation: {dict(original_counts)}\n")
    #     log_file.write(f"Label distribution after propagation: {dict(propagated_counts)}\n")
    # print(f"Log saved to {log_path}")

    # # Plotting UMAP with Ground Truth, HDBSCAN and LP labels
    # plt.figure(figsize=(24, 8))
    
    # plt.subplot(1, 3, 1)
    # sns.scatterplot(x=umap_data[:, 0], y=umap_data[:, 1], hue=y_labels_mapped, palette='tab10', legend='full')
    # plt.title('Ground Truth Labels')
    
    # plt.subplot(1, 3, 2)
    # sns.scatterplot(x=umap_data[:, 0], y=umap_data[:, 1], hue=hdb_labels, palette='tab10', legend='full')
    # plt.title('HDBSCAN Clustering')
    
    # plt.subplot(1, 3, 3)
    # sns.scatterplot(x=umap_data[:, 0], y=umap_data[:, 1], hue=y_pred, palette='tab10', legend='full')
    # plt.title('Label Propagation Results')
    
    # plt.tight_layout()
    # plt.savefig(output_folder_path / f"{run_id}_hdb_lp_gt_comparison.png", dpi=300, bbox_inches='tight')
    # plt.show()

    # # save in log each wav name and its predicted label
    # with open(log_path, "a") as log_file:
    #     log_file.write("\nDetailed Predictions:\n")
    #     for stem, gt, hdb, hdbp, lp, lpc in zip(wav_stems, y_labels, hdb_labels, hdb_probs, y_pred, confidence_scores):
    #         log_file.write(f"{stem}: GT={gt}, HDBSCAN={hdb} (p={hdbp:.2f}), LP={lp} (conf={lpc:.2f})\n")