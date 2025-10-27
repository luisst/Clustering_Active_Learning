from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import hdbscan as hdb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pprint
import mplcursors
import sys

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
import csv


def read_speaker_diarization(input_folder_AL):
    """
    Reads all speaker diarization CSV files from a folder and returns a dictionary
    mapping SampleIndex to SpeakerLP.
    
    Args:
        input_folder_AL (str or Path): Path to the folder containing CSV files
        
    Returns:
        dict: Dictionary with SampleIndex as keys and SpeakerLP as values
        
    Example:
        >>> speaker_dict = read_speaker_diarization('path/to/csv/folder')
        >>> print(speaker_dict)
        {101: 'S0', 7: 'S1', 81: 'S2', ...}
    """
    from pathlib import Path
    import csv
    import sys
    
    speaker_dict = {}
    input_folder = Path(input_folder_AL)
    
    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Error: Folder '{input_folder}' does not exist or is not a directory.")
        sys.exit(1)
    
    # Find all CSV files in the folder
    csv_files = list(input_folder.glob("*.csv"))
    
    if not csv_files:
        print(f"Error: No CSV files found in '{input_folder}'")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files in '{input_folder}'")
    
    # Process each CSV file
    for csv_file_path in csv_files:
        print(f"Processing: {csv_file_path.name}")
        
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file, delimiter='\t')
                
                file_count = 0
                for row in csv_reader:
                    try:
                        sample_index = int(row['SampleIndex'])
                        speaker_lp = row['SpeakerLP']
                        
                        # Check for duplicate indices
                        if sample_index in speaker_dict:
                            print(f"Warning: Duplicate SampleIndex {sample_index} found in {csv_file_path.name}. "
                                  f"Existing: {speaker_dict[sample_index]}, New: {speaker_lp}")
                        
                        speaker_dict[sample_index] = speaker_lp
                        file_count += 1
                        
                    except (ValueError, KeyError) as e:
                        print(f"Error processing row in {csv_file_path.name}: {e}")
                        print("Exiting program due to data processing error.")
                        sys.exit(1)
                
                print(f"  Added {file_count} entries from {csv_file_path.name}")
                        
        except FileNotFoundError:
            print(f"Error: File '{csv_file_path}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading CSV '{csv_file_path}': {e}")
            sys.exit(1)
    
    if not speaker_dict:
        print("Error: No valid data loaded from CSV files.")
        sys.exit(1)
    
    print(f"Total entries loaded: {len(speaker_dict)}")
    return speaker_dict

def calculate_mst_from_features(X_data, metric='euclidean', return_format='coo'):
    """
    Calculate Minimum Spanning Tree from feature data and return as COO matrix.
    
    Parameters:
    -----------
    X_data : numpy.ndarray
        Feature matrix of shape (n_samples, n_features)
        Use full 256 features for best results
    metric : str, default='euclidean'
        Distance metric for MST calculation
        Options: 'euclidean', 'cosine', 'manhattan', etc.
    return_format : str, default='coo'
        Return format: 'coo' for COO matrix, 'edges' for edge list
    
    Returns:
    --------
    mst_coo : scipy.sparse.coo_matrix or list
        MST as COO matrix or edge list [(i, j, weight), ...]
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.sparse import csr_matrix
    import numpy as np
    
    n_samples = X_data.shape[0]
    
    # Calculate pairwise distances using k-NN for efficiency
    # For MST, we need a fully connected graph, but we can start with k-NN
    k = min(50, n_samples - 1)  # Adjust k based on data size
    
    print(f"Calculating MST for {n_samples} samples with {X_data.shape[1]} features")
    print(f"Using metric: {metric}")
    
    # Method 1: Use k-NN graph and ensure connectivity
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1).fit(X_data)
    distances, indices = nbrs.kneighbors(X_data)
    
    # Build sparse distance matrix from k-NN
    rows = []
    cols = []
    data = []
    
    for i in range(n_samples):
        for j_idx, j in enumerate(indices[i]):
            if i != j:  # Skip self-loops
                rows.append(i)
                cols.append(j)
                data.append(distances[i, j_idx])
    
    # Create sparse distance matrix
    dist_matrix = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    
    # Make symmetric (take minimum distance)
    dist_matrix_sym = dist_matrix.minimum(dist_matrix.T)
    
    # Calculate MST
    mst = minimum_spanning_tree(dist_matrix_sym, overwrite=True)
    
    # Convert to COO format
    mst_coo = mst.tocoo()
    
    print(f"MST calculated with {len(mst_coo.data)} edges")
    print(f"Total MST weight: {np.sum(mst_coo.data):.4f}")
    
    if return_format == 'coo':
        return mst_coo
    elif return_format == 'edges':
        # Return as list of (i, j, weight) tuples
        edges = list(zip(mst_coo.row, mst_coo.col, mst_coo.data))
        return edges
    else:
        raise ValueError("return_format must be 'coo' or 'edges'")



LP_METHOD_NAME = "LP1"
DATASET_NAME = "TestAO-Irma"

base_path_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO','Unsupervised_Pipeline',DATASET_NAME)
stg3_pred_folders_ex = base_path_ex.joinpath('STG_3','STG3_EXP010-SHAS-DV-hdb','merged_wavs')
stg4_al_folder_ex = base_path_ex.joinpath('STG_4',f'STG4_{LP_METHOD_NAME}','webapp_results')

merged_data_clusters_pickle = stg3_pred_folders_ex.parent / 'merged_clustering_data.pickle'

output_folder_path = stg4_al_folder_ex.parent / 'lp_results'
output_folder_path.mkdir(parents=True, exist_ok=True)

log_path = output_folder_path / 'lp_log.txt'

run_id = 'RUN001'

with open(f'{merged_data_clusters_pickle}', "rb") as file:
    merged_clustering_data = pickle.load(file)


merged_X_data, \
merged_paths, \
merged_hdb_data, \
merged_tsne_2d, \
merged_y_labels, \
merged_sample_labels, \
merged_sample_probs, \
merged_sample_outliers = merged_clustering_data


k=5
metric='cosine'
add_mst=True
alpha=0.5
max_iter=100

tol=1e-6
# TODO: load human_labels from AL webapp
human_labels = read_speaker_diarization(stg4_al_folder_ex)
# human_labels = {0: 'Ari', 40: 'Ed', 75: 'Eve', 120: 'Jad', 170: 'Lan',
#                 25: 'Ari', 55: 'Ed', 96: 'Eve', 140: 'Jad'}  # Example: index to speaker ID

# X: (N, D) numpy array
nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1).fit(merged_X_data)
distances, indices = nbrs.kneighbors(merged_X_data)  # distances[:,0] is zero (self)
# drop self
distances = distances[:, 1:]  # (N, k)
indices = indices[:, 1:]      # (N, k)

n_samples = merged_X_data.shape[0]
# compute local scale sigma_i as distance to k-th neighbor
sigma = distances[:, -1]

# Avoid division by zero
sigma = np.maximum(sigma, 1e-8)

rows = []
cols = []
data = []

for i in range(n_samples):
    for j_idx, d in zip(indices[i], distances[i]):
        # Skip self-connections
        if i == j_idx:
            continue

        # compute weight using local scaling
        wij = np.exp(- (d**2) / (sigma[i] * sigma[j_idx] + 1e-12))
        rows.append(i)
        cols.append(j_idx)
        data.append(wij)

A = coo_matrix((data, (rows, cols)), shape=(n_samples, n_samples)).tocsr()

plt.figure(figsize=(16, 8))
plt.spy(A, markersize=2, aspect='auto')
plt.title('Sparsity Pattern of the CSR Matrix')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.grid(True)

dense_matrix = A.toarray()

plt.figure(figsize=(16, 8))
sns.heatmap(dense_matrix, cmap='viridis')
plt.title('Seaborn Heatmap of the CSR Matrix')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()

# symmetrize by intersection (mutual): keep only edges present both ways (A & A.T)
A_mutual = A.multiply(A.transpose())
# make it symmetric
A = 0.5 * (A_mutual + A_mutual.transpose())

# Add small self-connections before normalization
A.setdiag(0.1)  # Small self-weight



# ensure connectivity: add MST edges computed on Euclidean distances
if add_mst:
    
    # Extract edge information from HDBSCAN MST
    # MST is stored as a sparse matrix with edge weights
    mst_coo = calculate_mst_from_features(merged_X_data, metric='euclidean')
    
    # Add MST edges to our adjacency matrix
    for i, j, weight in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        # Convert MST distance to similarity weight
        # Use a small weight to ensure connectivity without dominating local structure
        mst_weight = np.exp(-weight / np.mean(sigma))  # normalize by average local scale
        
        # Add edge in both directions (symmetric)
        A[i, j] = max(A[i, j], mst_weight)
        A[j, i] = max(A[j, i], mst_weight)
    
    print(f"Added {len(mst_coo.data)} MST edges for connectivity")

# row-normalize if needed later for propagation
A_final = A.tocsr()

# Option 1: Using sklearn's normalize function (recommended)
W = normalize(A_final, norm='l1', axis=1)

# --- Step 3: Initialize label distributions with HDBSCAN weak priors ---
speaker_ids = list(set(human_labels.values()))
id_to_idx = {spk: idx for idx, spk in enumerate(speaker_ids)}
n_classes = len(speaker_ids)

# Create cluster-to-speaker mapping from manual labels
cluster_to_speaker = {}
for idx, spk in human_labels.items():
    hdb_cluster = merged_sample_labels[idx]
    if hdb_cluster != -1:  # Not noise
        if hdb_cluster in cluster_to_speaker:
            # If cluster already mapped, verify consistency
            if cluster_to_speaker[hdb_cluster] != spk:
                print(f"Warning: Cluster {hdb_cluster} has conflicting labels: {cluster_to_speaker[hdb_cluster]} vs {spk}")
        cluster_to_speaker[hdb_cluster] = spk

print(f"Cluster-to-speaker mapping: {cluster_to_speaker}")

# Initialize label matrix
Y = np.zeros((n_samples, n_classes))

# Set strong priors for manually labeled samples
for idx, spk in human_labels.items():
    Y[idx, id_to_idx[spk]] = 1.0  # Strong prior (will be anchored during propagation)

# Set weak priors for HDBSCAN predictions
weak_prior_strength = 0.5  # Adjust this value (0.1-0.7 range)
min_confidence_threshold = 0.7  # Only use HDBSCAN predictions above this confidence

for i in range(n_samples):
    # Skip if already manually labeled
    if i in human_labels:
        continue
    
    hdb_cluster = merged_sample_labels[i]
    hdb_confidence = merged_sample_probs[i]
    
    # Only add weak prior if:
    # 1. Not noise (-1)
    # 2. Cluster is mapped to a speaker
    # 3. HDBSCAN confidence is above threshold
    if (hdb_cluster != -1 and 
        hdb_cluster in cluster_to_speaker and 
        hdb_confidence >= min_confidence_threshold):
        
        speaker = cluster_to_speaker[hdb_cluster]
        speaker_idx = id_to_idx[speaker]
        
        # Set weak prior proportional to HDBSCAN confidence
        prior_strength = weak_prior_strength * hdb_confidence
        Y[i, speaker_idx] = prior_strength
        
        # Distribute remaining probability uniformly among other classes
        remaining_prob = 1.0 - prior_strength
        uniform_prob = remaining_prob / (n_classes - 1)
        for j in range(n_classes):
            if j != speaker_idx:
                Y[i, j] = uniform_prob

print(f"Label propagation setup:")
print(f"  Number of manually labeled samples: {len(human_labels)}")
print(f"  Number of HDBSCAN weak priors: {np.sum((merged_sample_labels != -1) & (merged_sample_probs >= min_confidence_threshold) & (~np.isin(np.arange(n_samples), list(human_labels.keys()))))}")
print(f"  Number of speaker classes: {n_classes}")
print(f"  Speaker IDs: {speaker_ids}")
print(f"  Weak prior strength: {weak_prior_strength}")
print(f"  Min confidence threshold: {min_confidence_threshold}")

# Propagated distributions start from Y
F = Y.copy()

# --- Step 4: Modified iterative propagation with different anchor strengths ---
manual_anchor_strength = 0.2  # Strong anchoring for manual labels (1 - alpha for manual)
weak_anchor_strength = 0.05   # Weak anchoring for HDBSCAN priors

for it in range(max_iter):
    F_new = alpha * W.dot(F)  # Smoothing term
    
    # Add strong anchoring for manually labeled samples
    for idx, spk in human_labels.items():
        speaker_idx = id_to_idx[spk]
        F_new[idx] = (1 - manual_anchor_strength) * F_new[idx]
        F_new[idx, speaker_idx] += manual_anchor_strength
    
    # Add weak anchoring for HDBSCAN priors
    for i in range(n_samples):
        if i in human_labels:
            continue
            
        hdb_cluster = merged_sample_labels[i]
        hdb_confidence = merged_sample_probs[i]
        
        if (hdb_cluster != -1 and 
            hdb_cluster in cluster_to_speaker and 
            hdb_confidence >= min_confidence_threshold):
            
            speaker = cluster_to_speaker[hdb_cluster]
            speaker_idx = id_to_idx[speaker]
            
            # Apply weak anchoring
            anchor_weight = weak_anchor_strength * hdb_confidence
            F_new[i] = (1 - anchor_weight) * F_new[i]
            F_new[i, speaker_idx] += anchor_weight
    
    # Normalize rows to maintain probability distributions
    row_sums = F_new.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    F_new = F_new / row_sums
    
    # Convergence check
    delta = np.abs(F_new - F).sum()
    F = F_new
    if delta < tol:
        print(f"Converged at iteration {it}")
        break
else:
    print(f"Reached maximum iterations ({max_iter})")


# --- Step 5: Assign final labels ---
y_pred = np.argmax(F, axis=1)
idx_to_id = {v: k for k, v in id_to_idx.items()}
y_pred = np.array([idx_to_id[i] for i in y_pred])

# Print some statistics
confidence_scores = np.max(F, axis=1)
print(f"Propagation results:")
print(f"  Average confidence: {np.mean(confidence_scores):.4f}")
print(f"  Min confidence: {np.min(confidence_scores):.4f}")
print(f"  Max confidence: {np.max(confidence_scores):.4f}")

# Use the human labels to map the y_labels numbers to speaker IDs
print(f'shape of human_labels: {len(human_labels)} \t type: {type(human_labels)}')
y_labels_dict = {} 
for sample_idx, lbl in enumerate(merged_y_labels):
    print(f"Sample idx: {sample_idx}, Label: {lbl}")
    if sample_idx in human_labels:
        print(f"   Mapped to speaker ID: {human_labels[sample_idx]}")
        y_labels_dict[lbl] = human_labels[sample_idx]

# Handle cases where merged_y_labels has more labels than y_labels_dict
y_labels_mapped = []
unmapped_labels = set()

for lbl in merged_y_labels:
    if lbl in y_labels_dict:
        y_labels_mapped.append(y_labels_dict[lbl])
    else:
        # Use a default label for unmapped ground truth labels
        y_labels_mapped.append('SXX')
        unmapped_labels.add(lbl)

y_labels_mapped = np.array(y_labels_mapped)

if unmapped_labels:
    print(f"\nWarning: Found {len(unmapped_labels)} unmapped ground truth labels: {sorted(unmapped_labels)}")
    print(f"These samples will be labeled as 'UNKNOWN'")

print(f"\nMapped Ground Truth Labels:")
pprint.pprint(y_labels_dict)
print(f'-------------------------------------\n')

# --- Step 6: Compare with Ground Truth Labels ---
print(f"\nGround Truth Comparison:")

# Calculate accuracy against ground truth
gt_accuracy = accuracy_score(y_labels_mapped, y_pred)
print(f"  Overall accuracy vs GT: {gt_accuracy*100:.2f}%")

# Get unique labels from both predictions and ground truth
unique_gt_labels = np.unique(y_labels_mapped)
unique_pred_labels = np.unique(y_pred)
all_labels = np.unique(np.concatenate([unique_gt_labels, unique_pred_labels]))

print(f"  Unique labels: {sorted(all_labels)}")
print(f"  Unique GT labels: {sorted(unique_gt_labels)}")
print(f"  Unique predicted labels: {sorted(unique_pred_labels)}")

# Classification report
try:
    class_report = classification_report(y_labels_mapped, y_pred, labels=all_labels, zero_division=0)
    print(f"  Classification Report:\n{class_report}")
except Exception as e:
    print(f"  Could not generate classification report: {e}")

# Confusion matrix
try:
    conf_matrix = confusion_matrix(y_labels_mapped, y_pred, labels=all_labels)
    print(f"  Confusion Matrix:")
    print(f"  GT\\Pred: {all_labels}")
    for i, gt_label in enumerate(all_labels):
        print(f"  {gt_label}: {conf_matrix[i]}")
except Exception as e:
    print(f"  Could not generate confusion matrix: {e}")

# Per-class accuracy
class_accuracies = {}
for label in unique_gt_labels:
    mask = y_labels_mapped == label
    if np.sum(mask) > 0:
        class_acc = accuracy_score(y_labels_mapped[mask], y_pred[mask])
        class_accuracies[label] = class_acc
        print(f"  Class '{label}' accuracy: {class_acc*100:.2f}% ({np.sum(mask)} samples)")

# Analyze label distribution changes
from collections import Counter
gt_counts = Counter(y_labels_mapped)
original_counts = Counter([human_labels.get(i, 'unlabeled') for i in range(n_samples)])
propagated_counts = Counter(y_pred)
print(f"GT label distribution: {gt_counts}")
print(f"Label distribution before propagation: {original_counts}")
print(f"Label distribution after propagation: {propagated_counts}")

merged_stem_paths = [Path(p).stem for p in merged_paths]

# Save results to a CSV
results_df = pd.DataFrame({
    'wav_stem': merged_stem_paths,
    'gt_label': y_labels_mapped,
    'hdbscan_label': merged_sample_labels,
    'hdbscan_prob': merged_sample_probs,
    'lp_label': y_pred,
    'lp_confidence': confidence_scores
})
results_df.to_csv(output_folder_path / f"{run_id}_hdb_lp_results.csv", index=False)
print(f"Results saved to {output_folder_path / f'{run_id}_hdb_lp_results.csv'}")

n_clusters = len(set(merged_sample_labels)) - (1 if -1 in merged_sample_labels else 0)
percentage_assigned = (np.sum(merged_sample_labels != -1) / n_samples) * 100

# Save log
with open(log_path, "w") as log_file:
    log_file.write(f"Number of clusters: {n_clusters}\n")
    log_file.write(f"Percentage of assigned samples: {percentage_assigned:.2f}%\n")
    log_file.write(f"Label propagation setup:\n")
    log_file.write(f"  Number of labeled samples: {len(human_labels)}\n")
    log_file.write(f"  Number of speaker classes: {n_classes}\n")
    log_file.write(f"  Speaker IDs: {speaker_ids}\n")
    log_file.write(f"Propagation results:\n")
    log_file.write(f"  Average confidence: {np.mean(confidence_scores):.4f}\n")
    log_file.write(f"  Min confidence: {np.min(confidence_scores):.4f}\n")
    log_file.write(f"  Max confidence: {np.max(confidence_scores):.4f}\n")
    log_file.write(f"Ground Truth Comparison:\n")
    log_file.write(f"  Overall accuracy vs GT: {gt_accuracy*100:.2f}%\n")
    log_file.write(f"  Unique GT labels: {sorted(unique_gt_labels.tolist())}\n")
    log_file.write(f"  Unique predicted labels: {sorted(unique_pred_labels.tolist())}\n")
    for label, acc in class_accuracies.items():
        gt_count = np.sum(y_labels_mapped == label)
        log_file.write(f"  Class '{label}' accuracy: {acc*100:.2f}% ({gt_count} samples)\n")
    log_file.write(f"GT label distribution: {dict(gt_counts)}\n")
    log_file.write(f"Label distribution before propagation: {dict(original_counts)}\n")
    log_file.write(f"Label distribution after propagation: {dict(propagated_counts)}\n")
print(f"Log saved to {log_path}")

# Plotting UMAP with Ground Truth, HDBSCAN and LP labels
plt.figure(figsize=(24, 8))

plt.subplot(1, 3, 1)
# Count samples per label
from collections import Counter
label_counts = Counter(y_labels_mapped)
unique_labels = np.unique(y_labels_mapped)
# Create consistent color mapping
palette = sns.color_palette('tab10', n_colors=len(unique_labels))
label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

# Create color array for each point
colors = [label_to_color[label] for label in y_labels_mapped]

# Plot with explicit colors
plt.scatter(merged_tsne_2d[:, 0], merged_tsne_2d[:, 1], c=colors, s=50, alpha=0.7)

# Create custom legend with matching colors
handles = [mlines.Line2D([], [], color=label_to_color[label], marker='o', linestyle='None', 
                         markersize=8, label=f"{label} (n={label_counts[label]})") for label in unique_labels]
plt.legend(handles=handles, title='Ground Truth', loc='best')
plt.title('Ground Truth Labels')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

plt.subplot(1, 3, 2)
# Count HDBSCAN labels
hdbscan_counts = Counter(merged_sample_labels)
unique_hdbscan = np.unique(merged_sample_labels)
palette_hdb = sns.color_palette('tab10', n_colors=len(unique_hdbscan))
hdb_to_color = {label: palette_hdb[i] for i, label in enumerate(unique_hdbscan)}

# Create color array for each point
colors_hdb = [hdb_to_color[label] for label in merged_sample_labels]

# Plot with explicit colors
plt.scatter(merged_tsne_2d[:, 0], merged_tsne_2d[:, 1], c=colors_hdb, s=50, alpha=0.7)

# Create custom legend with matching colors
handles = [mlines.Line2D([], [], color=hdb_to_color[label], marker='o', linestyle='None',
                         markersize=8, label=f"Cluster {label} (n={hdbscan_counts[label]})" if label != -1 else f"Noise (n={hdbscan_counts[label]})") 
           for label in unique_hdbscan]
plt.legend(handles=handles, title='HDBSCAN', loc='best')
plt.title('HDBSCAN Clustering')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

plt.subplot(1, 3, 3)
# Count LP predictions
lp_counts = Counter(y_pred)
unique_lp = np.unique(y_pred)
palette_lp = sns.color_palette('tab10', n_colors=len(unique_lp))
lp_to_color = {label: palette_lp[i] for i, label in enumerate(unique_lp)}

# Create color array for each point
colors_lp = [lp_to_color[label] for label in y_pred]

# Plot with explicit colors
plt.scatter(merged_tsne_2d[:, 0], merged_tsne_2d[:, 1], c=colors_lp, s=50, alpha=0.7)

# Create custom legend with matching colors
handles = [mlines.Line2D([], [], color=lp_to_color[label], marker='o', linestyle='None',
                         markersize=8, label=f"{label} (n={lp_counts[label]})") for label in unique_lp]
plt.legend(handles=handles, title='Label Propagation', loc='best')
plt.title('Label Propagation Results')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

plt.tight_layout()
plt.savefig(output_folder_path / f"{run_id}_hdb_lp_gt_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# save in log each wav name and its predicted label
with open(log_path, "a") as log_file:
    log_file.write("\nDetailed Predictions:\n")
    for stem, gt, hdb, hdbp, lp, lpc in zip(merged_stem_paths, merged_y_labels, merged_sample_labels, merged_sample_probs, y_pred, confidence_scores):
        log_file.write(f"{stem}: GT={gt}, HDBSCAN={hdb} (p={hdbp:.2f}), LP={lp} (conf={lpc:.2f})\n")