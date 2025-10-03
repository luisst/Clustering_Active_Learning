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

base_path_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO','Unsupervised_Pipeline','MiniClusters')
feats_pickle_ex = base_path_ex.joinpath('STG_2','STG2_EXP010-SHAS-DV','MiniClusters_SHAS_DV_featsEN.pickle')

output_folder_path_ex = Path('/home/luis/Dropbox/DATASETS_AUDIO/Unsupervised_Pipeline/dv2')
run_params_ex = f"pca{pca_elem}_mcs{min_cluster_size}_ms{min_samples}_{hdb_mode}"
Exp_name_ex = 'TestAO-IrmaR'
clusters_data_pickle_ex = output_folder_path_ex / 'clusters_data.pickle'
reduced_features_pickle_ex = output_folder_path_ex / 'reduced_features.pickle'
pred_label_pickle_ex = output_folder_path_ex / 'pred_label.pickle'


parser = argparse.ArgumentParser()
parser.add_argument('--input_feats_pickle', default=feats_pickle_ex, help='Path to the folder to store the D-vectors features')
parser.add_argument('--output_pred_folder', type=valid_path, default=output_folder_path_ex, help='Path to the folder to store the predictions')
parser.add_argument('--run_params', default=run_params_ex, help='string with the run params for HDBSCAN')
parser.add_argument('--exp_name', default=Exp_name_ex, help='string with the experiment name')

parser.add_argument('--data_clusters_pickle', default=clusters_data_pickle_ex, help='Output path for clustering labels')
parser.add_argument('--stg3_reduced_features', default=reduced_features_pickle_ex, help='Output path for clustering labels')
parser.add_argument('--stg3_pred_lbl', default=pred_label_pickle_ex, help='Output path for clustering labels')


args = parser.parse_args()

output_folder_path = Path(args.output_pred_folder)
feats_pickle_path = Path(args.input_feats_pickle)

data_clusters_pickle = Path(args.data_clusters_pickle)
reduced_features_pickle = Path(args.stg3_reduced_features)
pred_label_pickle = Path(args.stg3_pred_lbl)

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

# Join the data for next stage
clusters_data = [Mixed_X_paths, hdb_data_input, x_tsne_2d, Mixed_y_labels, samples_label, samples_prob, samples_outliers]

with open(data_clusters_pickle, 'wb') as handle:
    pickle.dump(clusters_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(reduced_features_pickle, 'wb') as handle:
    pickle.dump(hdb_data_input, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(pred_label_pickle, 'wb') as handle:
    pickle.dump(samples_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

plot_clustering_dual(x_tsne_2d, Mixed_y_labels,
                        samples_label, samples_prob,
                        current_run_id, output_folder_path,
                        plot_mode)

# Use PCA to plot 2D from UMAP features
pca = PCA(n_components=2)
hdb_data_input_2d = pca.fit_transform(hdb_data_input)

# Plot and store the hdb_data_input_2d
plt.figure(figsize=(10, 8))
plt.scatter(hdb_data_input_2d[:, 0], hdb_data_input_2d[:, 1], s=5)
plt.title("HDBSCAN Clustering (2D Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()
plt.savefig(f"{output_folder_path}/{current_run_id}_umap_2d.png")
plt.close()

organize_samples_by_label(Mixed_X_paths, samples_label, samples_prob, output_folder_path)