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

from sklearn.preprocessing import StandardScaler

import umap

warnings.filterwarnings('ignore', category=FutureWarning)

from clustering_utils import gen_tsne, check_0_clusters, plot_clustering_dual, organize_samples_by_label 


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

feats_pickle_ex = Path('/home/luis/Dropbox/DATASETS_AUDIO/Proposal_runs/TestAO-Liz/STG2_EXP001-SHAS-DV/TestAO-Liz_SHAS_DV_feats.pkl')
output_folder_path_ex = Path('/home/luis/Dropbox/DATASETS_AUDIO/Proposal_runs/littleTest')
run_params_ex = f"pca{pca_elem}_mcs{min_cluster_size}_ms{min_samples}_{hdb_mode}"
Exp_name_ex = 'TestAO-Liz_SHAS_DV'

pred_lbl_pickle_ex = Path('/home/luis/Dropbox/DATASETS_AUDIO/Proposal_runs/littleTest/TestAO-Liz_SHAS_DV_predlbl.pickle')
pred_reduced_feats_ex = Path('/home/luis/Dropbox/DATASETS_AUDIO/Proposal_runs/littleTest/TestAO-Liz_SHAS_DV_reduced_feats.pickle')


parser = argparse.ArgumentParser()
parser.add_argument('--input_feats_pickle', default=feats_pickle_ex, help='Path to the folder to store the D-vectors features')
parser.add_argument('--output_pred_folder', type=valid_path, default=output_folder_path_ex, help='Path to the folder to store the predictions')
parser.add_argument('--run_params', default=run_params_ex, help='string with the run params for HDBSCAN')
parser.add_argument('--exp_name', default=Exp_name_ex, help='string with the experiment name')

parser.add_argument('--pred_lbl_picke', default=pred_lbl_pickle_ex, help='Output path for clustering labels')
parser.add_argument('--pred_reduced_feats', default=pred_reduced_feats_ex, help='Output path for reduced features')
args = parser.parse_args()

output_folder_path = Path(args.output_pred_folder)
feats_pickle_path = Path(args.input_feats_pickle)

pred_lbl_pickle = Path(args.pred_lbl_picke)
pred_reduced_feats = Path(args.pred_reduced_feats)

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

hdb_data_input = None
# if pca_elem == None or pca_elem == 0:
#     hdb_data_input = Mixed_X_data
# else:
#     hdb_data_input = run_pca(Mixed_X_data, pca_elem) 

n_components = 15
data_standardized = StandardScaler().fit_transform(Mixed_X_data)

# Apply UMAP
umap_reducer = umap.UMAP(
    n_neighbors=5,  # Adjust based on dataset size
    min_dist=0.1,    # Controls compactness of clusters
    n_components=n_components,  # Reduced dimensionality
    metric='cosine',  # Good default for many feature types
)
hdb_data_input = umap_reducer.fit_transform(data_standardized)

### try cluster_selection_method = 'leaf' | default = 'eom'
hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,\
                        min_samples=min_samples,\
                    cluster_selection_method = hdb_mode).fit(hdb_data_input)

samples_outliers = hdb.outlier_scores_
samples_prob = hdb.probabilities_
samples_label = hdb.labels_

if check_0_clusters(samples_prob, samples_label, verbose = False):
    print(f'0 clusters: {current_run_id}')

df_mixed = gen_tsne(Mixed_X_data, Mixed_y_labels)
x_tsne_2d = np.array(list(zip(df_mixed['tsne-2d-one'], df_mixed['tsne-2d-two'])))

with open(str(pred_reduced_feats), 'wb') as handle:
    pickle.dump(hdb_data_input, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(pred_lbl_pickle, 'wb') as handle:
    pickle.dump(samples_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Store x_tsne_2d for later use
with open(f'{output_folder_path}/{current_run_id}_xtsne2d.pickle', 'wb') as handle:
    pickle.dump(x_tsne_2d, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Store Mixed_X_paths and Mixed_y_labels on a similar way
with open(f'{output_folder_path}/{current_run_id}_Xpaths.pickle', 'wb') as handle:
    pickle.dump(Mixed_X_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)

info_pred = [samples_label, x_tsne_2d, Mixed_X_paths]
# Store Mixed_X_paths and Mixed_y_labels on a similar way
with open(f'{output_folder_path}/{current_run_id}_predinfo.pickle', 'wb') as handle:
    pickle.dump(info_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)


plot_clustering_dual(x_tsne_2d, Mixed_y_labels,
                        samples_label, samples_prob,
                        current_run_id, output_folder_path,
                        plot_mode)


organize_samples_by_label(Mixed_X_paths, samples_label, samples_prob, output_folder_path)