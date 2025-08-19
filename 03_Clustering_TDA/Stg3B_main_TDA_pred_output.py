from __future__ import print_function
import os
import warnings
import re
from pathlib import Path
import pprint

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

import matplotlib.pyplot as plt
import warnings
import pickle
from sklearn.preprocessing import StandardScaler
import argparse
import sys

import hdbscan
import umap
import kmapper as km

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

from tda_utils import get_groups_alt, find_connected_nodes, copy_arrays_to_folder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

pca_elem = 16
min_cluster_size=5
hdb_mode = 'leaf'



perplexity_val = 15 
n_iter = 900

compute_flag = True

def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

root_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 'Proposal_runs','TestAO-Irmadb')

feats_pickle_ex = root_ex / Path('STG_2/STG2_EXP010-SHAS-DV/TestAO-Irmadb_SHAS_DV_feats.pkl')
output_folder_path_ex = root_ex / Path('STG_3/Km_irma_test')

# run_params_ex = 'pca16_mcs5_ms5_leaf'
run_params_ex = 'pca16_mcs12_ms5_eom'
exp_name_ex = 'TestAO-IrmaKM_SHAS_DV_feats'
nodes_th_ex = 1

n_cubes_ex = 10
p_overlap_ex = 0.5


parser = argparse.ArgumentParser()

parser.add_argument('--input_feats_pickle', default=feats_pickle_ex, help='Path to the folder to store the D-vectors features')
parser.add_argument('--output_pred_folder', type=valid_path, default=output_folder_path_ex, help='Path to the folder to store the predictions')
parser.add_argument('--run_params', default=run_params_ex, help='string with the HDB-SCAN run name')
parser.add_argument('--exp_name', default=exp_name_ex, help='string with the experiment name')
parser.add_argument('--nodes_th', type=int, default=nodes_th_ex, help='Threshold for the number of nodes in a connected component')
parser.add_argument('--km_cubes', type=int, default=n_cubes_ex, help='Number of cubes in the cover')
parser.add_argument('--km_overlap', type=float, default=p_overlap_ex, help='Overlap percentage in the cover')
args = parser.parse_args()

feats_pickle_path = Path(args.input_feats_pickle)
output_folder_path = Path(args.output_pred_folder)

n_cubes, perc_overlap = args.km_cubes, args.km_overlap
run_params = args.run_params
Exp_name = args.exp_name
nodes_th = int(args.nodes_th)


with open(f'{feats_pickle_path}.pickle', "rb") as file:
    X_data_and_labels = pickle.load(file)
X_train, X_train_paths, y_train = X_data_and_labels

print(f'run_params: {run_params}')

# #RUN_PARAMS="pca${pca_elem}_mcs${min_cluster_size}_ms${min_samples}_${hdb_mode}"
# #example "pca0_mcs10_ms5_eom"

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


run_id = f'{Exp_name}_TDA'
verbose =  False

data_standardized = StandardScaler().fit_transform(X_train)

# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper = km.KeplerMapper(verbose=2)

##Fit and transform data
# projected_data = mapper.fit_transform(X_train,
#                                       projection=[sklearn.decomposition.PCA(n_components=pca_elem),
#                                                 sklearn.manifold.TSNE(n_components=2,
#                                                                       verbose=False,
#                                                                       perplexity=perplexity_val,
#                                                                       n_iter=n_iter)])

umap_N_neighs = 15
umap_min_dist = 0.1
umap_N_comp = 1
umap_metric = 'cosine'

projected_data = mapper.fit_transform(X_train,
                                      projection=umap.UMAP(n_neighbors=umap_N_neighs,  
                                        min_dist=umap_min_dist,
                                        n_components=umap_N_comp, 
                                        metric=umap_metric),
                                      scaler=StandardScaler())

hdb_data_input = None
n_components = 15
data_standardized = StandardScaler().fit_transform(X_train)

# Apply UMAP
umap_reducer = umap.UMAP(
    n_neighbors=5,  # Adjust based on dataset size
    min_dist=0.1,    # Controls compactness of clusters
    n_components=n_components,  # Reduced dimensionality
    metric='cosine',  # Good default for many feature types
)
hdb_data_input = umap_reducer.fit_transform(data_standardized)

# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(
                    projected_data,
                    X=hdb_data_input,
                    clusterer=hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,\
                                        min_samples=min_samples,\
                                        cluster_selection_method = hdb_mode),
                    cover=km.Cover(n_cubes, perc_overlap),
                    )
 
current_nodes = graph['nodes']
if not current_nodes:
    sys.exit("Empty 'current_nodes' dictionary. Exiting program. \n Parameters produced no nodes")

mapper.visualize(
    graph,
    title=run_id,
    ## TO-DO: Change the path to the output folder
    path_html=f"{output_folder_path}/{run_id}_Hdbscan.html",
    color_values=y_train,
    color_function_name="labels",
)

my_nodes_dict = graph['links']
my_nodes_list = list(graph['nodes'].keys())
print(f'Node list len: {len(my_nodes_list)}')
print(f'Node dict len: {len(my_nodes_dict)}')
print(f'nodes_th: {nodes_th}')

my_representative_nodes = get_groups_alt(my_nodes_dict)
lbl_idx = 0
single_nodes_list = []
multiple_nodes_list = []
nodes_stats_dict = {}

if len(my_representative_nodes) == 0:
    print(f' >>>>>>>>>>>>>>>>>>>>>>> Empty my_representative_nodes dictionary. \n Parameters produced no connected components')
    sys.exit("Empty 'my_representative_nodes' dictionary. Exiting program. \n Parameters produced no nodes")
else:
    print(f'len(my_representative_nodes): {len(my_representative_nodes)}')
    for current_unique_name, current_group_len in my_representative_nodes:

        connected_nodes = find_connected_nodes(current_unique_name, my_nodes_dict)
        # print(f'\n\nConnected nodes {connected_nodes} \t current_group_len: {current_group_len} \n current_unique_name: {current_unique_name}')

        multiple_nodes_list.extend(connected_nodes)

        # Verify the single nodes to the list
        if current_group_len == 1:
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> found single node {current_unique_name}')
            continue

        # Skip the nodes that are connected to less than 4 nodes
        if current_group_len < nodes_th:
            print(f'\n\nSkipping node {current_unique_name} - lbl: {lbl_idx}')
            continue

        # print(f'\n\nProcessing node {current_unique_name} - lbl: {lbl_idx}')
        # print(f'len: {current_group_len} - connected nodes: {len(connected_nodes)}')

        my_unique_nodes = []
        for idx in connected_nodes:
            my_unique_nodes.extend(graph['nodes'][idx])

        # Remove duplicates
        my_unique_nodes = list(set(my_unique_nodes))

        # print(f'Unique list {my_unique_nodes}')

        folder_path = output_folder_path.joinpath(str(lbl_idx))
        lbl_idx += 1

        # Store the number of nodes in the connected component and the total number of elements in each node
        nodes_stats_dict[lbl_idx] = (current_group_len, len(my_unique_nodes))

        #Store the wavs from a given indexs
        copy_arrays_to_folder(X_train_paths, my_unique_nodes, folder_path)

print(f'\n\n\tNumber of Connected Components: {lbl_idx}')
pprint.pprint(nodes_stats_dict)


# Print if there is repeated nodes in multiple_nodes_list
if len(multiple_nodes_list) != len(set(multiple_nodes_list)):
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Repeated nodes in multiple_nodes_list')

# Generate the single nodes list from the difference between my_nodes_list and multiple_nodes_list
single_nodes_list = [node for node in my_nodes_list if node not in multiple_nodes_list]


single_node_stats_dict = {}

# Process the single nodes only if threshold is 1
if nodes_th == 1: 
    print(f'\n\n-------------------------------------------------')
    print(f'\tProcessing single nodes {len(single_nodes_list)}\n')
    for single_node in single_nodes_list:
        # print(f'\nProcessing single node {single_node} - lbl: {lbl_idx}')

        folder_path = output_folder_path.joinpath(str(lbl_idx))
        lbl_idx += 1

        samples_list = graph['nodes'][single_node]

        #Store the number of nodes in the connected component and the total number of elements in each node
        single_node_stats_dict[lbl_idx] = (1, len(samples_list))

        #Store the wavs from a given indexs
        copy_arrays_to_folder(X_train_paths, samples_list, folder_path)


elif nodes_th == 0:
    sys.exit("Invalid nodes_th value. Exiting program. \n Parameters produced no nodes")

pprint.pprint(single_node_stats_dict)

# Define the path to save the chart
current_fig_path = output_folder_path.joinpath(f'{run_id}_chart.png')

# Get the list of subfolders in the output folder
subfolders = [folder for folder in output_folder_path.iterdir() if folder.is_dir()]
if verbose:
    print(f'Connected Components {len(subfolders)}: {[subfolder.name for subfolder in subfolders]}')
# Get the number of wav files in each subfolder
num_wav_files = [len(list(subfolder.glob("*.wav"))) for subfolder in subfolders]


# Add a percentage of stored wavs to the total
my_percentage = sum(num_wav_files) / len(X_train_paths) * 100
my_title = f'Memberships wavs from total wavs: {my_percentage:.2f}%\n{run_id}' 

# Create a figure and axis
my_fig, ax = plt.subplots(figsize=(12, 6))

# Create the pie chart
ax.pie(num_wav_files, labels=[subfolder.name for subfolder in subfolders], autopct='%1.1f%%')
ax.set_title(my_title)
my_fig.savefig(current_fig_path, dpi=300)