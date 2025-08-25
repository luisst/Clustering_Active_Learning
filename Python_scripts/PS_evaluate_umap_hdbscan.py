from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import hdbscan
import umap
from itertools import product
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def log_print(*args, **kwargs):
    """Prints to stdout and also logs to log_path."""

    log_path = kwargs.pop('lp', 'default_log.txt')
    print_to_console = kwargs.pop('print', True)

    message = " ".join(str(a) for a in args)
    if print_to_console:
        print(message)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def load_data(pickle_file):
    with open(pickle_file, "rb") as file:
        X_data_and_labels = pickle.load(file)
    Mixed_X_data, _, _ = X_data_and_labels

    return Mixed_X_data

def gen_c3_tsne(Mixed_X_data,
             perplexity_val = 15, n_iter = 900,
             n_comp = 108):
    
    if n_comp == 0:
        tsne = TSNE(n_components=2, verbose=False, perplexity=perplexity_val, n_iter=n_iter)
        tsne_results = tsne.fit_transform(Mixed_X_data)
        print(f'PCA before t-snePRE skipped')
    else:
        data_standardized = StandardScaler().fit_transform(Mixed_X_data)
        # Numbers to try: 16, 75, 108
        pca_selected = PCA(n_components=108)
        x_low_dim = pca_selected.fit_transform(data_standardized)

        tsne = TSNE(n_components=2, verbose=False, perplexity=perplexity_val, n_iter=n_iter)
        tsne_results = tsne.fit_transform(x_low_dim)

    df_mixed = pd.DataFrame()
    df_mixed['tsne-2d-one'] = tsne_results[:,0]
    df_mixed['tsne-2d-two'] = tsne_results[:,1]

    x_tsne_2d = np.array(list(zip(df_mixed['tsne-2d-one'], df_mixed['tsne-2d-two'])))

    return x_tsne_2d

def plot_clustering(X, labels, probabilities=None, parameters=None, 
                    ground_truth=False, ax=None,
                    remove_outliers = False,
                    add_gt_prd_flag = True):

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
            if remove_outliers:
                continue

        class_index = np.where(labels == k)[0]
        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                "x" if k == -1 else "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=4 if k == -1 else 1 + 5 * proba_map[ci],
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if ground_truth:
        if n_clusters_ == 1:
            title = f"Unlabeled total samples: {len(labels)}"
        else:
            title = f"GT #n: {n_clusters_} T: {len(labels)}"
    else:
        non_outliers_percentage = (len(labels[labels != -1]) / len(labels)) * 100
        title = f"Prd #n: {n_clusters_} | Mem %: {non_outliers_percentage:.2f}%"

    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    title = title 
    ax.set_title(title)

    # # Add legend with the number of labels for each cluster
    # legend_labels = [f"C{k}: {list(labels).count(k)}" for k in unique_labels]
    legend_labels = [f": {list(labels).count(k)}" for k in unique_labels]

    # Customizing the legend to not display the marker
    legend_without_symbol = []
    for idx, label_id in enumerate(unique_labels):
        if label_id != -1:
            current_lbl_color = colors[idx]
            current_label = legend_labels[idx]
            legend_without_symbol.append(mlines.Line2D([], [], color=current_lbl_color,
                                                        marker='o', 
                                                        # linestyle='solid', 
                                                        label=current_label))
    plt.legend(handles=legend_without_symbol)

    # ax.legend(legend_labels, labelcolor=colors)
    plt.tight_layout()

def evaluate_hdb(umap_data, m_size=25, m_samples=5, hdb_mode='eom'):
    hdb = hdbscan.HDBSCAN(min_cluster_size=m_size, min_samples=m_samples,
                    cluster_selection_method=hdb_mode).fit(umap_data)
    hdb_labels = hdb.labels_
    n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)  # Exclude noise label (-1)
    n_samples = len(hdb_labels)
    n_assigned = len(hdb_labels[hdb_labels != -1])
    percentage_assigned = (n_assigned / n_samples) * 100
    silhouette_avg = silhouette_score(umap_data, hdb_labels) if n_clusters > 1 else -1

    hdb_info = {'labels': hdb_labels, 'probabilities': hdb.probabilities_}

    return n_clusters, percentage_assigned, silhouette_avg, hdb_info


if __name__ == "__main__":

    # try with the easy, med pretrained pickles
    #TODO: also include PCA to give a baseline

    pickle_file = Path(r"/home/luis/Dropbox/DATASETS_AUDIO/Unsupervised_Pipeline/TestAO-Irma/STG_2/STG2_EXP010-SHAS-DVn1/TestAO-Irma_SHAS_DVn1_featsEN.pickle")  # Update with the actual path
    run_id = "DVn1"

    output_folder_path = Path.cwd() / "umap_hdb_loop"  # Update with the actual path
    log_path = output_folder_path / f"{run_id}_log.txt"
    output_folder_path.mkdir(parents=True, exist_ok=True)
    X_data = load_data(pickle_file)

    n_repetitions = 4

    # U-map num_components
    a_options = [15, 20, 25]

    # U-map num_neighbors
    b_options = [5, 10]

    # U-map min_dist
    c_options = [0.1]

    # U-map metric
    d_options = ['manhattan', 'cosine']

    all_idx = 0


    for rep_indx in range(0, n_repetitions): 
        for a, b, c, d in product(a_options, b_options, c_options, d_options):
            # You can perform some action or function with a, b, and c here
            # print(f"\n\n N_comp a: {a}, N_neighb b: {b}, min_dist c: {c}, Metric d: {d}")

            umap_N_comp = a
            umap_N_neighs = b
            umap_min_dist = c
            umap_metric = d

            run_params = f"Ncp{umap_N_comp}_Nne{umap_N_neighs}_md{umap_min_dist}_{umap_metric}" 

            data_standardized = StandardScaler().fit_transform(X_data)

            # Apply UMAP
            umap_reducer = umap.UMAP(
                n_neighbors=umap_N_neighs,  # Adjust based on dataset size
                min_dist=umap_min_dist,    # Controls compactness of clusters
                n_components=umap_N_comp,  # Reduced dimensionality
                metric=umap_metric  # Good default for many feature types
                # random_state=42  # For reproducibility
            )
            umap_data = umap_reducer.fit_transform(data_standardized)

            # Use PCA to plot 2D from UMAP features
            pca = PCA(n_components=2)
            hdb_data_input_2d = pca.fit_transform(umap_data)

            n_clusters, percentage_assigned, silhouette_avg, hdb_info = evaluate_hdb(umap_data)

            if n_clusters > 3:

                # Plot and store the hdb_data_input_2d
                plt.figure(figsize=(10, 8))
                plt.scatter(hdb_data_input_2d[:, 0], hdb_data_input_2d[:, 1], s=5)
                plt.title("HDBSCAN Clustering (2D Projection)")
                plt.xlabel("PCA Component 1")
                plt.ylabel("PCA Component 2")
                plt.grid()
                plt.savefig(f"{output_folder_path}/umap_{run_params}_{all_idx}_2d.png")
                plt.close()

                log_print(f"\n\n{all_idx}|{rep_indx} n_neighbors: {umap_N_neighs} \t min_dist: {umap_min_dist} \t n_comp: {umap_N_comp} \t m:{umap_metric}, "
                    f"\nn_clusters: {n_clusters} \t"
                    f"percentage_assigned: {percentage_assigned:.2f}%, "
                    f"\tsilhouette_score: {silhouette_avg:.4f}", lp=log_path)
                    
                x_tsne_2d = gen_c3_tsne(X_data)

                combined_fig, axes = plt.subplots(1, 1, figsize=(12, 6))
                plot_clustering(x_tsne_2d, labels=hdb_info['labels'],
                                probabilities = hdb_info['probabilities'],
                                remove_outliers = True, ax=axes)

                current_fig_path = output_folder_path.joinpath(f'{run_params}_{all_idx}.png')

                combined_fig.suptitle(f'{run_params}', fontsize=14)
                plt.tight_layout()
                combined_fig.savefig(current_fig_path, dpi=300)
            else:
                log_print(f"\n\n{all_idx}|{rep_indx} n_neighbors: {umap_N_neighs} \t min_dist: {umap_min_dist} \t n_comp: {umap_N_comp} \t m:{umap_metric},  "
                    f"\n>>>FAILED (n<3), {n_clusters}", lp=log_path)

            all_idx += 1