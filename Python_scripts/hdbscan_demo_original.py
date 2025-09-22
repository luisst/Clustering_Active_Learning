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
import mplcursors

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

import matplotlib.lines as mlines
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.exceptions import UndefinedMetricWarning
from scipy.stats import entropy


def calculate_entropy(labels):
    value, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return entropy(probs, base=2)


def plot_prediction_clusters(X, labels, wav_names=None, probabilities=None, parameters=None, 
                           ax=None, txt_title=None, remove_outliers=True):
    """
    Plot clustering results with interactive hover tooltips showing wav file names.
    
    Args:
        X: 2D coordinates for plotting
        labels: cluster labels for each point
        wav_names: list of wav file names (stems) corresponding to each point
        probabilities: probability/confidence values for marker sizing (0.0 - 1.0)
        parameters: dict of parameters to show in title
        ax: matplotlib axis to plot on
        txt_title: title prefix
        remove_outliers: whether to hide outlier points (-1 labels)
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # Filter outliers if requested
    if remove_outliers:
        mask = labels != -1
        X_plot = X[mask]
        labels_plot = labels[mask]
        wav_names_plot = np.array(wav_names)[mask] if wav_names is not None else None
        probabilities_plot = probabilities[mask] if probabilities is not None else None
    else:
        X_plot = X
        labels_plot = labels
        wav_names_plot = wav_names
        probabilities_plot = probabilities
    
    # Get unique clusters
    unique_labels = np.unique(labels_plot)
    n_clusters = len(unique_labels)
    
    # Generate colors for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))
    
    # Calculate marker sizes based on probabilities
    if probabilities_plot is not None:
        # Scale probabilities to reasonable marker sizes (20-100)
        sizes = 20 + (probabilities_plot * 80)
    else:
        sizes = 50  # Default size
    
    # Plot each cluster
    scatter_plots = []
    for i, label in enumerate(unique_labels):
        mask = labels_plot == label
        cluster_points = X_plot[mask]
        cluster_sizes = sizes[mask] if hasattr(sizes, '__len__') else sizes
        
        if label == -1:  # Outliers (if not removed)
            color = 'black'
            alpha = 0.5
        else:
            color = colors[i % len(colors)]
            alpha = 0.7
        
        scatter = ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                           c=[color], s=cluster_sizes, alpha=alpha, 
                           label=f'Cluster {label}' if label != -1 else 'Outliers',
                           edgecolors='white', linewidth=0.5)
        scatter_plots.append((scatter, mask))
    
    # Set up the plot
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Create title
    title_parts = []
    if txt_title:
        title_parts.append(txt_title)
    if parameters:
        param_str = ', '.join([f'{k}={v}' for k, v in parameters.items()])
        title_parts.append(f'({param_str})')
    
    if title_parts:
        ax.set_title(' '.join(title_parts))
    
    # Set up hover functionality if wav_names are provided
    if wav_names_plot is not None:
        # Create annotation box for displaying wav names
        annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                           bbox=dict(boxstyle="round", fc="white", alpha=0.8, edgecolor="black"),
                           arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        
        def update_annot(ind, scatter_obj, cluster_mask):
            """Update annotation with wav name for hovered point."""
            # Get the point index within the full dataset
            cluster_indices = np.where(cluster_mask)[0]
            point_idx = cluster_indices[ind]
            
            # Get position of the point
            pos = scatter_obj.get_offsets()[ind]
            annot.xy = pos
            
            # Set the wav name as annotation text
            wav_name = wav_names_plot[point_idx]
            annot.set_text(f"{wav_name}")
            annot.get_bbox_patch().set_facecolor('white')
            annot.get_bbox_patch().set_alpha(0.9)
        
        def hover(event):
            """Handle mouse hover events."""
            if event.inaxes == ax:
                # Check each scatter plot to see if mouse is over a point
                for scatter_obj, cluster_mask in scatter_plots:
                    cont, ind = scatter_obj.contains(event)
                    if cont:
                        # Mouse is over a point
                        point_ind = ind["ind"][0]  # Get first point if multiple
                        update_annot(point_ind, scatter_obj, cluster_mask)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return
                
                # Mouse is not over any point
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
        
        # Connect the hover event
        fig.canvas.mpl_connect("motion_notify_event", hover)
    
    plt.tight_layout()
    return ax


def log_print(*args, **kwargs):
    """Prints to stdout and also logs to log_path."""

    log_path = kwargs.pop('lp', 'default_log.txt')
    print_to_console = kwargs.pop('print', True)

    message = " ".join(str(a) for a in args)
    if print_to_console:
        print(message)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def calculate_internal_metrics(features, labels):
    """
    Calculate internal clustering evaluation metrics.
    
    Args:
        features: numpy array of feature vectors
        labels: numpy array of cluster labels
    
    Returns:
        dict: Dictionary containing calculated metrics
    """
    results = {}
    
    # Check if we have enough data and clusters
    if len(features) < 2 or len(np.unique(labels)) < 2:
        print("Warning: Not enough data or clusters for internal metrics calculation")
        return {
            'silhouette_score': None,
            'davies_bouldin_score': None,
            'calinski_harabasz_score': None
        }
    
    # Calculate Silhouette Score
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            results['silhouette_score'] = silhouette_score(features, labels)
    except Exception as e:
        print(f"Error calculating Silhouette Score: {e}")
        results['silhouette_score'] = None
    
    # Calculate Davies-Bouldin Index
    try:
        results['davies_bouldin_score'] = davies_bouldin_score(features, labels)
    except Exception as e:
        print(f"Error calculating Davies-Bouldin Index: {e}")
        results['davies_bouldin_score'] = None
    
    # Calculate Calinski-Harabasz Index
    try:
        results['calinski_harabasz_score'] = calinski_harabasz_score(features, labels)
    except Exception as e:
        print(f"Error calculating Calinski-Harabasz Index: {e}")
        results['calinski_harabasz_score'] = None
    
    return results

def gen_c3_tsne(Mixed_X_data,
             perplexity_val = 15, n_iter = 900,
             n_comp = 108):
    
    if (n_comp == 0) or(n_comp > Mixed_X_data.shape[1]):
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

    # df_mixed = pd.DataFrame()
    # df_mixed['tsne-2d-one'] = tsne_results[:,0]
    # df_mixed['tsne-2d-two'] = tsne_results[:,1]

    # x_tsne_2d = np.array(list(zip(df_mixed['tsne-2d-one'], df_mixed['tsne-2d-two'])))

    return tsne_results

def membership_curve(w, slope_plateau=0.01, sharpness=1.5, H=1.0):
    """
    Generate a curve in [0,100]:
      - Near 0 until ~40
      - Rise until ~60
      - Consistent linear slope 60-85
      - Gradual drop to almost 0 by 95%
    
    Parameters
    ----------
    w : float or array-like
        Input values in [0,100].
    slope_plateau : float
        Linear slope for the plateau region (rise per unit w).
    sharpness : float
        Controls how sharp the rise/drop transitions are (lower = sharper).
    H : float
        Peak height scaling.
    """
    w = np.asarray(w, dtype=float)
    
    # Smooth step-up from 40 to 60
    rise = 1 / (1 + np.exp(-(w-60)/sharpness))   # sigmoid centered at 50
    
    # More gradual step-down starting from 95, nearly zero by 100%
    fall = 1 / (1 + np.exp((w-95)/2.0))    # gentler sigmoid centered at 95 
    
    # Base curve from rise and fall
    base = rise * fall

    # Create consistent linear slope in plateau region (60-100)
    plateau_mask = (w >= 60) & (w <= 100)
    slope_adjustment = np.ones_like(w)
    
    # Add linear slope only in plateau region
    slope_adjustment[plateau_mask] = 1 + slope_plateau * (w[plateau_mask] - 60)
    
    y = H * base * slope_adjustment
    
    # substract a constant of 0.25
    y -= 0.25
    
    return y

def n_clusters_curve(k, a=0.3, m=6, b=40, H=1.0):
    """
    Generate a non-negative right-skewed hump-shaped curve.
    
    Parameters
    ----------
    k : int or array-like
        Input value(s). k should be >= 0.
    a : float
        Controls steepness of the rise before the peak.
    m : float
        Location of the peak (mode).
    b : float
        Controls the rate of decay after the peak.
    H : float
        Peak height (scaling factor).

    Returns
    -------
    y : float or ndarray
        Value(s) of the curve at k.
    """
    k = np.asarray(k, dtype=float)
    y = np.zeros_like(k, dtype=float)
    
    mask = k > 0
    
    # Modified formula to ensure peak is always at k=m
    # Use different behavior before and after the peak
    before_peak = (k <= m) & mask
    after_peak = (k > m) & mask
    
    # Before peak: power function
    if np.any(before_peak):
        y[before_peak] = H * (k[before_peak] / m) ** a
    
    # After peak: exponential decay
    if np.any(after_peak):
        y[after_peak] = H * np.exp(-(k[after_peak] - m) / b)
    
    return y

def load_data(pickle_file):
    with open(pickle_file, "rb") as file:
        X_data_and_labels = pickle.load(file)
    Mixed_X_data, X_paths, y_labels = X_data_and_labels

    return Mixed_X_data, X_paths, y_labels

def plot_clustering_hdb(X, labels, probabilities=None, parameters=None, 
                    ax=None, txt_title=None,
                    remove_outliers = True):

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
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

    non_outliers_percentage = (len(labels[labels != -1]) / len(labels)) * 100
    title = f"{txt_title} #n: {n_clusters_} | {non_outliers_percentage:.2f}%"

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




if __name__ == "__main__":

    pickle_file = Path(r"C:\Users\luis2\Dropbox\DATASETS_AUDIO\Unsupervised_Pipeline\MiniClusters\STG_2\STG2_EXP010-SHAS-DV\MiniClusters_SHAS_DV_feats.pickle")
    run_id = "distx"

    output_folder_path = Path.cwd() / "hdb_distx"  # Update with the actual path
    log_path = output_folder_path / f"{run_id}_log.txt"
    output_folder_path.mkdir(parents=True, exist_ok=True)

    umap_pickle_path = output_folder_path / f"{run_id}_umap_data.pickle"

    X_data, X_paths, y_labels  = load_data(pickle_file)

    # Extract wav file stems from paths
    wav_stems = [Path(path).stem for path in X_paths]

    # If umap pickle path exists, then do not compute umap again
    if umap_pickle_path.exists():
        with open(umap_pickle_path, "rb") as f:
            umap_data = pickle.load(f)
    else:
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

    # --- Extract point-level info ---
    core_dists = hdb._prediction_data.core_distances

    max_lambdas_dict = hdb._prediction_data.max_lambdas
    max_lambdas = [max_lambdas_dict.get(i, np.nan) for i in range(len(X_data))]


    # Approximate lambda values (higher = denser region)
    lambda_val = 1.0 / core_dists
    lambda_val[np.isinf(lambda_val)] = np.nan  # handle 0 distances safely

    # Map cluster persistence (per cluster → to each sample)
    cluster_persistence = {i: p for i, p in enumerate(hdb.cluster_persistence_)}
    sample_persistence = [cluster_persistence[label] if label != -1 else 0 
                        for label in hdb.labels_]

    # --- Build initial dataframe ---
    df = pd.DataFrame({
        "sample_ID": np.arange(len(X_data)),
        "gt_label": y_labels,
        "pred_label": hdb.labels_,
        "probability": hdb.probabilities_,
        "outlier_score": hdb.outlier_scores_,
        "core_distance": core_dists,
        "lambda_value": lambda_val,
        "lambda_local_max": max_lambdas,
        "cluster_persistence": sample_persistence
    })

    # --- Compute per-cluster statistics ---
    cluster_stats = (
        df[df["label"] != -1]  # exclude noise points (-1)
        .groupby("label")
        .agg(
            cluster_size=("label", "size"),
            cluster_mean_prob=("probability", "mean"),
            cluster_mean_outlier=("outlier_score", "mean"),
            cluster_mean_core_dist=("core_distance", "mean")
        )
        .reset_index()
    )

    # Add cluster persistence from HDBSCAN model
    cluster_stats["cluster_persistence"] = cluster_stats["label"].map(cluster_persistence)

    # --- Compute cluster centroids ---
    centroids = {}
    for label in cluster_stats["label"]:
        points = X_data[hdb.labels_ == label]
        centroids[label] = points.mean(axis=0)
    

    # Compute distance to centroid for each sample
    dist_to_centroid = []
    for i, label in enumerate(hdb.labels_):
        if label == -1:
            dist_to_centroid.append(np.nan)  # noise has no centroid
        else:
            dist = np.linalg.norm(X_data[i] - centroids[label])
            dist_to_centroid.append(dist)

    df["dist_to_centroid"] = dist_to_centroid

    # --- Compute k-nearest neighbor distance (k=min_samples=5) ---
    k = 5  # min_samples value
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='manhattan').fit(umap_data)  # k+1 because first neighbor is the point itself
    distances, indices = nbrs.kneighbors(umap_data)
    
    # Take the distance to the k-th nearest neighbor (index k, since index 0 is the point itself)
    knn_distances = distances[:, k]
    df["knn_distance_k5"] = knn_distances.round(4)

    # Use indices to extract the 5 nearest neighbors' labels
    knn_labels_k5 = [hdb.labels_[indices[i, 1:]] for i in range(len(indices))]
    df["knn_labels_k5"] = knn_labels_k5

    # Calculate the entropy for those 5 labels
    knn_entropy_k5 = [calculate_entropy(labels) for labels in knn_labels_k5]
    df["knn_entropy_k5"] = knn_entropy_k5

    # --- Merge back cluster-level stats ---
    df = df.merge(cluster_stats, on="label", how="left")

    # --- Soft clustering (secondary memberships) ---
    membership_vectors = hdbscan.all_points_membership_vectors(hdb)

    top_clusters = []
    top_probs = []

    for row in membership_vectors:
        # get top 5 (cluster, probability) pairs
        top5 = sorted(list(enumerate(row)), key=lambda x: x[1], reverse=True)[:5]
        top_clusters.append([c for c, p in top5])
        top_probs.append([p for c, p in top5])

    # Expand into columns
    for k in range(5):
        df[f"soft_cluster_{k+1}"] = [clusters[k] if len(clusters) > k else np.nan for clusters in top_clusters]
        df[f"soft_prob_{k+1}"] = [probs[k] if len(probs) > k else np.nan for probs in top_probs]


    # Round numeric columns to 4 decimal places
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(4)

    # --- Sort for readability ---
    df = df.sort_values(by=["label", "sample_ID"]).reset_index(drop=True)

    # --- Export to CSV ---
    output_file = output_folder_path.joinpath("hdbscan_results.csv")
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Take core_distances and normalize them into 0.0 to 1.0 and store in new list
    core_distances = df["core_distance"].tolist()
    min_core = min(core_distances)
    max_core = max(core_distances)
    normalized_core = [(x - min_core) / (max_core - min_core) if max_core > min_core else 0 for x in core_distances]

    # Generate t-SNE representation
    tsne_umap = gen_c3_tsne(umap_data)
    tsne_PCAdvectors = gen_c3_tsne(X_data)

    # Calculate hdb score
    current_hdb_score = n_clusters_curve(n_clusters)/2 + membership_curve(percentage_assigned)/2

    combined_fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_clustering_hdb(tsne_PCAdvectors, labels=hdb_labels,
                    probabilities = normalized_core,
                    remove_outliers = False,
                    txt_title='DdvecPCA', ax=axes[0])

    plot_clustering_hdb(tsne_umap, labels=hdb_labels,
                    probabilities = normalized_core,
                    remove_outliers = False,
                    txt_title='Dumap', ax=axes[1])

    current_fig_path = output_folder_path.joinpath(f'minitest_hdb.png')

    # combined_fig.suptitle(f'minitest-{current_hdb_score:.3f}', fontsize=14)
    plt.tight_layout()
    combined_fig.savefig(current_fig_path, dpi=300)

    combined_fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_clustering_hdb(tsne_PCAdvectors, labels=y_labels,
                    probabilities = None,
                    remove_outliers = True,
                    txt_title='GTdvectorsPCA', ax=axes[0])

    plot_clustering_hdb(tsne_umap, labels=y_labels,
                    probabilities = None,
                    remove_outliers = True,
                    txt_title='GTumap', ax=axes[1])

    current_fig_path = output_folder_path.joinpath(f'minitest_hdb_gt.png')

    # combined_fig.suptitle(f'minitest-{current_hdb_score:.3f}', fontsize=14)
    plt.tight_layout()
    combined_fig.savefig(current_fig_path, dpi=300)


    # Plot minimum spanning tree and store fig
    fig, ax = plt.subplots(figsize=(16, 10))
    hdb.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                    edge_alpha=0.4,
                                    node_size=30,
                                    edge_linewidth=2,
                                    axis=ax)
    min_spanning_tree_path = output_folder_path.joinpath(f'minitest_hdb_min_spanning_tree.png')
    fig.savefig(min_spanning_tree_path, dpi=300)

    # Plot single linkage tree and store fig
    fig, ax = plt.subplots(figsize=(16, 10))
    hdb.single_linkage_tree_.plot(cmap='viridis', colorbar=True, axis=ax)
    single_linkage_tree_path = output_folder_path.joinpath(f'minitest_hdb_single_linkage_tree.png')
    fig.savefig(single_linkage_tree_path, dpi=300)

    # Plot condensed tree and store fig
    fig, ax = plt.subplots(figsize=(16, 16))
    hdb.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette(), axis=ax)
    condensed_tree_path = output_folder_path.joinpath(f'minitest_hdb_condensed_tree.png')
    fig.savefig(condensed_tree_path, dpi=300)

    log_path = output_folder_path.joinpath(f'minitest_hdb_log.txt')

    # Calculate internal cluster metrics
    internal_metrics = calculate_internal_metrics(X_data, hdb_labels)
    log_print("\nInternal Cluster Metrics:", lp=log_path)
    for metric, value in internal_metrics.items():
        log_print(f"  {metric}: {value}", lp=log_path)


    # Plot knn_entropy_k5 as y-axis and knn_distance_k5 as x_axis
    plt.figure(figsize=(12, 8))
    # Use the predicted cluster for coloring
    plt.scatter(df["knn_distance_k5"], df["knn_entropy_k5"], c=hdb.labels_[df.index], cmap='viridis', alpha=0.5)
    # Add colorbar
    plt.colorbar(label='Cluster')
    plt.title("KNN Entropy vs Distance (k=5)")
    plt.xlabel("KNN Distance (k=5)")
    plt.ylabel("KNN Entropy (k=5)")
    plt.grid()
    plt.show()

    # Use the new plotting function with hover functionality
    combined_fig, axes = plt.subplots(figsize=(15, 8))
    
    plot_prediction_clusters(tsne_PCAdvectors, labels=hdb_labels,
                            wav_names=wav_stems,
                            probabilities=hdb_probs,
                            remove_outliers=False,
                            txt_title='DdvecPCA', ax=axes)

    current_fig_path = output_folder_path.joinpath(f'minitest_hdb_interactive.png')
    plt.tight_layout()
    combined_fig.savefig(current_fig_path, dpi=300)
    plt.show()  # Show interactive plot