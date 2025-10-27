import numpy as np
import shutil
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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

def check_number_clusters(samples_prob, hdb_labels, verbose = True):
    # print number of clusters:

    n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)  # Exclude noise label (-1)
    n_samples = len(hdb_labels)
    n_assigned = len(hdb_labels[hdb_labels != -1])
    membership_percentage = (n_assigned / n_samples) * 100

    if verbose:
        print(f'Number of clusters: {n_clusters}')

    return n_clusters, membership_percentage

def gen_tsne(Mixed_X_data, Mixed_y_labels,
             perplexity_val = 15, max_iter = 900,
             n_comp = 108):
    
    if (n_comp == 0) or(n_comp > Mixed_X_data.shape[1]):
        tsne = TSNE(n_components=2, verbose=False, perplexity=perplexity_val, max_iter=max_iter)
        tsne_results = tsne.fit_transform(Mixed_X_data)
        print(f'PCA before t-snePRE skipped')
    
    else:
        data_standardized = StandardScaler().fit_transform(Mixed_X_data)
        # Numbers to try: 16, 75, 108
        pca_selected = PCA(n_components=108)
        x_low_dim = pca_selected.fit_transform(data_standardized)

        tsne = TSNE(n_components=2, verbose=False, perplexity=perplexity_val, max_iter=max_iter)
        tsne_results = tsne.fit_transform(x_low_dim)

    df_mixed = pd.DataFrame()
    df_mixed['y'] = Mixed_y_labels
    df_mixed['tsne-2d-one'] = tsne_results[:,0]
    df_mixed['tsne-2d-two'] = tsne_results[:,1]

    return df_mixed

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


def plot_clustering_dual(x_tsne_2d, Mixed_y_labels,
                         samples_label, samples_prob,
                         run_id, output_folder_path,
                         plot_mode):

    ## Available options: 
    ## 'show' : only plot
    ## 'store' : only store
    ## 'show_store' : plot and store fig

    combined_fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_clustering(x_tsne_2d, labels=Mixed_y_labels, ground_truth=True,
                    ax=axes[0])

    plot_clustering(x_tsne_2d, labels=samples_label,
                     probabilities = samples_prob,
                    remove_outliers = True, ax=axes[1])

    current_fig_path = output_folder_path.joinpath(f'{run_id}.png') 

    combined_fig.suptitle(f'{run_id}', fontsize=14)
    plt.tight_layout()


    if plot_mode == 'show':
        plt.show()
    elif plot_mode == 'store':
        combined_fig.savefig(current_fig_path, dpi=300)
    elif plot_mode == 'show_store':
        combined_fig.savefig(current_fig_path, dpi=300)
        plt.show()
    else:
        print(f'Error! plot_histogam plot_mode')


def organize_samples_by_label(X_test_paths, samples_label, samples_prob, wav_chunks_output_path):
    # Loop over all paths in X_test_paths
    for idx, path in enumerate(X_test_paths):
        # Create a Path object
        path_obj = Path(path)
        
        # Get the label for the current sample
        label = str(samples_label[idx])
        
        # Create a new directory path for the label if it doesn't exist
        new_dir = wav_chunks_output_path.joinpath(label)
        new_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a new name for the file with the prob appended
        new_name = f"{path_obj.stem}_{samples_prob[idx]:.2f}{path_obj.suffix}" 

        # Create a new path for the file in the new directory
        new_path = new_dir / new_name 
        
        # Copy the file to the new directory
        shutil.copy(path, new_path)



