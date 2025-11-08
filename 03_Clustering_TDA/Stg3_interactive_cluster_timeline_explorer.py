"""
All Clusters t-SNE Explorer with Temporal Information and Time Windows

This script allows you to explore how samples from ALL clusters relate to each other
in both the feature space (t-SNE) and time domain within a 20-second time window,
filtered by base long wav file.

Features:
- Interactive selection of base long wav file (since segments come from different source files)
- Displays ALL clusters simultaneously in t-SNE space for the selected base file
- 20-second time window constraint with valid range validation
- Different colors for different clusters (up to 20 with tab20 colormap)
- Different shapes for different GT labels (up to 9)
- Alpha gradient by temporal order (oldest=solid, newest=transparent)
- Hover tooltips showing sample index and start time
- Helps identify if temporal proximity correlates with feature similarity

Usage:
1. Run the script
2. Select which base long wav file to explore (e.g., D0, D1, D2, etc.)
3. View the available time range for that file
4. Enter a start time (script validates it allows a full 20s window)
5. View the t-SNE visualization with all clusters in that time window
6. Run again to explore different files or time windows
"""

from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from collections import Counter
import sys
import json


# Numpy compatibility for pickle loading
class NumpyCompatUnpickler(pickle.Unpickler):
    """Handle numpy version differences when loading pickle files."""
    def find_class(self, module, name):
        if 'numpy.core' in module or 'numpy._core' in module:
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                if 'numpy.core' in module:
                    module_alt = module.replace('numpy.core', 'numpy._core')
                elif 'numpy._core' in module:
                    module_alt = module.replace('numpy._core', 'numpy.core')
                return super().find_class(module_alt, name)
        return super().find_class(module, name)


# Setup numpy compatibility at module level
try:
    import numpy._core as numpy_core
    sys.modules['numpy.core'] = numpy_core
    sys.modules['numpy.core.multiarray'] = numpy_core.multiarray
    sys.modules['numpy.core.numeric'] = numpy_core.numeric
    sys.modules['numpy.core._multiarray_umath'] = numpy_core._multiarray_umath
except (ImportError, AttributeError):
    pass


def extract_base_wav_name(path_stem):
    """
    Extract base long wav filename from segment path.

    Format examples:
    - D0_1_7.6_9.5 -> D0
    - D1_2_54.11_55.05 -> D1
    - audio_A_3_100.5_102.3 -> audio_A

    Returns:
    --------
    base_name : str or None
        Base long wav filename
    """
    parts = path_stem.split('_')

    # The base name is everything before the cluster number and times
    # Assuming format: base_cluster_start_end or base_part1_part2_cluster_start_end
    # We need to find where the numeric parts start (cluster, start, end)

    # Find the first part that is purely numeric (cluster number)
    for i in range(len(parts)):
        try:
            # Try to convert to int (cluster number should be integer)
            int(parts[i])
            # If successful, everything before this is the base name
            if i > 0:
                return '_'.join(parts[:i])
        except ValueError:
            continue

    return None


def extract_time_from_path(path_stem):
    """
    Extract start_time and end_time from merged file path.

    Format examples:
    - D0_1_7.6_9.5
    - D0_2_54.11_55.05

    Returns:
    --------
    start_time : float or None
    end_time : float or None
    """
    parts = path_stem.split('_')

    # Try to extract from standard format: base_cluster_start_end
    if len(parts) >= 4:
        try:
            start_time = float(parts[-2])
            end_time = float(parts[-1])
            return start_time, end_time
        except (ValueError, IndexError):
            pass

    return None, None


def load_clustering_data(pickle_path):
    """
    Load clustering_data.pickle file.

    Returns:
    --------
    data_dict : dict
        Dictionary with all arrays and metadata
    """
    print(f"\nLoading clustering data: {pickle_path}")

    try:
        with open(pickle_path, "rb") as file:
            clustering_data = NumpyCompatUnpickler(file).load()
    except Exception as e:
        print(f"Error with NumpyCompatUnpickler: {e}")
        print("Trying standard pickle.load()...")
        with open(pickle_path, "rb") as file:
            clustering_data = pickle.load(file)

    # Check if it's 7-element (clustering_data) or 8-element (merged_clustering_data)
    if len(clustering_data) == 7:
        print("✓ Detected 7-element pickle (clustering_data.pickle)")
        Mixed_X_paths, hdb_data_input, x_tsne_2d, Mixed_y_labels, \
        samples_label, samples_prob, samples_outliers = clustering_data

        data_dict = {
            'paths': Mixed_X_paths,
            'hdb_data': hdb_data_input,
            'tsne_2d': x_tsne_2d,
            'y_labels': Mixed_y_labels,
            'sample_labels': samples_label,
            'sample_probs': samples_prob,
            'sample_outliers': samples_outliers
        }

    elif len(clustering_data) == 8:
        print("✓ Detected 8-element pickle (merged_clustering_data.pickle)")
        merged_X_data, merged_paths, merged_hdb_data, merged_tsne_2d, \
        merged_y_labels, merged_sample_labels, merged_sample_probs, \
        merged_sample_outliers = clustering_data

        data_dict = {
            'x_data': merged_X_data,
            'paths': merged_paths,
            'hdb_data': merged_hdb_data,
            'tsne_2d': merged_tsne_2d,
            'y_labels': merged_y_labels,
            'sample_labels': merged_sample_labels,
            'sample_probs': merged_sample_probs,
            'sample_outliers': merged_sample_outliers
        }

    elif len(clustering_data) == 9:
        print("✓ Detected 9-element pickle (merged_clustering_data_with_labels.pickle)")
        merged_X_data, merged_paths, merged_hdb_data, merged_tsne_2d, \
        merged_y_labels, merged_sample_labels, merged_sample_probs, \
        merged_sample_outliers, speaker_lp_column = clustering_data

        data_dict = {
            'x_data': merged_X_data,
            'paths': merged_paths,
            'hdb_data': merged_hdb_data,
            'tsne_2d': merged_tsne_2d,
            'y_labels': merged_y_labels,
            'sample_labels': merged_sample_labels,
            'sample_probs': merged_sample_probs,
            'sample_outliers': merged_sample_outliers,
            'speaker_lp': speaker_lp_column
        }

    else:
        print(f"ERROR: Unexpected pickle structure with {len(clustering_data)} elements")
        sys.exit(1)

    print(f"Loaded {len(data_dict['paths'])} samples")
    return data_dict


def get_base_wav_files_info(data_dict):
    """
    Get information about all base long wav files in the dataset.

    Returns:
    --------
    base_wav_info : dict
        Dictionary mapping base_wav_name -> {
            'count': sample count,
            'min_time': earliest start time,
            'max_time': latest end time,
            'indices': list of sample indices
        }
    """
    paths = data_dict['paths']
    base_wav_info = {}

    for idx, path in enumerate(paths):
        path_stem = Path(path).stem
        base_name = extract_base_wav_name(path_stem)
        start_time, end_time = extract_time_from_path(path_stem)

        if base_name is not None and start_time is not None:
            if base_name not in base_wav_info:
                base_wav_info[base_name] = {
                    'count': 0,
                    'min_time': float('inf'),
                    'max_time': float('-inf'),
                    'indices': []
                }

            base_wav_info[base_name]['count'] += 1
            base_wav_info[base_name]['min_time'] = min(base_wav_info[base_name]['min_time'], start_time)
            base_wav_info[base_name]['max_time'] = max(base_wav_info[base_name]['max_time'], end_time)
            base_wav_info[base_name]['indices'].append(idx)

    return base_wav_info


def get_cluster_info(data_dict):
    """
    Get information about all clusters.

    Returns:
    --------
    cluster_info : dict
        Dictionary mapping cluster_id -> {count, gt_labels, prob_stats}
    """
    sample_labels = data_dict['sample_labels']
    y_labels = data_dict['y_labels']
    sample_probs = data_dict['sample_probs']

    unique_clusters = np.unique(sample_labels)
    cluster_info = {}

    for cluster_id in unique_clusters:
        mask = sample_labels == cluster_id
        count = np.sum(mask)

        # Get GT labels in this cluster
        gt_in_cluster = y_labels[mask]
        gt_counter = Counter(gt_in_cluster)

        # Get probability stats
        probs_in_cluster = sample_probs[mask]
        prob_mean = np.mean(probs_in_cluster)
        prob_min = np.min(probs_in_cluster)
        prob_max = np.max(probs_in_cluster)

        cluster_info[cluster_id] = {
            'count': count,
            'gt_labels': dict(gt_counter),
            'prob_mean': prob_mean,
            'prob_min': prob_min,
            'prob_max': prob_max
        }

    return cluster_info


def select_cluster_interactive(cluster_info):
    """
    Interactive cluster selection from command line.

    Returns:
    --------
    cluster_id : int
        Selected cluster ID
    """
    print("\n" + "="*80)
    print("AVAILABLE CLUSTERS")
    print("="*80)

    # Sort clusters by ID (excluding -1 noise)
    sorted_clusters = sorted([c for c in cluster_info.keys() if c != -1])
    if -1 in cluster_info:
        sorted_clusters.append(-1)  # Add noise at the end

    for cluster_id in sorted_clusters:
        info = cluster_info[cluster_id]
        cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"

        print(f"\n{cluster_name}:")
        print(f"  Samples: {info['count']}")
        print(f"  GT labels: {info['gt_labels']}")
        print(f"  Prob: mean={info['prob_mean']:.3f}, min={info['prob_min']:.3f}, max={info['prob_max']:.3f}")

    print("\n" + "="*80)

    while True:
        try:
            cluster_id = int(input("Enter cluster ID to visualize (-1 for noise): "))
            if cluster_id in cluster_info:
                return cluster_id
            else:
                print(f"Error: Cluster {cluster_id} not found. Please try again.")
        except ValueError:
            print("Error: Please enter a valid integer.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


def plot_all_clusters_in_time_window(data_dict, base_wav_name, time_window_start, time_window_end):
    """
    Create interactive t-SNE plot for ALL clusters within a time window for a specific base wav file.

    Parameters:
    -----------
    data_dict : dict
        Clustering data dictionary
    base_wav_name : str
        Base long wav filename to filter samples
    time_window_start : float
        Start of time window in seconds
    time_window_end : float
        End of time window in seconds
    """
    print(f"\nPreparing t-SNE visualization for base file '{base_wav_name}'")
    print(f"Time window: {time_window_start}s - {time_window_end}s...")

    # Extract data
    paths = data_dict['paths']
    y_labels = data_dict['y_labels']
    sample_labels = data_dict['sample_labels']
    sample_probs = data_dict['sample_probs']
    sample_outliers = data_dict['sample_outliers']
    tsne_2d = data_dict['tsne_2d']

    # Parse all samples and filter by base wav file and time window
    all_sample_data = []
    for idx in range(len(paths)):
        path_stem = Path(paths[idx]).stem
        base_name = extract_base_wav_name(path_stem)
        start_time, end_time = extract_time_from_path(path_stem)

        if start_time is not None and base_name == base_wav_name:
            # Filter by time window
            if time_window_start <= start_time < time_window_end:
                all_sample_data.append({
                    'index': idx,
                    'path': path_stem,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'gt_label': y_labels[idx],
                    'cluster_id': sample_labels[idx],
                    'prob': sample_probs[idx],
                    'outlier': sample_outliers[idx],
                    'tsne_x': tsne_2d[idx, 0],
                    'tsne_y': tsne_2d[idx, 1]
                })

    if not all_sample_data:
        print(f"Error: No samples found in time window {time_window_start}s - {time_window_end}s for base file '{base_wav_name}'")
        return

    # Sort by start_time for alpha gradient
    all_sample_data.sort(key=lambda x: x['start_time'])

    print(f"Found {len(all_sample_data)} samples in time window")

    # Get unique clusters and GT labels
    clusters_in_window = sorted(set([s['cluster_id'] for s in all_sample_data]))
    gt_labels_in_window = sorted(set([s['gt_label'] for s in all_sample_data]))

    print(f"Clusters in window: {clusters_in_window}")
    print(f"GT labels in window: {gt_labels_in_window}")

    # Define marker shapes for GT labels (up to 9)
    marker_shapes = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P']
    gt_to_marker = {gt: marker_shapes[i % len(marker_shapes)]
                    for i, gt in enumerate(gt_labels_in_window)}

    # Define colors for clusters
    import matplotlib.cm as cm
    n_clusters = len(clusters_in_window)
    colors = cm.get_cmap('tab20', n_clusters)
    cluster_to_color = {cluster: colors(i) for i, cluster in enumerate(clusters_in_window)}

    # Calculate alpha values based on time order
    n_samples = len(all_sample_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 12))

    # Plot samples with alpha gradient and cluster colors
    scatter_objects = []

    for i, sample in enumerate(all_sample_data):
        # Alpha decreases linearly from 1.0 (first/oldest) to 0.3 (last/newest)
        alpha = 1.0 - (i / max(n_samples - 1, 1)) * 0.7

        gt_label = sample['gt_label']
        cluster_id = sample['cluster_id']
        marker = gt_to_marker[gt_label]
        color = cluster_to_color[cluster_id]

        # Plot point using t-SNE coordinates
        scatter = ax.scatter(
            sample['tsne_x'],
            sample['tsne_y'],
            c=[color],
            marker=marker,
            s=200,
            alpha=alpha,
            edgecolors='black',
            linewidths=1.5,
            label=None  # We'll create custom legend
        )

        scatter_objects.append({
            'scatter': scatter,
            'sample': sample,
            'marker': marker,
            'alpha': alpha
        })

    # Create custom legend - Part 1: Clusters
    from matplotlib.lines import Line2D
    legend_elements = []

    # Add cluster legend entries
    legend_elements.append(Line2D([0], [0], color='w', label='CLUSTERS:', marker=None, markersize=0))
    for cluster_id in clusters_in_window:
        color = cluster_to_color[cluster_id]
        cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"

        # Count samples in this cluster
        count = sum(1 for s in all_sample_data if s['cluster_id'] == cluster_id)

        legend_elements.append(
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color, markeredgecolor='black',
                   markersize=10, linewidth=0,
                   label=f'  {cluster_name} (n={count})')
        )

    # Add separator
    legend_elements.append(Line2D([0], [0], color='w', label='', marker=None, markersize=0))

    # Add GT label legend entries
    legend_elements.append(Line2D([0], [0], color='w', label='GT LABELS:', marker=None, markersize=0))
    for gt_label in gt_labels_in_window:
        marker = gt_to_marker[gt_label]

        # Count samples with this GT label
        count = sum(1 for s in all_sample_data if s['gt_label'] == gt_label)

        legend_elements.append(
            Line2D([0], [0], marker=marker, color='w',
                   markerfacecolor='gray', markeredgecolor='black',
                   markersize=10, linewidth=0,
                   label=f'  GT {gt_label} (n={count})')
        )

    # Add separator
    legend_elements.append(Line2D([0], [0], color='w', label='', marker=None, markersize=0))

    # Add temporal order legend entries
    legend_elements.append(Line2D([0], [0], color='w', label='TEMPORAL ORDER:', marker=None, markersize=0))
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='gray', markeredgecolor='black',
               markersize=10, linewidth=0, alpha=1.0,
               label='  Oldest (solid)')
    )
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='gray', markeredgecolor='black',
               markersize=10, linewidth=0, alpha=0.3,
               label='  Newest (transparent)')
    )

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=9, framealpha=0.9)

    # Labels and title
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(f'All Clusters - t-SNE Feature Space with Temporal Order\n'
                 f'Base File: {base_wav_name} | Time Window: {time_window_start:.1f}s - {time_window_end:.1f}s '
                 f'({time_window_end - time_window_start:.1f}s span)\n'
                 f'{n_samples} samples | {n_clusters} clusters | {len(gt_labels_in_window)} GT labels',
                 fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle='--')

    # Add interactive cursor with hover tooltips
    cursor = mplcursors.cursor(hover=True)

    @cursor.connect("add")
    def on_add(sel):
        # Find which sample was selected
        for i, obj in enumerate(scatter_objects):
            if sel.artist == obj['scatter']:
                sample = obj['sample']

                # Create tooltip text (simplified)
                tooltip_text = (
                    f"Index: {sample['index']}\n"
                    f"Start: {sample['start_time']:.2f}s"
                )

                sel.annotation.set_text(tooltip_text)
                sel.annotation.get_bbox_patch().set(facecolor='wheat', alpha=0.9)
                break

    plt.tight_layout()

    # Save figure
    output_folder = Path.home().joinpath('Dropbox', 'DATASETS_AUDIO', 'Unsupervised_Pipeline')
    output_path = output_folder / f"{base_wav_name}_all_clusters_tsne_{time_window_start:.0f}s_{time_window_end:.0f}s.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")

    plt.show()


def main():
    """Main execution function."""

    # Configuration
    DATASET_NAME = "TestAO-Irma"
    STG3_METHOD = "STG3_EXP011-SHAS-DV-hdb"

    # Paths
    base_path = Path.home().joinpath('Dropbox', 'DATASETS_AUDIO', 'Unsupervised_Pipeline', DATASET_NAME)
    stg3_folder = base_path.joinpath('STG_3', STG3_METHOD)

    # Input pickle - try merged first, then clustering
    merged_pickle_path = stg3_folder / 'merged_clustering_data.pickle'
    clustering_pickle_path = stg3_folder / 'clustering_data.pickle'

    # Check which pickle exists
    if merged_pickle_path.exists():
        pickle_path = merged_pickle_path
    elif clustering_pickle_path.exists():
        pickle_path = clustering_pickle_path
    else:
        print(f"ERROR: No pickle file found in {stg3_folder}")
        print(f"  Tried: {merged_pickle_path}")
        print(f"  Tried: {clustering_pickle_path}")
        sys.exit(1)

    print("="*80)
    print("ALL CLUSTERS TIMELINE EXPLORER - TIME WINDOW VIEW")
    print("="*80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"STG3 Method: {STG3_METHOD}")
    print(f"Pickle file: {pickle_path}")
    print("="*80)

    # Load data
    data_dict = load_clustering_data(pickle_path)

    # Get information about base wav files
    print("\n" + "="*80)
    print("STEP 1: Select Base Long Wav File")
    print("="*80)
    base_wav_info = get_base_wav_files_info(data_dict)

    if not base_wav_info:
        print("ERROR: No valid base wav files found in the dataset")
        sys.exit(1)

    # Sort base wav files by name
    sorted_base_wavs = sorted(base_wav_info.keys())

    # Display available base wav files
    print("\nAvailable long wav files:\n")
    for i, base_name in enumerate(sorted_base_wavs, 1):
        info = base_wav_info[base_name]
        time_span = info['max_time'] - info['min_time']
        print(f"  {i}. {base_name}")
        print(f"     - Samples: {info['count']}")
        print(f"     - Time range: {info['min_time']:.1f}s to {info['max_time']:.1f}s (span: {time_span:.1f}s)")
        print()

    # Ask user to select base wav file
    while True:
        try:
            choice = int(input(f"Select a long wav file (1-{len(sorted_base_wavs)}): "))
            if 1 <= choice <= len(sorted_base_wavs):
                selected_base_wav = sorted_base_wavs[choice - 1]
                break
            else:
                print(f"Error: Please enter a number between 1 and {len(sorted_base_wavs)}")
        except ValueError:
            print("Error: Please enter a valid integer")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

    # Get time range information for selected file
    selected_info = base_wav_info[selected_base_wav]
    min_time = selected_info['min_time']
    max_time = selected_info['max_time']

    # Calculate maximum valid start time (allowing for 20-second window)
    max_start_time = max_time - 20
    if max_start_time < min_time:
        max_start_time = min_time

    print("\n" + "="*80)
    print("STEP 2: Select Time Window")
    print("="*80)
    print(f"Selected file: {selected_base_wav}")
    print(f"Available time range: {min_time:.1f}s to {max_time:.1f}s")
    print(f"Valid start time range for 20s window: {min_time:.1f}s to {max_start_time:.1f}s")
    print()

    # Ask user for start time
    while True:
        try:
            user_time_start = float(input(f"Enter start time in seconds ({min_time:.1f} to {max_start_time:.1f}): "))
            if min_time <= user_time_start <= max_start_time:
                break
            else:
                print(f"Error: Start time must be between {min_time:.1f}s and {max_start_time:.1f}s")
        except ValueError:
            print("Error: Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

    time_window_start = user_time_start
    time_window_end = user_time_start + 20

    print("\n" + "="*80)
    print("VISUALIZATION")
    print("="*80)
    print(f"Base file: {selected_base_wav}")
    print(f"Time window: {time_window_start:.1f}s - {time_window_end:.1f}s (20s span)")
    print("="*80)

    # Plot all clusters in time window for selected base wav file
    plot_all_clusters_in_time_window(data_dict, selected_base_wav, time_window_start, time_window_end)


if __name__ == "__main__":
    main()
