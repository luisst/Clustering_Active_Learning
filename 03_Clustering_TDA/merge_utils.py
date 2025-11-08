from pathlib import Path
import subprocess as subp
import json
import sys
import time
import matplotlib.pyplot as plt
import pandas as pd

def create_folder_if_missing(folder_path):
    """Create folder if it doesn't exist"""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

def get_platform():
    platforms = {
        'linux1' : 'Linux',
        'linux2' : 'Linux',
        'darwin' : 'OS X',
        'win32' : 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform
    
    return platforms[sys.platform]

def get_total_video_length(input_video_path):
    script_out = subp.check_output(["ffprobe", "-v", "quiet", "-show_format", "-print_format", "json", input_video_path])
    ffprobe_data = json.loads(script_out)
    video_duration_seconds = float(ffprobe_data["format"]["duration"])

    return video_duration_seconds

def ffmpeg_split_audio(input_video, output_pth,
            start_time_csv = '0.00',
            stop_time_csv = 'default',
            sr = 16000,
            verbose = False,
            formatted = False,
            output_video_flag = False,
            times_as_integers = False):

    if times_as_integers:
        start_time_csv = str(start_time_csv)
        stop_time_csv = str(stop_time_csv)

    if formatted:
        (hstart, mstart, sstart) = start_time_csv.split(':')
        start_time_csv = str(float(hstart) * 3600 + float(mstart) * 60 + float(sstart))

        (hstop, mstop, sstop) = stop_time_csv.split(':')
        stop_time_csv = str(float(hstop) * 3600 + float(mstop) * 60 + float(sstop))

    if verbose:
        if stop_time_csv == 'default':
            if get_platform() == 'Linux':
                cmd = f"ffmpeg -i '{input_video}' -acodec pcm_s16le -ac 1 -ar {sr} '{output_pth}'"
            else:
                cmd = f"ffmpeg -i {input_video} -acodec pcm_s16le -ac 1 -ar {sr} {output_pth}"
            subp.run(cmd, shell=True)
            return 'non_valid', 'non_valid'
    else:
        if stop_time_csv == 'default':
            if get_platform() == 'Linux':
                cmd = f"ffmpeg -i '{input_video}' -hide_banner -loglevel error -acodec pcm_s16le -ac 1 -ar {sr} '{output_pth}'"
            else:
                cmd = f"ffmpeg -i {input_video} -hide_banner -loglevel error -acodec pcm_s16le -ac 1 -ar {sr} {output_pth}"
            subp.run(cmd, shell=True)
            return 'non_valid', 'non_valid'

    video_duration_seconds = get_total_video_length(input_video)

    # Check stop time is larger than start time
    if float(start_time_csv) >= float(stop_time_csv):
        print(f'Error! Start time {start_time_csv} is larger than stop time {stop_time_csv}')

    # Check stop time is less than duration of the video
    if float(stop_time_csv) > video_duration_seconds:
        print(f'Warning! [changed] Stop time {stop_time_csv} is larger than video duration {video_duration_seconds}')
        stop_time_csv = str(video_duration_seconds)
    
    # convert the starting time/stop time from seconds -> 00:00:00
    start_time_format = time.strftime("%H:%M:%S", time.gmtime(int(start_time_csv.split('.')[0]))) + \
        '.' + start_time_csv.split('.')[-1][0:2]
    stop_time_format = time.strftime("%H:%M:%S", time.gmtime(int(stop_time_csv.split('.')[0]))) + \
        '.' + stop_time_csv.split('.')[-1][0:2]

    if verbose:
        print(f'{start_time_format} - {stop_time_format}')
        if output_video_flag:
            ffmpeg_params = f' -c:v libx264 -crf 30 '
        else:
            ffmpeg_params = f' -acodec pcm_s16le -ac 1 -ar {sr} '
    else:
        if output_video_flag:
            ffmpeg_params = f' -hide_banner -loglevel error -c:v libx264 -crf 30 '
        else:
            ffmpeg_params = f' -hide_banner -loglevel error -acodec pcm_s16le -ac 1 -ar {sr} '

    if get_platform() == 'Linux':
        cmd = f"ffmpeg -i '{input_video}' '{ffmpeg_params}' -ss '{start_time_format}' -to  '{stop_time_format}' '{output_pth}'"
    else:
        cmd = f"ffmpeg -i {input_video}  {ffmpeg_params} -ss {start_time_format} -to  {stop_time_format} {output_pth}"

    # print(cmd)

    subp.run(cmd, shell=True)

    return start_time_csv, stop_time_csv


def active_learning_sample_selection(hdb_labels, hdb_probs, umap_data, output_folder_al, x_tsne_2d, n_samples_per_cluster=3, plot_flag=False):
    """
    Select samples for manual labeling using Active Learning strategies.

    Parameters:
    -----------
    hdb_labels : array
        HDBSCAN cluster labels
    hdb_probs : array
        HDBSCAN membership probabilities
    umap_data : array
        UMAP-reduced data for distance calculations
    output_folder_al : Path
        Output folder for saving plots
    x_tsne_2d : array
        t-SNE 2D coordinates for visualization (ensures consistency with other plots)
    n_samples_per_cluster : int
        Number of samples to select per cluster (max 3)
    plot_flag : bool
        Whether to generate and save plots

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
    
    # Adjust samples per cluster based on total number of clusters
    if len(unique_clusters) > 10:
        effective_samples_per_cluster = 2  # Only strategies 1 & 2
        print(f"Active Learning: {len(unique_clusters)} clusters detected (>10), limiting to 2 samples per cluster")
    else:
        effective_samples_per_cluster = n_samples_per_cluster  # Use original parameter
        print(f"Active Learning: {len(unique_clusters)} clusters detected (≤10), selecting up to {effective_samples_per_cluster} samples per cluster")
    
    print(f"Active Learning: Selecting up to {effective_samples_per_cluster} samples from {len(unique_clusters)} clusters")
    
    for cluster_id in unique_clusters:
        cluster_mask = hdb_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_probs = hdb_probs[cluster_mask]
        cluster_umap = umap_data[cluster_mask]
        
        if len(cluster_indices) < effective_samples_per_cluster:
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
        
        # Strategy 3: Boundary sample (diverse/informative) - Only if ≤10 clusters
        if len(cluster_indices) > 2 and len(selected_for_cluster) < effective_samples_per_cluster and effective_samples_per_cluster > 2:
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
        
        # Alternative Strategy 3: Edge/Boundary sample - Only if ≤10 clusters
        if len(selected_for_cluster) < effective_samples_per_cluster and len(cluster_indices) > 5 and effective_samples_per_cluster > 2:
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
        
        selected_samples[cluster_id] = selected_for_cluster[:effective_samples_per_cluster]
        selection_reasons[cluster_id] = reasons_for_cluster[:effective_samples_per_cluster]
        
        print(f"  Cluster {cluster_id}: Selected {len(selected_for_cluster)} samples "
              f"from {len(cluster_indices)} total ({reasons_for_cluster})")
        
    if plot_flag:
        # Plot t-SNE 2D with selected samples highlighted (for visual consistency)
        plt.figure(figsize=(12, 10))

        # Plot all samples in light gray
        plt.scatter(x_tsne_2d[:, 0], x_tsne_2d[:, 1], c='lightgray', s=30, alpha=0.5, label='All samples', edgecolors='none')

        # Color palette for different clusters
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta',
                  'lime', 'pink', 'olive', 'navy', 'teal', 'maroon', 'gold']
        color_map = {cid: colors[i % len(colors)] for i, cid in enumerate(unique_clusters)}

        # Plot selected samples per cluster with larger markers
        for cluster_id, sample_indices in selected_samples.items():
            cluster_tsne = x_tsne_2d[sample_indices]
            plt.scatter(cluster_tsne[:, 0], cluster_tsne[:, 1],
                        c=color_map[cluster_id], s=150, marker='o',
                        label=f'Cluster {cluster_id} (n={len(sample_indices)})',
                        edgecolors='none', linewidths=1.5, zorder=5)

        plt.title('t-SNE 2D Projection with Selected Active Learning Samples', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(loc='best', fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_folder_al / 'tsne_active_learning_selected_samples.png', dpi=150)
        plt.close()

        print(f"  Plot saved: {output_folder_al / 'tsne_active_learning_selected_samples.png'}")
    
    return selected_samples, selection_reasons


def format_active_learning_results(selected_samples, selection_reasons, wav_stems, hdb_labels, hdb_probs, al_input_csv, run_id):
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
    summary_df.to_csv(al_input_csv, index=False)
    
    print(f"\nActive Learning Results:")
    print(f"  Total samples selected: {len(summary_data)}")
    print(f"  Samples saved to: {al_input_csv}")
    print(f"  Please manually label the 'suggested_label' column with actual speaker IDs")
    
    # Print summary by strategy
    strategy_counts = summary_df['selection_reason'].value_counts()
    print(f"  Selection strategy breakdown: {dict(strategy_counts)}")
    
    return summary_df
