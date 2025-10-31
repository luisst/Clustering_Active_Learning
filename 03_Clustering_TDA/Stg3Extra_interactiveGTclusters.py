import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Cursor
from pathlib import Path
import seaborn as sns
from collections import Counter

class InteractiveTSNEPlot:
    def __init__(self, tsne_2d, labels, paths, title="Interactive t-SNE Plot"):
        self.tsne_2d = tsne_2d
        self.labels = labels
        self.paths = [Path(p).stem for p in paths]  # Extract filename stems
        self.title = title
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle(self.title, fontsize=14, fontweight='bold')
        
        # Initialize selected point tracking
        self.selected_point = None
        self.selected_scatter = None
        
        # Setup colors and plotting
        self.setup_colors()
        self.plot_data()
        self.setup_interactivity()
        
        # Create legend and info text
        self.create_legend()
        self.create_info_text()
        
    def setup_colors(self):
        """Setup consistent color mapping for labels"""
        unique_labels = np.unique(self.labels)
        
        # Use seaborn color palette for consistency
        if len(unique_labels) <= 10:
            palette = sns.color_palette('tab10', n_colors=len(unique_labels))
        else:
            palette = sns.color_palette('husl', n_colors=len(unique_labels))
        
        # Create label to color mapping
        self.label_to_color = {}
        self.label_to_idx = {}
        
        # Handle noise points (-1) with gray color if present
        if -1 in unique_labels:
            self.label_to_color[-1] = 'lightgray'
            self.label_to_idx[-1] = -1
            other_labels = [l for l in unique_labels if l != -1]
        else:
            other_labels = unique_labels
        
        # Assign colors to other labels
        for i, label in enumerate(sorted(other_labels)):
            self.label_to_color[label] = palette[i % len(palette)]
            self.label_to_idx[label] = i
    
    def plot_data(self):
        """Plot the t-SNE data with colors"""
        # Count labels for legend
        self.label_counts = Counter(self.labels)
        
        # Create color array for all points
        colors = [self.label_to_color[label] for label in self.labels]
        
        # Plot all points
        self.scatter = self.ax.scatter(
            self.tsne_2d[:, 0], 
            self.tsne_2d[:, 1], 
            c=colors, 
            s=50, 
            alpha=0.7,
            picker=True,
            pickradius=5
        )
        
        self.ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        self.ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        self.ax.grid(True, alpha=0.3)
    
    def create_legend(self):
        """Create legend with label counts"""
        legend_elements = []
        
        # Sort labels for consistent legend order
        sorted_labels = sorted(self.label_counts.keys(), key=lambda x: (x == -1, x))
        
        for label in sorted_labels:
            count = self.label_counts[label]
            color = self.label_to_color[label]
            
            if label == -1:
                label_text = f"Noise (n={count})"
            else:
                label_text = f"Cluster {label} (n={count})"
            
            legend_elements.append(
                mpatches.Patch(color=color, label=label_text)
            )
        
        self.legend = self.ax.legend(
            handles=legend_elements,
            title='Clusters',
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            fontsize=10
        )
    
    def create_info_text(self):
        """Create text area for displaying point information"""
        self.info_text = self.ax.text(
            0.02, 0.98, 
            "Hover over points to see filename\nClick to select/deselect",
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            zorder=1000
        )
    
    def setup_interactivity(self):
        """Setup mouse interaction events"""
        # Connect events
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add cursor
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
    
    def find_closest_point(self, event):
        """Find the closest point to mouse position"""
        if event.inaxes != self.ax:
            return None
            
        # Get mouse position in data coordinates
        mouse_x, mouse_y = event.xdata, event.ydata
        
        if mouse_x is None or mouse_y is None:
            return None
        
        # Calculate distances to all points
        distances = np.sqrt(
            (self.tsne_2d[:, 0] - mouse_x)**2 + 
            (self.tsne_2d[:, 1] - mouse_y)**2
        )
        
        # Find closest point within reasonable distance
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        # Only return if close enough (adjust threshold as needed)
        if min_distance < 0.1 * (np.max(self.tsne_2d) - np.min(self.tsne_2d)):
            return min_idx
        
        return None
    
    def on_hover(self, event):
        """Handle mouse hover events"""
        closest_idx = self.find_closest_point(event)
        
        if closest_idx is not None:
            # Update info text with point details
            filename = self.paths[closest_idx]
            label = self.labels[closest_idx]
            x, y = self.tsne_2d[closest_idx]
            
            if label == -1:
                label_text = "Noise"
            else:
                label_text = f"Cluster {label}"
            
            info_text = (
                f"File: {filename}\n"
                f"Label: {label_text}\n" 
                f"Position: ({x:.2f}, {y:.2f})\n"
                f"Index: {closest_idx}\n"
                "Click to select/deselect"
            )
            
            self.info_text.set_text(info_text)
            self.fig.canvas.draw_idle()
        else:
            # Reset info text
            self.info_text.set_text("Hover over points to see filename\nClick to select/deselect")
            self.fig.canvas.draw_idle()
    
    def on_click(self, event):
        """Handle mouse click events"""
        closest_idx = self.find_closest_point(event)
        
        if closest_idx is not None:
            if self.selected_point == closest_idx:
                # Deselect if clicking on already selected point
                self.deselect_point()
            else:
                # Select new point
                self.select_point(closest_idx)
    
    def select_point(self, idx):
        """Highlight a selected point"""
        # Remove previous selection
        self.deselect_point()
        
        # Add new selection highlight
        x, y = self.tsne_2d[idx]
        self.selected_scatter = self.ax.scatter(
            x, y, 
            s=200, 
            c='red', 
            marker='o', 
            edgecolors='black', 
            linewidths=2,
            alpha=0.8,
            zorder=1000
        )
        
        self.selected_point = idx
        
        # Update info text
        filename = self.paths[idx]
        label = self.labels[idx]
        
        if label == -1:
            label_text = "Noise"
        else:
            label_text = f"Cluster {label}"
        
        info_text = (
            f"SELECTED:\n"
            f"File: {filename}\n"
            f"Label: {label_text}\n" 
            f"Position: ({x:.2f}, {y:.2f})\n"
            f"Index: {idx}\n"
            "Click again to deselect"
        )
        
        self.info_text.set_text(info_text)
        self.fig.canvas.draw()
    
    def deselect_point(self):
        """Remove selection highlight"""
        if self.selected_scatter is not None:
            self.selected_scatter.remove()
            self.selected_scatter = None
        
        self.selected_point = None
        self.info_text.set_text("Hover over points to see filename\nClick to select/deselect")
        self.fig.canvas.draw()
    
    def show(self):
        """Display the interactive plot"""
        plt.tight_layout()
        plt.show()

def load_clustering_data(pickle_path):
    """Load clustering data from pickle file"""
    try:
        with open(pickle_path, 'rb') as f:
            clustering_data = pickle.load(f)
        return clustering_data
    except Exception as e:
        print(f"Error loading clustering data: {e}")
        return None

def detect_data_format(clustering_data):
    """
    Detect whether the data is in original format (7 elements) or merged format (8 elements)
    
    Returns:
        str: 'original' or 'merged'
    """
    if len(clustering_data) == 7:
        return 'original'
    elif len(clustering_data) == 8:
        return 'merged' 
    else:
        raise ValueError(f"Unexpected clustering data format. Expected 7 or 8 elements, got {len(clustering_data)}")

def unpack_clustering_data(clustering_data, data_format):
    """
    Unpack clustering data based on format
    
    Returns:
        tuple: (paths, tsne_2d, labels, title_suffix)
    """
    if data_format == 'original':
        # Original format: 7 elements
        Mixed_X_paths, hdb_data_input, x_tsne_2d, Mixed_y_labels, samples_label, samples_prob, samples_outliers = clustering_data
        return Mixed_X_paths, x_tsne_2d, Mixed_y_labels, "Original Clustering"
    
    elif data_format == 'merged':
        # Merged format: 8 elements
        merged_x_data, merged_paths, merged_hdb_data, merged_tsne_2d, merged_y_labels, merged_sample_labels, merged_sample_probs, merged_sample_outliers = clustering_data
        return merged_paths, merged_tsne_2d, merged_y_labels, "Merged Clustering"

def show_plot(pickle_path, title_prefix, data_format_hint=None):
    """
    Load data and show interactive plot
    
    Args:
        pickle_path: Path to pickle file
        title_prefix: Prefix for plot title
        data_format_hint: Optional hint about expected data format
    """
    print(f"\nLoading clustering data from: {pickle_path}")
    clustering_data = load_clustering_data(pickle_path)
    
    if clustering_data is None:
        print("Failed to load clustering data.")
        return False
    
    try:
        # Detect data format
        data_format = detect_data_format(clustering_data)
        print(f"Detected data format: {data_format}")
        
        if data_format_hint and data_format != data_format_hint:
            print(f"Warning: Expected {data_format_hint} format but detected {data_format}")
        
        # Unpack data
        paths, tsne_2d, labels, title_suffix = unpack_clustering_data(clustering_data, data_format)
        
        print(f"Loaded data shapes:")
        print(f"  Paths: {len(paths)}")
        print(f"  t-SNE: {tsne_2d.shape}")
        print(f"  Labels: {len(labels)} samples, {len(np.unique(labels))} unique labels")
        
        # Create title
        full_title = f"{title_prefix} - {title_suffix}"
        
        # Create and show interactive plot
        print("Creating interactive plot...")
        interactive_plot = InteractiveTSNEPlot(
            tsne_2d=tsne_2d,
            labels=labels,
            paths=paths,
            title=full_title
        )
        
        print("Showing interactive plot. Instructions:")
        print("  - Hover over points to see filename and details")
        print("  - Click on points to select/highlight them")
        print("  - Click again on selected point to deselect")
        print("  - Close the window to continue")
        
        interactive_plot.show()
        return True
        
    except Exception as e:
        print(f"Error processing clustering data: {e}")
        return False

def main():
    DATASET_NAME = "TestAO-Irma"
    base_ex = Path.home() / "Dropbox" / "DATASETS_AUDIO" / "Unsupervised_Pipeline" / DATASET_NAME
    
    # Default paths
    cluster_data_pickle_ex = base_ex / "STG_3" / "STG3_EXP011-SHAS-DV-hdb" / "clustering_data.pickle"
    merged_data_pickle_ex = base_ex / "STG_3" / "STG3_EXP011-SHAS-DV-hdb" / "merged_clustering_data.pickle"

    parser = argparse.ArgumentParser(
        description="Interactive t-SNE visualization for clustering results"
    )
    
    parser.add_argument(
        '--clustering_pickle',
        type=str,
        default=cluster_data_pickle_ex,
        help='Path to the original clustering data pickle file'
    )
    
    parser.add_argument(
        '--merged_pickle',
        type=str,
        default=merged_data_pickle_ex,
        help='Path to the merged clustering data pickle file'
    )
    
    parser.add_argument(
        '--title_prefix',
        type=str,
        default=f"GT {DATASET_NAME} t-SNE HDBSCAN Clusters",
        help='Prefix for the plot titles'
    )
    
    
    args = parser.parse_args()
    
    
    print("="*60)
    print("INTERACTIVE t-SNE CLUSTERING VISUALIZATION")
    print("="*60)
    
    # Show original clustering plot
    print("\n" + "="*40)
    print("SHOWING ORIGINAL CLUSTERING DATA")
    print("="*40)
    

    success = show_plot(args.clustering_pickle, args.title_prefix, 'original')
    if not success:
        print("Failed to show original clustering plot")
    
    # Show merged clustering plot
    print("\n" + "="*40)
    print("SHOWING MERGED CLUSTERING DATA")
    print("="*40)
    
    success = show_plot(args.merged_pickle, args.title_prefix, 'merged')
    if not success:
        print("Failed to show merged clustering plot")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()