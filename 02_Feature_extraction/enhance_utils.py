import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies, output_folder, run_id):
    """
    Plot training and validation loss, and training and validation accuracy side by side.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accuracies: List of training accuracies per epoch
        val_accuracies: List of validation accuracies per epoch
        output_folder: Path to save the plot
        run_id: Run identifier for filename
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot Loss
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=6)
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=6)
    ax1.set_title('Training and Validation Loss', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot Accuracy
    ax2.plot(epochs, train_accuracies, 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=6)
    ax2.plot(epochs, val_accuracies, 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=6)
    ax2.set_title('Training and Validation Accuracy', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Accuracy', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Set y-axis limits for accuracy (0 to 1)
    ax2.set_ylim(0, 1)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save plot with high resolution
    plot_path = output_folder / f'{run_id}_training_metrics.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return plot_path

