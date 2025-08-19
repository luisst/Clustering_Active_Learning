import torch
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from tqdm import tqdm
import pickle

import sys
import re
import os

from Stg2_models import SimpleClassifier
from Stg2_dataloaders import create_dataloaders


# from pipeline_utilities import log_print, valid_path

def log_print(*args, **kwargs):
    """Prints to stdout and also logs to log_path."""

    log_path = kwargs.pop('lp', 'default_log.txt')
    print_to_console = kwargs.pop('print', True)

    message = " ".join(str(a) for a in args)
    if print_to_console:
        print(message)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

root_path = Path.home() / 'Dropbox' / 'DATASETS_AUDIO' / 'Dvectors'
main_folder_path = root_path / 'TTS4_easy_40-200'
pickle_folder_path = main_folder_path / 'iterative_speakers'

exp_name = 'EasySpeakers2to70C'
run_params = 'mask04_lr-5_ep180'

# Create output folder in the same directory as feats_pickle_path
output_folder_path = main_folder_path / f'{exp_name}_{run_params}_output'
output_folder_path.mkdir(exist_ok=True)

pattern = r"mask(\d+)_lr-(\d+)_ep(\d+)"
match = re.match(pattern, run_params)

if match:
    mask_prob = float(match.group(1))/10
    learning_rate = float('1e-' + str(int(match.group(2))))  # Convert to scientific notation
    max_epochs = int(match.group(3))
else:
    sys.exit("Invalid run_name format")

# Print the parsed parameters
print(f"Mask Probability: {mask_prob}")
print(f"Learning Rate: {learning_rate}")
print(f"Max Epochs: {max_epochs}")

patience = 15  # Number of epochs to wait before stopping
min_delta = 1e-4  # Minimum change to qualify as an improvement

current_run_id = f'{exp_name}_{run_params}'
log_path = output_folder_path / 'train_log.txt'

# List all pkl files in the pickle folder
feats_pickle_path_list = sorted(list(pickle_folder_path.glob("*.pkl")))

# List of all metrics from all features pickles
all_train_losses = []
all_val_losses = []
all_train_accuracies = []
all_val_accuracies = []

n_speakers = 2

for feats_pickle_path in feats_pickle_path_list:

    # Print the current pickle file being processed
    log_print(f"\n\n\tProcessing: {feats_pickle_path.name}", lp=log_path)

    ### Start Processing ###
    train_loader, test_loader, feature_dim, num_speakers = create_dataloaders(
        feats_pickle_path, 
        log_path,
        batch_size=16, 
        test_size=0.2,
        augment_training=True, 
        noise_std=0.05, 
        mask_prob=mask_prob,
        drop_last=True
    )

    log_print(f"Feature Dimension: {feature_dim}, Number of Speakers: {num_speakers}")

    model = SimpleClassifier(dim=feature_dim, hidden_dim=128, num_classes=num_speakers).to(DEVICE)
    loss_function = nn.CrossEntropyLoss().to(DEVICE) 

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Tracking metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch_idx in range(0, max_epochs):
        model.train()
        total_train_loss = 0.0
        train_correct_predictions = 0
        train_total_samples = 0

        for batch_x, batch_labels, batch_wavs in tqdm(train_loader, desc=f"Epoch {epoch_idx}"):
            batch_x = batch_x.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            # # Print shape of current batch data
            # print(f"\n{num_speakers} Batch Data Shape: {batch_x.shape}")

            optimizer.zero_grad()
            refined_x = model(batch_x)

            # # Print shape of refined features
            # print(f"{num_speakers} Refined Features Shape: {refined_x.shape}")

            classifier_output = model.fc_classifier(refined_x)
            # Compute loss
            loss = loss_function(classifier_output, batch_labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(classifier_output, 1)
            train_correct_predictions += (predicted == batch_labels).sum().item()
            train_total_samples += batch_labels.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct_predictions / train_total_samples
        # log_print(f"Epoch {epoch_idx}, Average Training Loss: {avg_train_loss:.3f}, Training Accuracy: {train_accuracy:.3f}", lp=log_path)
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_labels, batch_wavs in tqdm(test_loader, desc=f"Validation Epoch {epoch_idx}"):
                batch_x = batch_x.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)

                refined_x = model(batch_x)
                classifier_output = model.fc_classifier(refined_x)
                loss = loss_function(classifier_output, batch_labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(classifier_output, 1)
                correct_predictions += (predicted == batch_labels).sum().item()
                total_samples += batch_labels.size(0)
        
        avg_val_loss = total_val_loss / len(test_loader)
        val_accuracy = correct_predictions / total_samples
        # log_print(f"Epoch {epoch_idx}, Average Validation Loss: {avg_val_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}", lp=log_path)

        # Store metrics for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Early stopping logic
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            # log_print(f"New best validation loss: {best_val_loss:.4f}", lp=log_path)
        else:
            patience_counter += 1
            # log_print(f"No improvement. Patience counter: {patience_counter}/{patience}", lp=log_path)
            
            if patience_counter >= patience:
                # log_print(f"Early stopping triggered after {epoch_idx + 1} epochs", lp=log_path)
                break
        
        
        # Step the scheduler
        scheduler.step()

    # Load the best model state if early stopping was triggered
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        log_print("Loaded best model state", lp=log_path)
    
    # Extract average value of last epochs
    avg_train_loss = sum(train_losses[-5:]) / min(5, len(train_losses))
    avg_val_loss = sum(val_losses[-5:]) / min(5, len(val_losses))
    avg_train_accuracy = sum(train_accuracies[-5:]) / min(5, len(train_accuracies))
    avg_val_accuracy = sum(val_accuracies[-5:]) / min(5, len(val_accuracies))

    # log_print(f"Average Training Loss (last 5 epochs): {avg_train_loss:.3f}", lp=log_path)
    # log_print(f"Average Validation Loss (last 5 epochs): {avg_val_loss:.3f}", lp=log_path)
    # log_print(f"Average Training Accuracy (last 5 epochs): {avg_train_accuracy:.3f}", lp=log_path)
    # log_print(f"Average Validation Accuracy (last 5 epochs): {avg_val_accuracy:.3f}", lp=log_path)

    all_train_losses.append(avg_train_loss)
    all_val_losses.append(avg_val_loss)
    all_train_accuracies.append(avg_train_accuracy)
    all_val_accuracies.append(avg_val_accuracy)

    del model
    torch.cuda.empty_cache()

# Store all_train metrics into a pickle file

all_train_val_metrics = [all_train_losses, all_val_losses, all_train_accuracies, all_val_accuracies]
all_train_val_path = output_folder_path / 'all_train_val_metrics.pickle'

with open(all_train_val_path, 'wb') as handle:
    pickle.dump(all_train_val_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # Plot all metrics in 4 subplots with N of speakers as X-axis
# import matplotlib.pyplot as plt

# num_speakers = len(all_train_losses)
# speaker_counts = list(range(2, 2 + num_speakers))  # X-axis: 2 up to N speakers

# fig, axs = plt.subplots(1, 2, figsize=(18, 8))
# axs[0].plot(speaker_counts, all_train_losses, label='Train Loss')
# axs[0].plot(speaker_counts, all_val_losses, label='Val Loss')
# axs[0].set_title('Loss')
# axs[0].set_xlabel('Number of Speakers')
# axs[0].legend()

# axs[1].plot(speaker_counts, all_train_accuracies, label='Train Accuracy')
# axs[1].plot(speaker_counts, all_val_accuracies, label='Val Accuracy')
# axs[1].set_title('Accuracy')
# axs[1].set_xlabel('Number of Speakers')
# axs[1].legend()

# for ax in axs.flat:
#     ax.label_outer()

# plt.tight_layout()
# plt.savefig(output_folder_path / 'training_metrics.png')
# plt.show()

