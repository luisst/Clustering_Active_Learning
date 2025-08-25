import torch
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from tqdm import tqdm
import pickle
import argparse
import sys
import re
import os

from Stg2_models import SimpleClassifier
from Stg2_dataloaders import create_dataloaders
from enhance_utils import plot_training_metrics 

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


def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

root_path = Path.home() / 'Dropbox' / 'DATASETS_AUDIO' / 'Dvectors'
main_folder_path = root_path / 'TTS4_easy_40-200'
# feats_pickle_path_ex = main_folder_path / 'd_vectors_feats_2spk.pickle'
feats_pickle_path_ex = main_folder_path / 'dvec_easy40-200.pickle'

exp_name_ex = 'S1_easyN1'
# exp_name_ex = 'S1_600_sc'
run_params_ex = 'mask00_lr-5_ep180'

# Create output folder in the same directory as feats_pickle_path
output_folder_ex = main_folder_path / f'{exp_name_ex}_{run_params_ex}_output'
output_folder_ex.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--input_feats_pickle', default=feats_pickle_path_ex, help='Path to the folder to store the D-vectors features')
parser.add_argument('--output_pred_folder', type=valid_path, default=output_folder_ex, help='Path to the folder to store the predictions')
parser.add_argument('--run_params', default=run_params_ex, help='string with the run params for HDBSCAN')
parser.add_argument('--exp_name', default=exp_name_ex, help='string with the experiment name')
args = parser.parse_args()

output_folder_path = Path(args.output_pred_folder)
feats_pickle_path = Path(args.input_feats_pickle)

run_params = args.run_params
exp_name = args.exp_name

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


enhanced_feats_and_labels = output_folder_path / f'val_{current_run_id}_enhanced.pickle'
enhanced_2d_labels = output_folder_path / f'val_{current_run_id}_enhanced_2d.pickle'

### Start Processing ###
train_loader, test_loader, feature_dim, num_speakers = create_dataloaders(
    feats_pickle_path, 
    log_path,
    batch_size=16, 
    test_size=0.2,
    augment_training=False, 
    noise_std=0.05, 
    mask_prob=mask_prob
)

log_print(f"Feature Dimension: {feature_dim}, Number of Speakers: {num_speakers}")

model = SimpleClassifier(dim=feature_dim, hidden_dim=256, num_classes=num_speakers).to(DEVICE)
loss_function = nn.CrossEntropyLoss().to(DEVICE) 

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

checkpoint_path = output_folder_path / f'model_{current_run_id}_{num_speakers}.pth'

# Early stopping variables
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

# Tracking metrics for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Store enhanced features and labels from the validation set
enhanced_features_all = []
enhanced_labels_all = []
enhanced_wavs_paths = []
classifier_2dim_list = []

for epoch_idx in range(0, max_epochs):
    model.train()
    total_train_loss = 0.0
    train_correct_predictions = 0
    train_total_samples = 0

    for batch_x, batch_labels, batch_wavs in tqdm(train_loader, desc=f"Epoch {epoch_idx}"):
        batch_x = batch_x.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)

        optimizer.zero_grad()
        refined_x = model(batch_x)
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
    log_print(f"Epoch {epoch_idx}, Average Training Loss: {avg_train_loss:.3f}, Training Accuracy: {train_accuracy:.3f}", lp=log_path)
    
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
    log_print(f"Epoch {epoch_idx}, Average Validation Loss: {avg_val_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}", lp=log_path)

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
        log_print(f"New best validation loss: {best_val_loss:.4f}", lp=log_path)
    else:
        patience_counter += 1
        log_print(f"No improvement. Patience counter: {patience_counter}/{patience}", lp=log_path)
        
        if patience_counter >= patience:
            log_print(f"Early stopping triggered after {epoch_idx + 1} epochs", lp=log_path)
            break
    
    # Step the scheduler
    scheduler.step()

# Load the best model state if early stopping was triggered
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    log_print("Loaded best model state", lp=log_path)

# Save the final model checkpoint
torch.save(model.state_dict(), checkpoint_path)
log_print(f"Final model saved to {checkpoint_path}", lp=log_path)

# Plot training metrics
if len(train_losses) > 0:  # Only plot if we have metrics
    plot_path = plot_training_metrics(
        train_losses, val_losses, train_accuracies, val_accuracies,
        output_folder_path, current_run_id
    )
    log_print(f"Training metrics plot saved to {plot_path}", lp=log_path)

# Generate enhanced features using the best model
log_print("Generating enhanced features from validation set with best model...", lp=log_path)
model.eval()
enhanced_features_all = []
enhanced_labels_all = []
enhanced_wavs_paths = []
classifier_2dim_list = []

with torch.no_grad():
    for batch_x, batch_labels, batch_wavs in tqdm(test_loader, desc="Generating enhanced features"):
        batch_x = batch_x.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)

        refined_x = model(batch_x)
        classifier_output = model.fc_classifier(refined_x)

        # Store enhanced features
        enhanced_features_all.extend(refined_x.cpu().numpy())
        enhanced_labels_all.extend(batch_labels.cpu().numpy())
        enhanced_wavs_paths.extend(batch_wavs)
        classifier_2dim_list.extend(classifier_output.cpu().numpy())  # Store all dimensions

# store enhanced features and labels from the validation set
X_data_and_labels = [enhanced_features_all, enhanced_wavs_paths, enhanced_labels_all]
with open(f'{enhanced_feats_and_labels}', "wb") as file:
    pickle.dump(X_data_and_labels, file)    


# store enhanced features and labels from the validation set
X_2d_and_labels = [classifier_2dim_list, enhanced_wavs_paths, enhanced_labels_all]
with open(f'{enhanced_2d_labels}', "wb") as file:
    pickle.dump(X_2d_and_labels, file)
