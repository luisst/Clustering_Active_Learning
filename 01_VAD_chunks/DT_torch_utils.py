import numpy as np
import shutil
from pathlib import Path
import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Constants
SAMPLE_RATE = 16000  # Default sample rate, adjust if your files differ
BATCH_SIZE = 32 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
binary_th = 0.5  # Threshold for binary classification

class XvectorDataset(Dataset):
    """Dataset for handling varying length tensors"""
    
    def __init__(self, tensor_list, wavs_path_list):
        """
        Args:
            tensor_list: List of varying length tensors of shape [1, seq_len, 768]
        """
        self.tensor_list = tensor_list
        self.wavs_path_list = wavs_path_list
        
        # Remove the batch dimension (1) from each tensor if it exists
        self.processed_tensors = []
        for tensor in tensor_list:
            if tensor.dim() == 3 and tensor.size(0) == 1:
                # Remove the batch dimension
                self.processed_tensors.append(tensor.squeeze(0))
            else:
                self.processed_tensors.append(tensor)
    
    def __len__(self):
        return len(self.processed_tensors)
    
    def __getitem__(self, idx):
        return self.processed_tensors[idx], self.wavs_path_list[idx]

def collate_fn_inference(batch):
    """
    Custom collate function to handle tensors of different lengths
    Args:
        batch: List of tensors of shape [seq_len, 768]
    
    Returns:
        padded_batch: Tensor of shape [batch_size, max_seq_len, 768]
        lengths: List of sequence lengths
    """
    features = [item[0] for item in batch]
    wav_names = [item[1] for item in batch]
    
    # Pad sequences
    padded_feats = pad_sequence(features, batch_first=True)
    
    return padded_feats, wav_names 

def create_inference_dataloader(tensor_list, wavs_list, batch_size=8, shuffle=False, num_workers=0):
    """
    Create a DataLoader for varying length tensors
    
    Args:
        tensor_list: List of tensors with shapes [1, seq_len, 768] where seq_len varies
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the dataset
        num_workers: Number of workers for data loading
    
    Returns:
        DataLoader object
    """
    dataset = XvectorDataset(tensor_list, wavs_list)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_inference,
        pin_memory=torch.cuda.is_available()  # Speeds up the host to device transfer
    )
    
    return dataloader


class WavLMFeatureDataset(Dataset):
    def __init__(self, feature_paths):
        self.feature_paths = feature_paths
    
    def __len__(self):
        return len(self.feature_paths)
    
    def __getitem__(self, idx):
        # Load pre-computed features and label
        data = torch.load(self.feature_paths[idx])
        wav_name = Path(data['original_path']).name
        return data['features'], data['label'], wav_name


def collate_fn_train(batch):
    """
    Custom collate function to handle variable length features
    """
    # Extract features and labels
    features = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    wav_names = [item[2] for item in batch]
    
    # Pad features to the maximum length in the batch
    max_length = max([f.size(1) for f in features])
    padded_features = torch.zeros(len(features), 1, max_length, features[0].size(2))
    for i, feature in enumerate(features):
        padded_features[i, :, :feature.size(1), :] = feature
    
    return padded_features, labels, wav_names

def create_dataloaders(features_folder):
    """Create DataLoader objects for training, validation, and testing"""

    # Get all feature files
    feature_files = list(features_folder.glob('*.pt'))
    print(f"Found {len(feature_files)} feature files")
    
    # Create dataset
    dataset = WavLMFeatureDataset(feature_files)
    
    # Split dataset into train, validation, and test sets
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.2 * total_size)
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_train,
        pin_memory=True  # Speeds up host to device transfers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_inference,
        pin_memory=True
    )
    
    return train_loader, val_loader


def generate_features(input_folder, output_features_folder):
    """
    Generate features from WAV files in the input folder and save them in the output folder.
    Each feature file will be named after the original WAV file but with a .pt extension.
    """
    # Get all wav files
    wav_files = list(input_folder.glob("*.wav"))
    print(f"Found {len(wav_files)} wav files")

    # Load WavLM model
    wavlm_model = torchaudio.pipelines.WAVLM_BASE.get_model().to(DEVICE)
    wavlm_model.eval()

    # Process each wav file
    for wav_path in tqdm(wav_files):
        # Extract label from filename
        filename = wav_path.stem
        if 'overlap' in filename:
            label = 1
        elif 'single' in filename:
            label = 0
        else:
            print(f"Warning: Could not determine label for {filename}, skipping")
            continue

        # Load waveform
        waveform, sample_rate = torchaudio.load(wav_path)
        
        # Extract features
        with torch.no_grad():
            features = wavlm_model(waveform.to(DEVICE))
            
        # Create output filename (replace .wav with .pt)
        output_filename = filename + '.pt'
        output_path = output_features_folder / output_filename
        
        # Save features and label
        torch.save({
            'features': features[0].cpu(),
            'label': label,
            'original_path': wav_path
        }, output_path)

    print(f"Feature extraction completed. Features saved to {output_features_folder}")


# Model architecture
class OverlapDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN for feature extraction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # self.dropout1 = nn.Dropout2d(0.3)  # Dropout after first convolutional layer
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout2d(0.3)  # Dropout after second convolutional layer
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # self.dropout3 = nn.Dropout2d(0.3)  # Dropout after third convolutional layer
        
        # Calculate flattened size

        # self._to_linear = self._get_conv_output((1, 34, 768))  # For 0.7 seconds of audioyy0
        self._to_linear = self._get_conv_output((1, 50, 768))  # For 1.0 seconds of audioyy0
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.dropout_fc1 = nn.Dropout(0.5)  # Dropout in fully connected layer
        self.fc2 = nn.Linear(256, 1)
        
    def _get_conv_output(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output = self._forward_conv(input)
        return int(np.prod(output.shape))
    
    def _forward_conv(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2))
        # x = self.dropout1(x)  # Apply dropout
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
        x = self.dropout2(x)  # Apply dropout
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (2, 2))
        # x = self.dropout3(x)  # Apply dropout
        return x
    
    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)  # Apply dropout
        x = self.fc2(x)
        return torch.sigmoid(x)

def plot_training_history(history, output_dir):
    """Plot training and validation loss/accuracy over epochs"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    training_history_path = output_dir / 'training_history.png'
    
    plt.tight_layout()
    plt.savefig(training_history_path)
    plt.close()


def evaluate_model(model, test_loader, output_dir, input_wavs_dir):
    """Evaluate the model on the test set"""
    model.eval()
    model.to(DEVICE)
    
    all_predictions = []
    all_targets = []
    all_wav_names = []
    all_confidences = []

    with torch.no_grad():
        for inputs, targets, current_wav_name in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            targets = targets.float().to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            predicted = (outputs >= binary_th).float().squeeze().cpu().numpy()
            confidences = outputs.squeeze().cpu().numpy()
            
            # Store predictions, targets, confidences, and filenames
            all_predictions.extend(predicted.tolist() if isinstance(predicted, np.ndarray) else [predicted])
            all_targets.extend(targets.cpu().numpy().tolist())
            all_confidences.extend(confidences.tolist() if isinstance(confidences, np.ndarray) else [confidences])
            all_wav_names.extend(current_wav_name)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, zero_division=0)
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    # Identify errors
    errors = []
    for i in range(len(all_predictions)):
        if all_predictions[i] != all_targets[i]:  # Error case
            confidence = all_confidences[i] if all_predictions[i] == 1 else 1 - all_confidences[i]
            errors.append((confidence, all_wav_names[i], all_predictions[i], all_targets[i]))
    
    # Sort errors by confidence in descending order
    errors = sorted(errors, key=lambda x: x[0], reverse=True)
    
    # Print the 10 most confident errors
    print("\nTop 10 Most Confident Errors:")
    for i, (confidence, wav_name, prediction, target) in enumerate(errors[:10]):
        print(f"{i+1}. WAV: {wav_name}, Confidence: {confidence:.4f}, Prediction: {prediction}, Target: {target}")

    # Copy TP, TN, FP, FN wavs files to respective folders
    tp_folder = output_dir / 'TP_wavs'
    tn_folder = output_dir / 'TN_wavs'
    fp_folder = output_dir / 'FP_wavs'
    fn_folder = output_dir / 'FN_wavs'

    tp_folder.mkdir(exist_ok=True)
    tn_folder.mkdir(exist_ok=True)
    fp_folder.mkdir(exist_ok=True)
    fn_folder.mkdir(exist_ok=True)

    for i, wav_name in enumerate(all_wav_names):
        wav_path = input_wavs_dir / wav_name
        if not wav_path.exists():
            print(f"Warning: {wav_path} does not exist. Skipping...")
            continue
        if all_predictions[i] == 1 and all_targets[i] == 1:
            shutil.copy(wav_path, tp_folder / Path(wav_name).name)
        elif all_predictions[i] == 0 and all_targets[i] == 0:
            shutil.copy(wav_path, tn_folder / Path(wav_name).name)
        elif all_predictions[i] == 1 and all_targets[i] == 0:
            shutil.copy(wav_path, fp_folder / Path(wav_name).name)
        elif all_predictions[i] == 0 and all_targets[i] == 1:
            shutil.copy(wav_path, fn_folder / Path(wav_name).name)
    
    
    print("\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    confusion_matrix_path = output_dir / 'confusion_matrix.png'

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Single', 'Multiple'],
                yticklabels=['Single', 'Multiple'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }


def train_model(output_dir, model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=100):
    """Train the model and validate after each epoch"""
    model.to(DEVICE)
    best_val_loss = float('inf')
    best_model_path = output_dir / 'best_overlap_detection_model.pth'

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, targets, current_wav_name in progress_bar:
            inputs = inputs.to(DEVICE)
            targets = targets.float().unsqueeze(1).to(DEVICE)

            # Add Gaussian noise to inputs
            noise = torch.randn_like(inputs) * 0.01  # Mean 0, standard deviation 0.01
            inputs = inputs + noise           

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs >= binary_th).float()
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': train_correct / train_total
            })
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for inputs, targets, current_wav_name in progress_bar:
                inputs = inputs.to(DEVICE)
                targets = targets.float().unsqueeze(1).to(DEVICE)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs >= binary_th).float()
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
                
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': val_correct / val_total
                })
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_correct / val_total
        
        # Update learning rate scheduler if provided
        if scheduler:
            scheduler.step(avg_val_loss)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        print("-" * 50)
        
        # Save model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print("Saved new best model!")
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
    
    return model, history

