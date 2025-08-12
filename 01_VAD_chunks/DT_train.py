from pathlib import Path
import torch
import torch.nn as nn

from DT_torch_utils import OverlapDetectionModel, create_dataloaders, generate_features, plot_training_history, train_model 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


NUM_EPOCHS = 80
LEARNING_RATE = 1e-4

root_dir = Path.home().joinpath('Dropbox/DATASETS_AUDIO/TTS3_dt/overlapping_gt/output_segments10')
input_wavs_dir = root_dir / "input_wavs" # Change this to your input directory

output_dir = root_dir / "CRl_15K05_1"
output_dir.mkdir(parents=True, exist_ok=True)

output_features_folder = input_wavs_dir / "X_features"

# Check if output features folder is populated, if not, generate features
if not output_features_folder.exists() or not any(output_features_folder.iterdir()):
    print(f"Generating features in {output_features_folder}")
    output_features_folder.mkdir(parents=True, exist_ok=True)
    generate_features(input_wavs_dir, output_features_folder)
else:
    print(f"Features already exist in {output_features_folder}, skipping generation.")

# Create data loaders
train_loader, val_loader, test_loader = create_dataloaders(output_features_folder)

# # Initialize model, loss function, and optimizer
model = OverlapDetectionModel()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Train model
model, history = train_model(
    output_dir, model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS
)

# Plot training history
plot_training_history(history, output_dir)

