import pickle
from pathlib import Path
import matplotlib.pyplot as plt

all_train_val_metrics_path = Path(r"C:\Users\luis2\Dropbox\DATASETS_AUDIO\Dvectors\TTS4_easy_40-200\EasySpeakers2to70C_mask04_lr-5_ep180_output\all_train_val_metrics.pickle")

with open(all_train_val_metrics_path, 'rb') as handle:
    all_train_val_metrics = pickle.load(handle)

all_train_losses, all_val_losses, all_train_accuracies, all_val_accuracies = all_train_val_metrics

# Plot all metrics in 4 subplots with N of speakers as X-axis

num_speakers = len(all_train_losses)
speaker_counts = list(range(2, 2 + num_speakers))  # X-axis: 2 up to N speakers

fig, axs = plt.subplots(1, 2, figsize=(16, 8))
axs[0].plot(speaker_counts, all_train_losses, label='Train Loss')
axs[0].plot(speaker_counts, all_val_losses, label='Val Loss')
axs[0].set_title('Loss')
axs[0].set_xlabel('Number of Speakers')
axs[0].legend()

axs[1].plot(speaker_counts, all_train_accuracies, label='Train Accuracy')
axs[1].plot(speaker_counts, all_val_accuracies, label='Val Accuracy')
axs[1].set_title('Accuracy')
axs[1].set_xlabel('Number of Speakers')
axs[1].legend()

# for ax in axs.flat:
#     ax.label_outer()

plt.tight_layout()
plt.show()

