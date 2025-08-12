from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import shutil
import os

# from pipeline_utilities import valid_path, log_print

def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

SAMPLE_RATE = 16000  # VAD expects 16kHz
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')


def compute_vad_score(wav_path, model):
    wav = read_audio(wav_path, sampling_rate=SAMPLE_RATE)
    
    # Silero VAD expects chunks of 512 samples for 16kHz
    chunk_size = 512
    all_probs = []
    
    # Use no_grad() for inference
    with torch.no_grad():
        # Process audio in chunks of 512 samples
        for i in range(0, len(wav), chunk_size):
            chunk = wav[i:i + chunk_size]
            
            # Pad the last chunk if it's shorter than chunk_size
            if len(chunk) < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
            
            speech_prob = model(chunk.unsqueeze(0).to(DEVICE), SAMPLE_RATE).cpu().numpy()
            all_probs.extend(speech_prob)
    
    all_probs = np.array(all_probs)
    return np.mean(all_probs), np.max(all_probs)


# Load silero VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
(get_speech_ts, _, read_audio, _, _) = utils

model.to(DEVICE)

# Example usage
root_dir_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO','Unsupervised_Pipeline')
main_dir_ex = root_dir_ex / 'TestAO-Irma' / 'STG_1' / 'STG1_SHAS'
input_chunks_folder_ex = main_dir_ex / 'wav_chunks_raw_mini'
filtered_wavs_folder_ex = main_dir_ex / 'wav_chunks'
keep_perc_ex = 90

# Define the command-line arguments
parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--input_chunks_folder', type=valid_path, default=input_chunks_folder_ex, help='Folder with chunks of audio files')
parser.add_argument('--filtered_wavs_folder', type=valid_path, default=filtered_wavs_folder_ex ,help='Folder with filtered chunks files')
parser.add_argument('--keep_perc', type=int, default=90 , help='Percentage of files to keep based on mean probability')
args = parser.parse_args()

input_chunks_folder = args.input_chunks_folder
filtered_wavs_folder = args.filtered_wavs_folder

keep_perc = args.keep_perc  # Percentage of files to keep based on mean probability

# List of all wav files in wav_folder
wav_files = list(input_chunks_folder.glob('*.wav'))

mean_probs = []
max_probs = []
file_paths = []
perc_val = 100 - keep_perc  # Percentile for thresholding

print("Processing WAVs...")
for current_wav_path in wav_files:
    try:
        mean_p, max_p = compute_vad_score(current_wav_path, model)
        mean_probs.append(mean_p)
        max_probs.append(max_p)
        file_paths.append(current_wav_path)
    except Exception as e:
        print(f"Error with {current_wav_path.name}: {e}")

mean_probs = np.array(mean_probs)
max_probs = np.array(max_probs)

plt.hist(mean_probs, bins=50, alpha=0.7, label='Mean Speech Prob')
plt.hist(max_probs, bins=50, alpha=0.7, label='Max Speech Prob')
plt.axvline(np.percentile(mean_probs, perc_val), color='red', linestyle='--', label=f'{perc_val}th Percentile Threshold')
plt.legend()
plt.title("Speech Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.savefig(filtered_wavs_folder.joinpath('speech_prob_distribution.png'))

threshold = np.percentile(mean_probs, perc_val)
print(f"\n Suggested threshold (mean prob): {threshold:.3f}")

# Select the top percent of files with the highest mean probability
sorted_wavs_list_probs = sorted(zip(file_paths, mean_probs), key=lambda x: x[1], reverse=True)
sorted_wavs_list = [f[0] for f in sorted_wavs_list_probs]

# Calculate 90% of overlap files
num_files = int(len(sorted_wavs_list) * keep_perc / 100)

selected_single_speech = sorted_wavs_list[:num_files]

# Speech files
for current_src_path in selected_single_speech:
    src = current_src_path
    dst = filtered_wavs_folder.joinpath(current_src_path.name)
    shutil.copy2(src, dst)

print(f"Copied {num_files} files to {filtered_wavs_folder}, from total {len(sorted_wavs_list)} files.")

