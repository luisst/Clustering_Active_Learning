import torch
import sys
from pathlib import Path
import pickle

from metaSR_utils import load_model_predict, get_d_vector_aolme, separate_dict_embeddings, extract_label
import random
from pathlib import Path
from collections import defaultdict
from typing import List

def progressive_speaker_sampling(folder):
    """
    Cumulatively samples speakers from a folder of pkl files.
    Starts with 2 random speakers, then keeps adding one more speaker each time.
    
    Args:
        folder_path: Path to folder containing pkl files with format 'DX_speakerID_index.pkl'
    
    Returns:
        List of lists, where each inner list contains Path objects for pkl files
        from a growing set of speakers (2, then those same 2 + 1 more, etc.)
    """
    
    # Get all pkl files
    pkl_files = list(folder.glob("*.pkl"))
    
    if not pkl_files:
        return []
    
    # Group files by speaker ID
    speaker_files = defaultdict(list)
    
    for file_path in pkl_files:
        # Parse filename: DX_speakerID_index.pkl
        parts = file_path.stem.split('_')
        if len(parts) >= 2:
            speaker_id = parts[1]  # Second substring is speaker ID
            speaker_files[speaker_id].append(file_path)
    
    # Get all unique speaker IDs
    all_speakers = list(speaker_files.keys())
    total_speakers = len(all_speakers)
    
    if total_speakers < 2:
        return []
    
    # Shuffle speakers for random sampling
    random.shuffle(all_speakers)
    
    result = []
    
    # Create groups by progressively adding speakers
    for current_idx in range(2, total_speakers + 1):
        # Collect all pkl files from currently selected speakers
        group_files = []
        for speaker_id in all_speakers[:current_idx]:
            group_files.extend(speaker_files[speaker_id])
        
        # Print the speakers IDs used
        print(f"Current speaker group (IDs): {all_speakers[:current_idx]}")

        result.append(group_files)
        
    return result


feats_folder = Path("/home/luis/Dropbox/DATASETS_AUDIO/Dvectors/TTS4_easy_40-200/input_feats/")
pretrained_path = Path("/home/luis/Dropbox/Source_2025/pre-trained/checkpoint_100_original_5994.pth")

output_folder = feats_folder.parent / 'iterative_speakers'

# Create output folder if it doesn't exist
output_folder.mkdir(exist_ok=True)

lists_feats_list = progressive_speaker_sampling(feats_folder)
n_classes = int(pretrained_path.stem.split('_')[-1])

## load model from checkpoint
model = load_model_predict(pretrained_path, n_classes, use_cuda=True)

num_speakers = 2

for list_of_feats in lists_feats_list:

    # Print length of current feature list
    print(f"Number of features for current speaker group: {len(list_of_feats)}")
    
    current_pickle_feats_path = output_folder / f"features_group_{num_speakers:02d}_{len(list_of_feats):05d}.pkl"

    num_speakers += 1

    # Get enroll d-vector and test d-vector per utterance
    label_dict = {}
    with torch.no_grad():
        for path_idx, current_feat_path in enumerate(list_of_feats):
            enroll_embedding, _ = get_d_vector_aolme(current_feat_path, model, norm_flag=True, use_cuda=True)
            speakerID_clusters = extract_label(current_feat_path, samples_flag=False)

            # Get the current wav path
            new_filename_wav = current_feat_path.stem + '.wav'
            current_wav_path = current_feat_path.parent / 'input_wavs' / new_filename_wav

            if speakerID_clusters in label_dict:
                label_dict[speakerID_clusters].append((enroll_embedding, current_wav_path))
            else:
                label_dict[speakerID_clusters] = [(enroll_embedding, current_wav_path)]


    current_package = separate_dict_embeddings(label_dict, 
                                    0.0,
                                    return_paths = True,
                                    verbose = False)

    X_data = current_package[0].cpu().numpy()
    y_data = current_package[1]
    X_paths_dummy = current_package[2]

    X_data_and_labels = [X_data, X_paths_dummy, y_data]

    with open(f'{current_pickle_feats_path}', "wb") as file:
        pickle.dump(X_data_and_labels, file)