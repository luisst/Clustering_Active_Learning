import os
from pathlib import Path
import sys
import argparse
import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_merged_hdf5_data(merged_h5_path):
    """
    Load merged samples HDF5 data including audio waveforms.

    Returns:
    --------
    dict with merged sample data
    """
    print(f"\nLoading merged HDF5 dataset: {merged_h5_path}")

    with h5py.File(merged_h5_path, 'r') as hf:
        n_merged = len(hf['merged_samples']['merged_unique_ids'])

        data = {
            'merged_unique_ids': [uid.decode() if isinstance(uid, bytes) else uid
                                 for uid in hf['merged_samples']['merged_unique_ids'][:]],
            'merged_wav_paths': [wp.decode() if isinstance(wp, bytes) else wp
                                for wp in hf['merged_samples']['merged_wav_paths'][:]],
            'cluster_labels': hf['merged_samples']['cluster_labels'][:],
            'gt_labels': hf['merged_samples']['gt_labels'][:],
            'n_constituents': hf['merged_samples']['n_constituents'][:],
            'audio_waveforms': [hf['merged_audio']['waveforms'][i][:]
                               for i in range(n_merged)],
            'sample_rates': hf['merged_audio']['sample_rates'][:]
        }

        print(f"✓ Loaded {len(data['merged_unique_ids'])} merged samples")
        print(f"  - Cluster distribution: {dict(zip(*np.unique(data['cluster_labels'], return_counts=True)))}")

    return data


def extract_features_from_audio_waveforms(merged_data):
    """
    Extract MFB features directly from audio waveforms in merged_data.

    Parameters:
    -----------
    merged_data : dict
        Merged sample data from HDF5 (including audio waveforms)

    Returns:
    --------
    dict with extracted features:
        - features_list: list of MFB feature arrays
        - unique_ids: list of sample unique IDs
    """
    print(f"\nExtracting MFB features from {len(merged_data['audio_waveforms'])} audio waveforms...")

    from metaSR_utils import fbank, normalize_frames, USE_LOGSCALE, USE_NORM, USE_SCALE, FILTER_BANK

    features_list = []
    failed_count = 0

    for i, (unique_id, audio, sample_rate) in enumerate(zip(
        merged_data['merged_unique_ids'],
        merged_data['audio_waveforms'],
        merged_data['sample_rates']
    )):
        if audio is None or len(audio) == 0:
            print(f"  Warning: No audio data for {unique_id}, skipping...")
            features_list.append(None)
            failed_count += 1
            continue

        try:
            # Extract MFB features (same as extract_MFB_aolme but from array)
            mfb_features, energies = fbank(audio, samplerate=int(sample_rate),
                                          nfilt=FILTER_BANK, winlen=0.025,
                                          winfunc=np.hamming)

            if USE_LOGSCALE:
                mfb_features = 20 * np.log10(np.maximum(mfb_features, 1e-5))

            if USE_NORM:
                mfb_features = normalize_frames(mfb_features, Scale=USE_SCALE)

            features_list.append(mfb_features)

            if (i + 1) % 50 == 0:
                print(f"  Extracted features for {i + 1}/{len(merged_data['audio_waveforms'])} samples...")

        except Exception as e:
            print(f"  Error extracting features for {unique_id}: {e}")
            features_list.append(None)
            failed_count += 1

    valid_count = len([f for f in features_list if f is not None])
    print(f"✓ Extracted features for {valid_count} samples")
    if failed_count > 0:
        print(f"  ⚠ Failed to extract {failed_count} features")

    return {
        'features_list': features_list,
        'unique_ids': merged_data['merged_unique_ids']
    }


def compute_dvectors_from_features(features_data, pretrained_path):
    """
    Compute D-vectors directly from MFB feature arrays using pretrained model.

    Parameters:
    -----------
    features_data : dict
        Dictionary with 'features_list' and 'unique_ids'
    pretrained_path : Path
        Path to pretrained D-vector model

    Returns:
    --------
    tuple: (X_data, unique_ids, y_labels)
        - X_data: numpy array of D-vectors (n_samples, n_features)
        - unique_ids: list of sample unique IDs
        - y_labels: numpy array of labels (all zeros for merged samples)
    """
    print(f"\nComputing D-vectors using pretrained model: {pretrained_path}")

    import torch
    import torch.nn.functional as F
    from torch.autograd import Variable
    from metaSR_utils import load_model_predict, normalize_frames, ToTensorTestInput, USE_SCALE

    # Load model
    n_classes = int(pretrained_path.stem.split('_')[-1])
    model = load_model_predict(pretrained_path, n_classes, use_cuda=True)

    dvector_list = []
    valid_ids = []
    failed_count = 0

    TT = ToTensorTestInput()

    for i, (unique_id, features) in enumerate(zip(
        features_data['unique_ids'],
        features_data['features_list']
    )):
        if features is None:
            print(f"  Warning: No features for {unique_id}, skipping...")
            failed_count += 1
            continue

        try:
            # Normalize features
            input_features = normalize_frames(features, Scale=USE_SCALE)

            # Convert to tensor
            input_tensor = TT(input_features)
            input_tensor = Variable(input_tensor)

            with torch.no_grad():
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()

                # Get D-vector from model
                activation = model(input_tensor)
                result_tensor = F.normalize(activation, p=2.0, dim=-1)

                dvector_list.append(result_tensor.cpu().numpy())
                valid_ids.append(unique_id)

            if (i + 1) % 50 == 0:
                print(f"  Computed D-vectors for {i + 1}/{len(features_data['unique_ids'])} samples...")

        except Exception as e:
            print(f"  Error computing D-vector for {unique_id}: {e}")
            failed_count += 1

    # Stack all D-vectors
    X_dvectors = np.vstack(dvector_list)

    # Create dummy labels
    y_labels = np.zeros(len(valid_ids), dtype='int32')

    # Fill y_labels with the debug value 77
    y_labels.fill(77)

    print(f"✓ Computed D-vectors")
    print(f"  - Shape: {X_dvectors.shape}")
    print(f"  - Number of samples: {len(valid_ids)}")
    if failed_count > 0:
        print(f"  ⚠ Failed to compute {failed_count} D-vectors")

    return X_dvectors, valid_ids, y_labels


def update_merged_hdf5_with_recalculated_features(
    merged_h5_path,
    recalculated_dvectors,
    recalculated_unique_ids,
    recalculated_labels
):
    """
    Update merged HDF5 dataset with recalculated D-vector features.

    Parameters:
    -----------
    merged_h5_path : Path
        Path to merged HDF5 file
    recalculated_dvectors : np.ndarray
        Recalculated D-vectors (n_samples, n_features)
    recalculated_unique_ids : list
        List of unique IDs for recalculated samples
    recalculated_labels : np.ndarray
        Labels from recalculation
    """
    print(f"\nUpdating merged HDF5 with recalculated features: {merged_h5_path}")

    with h5py.File(merged_h5_path, 'a') as hf:
        # Load existing unique IDs
        existing_ids = [uid.decode() if isinstance(uid, bytes) else uid
                       for uid in hf['merged_samples']['merged_unique_ids'][:]]

        # Create index mapping
        id_to_new_idx = {uid: i for i, uid in enumerate(recalculated_unique_ids)}

        # Reorder recalculated features to match HDF5 order
        n_merged = len(existing_ids)
        n_features = recalculated_dvectors.shape[1]
        reordered_dvectors = np.zeros((n_merged, n_features), dtype='float32')
        reordered_labels = np.zeros(n_merged, dtype='int32')

        matched_count = 0
        for i, unique_id in enumerate(existing_ids):
            if unique_id in id_to_new_idx:
                new_idx = id_to_new_idx[unique_id]
                reordered_dvectors[i] = recalculated_dvectors[new_idx]
                reordered_labels[i] = recalculated_labels[new_idx]
                matched_count += 1
            else:
                print(f"  Warning: No recalculated features for {unique_id}")

        print(f"  Matched {matched_count}/{n_merged} samples")

        # Create or update /recalculated_features/ group
        if 'recalculated_features' in hf:
            del hf['recalculated_features']

        recalc_group = hf.create_group('recalculated_features')

        # Store recalculated D-vectors
        recalc_group.create_dataset(
            'dvectors',
            data=reordered_dvectors,
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )

        # Store recalculated labels
        recalc_group.create_dataset(
            'recalc_labels',
            data=reordered_labels,
            dtype='int32',
            compression='gzip',
            compression_opts=4
        )

        # Add metadata
        recalc_group.attrs['n_features'] = n_features
        recalc_group.attrs['n_samples'] = n_merged
        recalc_group.attrs['matched_samples'] = matched_count
        recalc_group.attrs['source'] = 'STG3E_RECALC_MERGED_FEATS'
        recalc_group.attrs['description'] = 'Recalculated D-vector features for merged samples'

        print(f"✓ Updated HDF5 with recalculated features")
        print(f"  - /recalculated_features/dvectors: {reordered_dvectors.shape}")
        print(f"  - /recalculated_features/recalc_labels: {reordered_labels.shape}")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================
base_path_ex = Path.home().joinpath('Dropbox', 'DATASETS_AUDIO', 'Unsupervised_Pipeline', 'TestAO-Irma')
stg3_folder_ex = base_path_ex.joinpath('STG_3', 'STG3_EXP011-SHAS-DV-hdb')
merged_h5_ex = stg3_folder_ex / 'merged_dataset.h5'

# Subfolders for Stage 4 (recalculated features)
stg4_folder_ex = base_path_ex.joinpath('STG_4', 'STG4_EXP011-SHAS-DV-hdb')

pretrained_path_ex = Path.home().joinpath('Dropbox', 'Source_2025', 'pre-trained', 'checkpoint_100_original_5994.pth')

parser = argparse.ArgumentParser(
    description='Stage 3e: Recalculate features for merged samples'
)
parser.add_argument(
    '--merged_dataset_h5',
    default=merged_h5_ex,
    help='Input path for merged samples HDF5 dataset (contains audio waveforms)'
)
parser.add_argument(
    '--pretrained_model_path',
    default=pretrained_path_ex,
    help='Path to pretrained D-vector model'
)

args = parser.parse_args()
merged_h5_path = Path(args.merged_dataset_h5)
pretrained_path = Path(args.pretrained_model_path)

# Verify paths exist
if not merged_h5_path.exists():
    sys.exit(f"Error: Merged HDF5 file not found: {merged_h5_path}")

if not pretrained_path.exists():
    sys.exit(f"Error: Pretrained model not found: {pretrained_path}")

# ============================================================================
# MAIN WORKFLOW
# ============================================================================
print("=" * 80)
print("STAGE 3E: RECALCULATE FEATURES FOR MERGED SAMPLES")
print("=" * 80)

# Step 1: Load merged HDF5 data (including audio waveforms)
merged_data = load_merged_hdf5_data(merged_h5_path)

# Step 2: Extract MFB features directly from audio waveforms in memory
features_data = extract_features_from_audio_waveforms(merged_data)

# Step 3: Compute D-vectors directly from feature arrays
recalculated_dvectors, recalculated_unique_ids, recalculated_labels = compute_dvectors_from_features(
    features_data,
    pretrained_path
)

# Step 4: Update merged HDF5 with recalculated features
update_merged_hdf5_with_recalculated_features(
    merged_h5_path,
    recalculated_dvectors,
    recalculated_unique_ids,
    recalculated_labels
)

print("\n" + "=" * 80)
print("STAGE 3E COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"Merged HDF5 updated with recalculated features: {merged_h5_path}")
print(f"  - Recalculated D-vectors stored in: /recalculated_features/dvectors")
print(f"  - All processing done in memory (no intermediate files created)")
