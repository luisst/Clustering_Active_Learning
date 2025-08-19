
import torch
import torchvision.transforms as transforms
from metaSR_utils import TruncatedInputfromMFB, ToTensorInput, FILTER_BANK
import random
import numpy as np
import pprint
import pickle
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
# from pipeline_utilities import log_print

def log_print(*args, **kwargs):
    """Prints to stdout and also logs to log_path."""

    log_path = kwargs.pop('lp', 'default_log.txt')
    print_to_console = kwargs.pop('print', True)

    message = " ".join(str(a) for a in args)
    if print_to_console:
        print(message)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")

class metaGenerator(object):

    def __init__(self, data_DB, file_loader, nb_classes=100, nb_samples_per_class=3,
                  max_iter=100, xp=np):
        super(metaGenerator, self).__init__()

        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.max_iter = max_iter
        self.xp = xp
        self.num_iter = 0
        self.data = self._load_data(data_DB)
        self.file_loader = file_loader
        self.transform = transforms.Compose([
            TruncatedInputfromMFB(),  # numpy array:(1, n_frames, n_dims)
            ToTensorInput()  # torch tensor:(1, n_dims, n_frames)
        ])


    def _load_data(self, data_DB):
        # Filter groups with sufficient samples
        filtered_data = data_DB.groupby('labels').filter(lambda x: len(x) >= self.nb_samples_per_class)
        
        # Assign keys to a consecutive range of integers
        data_dict = {
            idx: group['filename'].values
            for idx, (label, group) in enumerate(filtered_data.groupby('labels'))
        }
        
        return data_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            images, labels = self.sample(self.nb_classes, self.nb_samples_per_class)

            return (self.num_iter - 1), (images, labels)
        else:
            raise StopIteration()

    def sample(self, nb_classes, nb_samples_per_class):

        picture_list = sorted(set(self.data.keys()))
        sampled_characters = random.sample(list(self.data.keys()), nb_classes)
        labels_and_images = []
        for (k, char) in enumerate(sampled_characters):
            label = picture_list[char]
            _imgs = self.data[char]
            _ind = random.sample(range(len(_imgs)), nb_samples_per_class)
            labels_and_images.extend([(label, self.transform(self.file_loader(_imgs[i]))) for i in _ind])
        arg_labels_and_images = []
        for i in range(self.nb_samples_per_class):
            for j in range(self.nb_classes):
                arg_labels_and_images.extend([labels_and_images[i+j*self.nb_samples_per_class]])

        labels, images = zip(*arg_labels_and_images)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        return images, labels


class metaGenerator_test(object):

    def __init__(self, test_DB, file_loader, enroll_length, test_length,
                 nb_classes=100, n_support=1, n_query=2, max_iter=100, xp=np):
        super(metaGenerator_test, self).__init__()

        self.nb_classes = nb_classes
        self.n_support = n_support
        self.n_query = n_query
        self.nb_samples_per_class = n_support+ n_query

        self.enroll_length = enroll_length
        self.test_length = test_length

        self.max_iter = max_iter
        self.xp = xp
        self.num_iter = 0
        self.test_data = self._load_data(test_DB)
        self.file_loader = file_loader
        self.transform = transforms.Compose([
            ToTensorInput()  # torch tensor:(1, n_dims, n_frames)
        ])

    def _load_data(self, data_DB):
        nb_speaker = len(set(data_DB['labels']))

        return {key: np.array(data_DB.loc[data_DB['labels']==key]['filename']) for key in range(nb_speaker)}

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            images, labels, filenames = self.sample(self.nb_classes, self.nb_samples_per_class)

            return (self.num_iter - 1), (images, labels, filenames)
        else:
            raise StopIteration()

    def cut_frames(self, frames_features, mode='enroll'):
        # Normalizing before slicing
        network_inputs = []
        num_frames = len(frames_features)

        if mode == 'enroll': win_size = self.enroll_length
        elif mode == 'test': win_size = self.test_length

        half_win_size = int(win_size / 2)
        # if num_frames - half_win_size < half_win_size:
        while num_frames <= win_size:
            frames_features = np.append(frames_features, frames_features[:num_frames, :], axis=0)
            num_frames = len(frames_features)


        j = random.randrange(half_win_size, num_frames - half_win_size)
        if not j:
            frames_slice = np.zeros(num_frames, FILTER_BANK, 'float64')
            frames_slice[0:(frames_features.shape)[0]] = frames_features.shape
        else:
            frames_slice = frames_features[j - half_win_size:j + half_win_size]
        network_inputs.append(frames_slice)

        return np.array(network_inputs)

    def sample(self, nb_classes, nb_samples_per_class):

        picture_list = sorted(set(self.test_data.keys()))
        sample_classes = random.sample(list(self.test_data.keys()), nb_classes)
        labels_and_images = []
        # print(f'Sampling {nb_classes} classes with {nb_samples_per_class} samples each.')
        for (k, char) in enumerate(sample_classes):
            label = picture_list[char]
            # support(Enroll data) / query(Test data)
            data = self.test_data[char]
            # print(f'\tClass {k+1}/{nb_classes}. label:{label} with {len(data)} samples.')
            _ind = random.sample(range(len(data)), nb_samples_per_class)
            # sample support
            current_sample_list = []
            for i in _ind[:self.n_support]:
                single_data_loaded = self.file_loader(data[i])[0]
                frames_cut_loaded = self.cut_frames(single_data_loaded, mode='enroll')
                current_filename = data[i].split('/')[-1]  # Extract filename from path
                current_sample_list.append((label, self.transform(frames_cut_loaded), current_filename))
            labels_and_images.extend(current_sample_list)
            # sample query
            current_query_list = []
            for i in _ind[self.n_support:]:
                single_data_loaded = self.file_loader(data[i])[0]
                frames_cut_loaded = self.cut_frames(single_data_loaded, mode='test')
                current_filename = data[i].split('/')[-1]  # Extract filename from path
                current_query_list.append((label, self.transform(frames_cut_loaded), current_filename))
            labels_and_images.extend(current_query_list)

            # labels_and_images.extend([(label, self.transform(self.cut_frames(self.file_loader(data[i])[0], mode='test'))) for i in _ind[self.n_support:]])

        arg_labels_and_images = []
        for i in range(self.nb_samples_per_class):
            for j in range(self.nb_classes):
                arg_idx = i + j * self.nb_samples_per_class
                arg_labels_and_images.extend([labels_and_images[arg_idx]])
                # print(f'\rProcessing {arg_idx + 1}/{self.nb_samples_per_class * self.nb_classes} samples.')

        labels, images, filenames = zip(*arg_labels_and_images)

        support = torch.stack(images[:self.n_support * self.nb_classes], dim=0)
        query = torch.stack(images[self.n_support*self.nb_classes:], dim=0)

        labels = torch.tensor(labels, dtype=torch.long)

        return (support, query), labels, filenames


def custom_collate_fn(batch):
    """
    Custom collate function to handle batches containing tensors and path objects.
    
    Args:
        batch: List of tuples (features, labels, wav_path)
        
    Returns:
        Tuple of (batched_features, batched_labels, list_of_paths)
    """
    features, labels, wav_paths = zip(*batch)
    
    # Stack tensors normally
    batched_features = torch.stack(features)
    batched_labels = torch.stack(labels)
    
    # Convert paths to strings to avoid cross-platform issues
    wav_paths_str = []
    for path in wav_paths:
        if hasattr(path, '__fspath__'):  # It's a path-like object
            wav_paths_str.append(str(path))
        else:  # It's already a string
            wav_paths_str.append(path)
    
    return batched_features, batched_labels, wav_paths_str

def safe_load_pickle_with_paths(pickle_path):
    """
    Safely load a pickle file that may contain path objects from different platforms.
    
    Args:
        pickle_path: Path to the pickle file
        
    Returns:
        The loaded data with paths converted to strings
    """
    from pathlib import Path, PosixPath, WindowsPath
    
    # Custom unpickler to handle cross-platform path issues
    class CrossPlatformUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'pathlib' and name in ['PosixPath', 'WindowsPath']:
                # Convert any path to generic Path which works cross-platform
                return Path
            return super().find_class(module, name)
    
    try:
        with open(pickle_path, 'rb') as f:
            return CrossPlatformUnpickler(f).load()
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        # Fallback to regular pickle load
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            
        # Convert any path objects to strings
        def convert_paths_to_strings(obj):
            if hasattr(obj, '__fspath__'):  # It's a path-like object
                return str(obj)
            elif isinstance(obj, list):
                return [convert_paths_to_strings(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_paths_to_strings(item) for item in obj)
            else:
                return obj
                
        return convert_paths_to_strings(data)


class DataAugmentor:
    """
    Data augmentation class for speaker features.
    """
    
    def __init__(self, noise_std=0.01, mask_prob=0.1):
        self.noise_std = noise_std
        self.mask_prob = mask_prob
    
    def add_noise(self, features):
        """
        Add Gaussian noise to features.
        
        Args:
            features: torch tensor of shape (batch_size, feature_dim)
            
        Returns:
            Noisy features
        """
        noise = torch.randn_like(features) * self.noise_std
        return features + noise
    
    def apply_mask(self, features):
        """
        Apply random masking to features.
        
        Args:
            features: torch tensor of shape (batch_size, feature_dim)
            
        Returns:
            Masked features
        """
        mask = torch.rand_like(features) > self.mask_prob
        return features * mask.float()
    
    def augment(self, features, apply_noise=True, apply_mask=True):
        """
        Apply data augmentation to features.
        
        Args:
            features: torch tensor of shape (batch_size, feature_dim)
            apply_noise: whether to add noise
            apply_mask: whether to apply masking
            
        Returns:
            Augmented features
        """
        augmented = features.clone()
        
        if apply_noise:
            augmented = self.add_noise(augmented)
        
        if apply_mask:
            augmented = self.apply_mask(augmented)
            
        return augmented


class AudioDataset(Dataset):
    def __init__(self, features, labels, wavs_paths, augmentor=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.wavs_paths = wavs_paths
        self.augmentor = augmentor
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        labels = self.labels[idx]
        wav_path = self.wavs_paths[idx]
        
        # Apply augmentation if augmentor is provided
        if self.augmentor is not None:
            features = self.augmentor.augment(features)

        return features, labels, wav_path


def create_dataloaders(pickle_file_path, log_path, batch_size=32, test_size=0.2, random_state=42, num_workers=4, 
                      augment_training=True, noise_std=0.01, mask_prob=0.1, drop_last=True):
    """
    Create training and validation DataLoaders from a pickle file.
    
    Args:
        pickle_file_path (str): Path to the pickle file containing 'features', 'wav_paths', 'labels'
        batch_size (int): Batch size for DataLoaders
        test_size (float): Proportion of data to use for validation (default: 0.2 for 80-20 split)
        random_state (int): Random seed for reproducible splits
        num_workers (int): Number of worker processes for data loading
        augment_training (bool): Whether to apply data augmentation to training data
        noise_std (float): Standard deviation for Gaussian noise in augmentation
        mask_prob (float): Probability of masking features in augmentation
    
    Returns:
        tuple: (train_loader, val_loader)
    """

    # Load the pickle file safely
    features, wavs_paths, labels = safe_load_pickle_with_paths(pickle_file_path)

    # Convert to numpy arrays if they aren't already
    if not isinstance(features, np.ndarray):
        features = np.array(features)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    feature_dim = features.shape[1]
    num_speakers = len(set(labels))  # Assuming labels are speaker IDs
    
    # Split the data with shuffling
    X_train, X_val, y_train, y_val, wavs_train, wavs_val = train_test_split(
        features, labels, wavs_paths, 
        test_size=test_size, 
        shuffle=True,
        stratify=labels  # Ensures balanced split across classes
    )

    # Print all the distinct speakers in y_train
    log_print(f"Distinct speakers in training set: {set(y_train)}", lp=log_path)
    log_print(f"Distinct speakers in validation set: {set(y_val)}", lp=log_path)

    # Create 2 dictionaries with speaker label and number of speakers in each label
    speaker_counts_train = {label: 0 for label in set(y_train)}
    speaker_counts_val = {label: 0 for label in set(y_val)}

    for label in y_train:
        speaker_counts_train[label] += 1

    for label in y_val:
        speaker_counts_val[label] += 1
    
    # Log_print each value from the dictionaries
    log_print("Speaker counts in training set:", lp=log_path)
    for label, count in speaker_counts_train.items():
        log_print(f"  - Speaker {label}: {count} samples", lp=log_path)

    log_print("\n\nSpeaker counts in validation set:", lp=log_path)
    for label, count in speaker_counts_val.items():
        log_print(f"  - Speaker {label}: {count} samples", lp=log_path)

    # Create augmentor for training data if requested
    augmentor = DataAugmentor(noise_std=noise_std, mask_prob=mask_prob) if augment_training else None
    
    # Create datasets
    train_dataset = AudioDataset(X_train, y_train, wavs_train, augmentor=augmentor)
    val_dataset = AudioDataset(X_val, y_val, wavs_val, augmentor=None)  # No augmentation for validation

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Additional shuffling for training
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        collate_fn=custom_collate_fn,
        drop_last=drop_last
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        drop_last=drop_last
    )
    
    train_samples_msg = f"Training samples: {len(train_dataset)}"
    val_samples_msg = f"Validation samples: {len(val_dataset)}"
    features_shape_msg = f"Features shape: {features.shape}"
    labels_shape_msg = f"Labels shape: {labels.shape}"
    augmentation_msg = f"Data augmentation: {'Enabled' if augment_training else 'Disabled'}"

    log_print(train_samples_msg, lp=log_path)
    log_print(val_samples_msg, lp=log_path)
    log_print(features_shape_msg, lp=log_path)
    log_print(labels_shape_msg, lp=log_path)
    log_print(augmentation_msg, lp=log_path)

    if augment_training:
        noise_msg = f"  - Noise std: {noise_std}"
        mask_msg = f"  - Mask probability: {mask_prob}"
        log_print(noise_msg, lp=log_path)
        log_print(mask_msg, lp=log_path)

    return train_loader, val_loader, feature_dim, num_speakers

def inference_dataloader(feats_pickle_path, batch_size=32, num_workers=4):

    # Load the pickle file safely
    features, wavs_paths, labels = safe_load_pickle_with_paths(feats_pickle_path)

    feature_dim = features.shape[1]
    num_speakers = len(set(labels))  # Assuming labels are speaker IDs

    # Convert to numpy arrays if they aren't already
    if not isinstance(features, np.ndarray):
        features = np.array(features)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    test_dataset = AudioDataset(features, labels, wavs_paths, augmentor=None)  # No augmentation for test
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    return test_loader, labels, wavs_paths, feature_dim, num_speakers