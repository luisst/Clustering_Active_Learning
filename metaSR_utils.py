# This file includes routines for basic signal processing including framing and computing power spectra.
# Author: James Lyons 2012
import numpy as np
import decimal
import math
import logging
import scipy.io as sio
import scipy.io.wavfile
import pickle
import random
from pathlib import Path
import sys
import pandas as pd
import torch
import argparse
import os
import torch.nn.functional as F
from torch.autograd import Variable

from Stg2_models import background_resnet
from Stg2_models import background_resnet_ext
from pipeline_utilities import log_print, valid_path

USE_LOGSCALE = True
USE_NORM=  True
USE_DELTA = False
USE_SCALE = False
# Training context window size
NUM_WIN_SIZE = 200 # 200ms == 2 seconds
SHORT_SIZE = 100   # 100ms == 1 seconds

# Settings for feature extraction
USE_NORM = True
USE_SCALE = False


SAMPLE_RATE = 16000
FILTER_BANK = 40
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def rolling_window(a, window, step=1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]

def preemphasis(signal, coeff=0.95):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,)), stride_trick=True):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
            np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        win = np.tile(winfunc(frame_len), (numframes, 1))

    return frames * win

def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if np.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            np.shape(frames)[1], NFFT)
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)

def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))

def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)

def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the sample rate of the signal we are working with, in Hz. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

def fbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
          winfunc=lambda x:np.ones((x,))):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq= highfreq or samplerate/2
    signal = preemphasis(signal,preemph)
    frames = framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = powspec(frames,nfft)
    energy = np.sum(pspec,1) # this stores the total energy in each frame
    energy = np.where(energy == 0,np.finfo(float).eps,energy) # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = np.dot(pspec,fb.T) # compute the filterbank energies
    feat = np.where(feat == 0,np.finfo(float).eps,feat) # if feat is zero, we get problems with log

    return feat,energy

def delta(feat, N):
    """Compute delta features from a feature vector sequence.

    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat

def read_MFB_train(filename):
    with open(filename, 'rb') as f:
        feat_and_label = pickle.load(f)

    feature = feat_and_label['feat']  # size : (n_frames, dim=40)

    return feature

def read_MFB(filename):
    with open(filename, 'rb') as f:
        feat_and_label = pickle.load(f)
        
    feature = feat_and_label['feat'] # size : (n_frames, dim=40)
    label = feat_and_label['label']

    return feature, label

def read_feats2(feat_path_dir, n_shot, n_query, dataset_id='tts3', log_path=None):
    DB = pd.DataFrame()
    # List all files with *.pkl in directory, pathlib style
    DB['filename'] = list(feat_path_dir.glob('*.pkl'))

    DB['speaker_id'] = DB['filename'].apply(lambda x: x.stem.split('_')[1]) # speaker name
    DB['dataset_id'] = dataset_id # dataset name

    # Convert to string
    DB['filename'] = DB['filename'].astype(str)

    speaker_list = sorted(set(DB['speaker_id']))  # len(speaker_list) == n_speakers
    spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
    DB['labels'] = DB['speaker_id'].apply(lambda x: spk_to_idx[x])  # dataset folder name


    # Filter out speakers with less than n_shot + n_query samples
    speaker_counts = DB['labels'].value_counts()
    valid_speakers = speaker_counts[speaker_counts >= (n_shot + n_query)].index
    DB = DB[DB['labels'].isin(valid_speakers)]

    # Print the deleted speakers
    deleted_speakers = speaker_counts[speaker_counts < (n_shot + n_query)].index
    if len(deleted_speakers) > 0:
        log_print('Deleted speakers with less than {} samples: {}'.format(n_shot + n_query, ', '.join(deleted_speakers.astype(str))), log_path=log_path)
    else:
        log_print('No speakers were deleted. All speakers have at least {} samples.'.format(n_shot + n_query), log_path=log_path)

    # Update speaker list after filtering
    valid_speakers = sorted(set(DB['speaker_id']))
    spk_to_idx = {spk: i for i, spk in enumerate(valid_speakers)}
    DB['labels'] = DB['speaker_id'].apply(lambda x: spk_to_idx[x])  # dataset folder name
    num_speakers = len(DB['speaker_id'].unique())
    log_print('Found {} files with {} different speakers.'.format(str(len(DB)).zfill(7), str(num_speakers).zfill(5)), log_path=log_path)


    log_print('Filtered DB to {} files with {} speakers having at least {} samples each.'.format(
        str(len(DB)).zfill(7), str(len(valid_speakers)).zfill(5), n_shot + n_query), log_path=log_path)


    return DB, len(DB), num_speakers

class TruncatedInputfromMFB(object):
    """
    input size : (n_frames, dim=40)
    output size : (1, n_win=40, dim=40) => one context window is chosen randomly
    """

    def __init__(self, input_per_file=1):
        super(TruncatedInputfromMFB, self).__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features):
        # Normalizing before slicing
        network_inputs = []
        num_frames = len(frames_features)
        win_size = NUM_WIN_SIZE
        half_win_size = int(win_size / 2)
        # if num_frames - half_win_size < half_win_size:
        while num_frames <= win_size:
            frames_features = np.append(frames_features, frames_features[:num_frames, :], axis=0)
            num_frames = len(frames_features)

        for i in range(self.input_per_file):
            j = random.randrange(half_win_size, num_frames - half_win_size)
            if not j:
                frames_slice = np.zeros(num_frames, FILTER_BANK, 'float64')
                frames_slice[0:(frames_features.shape)[0]] = frames_features.shape
            else:
                frames_slice = frames_features[j - half_win_size:j + half_win_size]
            network_inputs.append(frames_slice)
        return np.array(network_inputs)

class ToTensorInput(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, np_feature):
        """
        Args:
            feature (numpy.ndarray): feature to be converted to tensor.
        Returns:
            Tensor: Converted feature.
        """
        if isinstance(np_feature, np.ndarray):
            # handle numpy array
            ten_feature = torch.from_numpy(np_feature.transpose((0,2,1))).float() # output type => torch.FloatTensor, fast
            
            # input size : (1, n_win=200, dim=40)
            # output size : (1, dim=40, n_win=200)
            return ten_feature



class ToTensorTestInput(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, np_feature):
        """
        Args:
            feature (numpy.ndarray): feature to be converted to tensor.
        Returns:
            Tensor: Converted feature.
        """
        if isinstance(np_feature, np.ndarray):
            # handle numpy array
            np_feature = np.expand_dims(np_feature, axis=0)
            np_feature = np.expand_dims(np_feature, axis=1)
            assert np_feature.ndim == 4, 'Data is not a 4D tensor. size:%s' %(np.shape(np_feature),)
            ten_feature = torch.from_numpy(np_feature.transpose((0,1,3,2))).float() # output type => torch.FloatTensor, fast
            # input size : (1, 1, n_win=200, dim=40)
            # output size : (1, 1, dim=40, n_win=200)
            return ten_feature


def normalize_frames(m,Scale=False):
    if Scale:
        return (m - np.mean(m, axis = 0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))


def extract_MFB_aolme(current_input_path, output_feats_folder):

    sr, audio = sio.wavfile.read(current_input_path)
    features, energies = fbank(audio, samplerate=SAMPLE_RATE, nfilt=FILTER_BANK, winlen=0.025, winfunc=np.hamming)

    if USE_LOGSCALE:
        features = 20 * np.log10(np.maximum(features,1e-5))
        
    if USE_DELTA:
        delta_1 = delta(features, N=1)
        delta_2 = delta(delta_1, N=1)
        
        features = normalize_frames(features, Scale=USE_SCALE)
        delta_1 = normalize_frames(delta_1, Scale=USE_SCALE)
        delta_2 = normalize_frames(delta_2, Scale=USE_SCALE)
        features = np.hstack([features, delta_1, delta_2])

    if USE_NORM:
        features = normalize_frames(features, Scale=USE_SCALE)
        total_features = features

    else:
        total_features = features


    curent_output_path = output_feats_folder / (current_input_path.stem + '.pkl')

    # Verify curent_input_path has 3 substrings separated by '_'
    current_label = current_input_path.stem.split('_')[-3]
    print(f'current_label: {current_label}')

    feat_and_label = {'feat':total_features, 'label':current_label}

    with open(curent_output_path, 'wb') as fp:
        pickle.dump(feat_and_label, fp)
    
    return current_label, total_features


def load_model_predict(pretrained_path, n_classes, use_cuda = True):
    model = background_resnet(num_classes=n_classes)

    if use_cuda:
        model.cuda()
    print('=> loading checkpoint')
    # load pre-trained parameters
    checkpoint = torch.load(pretrained_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def normalize_frames(m, Scale=False):
    if Scale:
        return (m - np.mean(m, axis = 0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))

def get_d_vector_aolme(filename, model, use_cuda = True, norm_flag = False):

    with open(filename, 'rb') as f:
        feat_and_label = pickle.load(f)
        
    input = feat_and_label['feat'] # size : (n_frames, dim=40)
    label = feat_and_label['label']

    input = normalize_frames(input, Scale=USE_SCALE)
    TT = ToTensorTestInput()  # torch tensor:(1, n_dims, n_frames)
    input = TT(input)  # size : (n_frames, 1, n_filter, T)
    input = Variable(input)
    with torch.no_grad():
        if use_cuda:
            #load gpu
            input = input.cuda()

        activation = model(input) #scoring function is cosine similarity so, you don't need to normalization

        if norm_flag:
            result_tensor = F.normalize(activation, p=2.0, dim=-1)
        else:
            result_tensor = activation

    return result_tensor, label


def d_vector_dict_lbls(list_of_feats, model, 
                       list_of_wavs,
                       norm_flag = False, use_pkl_label = False):

    if len(list_of_feats) != len(list_of_wavs):
        sys.exit('Error! Length of list_of_feats and wavs_paths are not the same')

    # Get enroll d-vector and test d-vector per utterance
    label_dict = {}
    with torch.no_grad():
        for path_idx, current_feat_path in enumerate(list_of_feats):
            enroll_embedding, pkl_label = get_d_vector_aolme(current_feat_path, model, norm_flag=norm_flag)
            if use_pkl_label:
                speakerID_clusters = pkl_label
            else:
                speakerID_clusters = 0

            # Get the current wav path
            current_wav_path = list_of_wavs[path_idx]
            if current_wav_path.stem != current_feat_path.stem:
                sys.exit('Error! Wav and Feat file names do not match')


            if speakerID_clusters in label_dict:
                label_dict[speakerID_clusters].append((enroll_embedding, current_wav_path))
            else:
                label_dict[speakerID_clusters] = [(enroll_embedding, current_wav_path)]

    return label_dict


def convert_dict_to_tensor(dict_data_input):
    # Initialize lists to store labels and concatenated data
    labels_list = []
    data_list = []
    path_list = []
    concatenated_tensor = torch.empty(0).cuda()

    # Iterate through the dictionary and concatenate tensors
    for label, tensor_and_path in dict_data_input.items():
        if len(tensor_and_path) != 0:
            # Repeat the label for each row in the tensor
            labels = [label] * len(tensor_and_path)
            
            # Append labels and data to the respective lists
            labels_list.extend(labels)
            for current_tuple in tensor_and_path:
                concatenated_tensor = torch.cat((concatenated_tensor, current_tuple[0]), dim=0)
                # data_list.append()
                path_list.append(str(current_tuple[1]))

    # # Concatenate the tensors in data_list along the first dimension (rows)
    # if len(data_list) != 0:
    #     X_data = torch.cat(data_list, dim=0)
    # else:
    #     X_data = torch.empty(0)

    speaker_labels_dict = dict([(y,x) for x,y in enumerate(sorted(set(labels_list)))])
    if 'noises' in speaker_labels_dict.keys():
        speaker_labels_dict['noises'] = 99 
    
    if 'spkNoise' in speaker_labels_dict.keys():
        speaker_labels_dict['spkNoise'] = 88 

    y_lbls = [speaker_labels_dict[x] for x in labels_list]
    y_data = np.array(y_lbls)

    return concatenated_tensor, y_data, path_list, speaker_labels_dict 



def separate_dict_embeddings(dict_embeddings, percentage_test,
                             return_paths = False,
                             verbose = False):
    # Calculate the total number of samples across all labels
    total_samples = sum(len(samples) for samples in dict_embeddings.values())
    
    # Calculate the label ratio based on the proportion of samples for each label
    labels_amounts = {label: len(samples) / total_samples for label, samples in dict_embeddings.items()}

    # Total number of samples in the Test
    test_samples = np.floor(percentage_test*total_samples)


    if verbose:
        print(f'\nTEST: Ratio of samples per class {labels_amounts}')

    for key in labels_amounts:
        labels_amounts[key] = int(np.floor(labels_amounts[key]*test_samples ))
    
    if verbose:
        print(f'TEST: Number of samples per class {labels_amounts}\n')


    # Initialize a dictionary to store the stratified samples
    dict_test_data = {}
    dict_train_data = {}

    # Perform stratified sampling for each label
    for label, samples in dict_embeddings.items():
        num_samples = len(samples)
        desired_num_samples = labels_amounts[label]

        # Check if there are enough samples for this label
        if num_samples >= desired_num_samples:
            test_indices = random.sample(range(num_samples), desired_num_samples)

            # List of indices that were not selected
            train_indices = [index for index in range(num_samples) if index not in test_indices]

            test_sampled_data = [samples[i] for i in test_indices]
            train_sampled_data = [samples[i] for i in train_indices]

        # Store the sampled data in the new dictionary
        dict_test_data[label] = test_sampled_data 

        # Define outliers
        if verbose:
            print(f'Key in prototypes: {label}')
        
        dict_train_data[label] = train_sampled_data 


    X_test, y_test, X_test_path, speaker_labels_dict_test = convert_dict_to_tensor(dict_test_data)
    X_train, y_train, X_train_path, speaker_labels_dict_train = convert_dict_to_tensor(dict_train_data)


    if return_paths:
        return X_train, y_train, X_train_path, X_test, y_test, X_test_path, speaker_labels_dict_train
    else:
        return X_train, y_train, X_test, y_test, speaker_labels_dict_train


def verify_matching_stems_to_file(wav_paths, pkl_paths, output_file):
    wav_stems = {Path(p).stem for p in wav_paths}
    pkl_stems = {Path(p).stem for p in pkl_paths}

    missing_pkl = wav_stems - pkl_stems
    missing_wav = pkl_stems - wav_stems

    lines = []
    if not missing_pkl and not missing_wav:
        lines.append("✅ All stems match between WAV and PKL lists.")
    else:
        if missing_pkl:
            lines.append("⚠ WAVs without matching PKL:")
            for stem in sorted(missing_pkl):
                lines.append(f"  - {stem}")
        if missing_wav:
            lines.append("⚠ PKLs without matching WAV:")
            for stem in sorted(missing_wav):
                lines.append(f"  - {stem}")

    # Write to file
    output_path = Path(output_file)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to {output_path}")

def d_vectors_pretrained_model(feats_folder, percentage_test,
                               wavs_paths, pretrained_path,
                               return_paths_flag = False,
                               norm_flag = False,
                               use_cuda=True,
                               use_pkl_label=False , verbose = False):

    list_of_feats = sorted(list(feats_folder.glob('*.pkl')))
    list_of_wavs = sorted(list(wavs_paths.glob('*.wav')))
    # n_classes = 5994 # from trained with vox2
    n_classes = int(pretrained_path.stem.split('_')[-1])

    # verify_matching_stems_to_file(list_of_wavs, list_of_feats, "stems_report.txt")

    ## load model from checkpoint
    model = load_model_predict(pretrained_path, n_classes, True)


    dict_embeddings = d_vector_dict_lbls(list_of_feats, model,
                                         list_of_wavs,
                                         norm_flag=norm_flag, use_pkl_label=use_pkl_label)

    # Extract keys from dict_embeddings
    list_of_keys = list(dict_embeddings.keys())

    # Export keys to a text file and counts of each file with that key
    output_keys_file = feats_folder.parent / 'dict_embeddings_keys_counts.txt'

    with open(output_keys_file, 'w', encoding='utf-8') as file:
        for key in list_of_keys:
            file.write(f"{key}: {len(dict_embeddings[key])}\n")

    return separate_dict_embeddings(dict_embeddings, 
                                    percentage_test,
                                    return_paths = return_paths_flag,
                                    verbose = verbose)
