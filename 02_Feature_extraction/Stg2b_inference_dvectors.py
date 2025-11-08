from __future__ import print_function
import warnings
from pathlib import Path
import warnings
import pickle
import argparse
import sys


sys.path.insert(0, str(Path(__file__).parent.parent))
from metaSR_utils import d_vectors_pretrained_model
from pipeline_utilities import valid_path

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings('ignore', category=FutureWarning)


min_cluster_size = 0
pca_elem = 0
hdb_mode = None
min_samples = 0

root_ex = Path('/home/luis/Dropbox/DATASETS_AUDIO')
wavs_folder_ex = root_ex / Path('Dvectors/TTS4_easy_40-200/input_wavs')
mfcc_folder_ex = root_ex / Path('Dvectors/TTS4_easy_40-200/input_feats')
feats_pickle_ex = mfcc_folder_ex.parent / Path('dvec_easy40-200.pickle')
pretrained_path_ex = root_ex.parent / 'Source_2025' / 'pre-trained' / f'checkpoint_100_original_5994.pth'
use_pkl_label_ex = False

parser = argparse.ArgumentParser()
parser.add_argument('--wavs_folder', type=valid_path, default=wavs_folder_ex, help='Path to the folder to input chunks wavs paths')
parser.add_argument('--input_mfcc_folder', type=valid_path, default=mfcc_folder_ex, help='Path to the folder to load the mfcc feats')
parser.add_argument('--output_feats_pickle', default=feats_pickle_ex, help='Path to the folder to store the D-vectors features')
parser.add_argument('--pretrained_model_path', default=pretrained_path_ex, help='Path to pretrained Dvector model')
parser.add_argument('--use_pkl_label', default=use_pkl_label_ex, help='Use pickle label for speaker ID extraction')
args = parser.parse_args()

#TODO: samples_flag is set to True by default for inferences

wavs_folder = Path(args.wavs_folder)
mfcc_folder_path = Path(args.input_mfcc_folder)
feats_pickle_path = Path(args.output_feats_pickle)
use_pkl_label = args.use_pkl_label


pretrained_path = Path(args.pretrained_model_path)

percentage_test = 0.0

print(f'Using pickle label: {use_pkl_label}')

dataset_dvectors = d_vectors_pretrained_model(mfcc_folder_path, percentage_test,
                                            wavs_folder,
                                            pretrained_path,
                                            return_paths_flag = True,
                                            norm_flag = True,
                                            use_cuda=True,
                                            use_pkl_label=use_pkl_label)

X_train = dataset_dvectors[0]
y_train = dataset_dvectors[1]
X_train_paths = dataset_dvectors[2]
X_test = dataset_dvectors[3]
y_test = dataset_dvectors[4]
X_test_paths = dataset_dvectors[5]
speaker_labels_dict_train = dataset_dvectors[6]

X_test = X_test.cpu().numpy()
X_train = X_train.cpu().numpy()

Mixed_X_data = X_train
Mixed_y_labels = y_train

sample_path = X_train_paths[0]

X_data_and_labels = [X_train, X_train_paths, y_train]
with open(f'{feats_pickle_path}', "wb") as file:
    pickle.dump(X_data_and_labels, file)

# Export dictionary keys and values speaker_labels_dict_train to a text file
list_of_keys = list(speaker_labels_dict_train.keys())
list_of_values = list(speaker_labels_dict_train.values())
dict_path = feats_pickle_path.parent / 'speaker_labels_dict.txt'
with open(dict_path, 'w', encoding='utf-8') as f:
    for key, value in zip(list_of_keys, list_of_values):
        f.write(f"{key}: {value}\n")

