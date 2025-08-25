from pathlib import Path
import pickle
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import warnings

from Stg2_models import SimpleClassifier
from Stg2_dataloaders import inference_dataloader

warnings.filterwarnings('ignore', category=FutureWarning)
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
main_folder_path = root_path / 'TTS4_clean_40-300'
inference_feats_pickle_path_ex = root_path / 'wavs_test_pairs' / 'd_vectors_aolme.pickle'

run_id_ex = 'S1_Bal300Clean_sc2'

# Create output folder in the same directory as feats_pickle_path
output_folder_ex = main_folder_path / f'{run_id_ex}_output'
num_speakers_pretrained_ex = 73  
model_pretrained_path_ex = output_folder_ex / f'model_.pth'
output_enhanced_pickle_ex = output_folder_ex / f'enhanced_{run_id_ex}.pickle'

parser = argparse.ArgumentParser()
parser.add_argument('--inference_feats_pickle', default=inference_feats_pickle_path_ex, help='Path to the folder to store the D-vectors features')
parser.add_argument('--pretrained_model_path', default=model_pretrained_path_ex, help='Path to the pretrained model')
parser.add_argument('--enhanced_feats_pickle', default=output_enhanced_pickle_ex, help='Path to the folder to store the predictions')
parser.add_argument('--run_id', default=run_id_ex, help='string with the experiment name')
args = parser.parse_args()

inference_feats_pickle_path = Path(args.inference_feats_pickle)
model_pretrained_path = Path(args.pretrained_model_path)
enhanced_feats_and_labels = Path(args.enhanced_feats_pickle)
run_id = args.run_id

batch_size = 16
num_workers = 4

output_folder_path = enhanced_feats_and_labels.parent 
log_path = output_folder_path / 'inference_log.txt'

number_speakers_pretrained = int(model_pretrained_path.stem.split('_')[-1])
print(f'Loaded {number_speakers_pretrained} number of speakers from pre-trained model')

log_print(f"Using pretrained model: {model_pretrained_path}", lp=log_path)

inference_loader, pickle_labels, wavs_paths, feature_dim, num_speakers_dataloader = inference_dataloader(inference_feats_pickle_path, batch_size=batch_size, num_workers=num_workers)

model = SimpleClassifier(dim=feature_dim, hidden_dim=256, num_classes=number_speakers_pretrained).to(DEVICE)

checkpoint = torch.load(model_pretrained_path)
model.load_state_dict(checkpoint)
log_print("Loaded best model state", lp=log_path)

log_print("=== Inference model ===", lp=log_path)
model.eval()
enhanced_features_all = []
with torch.no_grad():
    for batch_x, batch_labels, wav_paths in tqdm(inference_loader, desc="Inference run"):
        batch_x = batch_x.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)

        refined_x = model(batch_x)
        enhanced_features_all.extend(refined_x.cpu().numpy())

X_data_and_labels = [enhanced_features_all, wavs_paths, pickle_labels]
with open(f'{enhanced_feats_and_labels}', "wb") as file:
    pickle.dump(X_data_and_labels, file)