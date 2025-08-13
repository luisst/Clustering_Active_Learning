from pathlib import Path
import pickle
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import re

from Stg2_models import SimpleClassifier
from Stg2_dataloaders import inference_dataloader

from pipeline_utilities import log_print, valid_path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

root_path = Path.home() / 'Dropbox' / 'DATASETS_AUDIO' / 'Dvectors'
main_folder_path = root_path / 'TTS4_clean_40-300'
inference_feats_pickle_path_ex = root_path / 'wavs_test_pairs' / 'd_vectors_aolme.pickle'

exp_name_ex = 'S1_Bal300Clean_sc2'
run_params_ex = 'mask00_lr-5_ep180'


# Create output folder in the same directory as feats_pickle_path
output_folder_ex = main_folder_path / f'{exp_name_ex}_{run_params_ex}_output'

num_speakers_pretrained_ex = 73  # Assuming you have 2 speakers in the pretrained model
model_pretrained_path_ex = output_folder_ex / f'model_{exp_name_ex}_{run_params_ex}_{num_speakers_pretrained_ex}.pth'

parser = argparse.ArgumentParser()
parser.add_argument('--inference_feats_pickle', default=inference_feats_pickle_path_ex, help='Path to the folder to store the D-vectors features')
parser.add_argument('--pretrained_model_path', default=model_pretrained_path_ex, help='Path to the pretrained model')
parser.add_argument('--output_pred_folder', type=valid_path, default=output_folder_ex, help='Path to the folder to store the predictions')
parser.add_argument('--run_params', default=run_params_ex, help='string with the run params for HDBSCAN')
parser.add_argument('--exp_name', default=exp_name_ex, help='string with the experiment name')

args = parser.parse_args()
output_folder_path = Path(args.output_pred_folder)
inference_feats_pickle_path = Path(args.inference_feats_pickle)

run_params = args.run_params
exp_name = args.exp_name

batch_size = 16
num_workers = 4

log_path = output_folder_path / 'inference_log.txt'
enhanced_feats_and_labels = output_folder_path / f'inference_{exp_name}_{run_params}.pickle'
model_pretrained_path = args.pretrained_model_path 

# Verify the pretrained model path with run_params and exp_name
if model_pretrained_path.stem.split('_')[0] != 'model':
    sys.exit(f"Pretrained model path {model_pretrained_path} does not match expected format")

pattern = r"model_(.+)_(\d+)"
match = re.match(pattern, model_pretrained_path.stem)

if match:
    validate_exp_runparams = match.group(1)
    number_speakers_pretrained = int(match.group(2)) # Convert to scientific notation
else:
    sys.exit("Invalid run_name format")

if validate_exp_runparams != f'{exp_name}_{run_params}':
    sys.exit(f"Pretrained model path {model_pretrained_path} does not match expected experiment name and run parameters: {exp_name}_{run_params}")

log_print(f"Using pretrained model: {model_pretrained_path}", lp=log_path)

inference_loader, pickle_labels, wavs_paths, feature_dim, num_speakers_dataloader = inference_dataloader(inference_feats_pickle_path, batch_size=batch_size, num_workers=num_workers)

model = SimpleClassifier(dim=feature_dim, hidden_dim=128, num_classes=number_speakers_pretrained).to(DEVICE)

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