from pathlib import Path
import argparse
import os
import torch
import shutil
import torchaudio

# from pipeline_utilities import valid_path
from DT_torch_utils import OverlapDetectionModel, create_inference_dataloader

def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"Using device: {DEVICE}")
#TODO: If performance is not good, try metric on slots and new TTS4

# Example usage
root_dir_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO','Unsupervised_Pipeline')
main_dir_ex = root_dir_ex / 'TestAO-Irma' / 'STG_1' / 'STG1_SHAS'
chunks_wavs_ex = main_dir_ex / "wav_chunks"
filtered_chunks_wavs_ex = main_dir_ex / "wav_chunks_filtered"
dt_pretrained_ex = Path("/home/luis/Dropbox/Source_2025/pre-trained/best_overlap_detection_model_xvectors_May20.pth")

# Create the output folder if it doesn't exist with pathlib
filtered_chunks_wavs_ex.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--stg1_chunks_wavs', type=valid_path, default=chunks_wavs_ex, help='Stg1 chunks wavs folder path')
parser.add_argument('--stg1_dt_pretrained', type=valid_path, default=dt_pretrained_ex, help='Stg1 Pretrained CNN X-vectors model')
parser.add_argument('--stg1_filtered_chunks_wavs', type=valid_path, default=filtered_chunks_wavs_ex, help='Stg1 VAD csvs folder path')
parser.add_argument('--stg1_dt_th', type=float, default=0.8, help='Stg1 double talk detection threshold')

args = parser.parse_args()

wav_folder_path = args.stg1_chunks_wavs
pretrained_model_path = args.stg1_dt_pretrained 
output_wav_folder_path = args.stg1_filtered_chunks_wavs
binary_th = args.stg1_dt_th

# Temp extra output folder
overlap_output_wav_folder_path = output_wav_folder_path.parent / "overlap_segments"

# Create the output folder if it doesn't exist with pathlib
overlap_output_wav_folder_path.mkdir(parents=True, exist_ok=True)

verbose = False

"""Run inference on a single audio file"""
# Load model
model = OverlapDetectionModel()
model.load_state_dict(torch.load(pretrained_model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Get all wav files
wav_files = list(wav_folder_path.glob("*.wav"))
print(f"Found {len(wav_files)} wav files")

wavlm_model = torchaudio.pipelines.WAVLM_BASE.get_model().to(DEVICE)
wavlm_model.eval()

unpadded_features = []
# Process each wav file
for current_wav_path in wav_files:
    # Load waveform
    waveform, sample_rate = torchaudio.load(current_wav_path)

    with torch.no_grad():
        current_features = wavlm_model(waveform.to(DEVICE))
        unpadded_features.append(current_features[0].to('cpu'))

    
# Create dataloader
dataloader = create_inference_dataloader(unpadded_features, wav_files, batch_size=32)
    
# Example of inference loop
with torch.no_grad():
    for current_batch, current_wavs_paths in dataloader:
        # Move batch to device
        current_batch = current_batch.to(DEVICE)

        # Reshape to [batch_size, 1, seq_len, 768]
        current_batch = current_batch.unsqueeze(1)
        
        output = model(current_batch)

        # Calculate the prediction for the batch
        predictions = (output >= binary_th).int()

        # Confidence: if pred == 1, use x; if pred == 0, use 1 - x
        confidence = predictions * output + (1 - predictions) * (1 - output)

        if verbose:
            # Print predictions and confidence along with the wav file names
            for wav_path, pred, conf in zip(current_wavs_paths, predictions, confidence):
                print(f"Wav: {wav_path.stem}, Prediction: {pred.item()}, Confidence: {conf.item():.3f}")

        # Save the wav files with prediction = 1 in output folder
        for wav_path, pred in zip(current_wavs_paths, predictions):
            if pred.item() == 0:
                # Copy the wav file to the output folder
                shutil.copy(wav_path, output_wav_folder_path / wav_path.name)
            else:
                # Copy the wav file to the overlap output folder
                shutil.copy(wav_path, overlap_output_wav_folder_path / wav_path.name)
        
        # Store a csv file with the wav names and predictions
        csv_filename = os.path.join(output_wav_folder_path, "dt_predictions.csv")
        with open(csv_filename, 'a') as f:
            for wav_path, pred, conf in zip(current_wavs_paths, predictions, confidence):
                # start_time and end_time are the last 2 substrings of the wav_path name with underscore
                wav_path_parts = wav_path.stem.split('_')
                start_time = wav_path_parts[-2]
                end_time = wav_path_parts[-1]
                # write confidence with 2 decimal points
                current_conf = round(conf.item(), 2)
                f.write(f"{wav_path.name},{start_time},{end_time},{pred.item()},{current_conf}\n")


