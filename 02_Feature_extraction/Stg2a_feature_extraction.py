import os
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from metaSR_utils import extract_MFB_aolme

root_dir = Path.home().joinpath('Dropbox','DATASETS_AUDIO')
input_wavs_folder_ex = root_dir / 'Dvectors/TestAO-Irma/STG_1/STG1_SHAS/wav_chunks_filtered' 
output_feats_folder_ex = root_dir / 'Dvectors/TestAO-Irma/STG_2/STG2_EXP010-SHAS-DV/MFCC_files2'

if not output_feats_folder_ex.exists():
    os.makedirs(output_feats_folder_ex)

parser = argparse.ArgumentParser()
parser.add_argument('--wavs_folder', default=input_wavs_folder_ex , help='Path to the folder containing the WAV files')
parser.add_argument('--output_feats_folder', default=output_feats_folder_ex, help='Path to the folder to save the extracted features')
args = parser.parse_args()

wavs_folder = Path(args.wavs_folder)
output_feats_folder = Path(args.output_feats_folder)

list_of_wavs = sorted(list(wavs_folder.glob('*.wav')))

# Print the number of files to process
print(f'Number of files to process: {len(list_of_wavs)}')

if len(list_of_wavs) == 0:
    sys.exit("No files to process")

count = 0
list_of_labels = []
for current_wav_path in list_of_wavs:
    current_GT_label, _ = extract_MFB_aolme(current_wav_path, output_feats_folder)
    list_of_labels.append(current_GT_label)
    count = count + 1
    print(f'{count} - feature extraction: {current_wav_path.name}')

# Export labels to a text file
labels_file_path = output_feats_folder.parent / 'feat_extraction_labels.txt'
with open(labels_file_path, 'w', encoding='utf-8') as f:
    for label in list_of_labels:
        f.write(f"{label}\n")
print(f'Speaker labels saved to: {labels_file_path}') 