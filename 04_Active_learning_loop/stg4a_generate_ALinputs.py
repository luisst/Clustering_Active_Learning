import argparse
import os
import pandas as pd
from pathlib import Path

def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

base_path_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO','Unsupervised_Pipeline','MiniClusters')
stg4_al_input_ex = base_path_ex.joinpath('STG_4','AL_input')
method_name_ex = 'hdbscan'
stg3_pred_csv_ex = base_path_ex.joinpath('STG_3',f'{method_name_ex}_AL_input.csv')


if not stg4_al_input_ex.exists():
    print(f"Creating {stg4_al_input_ex}")
    stg4_al_input_ex.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--stg3_al_input', type=valid_path, default=stg3_pred_csv_ex, help='Stg3 prediction csv path')
parser.add_argument('--stg4_al_folder', default=stg4_al_input_ex, help='Stg4 AL input folder path')
args = parser.parse_args()

pred_csv = args.stg3_al_input
stg4_al_folder = Path(args.stg4_al_folder)

# Read csv AL input, columns: cluster_id,sample_index,wav_stem,selection_reason,hdbscan_prob,suggested_label
if not pred_csv.exists():
    raise FileNotFoundError(f"{pred_csv} not found.")
print(f"Reading AL input from: {pred_csv}")

df_pred = pd.read_csv(pred_csv)
print(f"  {len(df_pred)} rows read.")
print(f"  Unique cluster IDs: {df_pred['cluster_id'].nunique()}")
print(f"  Unique suggested labels: {df_pred['suggested_label'].nunique()}")

# for each wav_stem, extract the long wav name and start_time and end_time, divided by '_'
df_pred[['long_wav', 'lbl_filename', 'start_time', 'end_time']] = df_pred['wav_stem'].str.rsplit('_', n=3, expand=True)
df_pred['start_time'] = df_pred['start_time'].astype(float)
df_pred['end_time'] = df_pred['end_time'].astype(float)

# Store a new csv (tab separated) for each long_wav group with columns: cluster_id, start_time, end_time
grouped = df_pred.groupby('long_wav')
for long_wav, group in grouped:
    output_csv = stg4_al_folder / f"{long_wav}_ALinput.csv"
    group[['sample_index', 'cluster_id', 'start_time', 'end_time']].to_csv(output_csv, index=False, header=False)
    print(f"  {output_csv} created.")