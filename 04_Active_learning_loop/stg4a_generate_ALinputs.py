
import pandas as pd
from pathlib import Path

base_path_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO','Unsupervised_Pipeline','MiniClusters')

stg4_al_input_ex = base_path_ex.joinpath('STG_4','AL_input')
method_name_ex = 'distx'
prediction_csv_ex = base_path_ex.joinpath('STG_3',f'{method_name_ex}_AL_input.csv')

if not stg4_al_input_ex.exists():
    print(f"Creating {stg4_al_input_ex}")
    stg4_al_input_ex.mkdir(parents=True, exist_ok=True)

# Read csv AL input, columns: cluster_id,sample_index,wav_stem,selection_reason,hdbscan_prob,suggested_label
if not prediction_csv_ex.exists():
    raise FileNotFoundError(f"{prediction_csv_ex} not found.")
print(f"Reading AL input from: {prediction_csv_ex}")
df_pred = pd.read_csv(prediction_csv_ex)
print(f"  {len(df_pred)} rows read.")
print(f"  Unique cluster IDs: {df_pred['cluster_id'].nunique()}")
print(f"  Unique suggested labels: {df_pred['suggested_label'].nunique()}")

# for each wav_stem, extract the long wav name and start_time and end_time, divided by '_'
df_pred[['long_wav', 'start_time', 'end_time']] = df_pred['wav_stem'].str.rsplit('_', n=2, expand=True)
df_pred['start_time'] = df_pred['start_time'].astype(float)
df_pred['end_time'] = df_pred['end_time'].astype(float)

# Store a new csv (tab separated) for each long_wav group with columns: cluster_id, start_time, end_time
grouped = df_pred.groupby('long_wav')
for long_wav, group in grouped:
    output_csv = stg4_al_input_ex / f"{long_wav}_ALinput.csv"
    group[['sample_index', 'cluster_id', 'start_time', 'end_time']].to_csv(output_csv, index=False, header=False)
    print(f"  {output_csv} created.")