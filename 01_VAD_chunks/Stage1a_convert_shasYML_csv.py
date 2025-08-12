import argparse
from pathlib import Path
import re
import pdb

# Print current working directory
import os
from pipeline_utilities import log_print, valid_path

root_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO','SHAS')
yml_pth_ex = root_ex / 'shas_dev.yaml'
output_csv_folder_ex = root_ex.joinpath('STG_1','STG1_shas','csv_converted')

parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('yml_pth',  default=yml_pth_ex, type=valid_path, help='The path to the YML file')
parser.add_argument('output_csv_folder', default=output_csv_folder_ex, type=valid_path, help='The path to the output CSV folder')
args = parser.parse_args()

yml_pth = args.yml_pth
output_csv_folder = args.output_csv_folder

regex = r"duration: (\d+?.\d+?), offset: (\d+?.\d+?), rW: \d.?\d*?, speaker_id: (\w+?), uW: \d.?\d*?, wav: (.*?)}"

print(f'Processing {yml_pth.stem}...')

#open text file in read mode
with open(yml_pth, "r") as text_file:
    data = text_file.read()

matches = re.finditer(regex, data)

prev_name = '' 
first_time_flag = True
for matchNum, match in enumerate(matches, start=1):
    new_filename = match.group(4).split('.')[0]

    if new_filename != prev_name:
        if not(first_time_flag):
            new_file.close()
        new_transcr_path = output_csv_folder.joinpath(f'{new_filename}.txt')
        new_file = open(new_transcr_path, "w")
        first_time_flag = False

    stop_val = float(match.group(2)) + float(match.group(1))
    new_file.write(f'{new_filename}\t{match.group(2)}\t{stop_val}\n')

    prev_name = new_filename

new_file.close()
