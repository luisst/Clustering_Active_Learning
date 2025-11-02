from pathlib import Path
import csv

# Path to your input TSV file
input_path = Path(r"C:\Users\luis2\Dropbox\DATASETS_AUDIO\Unsupervised_Pipeline\TestAO-Irma\STG_1\STG1_SHAS\wav_chunks_filtered\dt_predictions.csv")
output_path = input_path.parent / "dt_predictions_comma.csv"

# Define the desired header
header = ['wav_name', 'start_time', 'end_time', 'prediction', 'confidence']

# Read the tab-separated file and write it as a comma-separated one
with input_path.open('r', newline='', encoding='utf-8') as infile, \
     output_path.open('w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile, delimiter='\t')
    writer = csv.writer(outfile, delimiter=',')
    
    # Write the header first
    writer.writerow(header)
    
    # Write all remaining rows
    for row in reader:
        writer.writerow(row)

print(f"Converted file saved as: {output_path}")
