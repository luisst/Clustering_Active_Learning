from pathlib import Path
import pandas as pd
import sys

# --- Get all .tsv files in the current working directory ---
cwd = Path.cwd()
tsv_files = list(cwd.glob("*.csv"))

output_folder = cwd / "modified_files"
output_folder.mkdir(exist_ok=True)

if not tsv_files:
    print("No .csv files found in the current directory.")
else:
    for csv_path in tsv_files:
        print(f"Processing: {csv_path.name}")
        
        # --- Load file ---
        df = pd.read_csv(csv_path, sep='\t', header=0)
        
        # --- Compute new columns ---
        filename = csv_path.stem  # e.g., "experiment_run1_sample"


        # Verify the filename suffix is 'GT'
        if not filename.endswith("_GT"):
            sys.exit(f"Warning: The file {csv_path.name} does not have the expected '_GT' suffix.")

        prefix = "_".join(filename.split("_")[:-1])
        
        # --- Add new columns at the end ---
        df["FileName_GT"] = prefix
        df["Constant"] = 0
        
        # --- Save modified file ---
        output_path = output_folder / (csv_path.stem + ".csv")
        df.to_csv(output_path, sep='\t', index=False)
        
        print(f" → Saved as: {output_path.name}")

print("All files processed.")
