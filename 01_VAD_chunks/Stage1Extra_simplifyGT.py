import argparse
import csv
import sys
from pathlib import Path
from collections import Counter
import pprint

def load_gt_labels(gt_csv_folder):
    """
    Load all GT labels from CSV files in the specified folder.
    
    Args:
        gt_csv_folder (str or Path): Path to folder containing GT CSV files
        
    Returns:
        dict: Dictionary with label counts across all CSV files
        list: List of tuples (csv_file_path, rows_data) for processing
    """
    gt_folder = Path(gt_csv_folder)
    
    if not gt_folder.exists() or not gt_folder.is_dir():
        print(f"Error: GT folder '{gt_folder}' does not exist or is not a directory.")
        sys.exit(1)
    
    # Find all CSV files in the folder
    csv_files = list(gt_folder.glob("*.csv"))
    
    if not csv_files:
        print(f"Error: No CSV files found in '{gt_folder}'")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files in GT folder")
    
    label_counts = Counter()
    all_csv_data = []
    
    # Process each CSV file to count labels
    for csv_file_path in csv_files:
        print(f"Processing: {csv_file_path.name}")
        
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file, delimiter='\t')
                rows_data = []
                
                for row in csv_reader:
                    if len(row) >= 1:  # Ensure we have at least one column
                        speaker_id = row[0]  # First column is speaker ID
                        label_counts[speaker_id] += 1
                        rows_data.append(row)
                    else:
                        print(f"Warning: Empty or invalid row in {csv_file_path.name}")
                
                all_csv_data.append((csv_file_path, rows_data))
                print(f"  Processed {len(rows_data)} rows from {csv_file_path.name}")
                        
        except FileNotFoundError:
            print(f"Error: File '{csv_file_path}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading CSV '{csv_file_path}': {e}")
            sys.exit(1)
    
    return dict(label_counts), all_csv_data

def filter_and_save_gt(all_csv_data, label_counts, threshold, output_folder):
    """
    Filter CSV files based on label threshold and save filtered versions.
    
    Args:
        all_csv_data (list): List of tuples (csv_file_path, rows_data)
        label_counts (dict): Dictionary with label counts
        threshold (int): Minimum count threshold for labels
        output_folder (Path): Output folder for filtered CSV files
    """
    # Determine which labels meet the threshold
    valid_labels = {label for label, count in label_counts.items() if count >= threshold}
    
    print(f"\nLabel filtering results:")
    print(f"  Total unique labels: {len(label_counts)}")
    print(f"  Labels meeting threshold ({threshold}): {len(valid_labels)}")
    print(f"  Valid labels: {sorted(valid_labels)}")
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    total_original_rows = 0
    total_filtered_rows = 0
    
    # Process each CSV file
    for csv_file_path, rows_data in all_csv_data:
        original_count = len(rows_data)
        total_original_rows += original_count
        
        # Filter rows based on valid labels
        filtered_rows = [row for row in rows_data if row[0] in valid_labels]
        filtered_count = len(filtered_rows)
        total_filtered_rows += filtered_count
        
        # Save filtered CSV
        output_file_path = output_folder / csv_file_path.name
        
        try:
            with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
                csv_writer = csv.writer(file, delimiter='\t')
                csv_writer.writerows(filtered_rows)
            
            print(f"  {csv_file_path.name}: {original_count} -> {filtered_count} rows "
                  f"({filtered_count/original_count*100:.1f}% retained)")
                  
        except Exception as e:
            print(f"Error saving filtered CSV '{output_file_path}': {e}")
            sys.exit(1)
    
    print(f"\nOverall filtering summary:")
    print(f"  Total original rows: {total_original_rows}")
    print(f"  Total filtered rows: {total_filtered_rows}")
    print(f"  Overall retention rate: {total_filtered_rows/total_original_rows*100:.1f}%")
    print(f"  Filtered files saved to: {output_folder}")

def main():
    # Default path for testing
    csv_gt_path_ex = Path("C:\\Users\\luis2\\Dropbox\\DATASETS_AUDIO\\Unsupervised_Pipeline\\TestAO-Irma\\GT_final")

    parser = argparse.ArgumentParser(
        description="Filter GT CSV files based on speaker label frequency threshold"
    )
    
    parser.add_argument(
        '--gt_csv_folder',
        type=str,
        default=csv_gt_path_ex,
        help='Path to folder containing GT CSV files'
    )
    
    parser.add_argument(
        '--threshold',
        type=int,
        default=5,
        help='Minimum count threshold for labels (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    gt_csv_folder = Path(args.gt_csv_folder)
    output_folder = gt_csv_folder / 'filtered_GT'
    
    print("="*60)
    print("GT CSV FILTERING SCRIPT")
    print("="*60)
    print(f"GT CSV folder: {gt_csv_folder}")
    print(f"Threshold: {args.threshold}")
    print(f"Output folder: {output_folder}")
    print("-"*60)
    
    # Load GT labels and count occurrences
    print("Step 1: Loading GT labels...")
    label_counts, all_csv_data = load_gt_labels(gt_csv_folder)
    
    # Print label statistics
    print(f"\nLabel count statistics:")
    print(f"Total unique labels found: {len(label_counts)}")
    
    # Sort labels by count (descending)
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\nLabels:")
    for i, (label, count) in enumerate(sorted_labels):
        print(f"  {i+1:2d}. {label}: {count} occurrences")
    
    # Show labels that will be filtered out
    filtered_out_labels = {label: count for label, count in label_counts.items() if count < args.threshold}
    if filtered_out_labels:
        print(f"\nLabels to be filtered out (count < {args.threshold}):")
        for label, count in sorted(filtered_out_labels.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {count} occurrences")
    
    # Filter and save
    print(f"\nStep 2: Filtering and saving CSV files...")
    filter_and_save_gt(all_csv_data, label_counts, args.threshold, output_folder)
    
    print("\n" + "="*60)
    print("FILTERING COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    main()