"""
Extract Speakers from JSON by Base Name

This script:
1. Reads images_june4.json with speaker info
2. Extracts base names from wav files in input_wavs folder
3. Validates that ONLY ONE base name exists in input_wavs (exits with error if multiple found)
4. Creates a new JSON organized by the detected base name with speaker info
5. Generates a modern HTML file with speaker images decoded from base64
6. Copies JSON to input_mp4s and GT_final folders (if they exist)
7. Copies HTML to input_wavs and input_mp4s folders (if they exist)

IMPORTANT: This script expects all wav files in input_wavs to belong to a SINGLE
conversation/recording (same base name). If multiple base names are detected,
the script will exit with an error.

Input JSON format:
{
    "wav_filename": [
        {
            "speaker_name": "...",
            "speaker_number": "...",
            "image_64": "..."
        },
        ...
    ]
}

Output JSON format (single base name only):
{
    "base_name": {
        "speakers": [
            {
                "speaker_name": "...",
                "speaker_number": "...",
                "image_64": "..."
            },
            ...
        ]
    }
}

Output HTML:
A modern, responsive HTML file with speaker images displayed in a grid layout.
Images are decoded from base64 and embedded directly in the HTML.
Only includes speakers from the single detected base name.
"""

import json
from pathlib import Path
from collections import defaultdict
import argparse
import shutil
import base64
import re
import time


def valid_path(path):
    """Validate that a path exists"""
    path = Path(path)
    if path.exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"Invalid path: {path}")


def extract_base_name(wav_filename):
    """
    Extract base name from wav filename using specific pattern

    Pattern: [A-Z]-[A-Za-z0-9]{5}-[A-Za-z]+[0-9]+-[A-Z]-[A-Za-z]+

    Examples:
        'G-C1L1P-Apr27-E-Irma_q2_03-08.wav' -> 'G-C1L1P-Apr27-E-Irma'
        'G-C1L1P-Apr27-E-Irma_q2_05-08-183.wav' -> 'G-C1L1P-Apr27-E-Irma'
        'A-B2C3D-Jan01-X-Test_segment1.wav' -> 'A-B2C3D-Jan01-X-Test'

    Args:
        wav_filename: Name of the wav file

    Returns:
        Base name matching the pattern, or None if pattern not found
    """
    # Remove .wav extension
    name = Path(wav_filename).stem

    # Pattern breakdown:
    # [A-Z]           : Single capital letter (e.g., G)
    # -               : Dash
    # [A-Za-z0-9]{5}  : Exactly 5 alphanumeric characters (e.g., C1L1P)
    # -               : Dash
    # [A-Za-z]+       : One or more letters (e.g., Apr)
    # [0-9]+          : One or more digits (e.g., 27)
    # -               : Dash
    # [A-Z]           : Single capital letter (e.g., E)
    # -               : Dash
    # [A-Za-z]+       : One or more letters (e.g., Irma)

    pattern = r'^([A-Z]-[A-Za-z0-9]{5}-[A-Za-z]+[0-9]+-[A-Z]-[A-Za-z]+)'

    match = re.match(pattern, name)

    if match:
        return match.group(1)
    else:
        # If pattern doesn't match, return None and let the calling code handle it
        return None


def load_input_wavs(input_wavs_folder):
    """
    Load all wav filenames from input_wavs folder

    Args:
        input_wavs_folder: Path to folder containing input wav files

    Returns:
        List of wav filenames
    """
    print(f"\nScanning input_wavs folder: {input_wavs_folder}")

    if not input_wavs_folder.exists():
        print(f"  ⚠ Folder not found: {input_wavs_folder}")
        return []

    wav_files = list(input_wavs_folder.glob('*.wav'))

    print(f"  ✓ Found {len(wav_files)} wav files")

    return [f.name for f in wav_files]


def create_speakers_html(organized_data, output_html_path):
    """
    Create a modern HTML file showing all speakers with their images

    Args:
        organized_data: Dictionary with base names and speaker info
        output_html_path: Path for output HTML file
    """
    print(f"\nCreating HTML speaker visualization...")

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speaker Reference</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        h1 {
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            font-size: 2.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        .base-name-section {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        .base-name-title {
            color: #667eea;
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #667eea;
        }

        .speakers-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1.5rem;
        }

        .speaker-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border: 2px solid transparent;
        }

        .speaker-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
            border-color: #667eea;
        }

        .speaker-image {
            width: 150px;
            height: 150px;
            object-fit: cover;
            border-radius: 50%;
            margin: 0 auto 1rem;
            border: 4px solid #667eea;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }

        .speaker-name {
            font-size: 1.25rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 0.5rem;
        }

        .speaker-number {
            font-size: 0.9rem;
            color: #666;
            background: #e9ecef;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            display: inline-block;
        }

        .no-speakers {
            text-align: center;
            color: #999;
            font-style: italic;
            padding: 2rem;
        }

        @media (max-width: 768px) {
            .speakers-grid {
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            }

            h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Speaker Reference Guide</h1>
"""

    # Add each base name section
    for base_name in sorted(organized_data.keys()):
        data = organized_data[base_name]
        speakers = data['speakers']

        html_content += f"""
        <div class="base-name-section">
            <h2 class="base-name-title">{base_name}</h2>
"""

        if speakers:
            html_content += """            <div class="speakers-grid">\n"""

            for speaker in speakers:
                speaker_name = speaker['speaker_name']
                speaker_number = speaker['speaker_number']
                image_64 = speaker['image_64']

                # Create data URI for image (same as JavaScript: "data:image/png;base64," + image_64)
                image_src = f"data:image/png;base64,{image_64}"

                html_content += f"""
                <div class="speaker-card">
                    <img src="{image_src}" alt="{speaker_name}" class="speaker-image">
                    <div class="speaker-name">{speaker_name}</div>
                    <div class="speaker-number">Speaker #{speaker_number}</div>
                </div>
"""

            html_content += """            </div>\n"""
        else:
            html_content += """            <div class="no-speakers">No speakers found</div>\n"""

        html_content += """        </div>\n"""

    html_content += """
    </div>
</body>
</html>
"""

    # Write HTML file
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  ✓ HTML file created: {output_html_path}")


def copy_files_to_dataset_folders(output_json_path, output_html_path, dataset_root_path, gt_folder_path):
    """
    Copy JSON and HTML files to dataset subfolders if they exist

    Args:
        output_json_path: Path to the JSON file to copy
        output_html_path: Path to the HTML file to copy
        dataset_root_path: Root path of the dataset (parent of input_wavs)
    """
    print(f"\n{'='*80}")
    print("COPYING FILES TO DATASET FOLDERS")
    print("="*80)

    # Define target folders
    input_mp4s = dataset_root_path / 'input_mp4s'
    gt_final = gt_folder_path
    input_wavs = dataset_root_path / 'input_wavs'

    print(f"\nDataset root: {dataset_root_path}")
    print(f"Input MP4s folder: {input_mp4s}")
    print(f"GT_final folder: {gt_final}")
    print(f"Input WAVs folder: {input_wavs}")

    def safe_copy_with_retry(src, dest, max_retries=3, delay=1.0):
        """Copy file with retry logic for Windows file locking issues"""
        for attempt in range(max_retries):
            try:
                # If destination exists and is same as source, skip
                if dest.exists() and dest.samefile(src):
                    print(f"  ⚠ Skipping copy: Source and destination are the same file")
                    return True
                
                shutil.copy2(src, dest)
                return True
            except PermissionError as e:
                if "being used by another process" in str(e):
                    if attempt < max_retries - 1:
                        print(f"  ⚠ File in use, retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"  ✗ File still in use after {max_retries} attempts. Please close any programs using the file.")
                        return False
                else:
                    print(f"  ✗ Permission error: {e}")
                    return False
            except Exception as e:
                print(f"  ✗ Error copying file: {e}")
                return False
        return False

    # Copy JSON files
    for folder_name, folder_path in [("input_mp4s", input_mp4s), ("GT_final", gt_final)]:
        if folder_path.exists():
            dest = folder_path / output_json_path.name
            if safe_copy_with_retry(output_json_path, dest):
                print(f"  ✓ JSON copied to: {dest}")
        else:
            print(f"  ⚠ Warning: {folder_name} folder not found: {folder_path}")

    # Copy HTML files
    for folder_name, folder_path in [("input_wavs", input_wavs), ("input_mp4s", input_mp4s)]:
        if folder_path.exists():
            dest = folder_path / output_html_path.name
            if safe_copy_with_retry(output_html_path, dest):
                print(f"  ✓ HTML copied to: {dest}")
        else:
            print(f"  ⚠ Warning: {folder_name} folder not found: {folder_path}")


def process_json(input_json_path, input_wavs_folder, output_json_path):
    """
    Process the input JSON and create organized output by base name

    Args:
        input_json_path: Path to images_june4.json
        input_wavs_folder: Path to folder with input wav files
        output_json_path: Path for output JSON
    """
    print(f"="*80)
    print(f"PROCESSING SPEAKER JSON")
    print(f"="*80)
    print(f"Input JSON: {input_json_path}")
    print(f"Output JSON: {output_json_path}")

    # Load input JSON
    print(f"\nLoading input JSON...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        speaker_data = json.load(f)

    print(f"  ✓ Loaded {len(speaker_data)} entries")

    # Load wav files from input folder
    wav_files = load_input_wavs(input_wavs_folder)

    if not wav_files:
        print(f"\n  ✗ ERROR: No wav files found in {input_wavs_folder}")
        print(f"  Cannot proceed without input wav files.")
        exit(1)

    # Extract base names from wav files
    print(f"\nExtracting base names from {len(wav_files)} wav files...")
    base_names = set()
    wav_to_base = {}
    failed_files = []

    for wav_file in wav_files:
        base_name = extract_base_name(wav_file)
        if base_name is not None:
            base_names.add(base_name)
            wav_to_base[wav_file] = base_name
            print(f"  {wav_file} -> {base_name}")
        else:
            failed_files.append(wav_file)
            print(f"  {wav_file} -> [PATTERN NOT MATCHED]")

    if failed_files:
        print(f"\n  ⚠ WARNING: {len(failed_files)} files did not match the expected pattern:")
        print(f"  Expected pattern: [A-Z]-[A-Za-z0-9]{{5}}-[A-Za-z]+[0-9]+-[A-Z]-[A-Za-z]+")
        print(f"  Example: G-C1L1P-Apr27-E-Irma_q2_05-08-183.wav")
        for ff in failed_files[:5]:  # Show max 5 examples
            print(f"    - {ff}")
        if len(failed_files) > 5:
            print(f"    ... and {len(failed_files) - 5} more")

    print(f"\n  ✓ Found {len(base_names)} unique base names")

    # Validate: Only one base name allowed
    if len(base_names) == 0:
        print(f"\n  ✗ ERROR: No base names could be extracted from wav files")
        print(f"  All files failed to match the expected pattern.")
        print(f"  Expected pattern: [A-Z]-[A-Za-z0-9]{{5}}-[A-Za-z]+[0-9]+-[A-Z]-[A-Za-z]+")
        print(f"  Example: G-C1L1P-Apr27-E-Irma_q2_05-08-183.wav")
        exit(1)
    elif len(base_names) > 1:
        print(f"\n  ✗ ERROR: Found multiple base names in input_wavs folder:")
        for bn in sorted(base_names):
            print(f"    - {bn}")
        print(f"\n  This script expects all wav files to belong to a single conversation/recording.")
        print(f"  Please ensure input_wavs contains only files from one base name.")
        exit(1)

    # Get the single base name
    target_base_name = list(base_names)[0]
    print(f"\n  ✓ Validated: Single base name detected: {target_base_name}")

    # Organize speaker data for the target base name only
    print(f"\nOrganizing speaker data for base name: {target_base_name}...")
    organized_data = {target_base_name: {"speakers": []}}

    for wav_filename, speakers in speaker_data.items():
        # Extract base name from this JSON entry
        base_name = extract_base_name(wav_filename)

        # Skip if pattern doesn't match
        if base_name is None:
            continue

        # Only process if it matches our target base name
        if base_name == target_base_name:
            for speaker in speakers:
                # Check if this speaker is already added
                existing_speakers = organized_data[target_base_name]["speakers"]
                speaker_exists = any(
                    s["speaker_number"] == speaker["speaker_number"]
                    for s in existing_speakers
                )

                if not speaker_exists:
                    organized_data[target_base_name]["speakers"].append({
                        "speaker_name": speaker["speaker_name"],
                        "speaker_number": speaker["speaker_number"],
                        "image_64": speaker["image_64"]
                    })
                    print(f"  Added speaker {speaker['speaker_number']} ({speaker['speaker_name']}) to {target_base_name}")

    num_speakers = len(organized_data[target_base_name]["speakers"])
    print(f"\n  ✓ Found {num_speakers} unique speakers for base name: {target_base_name}")

    if num_speakers == 0:
        print(f"\n  ⚠ WARNING: No speakers found in JSON for base name: {target_base_name}")
        print(f"  The output files will be created but will contain no speaker data.")

    # Save organized data
    print(f"\nSaving organized data to: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dict(organized_data), f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved!")

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"\nBase Name: {target_base_name}")
    print(f"  Total Speakers: {num_speakers}")
    if num_speakers > 0:
        print(f"\n  Speaker List:")
        for speaker in organized_data[target_base_name]['speakers']:
            print(f"    - Speaker #{speaker['speaker_number']}: {speaker['speaker_name']}")
    else:
        print(f"  (No speakers found)")

    return organized_data


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

src_path_ex = Path.home().joinpath('Dropbox', 'Source_2025', '04_Active_learning_loop')
input_json_ex = src_path_ex / 'images_june4.json'
root_data_ex = Path.home().joinpath('Dropbox', 'DATASETS_AUDIO', 'Unsupervised_Pipeline', 'TestAO-Irma')
input_wavs_ex = root_data_ex / 'input_wavs'
output_json_ex = input_wavs_ex / 'speakers_info.json'
gt_folder_ex = root_data_ex / 'GT_final' / 'filtered_GT'

parser = argparse.ArgumentParser(
    description='Extract speakers from JSON organized by wav base names'
)

parser.add_argument(
    '--input_json',
    type=valid_path,
    default=input_json_ex,
    help='Input JSON file (images_june4.json)'
)

parser.add_argument(
    '--output_json',
    default=output_json_ex,
    help='Output JSON file path'
)

parser.add_argument(
    '--gt_folder',
    default=gt_folder_ex,
    help='Ground truth folder path'
)

args = parser.parse_args()

input_json = args.input_json
gt_folder = args.gt_folder
output_json = Path(args.output_json)

input_wavs = output_json.parent  # input_wavs is the parent of output_json

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Process JSON and get organized data
organized_data = process_json(input_json, input_wavs, output_json)

# Create HTML file with speaker visualization
output_html = output_json.parent / f'{output_json.stem}.html'
create_speakers_html(organized_data, output_html)

# Copy files to dataset folders
dataset_root = input_wavs.parent  # Parent of input_wavs
copy_files_to_dataset_folders(output_json, output_html, dataset_root, gt_folder)

print(f"\n{'='*80}")
print(f"EXTRACTION COMPLETED")
print(f"{'='*80}")
print(f"JSON output: {output_json}")
print(f"HTML output: {output_html}")
