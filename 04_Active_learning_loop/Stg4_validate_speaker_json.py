"""
Validate Speaker JSON Consistency Across Folders

This script:
1. Checks for specified JSON file in provided folders (GT folder, input_wavs, and input_mp4s)
2. Validates that all JSON files contain identical speaker information
3. Exits with error if inconsistencies are found
4. Intended to be called from STG4_LP.sh bash script

Exit codes:
0 - All JSON files are consistent
1 - JSON files are missing or inconsistent
"""

import json
from pathlib import Path
import argparse
import sys


def valid_path(path):
    """Validate that a path exists"""
    path = Path(path)
    if path.exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"Invalid path: {path}")


def load_json(json_path):
    """
    Load JSON file and return its content

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary with JSON content, or None if file doesn't exist
    """
    if not json_path.exists():
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ✗ Error loading {json_path}: {e}")
        return None


def compare_speaker_data(data1, data2, label1, label2):
    """
    Compare two speaker data dictionaries

    Args:
        data1: First speaker data dictionary
        data2: Second speaker data dictionary
        label1: Label for first data (e.g., "GT_final")
        label2: Label for second data (e.g., "input_wavs")

    Returns:
        Tuple (is_equal, differences_list)
    """
    differences = []

    # Check if base names match
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())

    if keys1 != keys2:
        differences.append(f"Base names differ: {label1} has {keys1}, {label2} has {keys2}")
        return False, differences

    # Compare each base name
    for base_name in keys1:
        speakers1 = data1[base_name].get('speakers', [])
        speakers2 = data2[base_name].get('speakers', [])

        # Check number of speakers
        if len(speakers1) != len(speakers2):
            differences.append(
                f"Base name '{base_name}': {label1} has {len(speakers1)} speakers, "
                f"{label2} has {len(speakers2)} speakers"
            )
            continue

        # Sort speakers by speaker_number for consistent comparison
        speakers1_sorted = sorted(speakers1, key=lambda x: x['speaker_number'])
        speakers2_sorted = sorted(speakers2, key=lambda x: x['speaker_number'])

        # Compare each speaker
        for i, (s1, s2) in enumerate(zip(speakers1_sorted, speakers2_sorted)):
            if s1['speaker_number'] != s2['speaker_number']:
                differences.append(
                    f"Base name '{base_name}', speaker {i}: "
                    f"speaker_number differs ({s1['speaker_number']} vs {s2['speaker_number']})"
                )

            if s1['speaker_name'] != s2['speaker_name']:
                differences.append(
                    f"Base name '{base_name}', speaker {s1['speaker_number']}: "
                    f"speaker_name differs ({s1['speaker_name']} vs {s2['speaker_name']})"
                )

            if s1['image_64'] != s2['image_64']:
                differences.append(
                    f"Base name '{base_name}', speaker {s1['speaker_number']}: "
                    f"image_64 differs (images are different)"
                )

    return len(differences) == 0, differences


def validate_speaker_jsons(json_filename, gt_folder=None, input_wavs=None, input_mp4s=None):
    """
    Validate speaker JSON files across provided folders

    Args:
        json_filename: Name of the JSON file to validate (e.g., 'speakers_by_basename.json')
        gt_folder: Path to GT folder (optional)
        input_wavs: Path to input_wavs folder (optional)
        input_mp4s: Path to input_mp4s folder (optional)

    Returns:
        True if all files are consistent, False otherwise
    """
    print(f"="*80)
    print(f"VALIDATING SPEAKER JSON CONSISTENCY")
    print(f"="*80)
    print(f"JSON filename: {json_filename}\n")

    # Build list of folders to check
    folders = []
    if gt_folder:
        folders.append(('GT_folder', Path(gt_folder)))
    if input_wavs:
        folders.append(('input_wavs', Path(input_wavs)))
    if input_mp4s:
        folders.append(('input_mp4s', Path(input_mp4s)))

    if not folders:
        print(f"✗ ERROR: No folders provided for validation")
        return False

    # Check which folders exist
    print("Checking folder existence:")
    folders_exist = {}
    json_paths = {}

    for folder_name, folder_path in folders:
        exists = folder_path.exists()
        folders_exist[folder_name] = exists
        status = "✓" if exists else "✗"
        print(f"  {status} {folder_name}: {folder_path}")

        if exists:
            json_paths[folder_name] = folder_path / json_filename

    print()

    # Load JSON files
    print("Loading JSON files:")
    json_data = {}

    for folder_name in folders_exist:
        if folders_exist[folder_name]:
            json_path = json_paths[folder_name]
            data = load_json(json_path)
            if data is not None:
                json_data[folder_name] = data
                print(f"  ✓ Loaded {folder_name} JSON: {json_path}")
            else:
                # MP4s is optional, others are warnings
                if folder_name == 'input_mp4s':
                    print(f"  ⚠ {folder_name} JSON not found: {json_path} (optional)")
                else:
                    print(f"  ✗ {folder_name} JSON not found: {json_path}")

    print()

    # Validate that we have at least 2 JSON files to compare
    if len(json_data) < 2:
        print(f"✗ ERROR: Need at least 2 JSON files to compare, found {len(json_data)}")
        return False

    # Compare JSON files (compare all pairs)
    print("Comparing JSON files:")
    all_consistent = True

    # Get list of available folders
    available_folders = list(json_data.keys())

    # Compare each pair
    comparisons = []
    for i in range(len(available_folders)):
        for j in range(i + 1, len(available_folders)):
            comparisons.append((available_folders[i], available_folders[j]))

    for label1, label2 in comparisons:
        print(f"\n  Comparing {label1} vs {label2}:")
        is_equal, differences = compare_speaker_data(
            json_data[label1],
            json_data[label2],
            label1,
            label2
        )

        if is_equal:
            print(f"    ✓ Files are identical")
        else:
            print(f"    ✗ Files have differences:")
            for diff in differences:
                print(f"      - {diff}")
            all_consistent = False

    # Print summary
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*80}")

    if all_consistent:
        print(f"✓ All speaker JSON files are consistent")
        for folder in available_folders:
            print(f"  - {folder} matches")
        return True
    else:
        print(f"✗ Speaker JSON files have inconsistencies")
        print(f"  Please regenerate the JSON files or fix the inconsistencies")
        return False


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(
    description='Validate speaker JSON consistency across dataset folders'
)

parser.add_argument(
    '--json_filename',
    type=str,
    required=True,
    help='Name of the JSON file to validate (e.g., speakers_by_basename.json)'
)

parser.add_argument(
    '--gt_folder',
    type=str,
    required=False,
    default=None,
    help='Path to GT folder'
)

parser.add_argument(
    '--input_wavs',
    type=str,
    required=False,
    default=None,
    help='Path to input_wavs folder'
)

parser.add_argument(
    '--input_mp4s',
    type=str,
    required=False,
    default=None,
    help='Path to input_mp4s folder'
)

args = parser.parse_args()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

success = validate_speaker_jsons(
    json_filename=args.json_filename,
    gt_folder=args.gt_folder,
    input_wavs=args.input_wavs,
    input_mp4s=args.input_mp4s
)

if success:
    print(f"\n{'='*80}")
    print(f"VALIDATION PASSED")
    print(f"{'='*80}\n")
    sys.exit(0)
else:
    print(f"\n{'='*80}")
    print(f"VALIDATION FAILED")
    print(f"{'='*80}\n")
    sys.exit(1)
