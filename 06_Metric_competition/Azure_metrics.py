"""
Azure Metrics: Calculate metrics for Azure Speech Service diarization results

This script:
1. Loads Azure diarization TXT files (per long_wav file)
2. Loads Ground Truth CSV files
3. Maps Azure speaker IDs (1, 2, 3...) to GT speakers using Hungarian algorithm
4. Calculates per-file and overall metrics (accuracy, precision, recall, F1, DER)
5. Generates comprehensive reports and visualizations

Azure TXT format (tab-delimited):
    speaker_id    start_time    end_time    transcript    confidence
    1             1.00          2.60        text          0.532
"""

from pathlib import Path
import argparse
import os
import sys
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, classification_report, confusion_matrix)
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns


def valid_path(path):
    """Validate that a path exists"""
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"Invalid path: {path}")


def calculate_overlap(start1, end1, start2, end2):
    """Calculate temporal overlap between two intervals"""
    return max(0, min(end1, end2) - max(start1, start2))


def calculate_der(predictions_df, gt_df, collar=0.25):
    """
    Calculate Diarization Error Rate (DER)

    DER = (False Alarm + Missed Detection + Speaker Error) / Total Ground Truth Time

    Args:
        predictions_df: DataFrame with start_time, end_time, matched_gt_speaker
        gt_df: DataFrame with start_time, end_time, gt_speaker
        collar: Collar time in seconds (default: 0.25s)

    Returns:
        Dictionary with DER components
    """
    if gt_df is None or len(gt_df) == 0:
        return None

    # Calculate total GT time
    total_gt_time = sum(gt_df['end_time'] - gt_df['start_time'])

    if total_gt_time == 0:
        return None

    # Initialize error counters
    false_alarm_time = 0
    missed_detection_time = 0
    speaker_error_time = 0

    # For each GT segment, find overlapping predictions
    for _, gt_row in gt_df.iterrows():
        gt_start = gt_row['start_time'] - collar
        gt_end = gt_row['end_time'] + collar
        gt_speaker = gt_row['gt_speaker']
        gt_duration = gt_row['end_time'] - gt_row['start_time']

        # Find overlapping predictions (same file)
        overlapping_preds = predictions_df[
            (predictions_df['long_wav_filename'] == gt_row['long_wav_filename']) &
            (predictions_df['end_time'] > gt_start) &
            (predictions_df['start_time'] < gt_end)
        ]

        if len(overlapping_preds) == 0:
            missed_detection_time += gt_duration
        else:
            total_overlap = 0
            correct_overlap = 0

            for _, pred_row in overlapping_preds.iterrows():
                overlap = calculate_overlap(gt_row['start_time'], gt_row['end_time'],
                                          pred_row['start_time'], pred_row['end_time'])
                total_overlap += overlap

                if pred_row['matched_gt_speaker'] == gt_speaker:
                    correct_overlap += overlap

            speaker_error_time += (total_overlap - correct_overlap)
            missed_detection_time += max(0, gt_duration - total_overlap)

    # Calculate false alarms
    for _, pred_row in predictions_df.iterrows():
        pred_start = pred_row['start_time']
        pred_end = pred_row['end_time']
        pred_duration = pred_end - pred_start

        overlapping_gt = gt_df[
            (gt_df['long_wav_filename'] == pred_row['long_wav_filename']) &
            (gt_df['end_time'] > pred_start - collar) &
            (gt_df['start_time'] < pred_end + collar)
        ]

        if len(overlapping_gt) == 0:
            false_alarm_time += pred_duration
        else:
            total_overlap = 0
            for _, gt_row in overlapping_gt.iterrows():
                overlap = calculate_overlap(pred_start, pred_end,
                                          gt_row['start_time'], gt_row['end_time'])
                total_overlap += overlap

            false_alarm_time += max(0, pred_duration - total_overlap)

    # Calculate DER
    total_error_time = false_alarm_time + missed_detection_time + speaker_error_time
    der = total_error_time / total_gt_time if total_gt_time > 0 else 0

    return {
        'der': der,
        'false_alarm_time': false_alarm_time,
        'missed_detection_time': missed_detection_time,
        'speaker_error_time': speaker_error_time,
        'total_gt_time': total_gt_time,
        'false_alarm_rate': false_alarm_time / total_gt_time,
        'miss_rate': missed_detection_time / total_gt_time,
        'speaker_error_rate': speaker_error_time / total_gt_time
    }


def calculate_confidence_statistics(azure_df):
    """
    Calculate confidence statistics for Azure predictions

    Args:
        azure_df: DataFrame with confidence column

    Returns:
        Dictionary with confidence statistics
    """
    if 'confidence' not in azure_df.columns:
        return None

    confidences = azure_df['confidence'].values

    return {
        'mean_confidence': np.mean(confidences),
        'std_confidence': np.std(confidences),
        'median_confidence': np.median(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences),
        'q25_confidence': np.percentile(confidences, 25),
        'q75_confidence': np.percentile(confidences, 75)
    }


def calculate_label_distribution_entropy(predictions_df, gt_df):
    """
    Calculate entropy of label distributions

    Higher entropy = more uniform distribution
    Lower entropy = more skewed distribution

    Args:
        predictions_df: DataFrame with matched_gt_speaker column
        gt_df: DataFrame with gt_speaker column

    Returns:
        Dictionary with entropy metrics
    """
    # GT label distribution entropy
    gt_counts = gt_df['gt_speaker'].value_counts().values
    gt_probs = gt_counts / gt_counts.sum()
    gt_entropy = entropy(gt_probs)

    # Predicted label distribution entropy
    pred_counts = predictions_df['mapped_speaker'].value_counts().values
    pred_probs = pred_counts / pred_counts.sum()
    pred_entropy = entropy(pred_probs)

    # Matched GT label distribution entropy (only valid matches)
    valid_matches = predictions_df[predictions_df['matched_gt_speaker'] != 'UNKNOWN']
    if len(valid_matches) > 0:
        matched_gt_counts = valid_matches['matched_gt_speaker'].value_counts().values
        matched_gt_probs = matched_gt_counts / matched_gt_counts.sum()
        matched_gt_entropy = entropy(matched_gt_probs)
    else:
        matched_gt_entropy = 0

    # Normalized entropy
    n_classes_gt = len(gt_counts)
    n_classes_pred = len(pred_counts)
    max_entropy_gt = np.log(n_classes_gt) if n_classes_gt > 1 else 1
    max_entropy_pred = np.log(n_classes_pred) if n_classes_pred > 1 else 1

    return {
        'gt_entropy': gt_entropy,
        'pred_entropy': pred_entropy,
        'matched_gt_entropy': matched_gt_entropy,
        'gt_entropy_normalized': gt_entropy / max_entropy_gt if max_entropy_gt > 0 else 0,
        'pred_entropy_normalized': pred_entropy / max_entropy_pred if max_entropy_pred > 0 else 0,
        'n_classes_gt': n_classes_gt,
        'n_classes_pred': n_classes_pred
    }


def load_azure_txt_file(azure_txt_path):
    """
    Load Azure diarization TXT file

    Format: speaker_id    start_time    end_time    transcript    confidence

    Returns:
        DataFrame with columns: azure_speaker_id, start_time, end_time, transcript, confidence
    """
    print(f"  Loading Azure file: {azure_txt_path.name}")

    try:
        df = pd.read_csv(azure_txt_path, sep='\t', header=None)

        if df.shape[1] == 6:
            df.columns = ['azure_speaker_id', 'start_time', 'end_time', 'transcript', 'confidence', 'language']
        else:
            print(f"    Unexpected format: {df.shape[1]} columns")
            return None

        # Convert to appropriate types
        df['azure_speaker_id'] = df['azure_speaker_id'].astype(int)
        df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
        df['end_time'] = pd.to_numeric(df['end_time'], errors='coerce')
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')

        # Remove rows with NaN times
        df = df.dropna(subset=['start_time', 'end_time'])

        print(f"    Loaded {len(df)} segments")
        print(f"    Azure speaker IDs: {sorted(df['azure_speaker_id'].unique())}")

        return df

    except Exception as e:
        print(f"    Error loading file: {e}")
        return None


def load_all_azure_files(azure_folder):
    """
    Load all Azure TXT files from folder

    Returns:
        Dictionary: {long_wav_filename: DataFrame}
    """
    print(f"\nLoading Azure TXT files from: {azure_folder}")

    azure_folder = Path(azure_folder)

    if not azure_folder.exists():
        sys.exit(f"Error: Azure folder not found: {azure_folder}")

    # Find all TXT files
    txt_files = list(azure_folder.glob('*.txt'))

    if len(txt_files) == 0:
        sys.exit(f"Error: No TXT files found in: {azure_folder}")

    print(f"  Found {len(txt_files)} TXT files")

    azure_data = {}

    for txt_file in txt_files:
        # Extract long_wav_filename from file name
        long_wav_name = txt_file.stem

        df = load_azure_txt_file(txt_file)

        if df is not None:
            df['long_wav_filename'] = long_wav_name
            azure_data[long_wav_name] = df

    print(f"  Successfully loaded {len(azure_data)} Azure files")

    return azure_data


def load_gt_csvs(gt_csv_folder):
    """
    Load Ground Truth CSV files

    Expected GT CSV format (tab-delimited):
        gt_speaker    language    start_time    end_time    filename    rnd_idx

    Returns:
        DataFrame with columns: long_wav_filename, start_time, end_time, gt_speaker
    """
    print(f"\nLoading GT CSV files from: {gt_csv_folder}")

    gt_csv_folder = Path(gt_csv_folder)

    if not gt_csv_folder.exists():
        print(f"  GT folder not found: {gt_csv_folder}")
        return None

    # Find all CSV/TXT files
    gt_files = list(gt_csv_folder.glob('*.csv')) + list(gt_csv_folder.glob('*.txt'))

    if len(gt_files) == 0:
        print(f"  No GT files found in: {gt_csv_folder}")
        return None

    print(f"  Found {len(gt_files)} GT files")

    # Load all GT files
    all_gt_data = []

    for gt_file in gt_files:
        # Extract long_wav_filename from GT filename
        long_wav_name = gt_file.stem.replace('_gt', '').replace('_GT', '')

        try:
            # Try reading with tab delimiter
            df = pd.read_csv(gt_file, sep='\t', header=None)

            # Check if it has the expected format
            if df.shape[1] >= 4:
                # Format: gt_speaker, language, start_time, end_time, filename, rnd_idx
                df.columns = ['gt_speaker', 'language', 'start_time', 'end_time', 'filename', 'rnd_idx']
                df = df[['start_time', 'end_time', 'gt_speaker']]
            elif df.shape[1] == 3:
                # Format: start_time, end_time, speaker
                df.columns = ['start_time', 'end_time', 'gt_speaker']
            else:
                print(f"  Unexpected format in {gt_file.name}: {df.shape[1]} columns")
                continue

            # Convert times to float
            df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
            df['end_time'] = pd.to_numeric(df['end_time'], errors='coerce')

            # Remove NaN rows
            df = df.dropna(subset=['start_time', 'end_time'])

            # Add long_wav_filename
            df['long_wav_filename'] = long_wav_name

            all_gt_data.append(df)
            print(f"    {gt_file.name}: {len(df)} GT segments")

        except Exception as e:
            print(f"  Error reading {gt_file.name}: {e}")
            continue

    if len(all_gt_data) == 0:
        print(f"  No valid GT data loaded")
        return None

    # Concatenate all GT data
    gt_df = pd.concat(all_gt_data, ignore_index=True)
    print(f"  Total GT segments loaded: {len(gt_df)}")
    print(f"  GT speakers: {sorted(gt_df['gt_speaker'].unique())}")

    return gt_df


def map_azure_speakers_to_gt(azure_df, gt_df_file, long_wav_name):
    """
    Map Azure speaker IDs to GT speakers using Hungarian algorithm (maximum overlap)

    Azure assigns arbitrary speaker IDs (1, 2, 3...) per file.
    We need to map these to actual GT speaker names.

    Strategy:
    1. Create overlap matrix: Azure speaker ID x GT speaker
    2. Use Hungarian algorithm to find optimal mapping
    3. Return mapping dictionary

    Args:
        azure_df: DataFrame with Azure predictions for one file
        gt_df_file: DataFrame with GT segments for the same file
        long_wav_name: Name of the file for logging

    Returns:
        Dictionary: {azure_speaker_id: gt_speaker_name}
    """
    print(f"\n  Mapping Azure speakers to GT for: {long_wav_name}")

    # Get unique Azure speaker IDs and GT speakers
    azure_speakers = sorted(azure_df['azure_speaker_id'].unique())
    gt_speakers = sorted(gt_df_file['gt_speaker'].unique())

    print(f"    Azure speakers: {azure_speakers}")
    print(f"    GT speakers: {gt_speakers}")

    # Create overlap matrix: [n_azure_speakers x n_gt_speakers]
    # overlap_matrix[i, j] = total overlap time between Azure speaker i and GT speaker j
    n_azure = len(azure_speakers)
    n_gt = len(gt_speakers)

    overlap_matrix = np.zeros((n_azure, n_gt))

    # Calculate overlaps
    for i, azure_spk in enumerate(azure_speakers):
        azure_segments = azure_df[azure_df['azure_speaker_id'] == azure_spk]

        for j, gt_spk in enumerate(gt_speakers):
            gt_segments = gt_df_file[gt_df_file['gt_speaker'] == gt_spk]

            # Calculate total overlap time
            total_overlap = 0
            for _, azure_seg in azure_segments.iterrows():
                for _, gt_seg in gt_segments.iterrows():
                    overlap = calculate_overlap(
                        azure_seg['start_time'], azure_seg['end_time'],
                        gt_seg['start_time'], gt_seg['end_time']
                    )
                    total_overlap += overlap

            overlap_matrix[i, j] = total_overlap

    print(f"    Overlap matrix (seconds):")
    print(f"    Azure \\ GT: {gt_speakers}")
    for i, azure_spk in enumerate(azure_speakers):
        print(f"      {azure_spk}: {overlap_matrix[i]}")

    # Use Hungarian algorithm to find optimal assignment (maximize overlap)
    # Note: linear_sum_assignment minimizes, so we negate the matrix
    row_ind, col_ind = linear_sum_assignment(-overlap_matrix)

    # Create mapping
    speaker_mapping = {}
    for azure_idx, gt_idx in zip(row_ind, col_ind):
        azure_spk = azure_speakers[azure_idx]
        gt_spk = gt_speakers[gt_idx]
        overlap_time = overlap_matrix[azure_idx, gt_idx]

        speaker_mapping[azure_spk] = gt_spk
        print(f"    Azure speaker {azure_spk} -> GT speaker '{gt_spk}' (overlap: {overlap_time:.2f}s)")

    return speaker_mapping


def apply_speaker_mapping(azure_df, speaker_mapping):
    """
    Apply speaker mapping to Azure DataFrame

    Args:
        azure_df: DataFrame with Azure predictions
        speaker_mapping: Dict {azure_speaker_id: gt_speaker_name}

    Returns:
        azure_df with new column 'mapped_speaker'
    """
    azure_df['mapped_speaker'] = azure_df['azure_speaker_id'].map(speaker_mapping)

    # Handle unmapped speakers (if any)
    unmapped_mask = azure_df['mapped_speaker'].isna()
    if unmapped_mask.any():
        print(f"    Warning: {unmapped_mask.sum()} segments with unmapped speakers")
        azure_df.loc[unmapped_mask, 'mapped_speaker'] = 'UNKNOWN'

    return azure_df


def match_azure_to_gt_segments(azure_df, gt_df_file):
    """
    Match Azure segments to GT segments based on maximum overlap

    Args:
        azure_df: DataFrame with Azure predictions (with mapped_speaker)
        gt_df_file: DataFrame with GT segments

    Returns:
        azure_df with added columns: matched_gt_speaker, gt_overlap_duration, gt_overlap_pct
    """
    matched_gt_speakers = []
    matched_overlaps = []

    for _, azure_row in azure_df.iterrows():
        azure_start = azure_row['start_time']
        azure_end = azure_row['end_time']

        # Find GT segment with maximum overlap
        best_overlap = 0
        best_gt_speaker = 'UNKNOWN'

        for _, gt_row in gt_df_file.iterrows():
            overlap = calculate_overlap(azure_start, azure_end,
                                       gt_row['start_time'], gt_row['end_time'])

            if overlap > best_overlap:
                best_overlap = overlap
                best_gt_speaker = gt_row['gt_speaker']

        matched_gt_speakers.append(best_gt_speaker)
        matched_overlaps.append(best_overlap)

    azure_df['matched_gt_speaker'] = matched_gt_speakers
    azure_df['gt_overlap_duration'] = matched_overlaps

    # Calculate overlap percentage
    azure_df['duration'] = azure_df['end_time'] - azure_df['start_time']
    azure_df['gt_overlap_pct'] = (azure_df['gt_overlap_duration'] /
                                   azure_df['duration']) * 100

    return azure_df


def calculate_file_metrics(azure_df, min_overlap_pct=50):
    """
    Calculate metrics for a single file

    Args:
        azure_df: DataFrame with Azure predictions (with mapped_speaker and matched_gt_speaker)
        min_overlap_pct: Minimum overlap percentage for valid matches

    Returns:
        Dictionary with metrics
    """
    # Filter by minimum overlap
    valid_df = azure_df[azure_df['gt_overlap_pct'] >= min_overlap_pct].copy()

    # Remove UNKNOWN
    valid_df = valid_df[valid_df['matched_gt_speaker'] != 'UNKNOWN']

    if len(valid_df) == 0:
        return None

    y_true = valid_df['matched_gt_speaker'].values
    y_pred = valid_df['mapped_speaker'].values

    # Get all unique labels
    all_labels = sorted(set(y_true) | set(y_pred))

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=all_labels,
                                average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, labels=all_labels,
                         average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=all_labels,
                 average='macro', zero_division=0)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=all_labels)

    # Per-class metrics
    per_class_metrics = {}
    for label in all_labels:
        mask = y_true == label
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            per_class_metrics[label] = {
                'accuracy': class_acc,
                'count': np.sum(mask)
            }

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': conf_matrix,
        'labels': all_labels,
        'n_valid_samples': len(valid_df),
        'pred_counts': Counter(y_pred),
        'gt_counts': Counter(y_true)
    }


def calculate_overall_metrics(all_azure_df, gt_df=None, min_overlap_pct=50):
    """
    Calculate overall metrics across all files

    Args:
        all_azure_df: Combined DataFrame with all Azure predictions
        gt_df: Ground Truth DataFrame (for DER calculation)
        min_overlap_pct: Minimum overlap percentage

    Returns:
        Dictionary with overall metrics
    """
    print(f"\n{'='*80}")
    print(f"CALCULATING OVERALL METRICS (min overlap: {min_overlap_pct}%)")
    print(f"{'='*80}")

    # Filter by minimum overlap
    valid_df = all_azure_df[all_azure_df['gt_overlap_pct'] >= min_overlap_pct].copy()
    valid_df = valid_df[valid_df['matched_gt_speaker'] != 'UNKNOWN']

    print(f"Valid samples: {len(valid_df)} / {len(all_azure_df)}")

    if len(valid_df) == 0:
        print("  No valid samples for metric calculation")
        return None

    y_true = valid_df['matched_gt_speaker'].values
    y_pred = valid_df['mapped_speaker'].values

    # Get all unique labels
    all_labels = sorted(set(y_true) | set(y_pred))

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=all_labels,
                                average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, labels=all_labels,
                         average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=all_labels,
                 average='macro', zero_division=0)

    # Classification report
    class_report = classification_report(y_true, y_pred, labels=all_labels,
                                        zero_division=0)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=all_labels)

    # Per-class metrics
    per_class_metrics = {}
    for label in all_labels:
        mask = y_true == label
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            per_class_metrics[label] = {
                'accuracy': class_acc,
                'count': np.sum(mask)
            }

    # NEW: Calculate DER (Diarization Error Rate)
    der_metrics = calculate_der(all_azure_df, gt_df, collar=0.25)

    # NEW: Calculate confidence statistics
    conf_stats = calculate_confidence_statistics(all_azure_df)

    # NEW: Calculate label distribution entropy
    entropy_metrics = calculate_label_distribution_entropy(all_azure_df, gt_df) if gt_df is not None else None

    # Print metrics
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}% (macro)")
    print(f"  Recall:    {recall*100:.2f}% (macro)")
    print(f"  F1 Score:  {f1*100:.2f}% (macro)")

    if der_metrics:
        print(f"\nDiarization Error Rate (DER):")
        print(f"  DER:               {der_metrics['der']*100:.2f}%")
        print(f"  False Alarm Rate:  {der_metrics['false_alarm_rate']*100:.2f}%")
        print(f"  Miss Rate:         {der_metrics['miss_rate']*100:.2f}%")
        print(f"  Speaker Error Rate:{der_metrics['speaker_error_rate']*100:.2f}%")

    if conf_stats:
        print(f"\nConfidence Statistics:")
        print(f"  Mean:   {conf_stats['mean_confidence']:.4f}")
        print(f"  Median: {conf_stats['median_confidence']:.4f}")
        print(f"  Std:    {conf_stats['std_confidence']:.4f}")
        print(f"  Range:  [{conf_stats['min_confidence']:.4f}, {conf_stats['max_confidence']:.4f}]")

    if entropy_metrics:
        print(f"\nLabel Distribution Entropy:")
        print(f"  GT Entropy (normalized):   {entropy_metrics['gt_entropy_normalized']:.4f}")
        print(f"  Pred Entropy (normalized): {entropy_metrics['pred_entropy_normalized']:.4f}")
        print(f"  GT classes: {entropy_metrics['n_classes_gt']}, Pred classes: {entropy_metrics['n_classes_pred']}")

    print(f"\nPer-class Accuracy:")
    for label, metrics in per_class_metrics.items():
        print(f"  {label}: {metrics['accuracy']*100:.2f}% ({metrics['count']} samples)")

    print(f"\nClassification Report:")
    print(class_report)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': conf_matrix,
        'labels': all_labels,
        'classification_report': class_report,
        'n_valid_samples': len(valid_df),
        'pred_counts': Counter(y_pred),
        'gt_counts': Counter(y_true),
        'der_metrics': der_metrics,
        'confidence_stats': conf_stats,
        'entropy_metrics': entropy_metrics
    }


def plot_confusion_matrix(conf_matrix, labels, output_path, title):
    """Plot and save confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)

    ax.set_xlabel('Predicted Speaker', fontsize=12)
    ax.set_ylabel('Ground Truth Speaker', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def plot_metrics_comparison(per_file_metrics, output_path):
    """Plot per-file metrics comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    file_names = list(per_file_metrics.keys())
    accuracies = [per_file_metrics[f]['accuracy']*100 for f in file_names]
    precisions = [per_file_metrics[f]['precision']*100 for f in file_names]
    recalls = [per_file_metrics[f]['recall']*100 for f in file_names]
    f1_scores = [per_file_metrics[f]['f1']*100 for f in file_names]

    # Plot 1: Accuracy per file
    ax = axes[0, 0]
    ax.bar(range(len(file_names)), accuracies, color='#2ecc71', alpha=0.7)
    ax.set_xticks(range(len(file_names)))
    ax.set_xticklabels(file_names, rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy per File')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Precision per file
    ax = axes[0, 1]
    ax.bar(range(len(file_names)), precisions, color='#3498db', alpha=0.7)
    ax.set_xticks(range(len(file_names)))
    ax.set_xticklabels(file_names, rotation=45, ha='right')
    ax.set_ylabel('Precision (%)')
    ax.set_title('Precision per File (macro)')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Recall per file
    ax = axes[1, 0]
    ax.bar(range(len(file_names)), recalls, color='#9b59b6', alpha=0.7)
    ax.set_xticks(range(len(file_names)))
    ax.set_xticklabels(file_names, rotation=45, ha='right')
    ax.set_ylabel('Recall (%)')
    ax.set_title('Recall per File (macro)')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: F1 Score per file
    ax = axes[1, 1]
    ax.bar(range(len(file_names)), f1_scores, color='#e74c3c', alpha=0.7)
    ax.set_xticks(range(len(file_names)))
    ax.set_xticklabels(file_names, rotation=45, ha='right')
    ax.set_ylabel('F1 Score (%)')
    ax.set_title('F1 Score per File (macro)')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

base_path_ex = Path.home().joinpath('Dropbox', 'DATASETS_AUDIO', 'Unsupervised_Pipeline', 'TestAO-Irma')
azure_folder_ex = base_path_ex / 'azure_diarization_output'
gt_csv_folder_ex = base_path_ex / 'GT_final' / 'filtered_GT'
output_folder_ex = base_path_ex / 'STG_5' / 'azure_metrics'

parser = argparse.ArgumentParser(
    description='Calculate metrics for Azure Speech Service diarization results'
)

parser.add_argument(
    '--azure_folder',
    type=valid_path,
    default=azure_folder_ex,
    help='Input folder with Azure TXT files'
)

parser.add_argument(
    '--gt_csv_folder',
    default=gt_csv_folder_ex,
    help='Ground Truth CSV folder'
)

parser.add_argument(
    '--output_folder',
    default=output_folder_ex,
    help='Output folder for results'
)

parser.add_argument(
    '--min_overlap_pct',
    type=float,
    default=50.0,
    help='Minimum overlap percentage for GT matching (default: 50)'
)

args = parser.parse_args()

azure_folder = args.azure_folder
gt_csv_folder = args.gt_csv_folder
output_folder = Path(args.output_folder)
min_overlap_pct = args.min_overlap_pct

# Create output folder
output_folder.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("="*80)
print("AZURE SPEECH SERVICE DIARIZATION METRICS")
print("="*80)
print(f"Azure folder: {azure_folder}")
print(f"GT CSV folder: {gt_csv_folder}")
print(f"Output folder: {output_folder}")
print(f"Min overlap %: {min_overlap_pct}")
print("="*80)

# Step 1: Load Azure TXT files
azure_data = load_all_azure_files(azure_folder)

# Step 2: Load GT data
gt_df = load_gt_csvs(gt_csv_folder)

if gt_df is None:
    sys.exit("Error: No GT data loaded")

# Step 3: Process each file
print(f"\n{'='*80}")
print("PROCESSING INDIVIDUAL FILES")
print("="*80)

per_file_metrics = {}
all_processed_azure_data = []

for long_wav_name, azure_df in azure_data.items():
    print(f"\n{'='*60}")
    print(f"Processing: {long_wav_name}")
    print("="*60)

    # Get GT for this file
    gt_df_file = gt_df[gt_df['long_wav_filename'] == long_wav_name]

    if len(gt_df_file) == 0:
        print(f"  No GT data found for {long_wav_name}, skipping...")
        continue

    # Map Azure speaker IDs to GT speakers
    speaker_mapping = map_azure_speakers_to_gt(azure_df, gt_df_file, long_wav_name)

    # Apply mapping
    azure_df = apply_speaker_mapping(azure_df, speaker_mapping)

    # Match to GT segments
    azure_df = match_azure_to_gt_segments(azure_df, gt_df_file)

    # Calculate metrics for this file
    file_metrics = calculate_file_metrics(azure_df, min_overlap_pct)

    if file_metrics is not None:
        per_file_metrics[long_wav_name] = file_metrics
        print(f"\n  File Metrics:")
        print(f"    Accuracy:  {file_metrics['accuracy']*100:.2f}%")
        print(f"    Precision: {file_metrics['precision']*100:.2f}%")
        print(f"    Recall:    {file_metrics['recall']*100:.2f}%")
        print(f"    F1 Score:  {file_metrics['f1']*100:.2f}%")

    # Save processed data
    all_processed_azure_data.append(azure_df)

    # Save per-file CSV
    file_csv_path = output_folder / f'{long_wav_name}_azure_predictions.csv'
    azure_df.to_csv(file_csv_path, index=False)
    print(f"  Saved: {file_csv_path.name}")

# Step 4: Combine all data for overall metrics
if len(all_processed_azure_data) > 0:
    all_azure_df = pd.concat(all_processed_azure_data, ignore_index=True)

    # Save combined CSV
    combined_csv_path = output_folder / 'azure_all_predictions.csv'
    all_azure_df.to_csv(combined_csv_path, index=False)
    print(f"\nCombined results saved: {combined_csv_path}")

    # Calculate overall metrics
    overall_metrics = calculate_overall_metrics(all_azure_df, gt_df=gt_df, min_overlap_pct=min_overlap_pct)

    # Step 5: Generate plots
    if overall_metrics is not None:
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        # Overall confusion matrix
        conf_matrix_path = output_folder / 'confusion_matrix_overall.png'
        plot_confusion_matrix(overall_metrics['confusion_matrix'],
                             overall_metrics['labels'],
                             conf_matrix_path,
                             'Azure Diarization: Overall Confusion Matrix')

        # Per-file metrics comparison
        if len(per_file_metrics) > 1:
            per_file_plot_path = output_folder / 'per_file_metrics_comparison.png'
            plot_metrics_comparison(per_file_metrics, per_file_plot_path)

    # Step 6: Save metrics report
    if overall_metrics is not None:
        metrics_report_path = output_folder / 'azure_metrics_report.txt'
        with open(metrics_report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("AZURE SPEECH SERVICE DIARIZATION METRICS REPORT\n")
            f.write("="*80 + "\n\n")

            # Overall metrics
            f.write(f"Overall Metrics:\n")
            f.write(f"  Total samples: {len(all_azure_df)}\n")
            f.write(f"  Valid samples (>{min_overlap_pct}% overlap): {overall_metrics['n_valid_samples']}\n")
            f.write(f"  Number of classes: {len(overall_metrics['labels'])}\n")
            f.write(f"  Classes: {', '.join(overall_metrics['labels'])}\n\n")

            f.write(f"  Accuracy:  {overall_metrics['accuracy']*100:.2f}%\n")
            f.write(f"  Precision: {overall_metrics['precision']*100:.2f}% (macro)\n")
            f.write(f"  Recall:    {overall_metrics['recall']*100:.2f}% (macro)\n")
            f.write(f"  F1 Score:  {overall_metrics['f1']*100:.2f}% (macro)\n\n")

            # DER metrics
            if overall_metrics.get('der_metrics'):
                der = overall_metrics['der_metrics']
                f.write(f"Diarization Error Rate (DER):\n")
                f.write(f"  DER:               {der['der']*100:.2f}%\n")
                f.write(f"  False Alarm Rate:  {der['false_alarm_rate']*100:.2f}%\n")
                f.write(f"  Miss Rate:         {der['miss_rate']*100:.2f}%\n")
                f.write(f"  Speaker Error Rate:{der['speaker_error_rate']*100:.2f}%\n")
                f.write(f"  Total GT Time:     {der['total_gt_time']:.2f}s\n\n")

            # Confidence statistics
            if overall_metrics.get('confidence_stats'):
                conf = overall_metrics['confidence_stats']
                f.write(f"Confidence Statistics:\n")
                f.write(f"  Mean:      {conf['mean_confidence']:.4f}\n")
                f.write(f"  Median:    {conf['median_confidence']:.4f}\n")
                f.write(f"  Std Dev:   {conf['std_confidence']:.4f}\n")
                f.write(f"  Min:       {conf['min_confidence']:.4f}\n")
                f.write(f"  Max:       {conf['max_confidence']:.4f}\n")
                f.write(f"  Q25:       {conf['q25_confidence']:.4f}\n")
                f.write(f"  Q75:       {conf['q75_confidence']:.4f}\n\n")

            # Entropy metrics
            if overall_metrics.get('entropy_metrics'):
                ent = overall_metrics['entropy_metrics']
                f.write(f"Label Distribution Entropy:\n")
                f.write(f"  GT Entropy:        {ent['gt_entropy']:.4f} (normalized: {ent['gt_entropy_normalized']:.4f})\n")
                f.write(f"  Pred Entropy:      {ent['pred_entropy']:.4f} (normalized: {ent['pred_entropy_normalized']:.4f})\n")
                f.write(f"  Matched GT Entropy:{ent['matched_gt_entropy']:.4f}\n")
                f.write(f"  GT Classes:        {ent['n_classes_gt']}\n")
                f.write(f"  Pred Classes:      {ent['n_classes_pred']}\n\n")

            # Per-file metrics
            f.write(f"{'='*80}\n")
            f.write(f"PER-FILE METRICS\n")
            f.write(f"{'='*80}\n\n")

            for file_name, metrics in per_file_metrics.items():
                f.write(f"{file_name}:\n")
                f.write(f"  Accuracy:  {metrics['accuracy']*100:.2f}%\n")
                f.write(f"  Precision: {metrics['precision']*100:.2f}%\n")
                f.write(f"  Recall:    {metrics['recall']*100:.2f}%\n")
                f.write(f"  F1 Score:  {metrics['f1']*100:.2f}%\n")
                f.write(f"  Samples:   {metrics['n_valid_samples']}\n\n")

            # Per-class metrics
            f.write(f"{'='*80}\n")
            f.write(f"PER-CLASS METRICS (OVERALL)\n")
            f.write(f"{'='*80}\n\n")

            for label, class_metrics in overall_metrics['per_class_metrics'].items():
                f.write(f"{label}:\n")
                f.write(f"  Accuracy: {class_metrics['accuracy']*100:.2f}%\n")
                f.write(f"  Count:    {class_metrics['count']}\n\n")

            # Classification report
            f.write(f"{'='*80}\n")
            f.write(f"CLASSIFICATION REPORT\n")
            f.write(f"{'='*80}\n\n")
            f.write(overall_metrics['classification_report'])

            # Confusion matrix
            f.write(f"\n{'='*80}\n")
            f.write(f"CONFUSION MATRIX\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Labels: {overall_metrics['labels']}\n\n")
            for i, gt_label in enumerate(overall_metrics['labels']):
                f.write(f"{gt_label}: {overall_metrics['confusion_matrix'][i]}\n")

        print(f"\nMetrics report saved: {metrics_report_path}")

print(f"\n{'='*80}")
print("AZURE METRICS CALCULATION COMPLETED")
print("="*80)
print(f"All results saved to: {output_folder}")
