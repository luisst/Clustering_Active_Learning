"""
Stage 4f: Create CSV from Label Propagation results with timing information and GT metrics

This script:
1. Reads LP results CSV (merged_unique_id, lp_label, lp_confidence)
2. Loads merged HDF5 dataset to get wav paths and timing (start/end times)
3. Loads Ground Truth CSV files
4. Calculates comprehensive metrics (accuracy, precision, recall, F1, confusion matrix)
5. Generates detailed output CSVs with timing information
6. Creates visualization plots
"""

from pathlib import Path
import csv
import argparse
import os
import sys
import numpy as np
import pandas as pd
import h5py
from collections import Counter
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, classification_report, confusion_matrix)
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns


def valid_path(path):
    """Validate that a path exists"""
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"Invalid path: {path}")


def load_lp_results(lp_results_csv):
    """
    Load Label Propagation results from CSV

    Returns:
        DataFrame with columns: merged_unique_id, gt_label, hdbscan_label,
                               hdbscan_prob, lp_label, lp_confidence, human_label
    """
    print(f"\nLoading LP results from: {lp_results_csv}")
    df = pd.read_csv(lp_results_csv)
    print(f"  ✓ Loaded {len(df)} LP results")
    print(f"  Columns: {list(df.columns)}")
    return df


def load_merged_h5_timing_info(merged_h5_path):
    """
    Load timing and path information from merged HDF5 dataset

    Returns:
        DataFrame with columns: merged_unique_id, merged_wav_path,
                               long_wav_filename, start_time, end_time, duration
    """
    print(f"\nLoading timing info from HDF5: {merged_h5_path}")

    with h5py.File(merged_h5_path, 'r') as hf:
        merged_group = hf['merged_samples']

        # Load arrays
        merged_unique_ids = merged_group['merged_unique_ids'][:]
        merged_wav_paths = merged_group['merged_wav_paths'][:]
        start_times = merged_group['start_times'][:]
        end_times = merged_group['end_times'][:]
        durations = merged_group['durations'][:]

        # Decode bytes to strings if necessary
        if merged_unique_ids.dtype.kind == 'S' or merged_unique_ids.dtype.kind == 'O':
            merged_unique_ids = [uid.decode('utf-8') if isinstance(uid, bytes) else uid
                                for uid in merged_unique_ids]

        if merged_wav_paths.dtype.kind == 'S' or merged_wav_paths.dtype.kind == 'O':
            merged_wav_paths = [path.decode('utf-8') if isinstance(path, bytes) else path
                               for path in merged_wav_paths]

    # Extract long_wav_filename from merged_wav_path
    # Format: <base>/<long_wav>_<label>_<start>_<end>.wav
    long_wav_filenames = []
    for wav_path in merged_wav_paths:
        # Get filename without extension
        filename = Path(wav_path).stem
        # Split by underscore and remove last 3 parts (label, start, end)
        parts = filename.split('_')
        long_wav_name = '_'.join(parts[:-3])
        long_wav_filenames.append(long_wav_name)

    df = pd.DataFrame({
        'merged_unique_id': merged_unique_ids,
        'merged_wav_path': merged_wav_paths,
        'long_wav_filename': long_wav_filenames,
        'start_time': start_times,
        'end_time': end_times,
        'duration': durations
    })

    print(f"  ✓ Loaded timing info for {len(df)} merged samples")
    print(f"  ✓ Unique long wav files: {df['long_wav_filename'].nunique()}")

    return df


def load_gt_csvs(gt_csv_folder):
    """
    Load Ground Truth CSV files

    Expected GT CSV format (tab-delimited):
        filename    start_time    end_time    speaker_label

    Returns:
        DataFrame with columns: long_wav_filename, start_time, end_time, gt_speaker
    """
    print(f"\nLoading GT CSV files from: {gt_csv_folder}")

    gt_csv_folder = Path(gt_csv_folder)

    if not gt_csv_folder.exists():
        print(f"  ⚠ GT folder not found: {gt_csv_folder}")
        return None

    # Find all CSV/TXT files
    gt_files = list(gt_csv_folder.glob('*.csv')) + list(gt_csv_folder.glob('*.txt'))

    if len(gt_files) == 0:
        print(f"  ⚠ No GT files found in: {gt_csv_folder}")
        return None

    print(f"  Found {len(gt_files)} GT files")

    # Load all GT files
    all_gt_data = []

    for gt_file in gt_files:
        # Extract long_wav_filename from GT filename
        # Typically: <long_wav_name>_gt.csv or <long_wav_name>.txt
        long_wav_name = gt_file.stem.replace('_gt', '').replace('_GT', '')

        try:
            # Try reading with tab delimiter
            df = pd.read_csv(gt_file, sep='\t', header=None)

            # Check if it has the expected format
            if df.shape[1] >= 4:
                # Format: filename, start_time, end_time, speaker
                df.columns = ['gt_speaker', 'language', 'start_time', 'end_time', 'filename', 'rnd_idx']
                df = df[['start_time', 'end_time', 'gt_speaker']]
            else:
                print(f"  ⚠ Unexpected format in {gt_file.name}: {df.shape[1]} columns")
                continue

            # Convert times to float
            df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
            df['end_time'] = pd.to_numeric(df['end_time'], errors='coerce')

            # Add long_wav_filename
            df['long_wav_filename'] = long_wav_name

            all_gt_data.append(df)
            print(f"    ✓ {gt_file.name}: {len(df)} GT segments")

        except Exception as e:
            print(f"  ⚠ Error reading {gt_file.name}: {e}")
            continue

    if len(all_gt_data) == 0:
        print(f"  ⚠ No valid GT data loaded")
        return None

    # Concatenate all GT data
    gt_df = pd.concat(all_gt_data, ignore_index=True)
    print(f"  ✓ Total GT segments loaded: {len(gt_df)}")
    print(f"  ✓ GT speakers: {sorted(gt_df['gt_speaker'].unique())}")

    return gt_df


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
    false_alarm_time = 0  # Predicted speech when GT is non-speech
    missed_detection_time = 0  # GT speech when prediction is non-speech
    speaker_error_time = 0  # Correct speech detection but wrong speaker

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
            # Missed detection: GT segment with no prediction
            missed_detection_time += gt_duration
        else:
            # Calculate overlaps
            total_overlap = 0
            correct_overlap = 0

            for _, pred_row in overlapping_preds.iterrows():
                overlap = calculate_overlap(gt_row['start_time'], gt_row['end_time'],
                                          pred_row['start_time'], pred_row['end_time'])
                total_overlap += overlap

                # Check if speaker is correct
                if pred_row['matched_gt_speaker'] == gt_speaker:
                    correct_overlap += overlap

            # Speaker error: overlapping but wrong speaker
            speaker_error_time += (total_overlap - correct_overlap)

            # Missed detection: parts of GT not covered by predictions
            missed_detection_time += max(0, gt_duration - total_overlap)

    # Calculate false alarms: predictions not overlapping with any GT
    for _, pred_row in predictions_df.iterrows():
        pred_start = pred_row['start_time']
        pred_end = pred_row['end_time']
        pred_duration = pred_end - pred_start

        # Find overlapping GT segments
        overlapping_gt = gt_df[
            (gt_df['long_wav_filename'] == pred_row['long_wav_filename']) &
            (gt_df['end_time'] > pred_start - collar) &
            (gt_df['start_time'] < pred_end + collar)
        ]

        if len(overlapping_gt) == 0:
            # False alarm: prediction with no GT
            false_alarm_time += pred_duration
        else:
            # Calculate how much of prediction doesn't overlap with GT
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


def calculate_confidence_statistics(predictions_df):
    """
    Calculate confidence statistics for predictions

    Args:
        predictions_df: DataFrame with lp_confidence column

    Returns:
        Dictionary with confidence statistics
    """
    if 'lp_confidence' not in predictions_df.columns:
        return None

    confidences = predictions_df['lp_confidence'].values

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
    pred_counts = predictions_df['lp_label'].value_counts().values
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

    # Normalized entropy (entropy / log(n_classes))
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


def match_predictions_to_gt(predictions_df, gt_df):
    """
    Match predicted segments to GT segments based on maximum overlap

    Args:
        predictions_df: DataFrame with start_time, end_time, lp_label, long_wav_filename
        gt_df: DataFrame with start_time, end_time, gt_speaker, long_wav_filename

    Returns:
        predictions_df with added 'matched_gt_speaker', 'gt_overlap_duration', and 'gt_overlap_pct' columns
    """
    print(f"\nMatching predictions to GT segments...")

    if gt_df is None:
        print(f"  No GT data available")
        predictions_df['matched_gt_speaker'] = 'UNKNOWN'
        return predictions_df

    matched_gt_speakers = []
    matched_overlaps = []

    for idx, pred_row in predictions_df.iterrows():
        long_wav = pred_row['long_wav_filename']
        pred_start = pred_row['start_time']
        pred_end = pred_row['end_time']

        # Find GT segments for the same long wav
        gt_segments = gt_df[gt_df['long_wav_filename'] == long_wav]

        if len(gt_segments) == 0:
            matched_gt_speakers.append('UNKNOWN')
            matched_overlaps.append(0.0)
            continue

        # Calculate overlap with each GT segment
        best_overlap = 0
        best_gt_speaker = 'UNKNOWN'

        for _, gt_row in gt_segments.iterrows():
            overlap = calculate_overlap(pred_start, pred_end,
                                       gt_row['start_time'], gt_row['end_time'])

            if overlap > best_overlap:
                best_overlap = overlap
                best_gt_speaker = gt_row['gt_speaker']

        matched_gt_speakers.append(best_gt_speaker)
        matched_overlaps.append(best_overlap)

    predictions_df['matched_gt_speaker'] = matched_gt_speakers
    predictions_df['gt_overlap_duration'] = matched_overlaps

    # Calculate overlap percentage
    predictions_df['gt_overlap_pct'] = (predictions_df['gt_overlap_duration'] /
                                        predictions_df['duration']) * 100

    print(f"  ✓ Matched {len(predictions_df)} predictions to GT")
    n_matched = (predictions_df['matched_gt_speaker'] != 'UNKNOWN').sum()
    print(f"  ✓ Successfully matched: {n_matched} ({n_matched/len(predictions_df)*100:.1f}%)")

    return predictions_df


def calculate_metrics(predictions_df, gt_df=None, min_overlap_pct=50):
    """
    Calculate comprehensive metrics comparing LP predictions to GT

    Args:
        predictions_df: DataFrame with lp_label and matched_gt_speaker
        gt_df: DataFrame with GT segments (for DER calculation)
        min_overlap_pct: Minimum overlap percentage to consider a valid match

    Returns:
        Dictionary with metrics
    """
    print(f"\n{'='*80}")
    print(f"CALCULATING METRICS (min overlap: {min_overlap_pct}%)")
    print(f"{'='*80}")

    # Filter by minimum overlap
    valid_df = predictions_df[predictions_df['gt_overlap_pct'] >= min_overlap_pct].copy()

    # Remove UNKNOWN GT labels
    valid_df = valid_df[valid_df['matched_gt_speaker'] != 'UNKNOWN']

    print(f"Valid samples for metrics: {len(valid_df)} / {len(predictions_df)}")

    if len(valid_df) == 0:
        print("  ⚠ No valid samples for metric calculation")
        return None

    y_true_names = valid_df['matched_gt_speaker'].values

    # Map GT speaker names to 'S' + numeric format to match y_pred
    speaker_name_to_s_format = {
        'Juan16P': 'S0',
        'Herminio10P': 'S1',
        'Irma': 'S2',
        'Jacinto51P': 'S3',
        'Jorge17P': 'S4'
    }

    # Apply mapping to y_true
    y_true = np.array([speaker_name_to_s_format.get(name, name) for name in y_true_names])

    y_pred = valid_df['lp_label'].values

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Get all unique labels
    all_labels = sorted(set(y_true) | set(y_pred))

    # Precision, Recall, F1 (macro average)
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
            class_count = np.sum(mask)
            per_class_metrics[label] = {
                'accuracy': class_acc,
                'count': class_count
            }

    # Label distribution
    pred_counts = Counter(y_pred)
    gt_counts = Counter(y_true)

    # NEW: Calculate DER (Diarization Error Rate)
    der_metrics = calculate_der(predictions_df, gt_df, collar=0.25)

    # NEW: Calculate confidence statistics
    conf_stats = calculate_confidence_statistics(predictions_df)

    # NEW: Calculate label distribution entropy
    entropy_metrics = calculate_label_distribution_entropy(predictions_df, gt_df) if gt_df is not None else None

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

    print(f"\nLabel Distribution:")
    print(f"  Ground Truth:")
    for label, count in sorted(gt_counts.items()):
        print(f"    {label}: {count}")
    print(f"  Predictions:")
    for label, count in sorted(pred_counts.items()):
        print(f"    {label}: {count}")

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
        'pred_counts': pred_counts,
        'gt_counts': gt_counts,
        'der_metrics': der_metrics,
        'confidence_stats': conf_stats,
        'entropy_metrics': entropy_metrics
    }


def plot_confusion_matrix(conf_matrix, labels, output_path):
    """Plot and save confusion matrix"""
    print(f"\nPlotting confusion matrix...")

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('Ground Truth Label', fontsize=12)
    ax.set_title('Confusion Matrix: LP Predictions vs Ground Truth', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Confusion matrix saved: {output_path}")


def plot_metrics_summary(metrics, output_path):
    """Plot summary of metrics"""
    print(f"\nPlotting metrics summary...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Overall metrics bar chart
    ax = axes[0, 0]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [metrics['accuracy'], metrics['precision'],
                     metrics['recall'], metrics['f1']]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

    bars = ax.bar(metrics_names, [v*100 for v in metrics_values], color=colors, alpha=0.7)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title('Overall Metrics', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    # Plot 2: Per-class accuracy
    ax = axes[0, 1]
    class_labels = list(metrics['per_class_metrics'].keys())
    class_accs = [metrics['per_class_metrics'][label]['accuracy']*100
                  for label in class_labels]

    bars = ax.barh(class_labels, class_accs, color='#3498db', alpha=0.7)
    ax.set_xlabel('Accuracy (%)', fontsize=11)
    ax.set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 100])
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        count = metrics['per_class_metrics'][class_labels[i]]['count']
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}% (n={count})', ha='left', va='center', fontsize=9)

    # Plot 3: GT vs Predicted distribution
    ax = axes[1, 0]
    labels_sorted = sorted(metrics['labels'])
    gt_counts = [metrics['gt_counts'].get(label, 0) for label in labels_sorted]
    pred_counts = [metrics['pred_counts'].get(label, 0) for label in labels_sorted]

    x = np.arange(len(labels_sorted))
    width = 0.35

    ax.bar(x - width/2, gt_counts, width, label='Ground Truth', color='#2ecc71', alpha=0.7)
    ax.bar(x + width/2, pred_counts, width, label='Predictions', color='#e74c3c', alpha=0.7)

    ax.set_xlabel('Speaker', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Label Distribution: GT vs Predictions', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Text summary
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    LABEL PROPAGATION METRICS SUMMARY
    {'='*40}

    Total Samples: {metrics['n_valid_samples']}

    Overall Performance:
      • Accuracy:  {metrics['accuracy']*100:.2f}%
      • Precision: {metrics['precision']*100:.2f}%
      • Recall:    {metrics['recall']*100:.2f}%
      • F1 Score:  {metrics['f1']*100:.2f}%

    Number of Classes: {len(metrics['labels'])}
    Classes: {', '.join(metrics['labels'])}

    {'='*40}
    """

    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Metrics summary saved: {output_path}")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

base_path_ex = Path.home().joinpath('Dropbox', 'DATASETS_AUDIO', 'Unsupervised_Pipeline', 'TestAO-Irma')
stg4_folder_ex = base_path_ex.joinpath('STG_4', 'STG4_LP1')
stg5_folder_ex = base_path_ex.joinpath('STG_5')

lp_results_csv_ex = stg4_folder_ex / 'lp_results' / 'RUN001_lp_results.csv'
merged_h5_ex = stg4_folder_ex / 'updated_h5_data' / 'merged_dataset_with_labels.h5'
gt_csv_folder_ex = base_path_ex / 'GT_final' / 'filtered_GT'
output_folder_ex = stg5_folder_ex / 'LP_metrics'

parser = argparse.ArgumentParser(
    description='Stage 5a: Create final CSV from LP results with timing and GT metrics'
)

parser.add_argument(
    '--lp_results_csv',
    type=valid_path,
    default=lp_results_csv_ex,
    help='Input LP results CSV file from Stage 4e'
)

parser.add_argument(
    '--merged_dataset_h5',
    type=valid_path,
    default=merged_h5_ex,
    help='Input merged HDF5 dataset with timing information'
)

parser.add_argument(
    '--gt_csv_folder',
    default=gt_csv_folder_ex,
    help='Ground Truth CSV folder (REQUIRED for metrics calculation)'
)

parser.add_argument(
    '--output_folder',
    type=valid_path,
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

lp_results_csv = args.lp_results_csv
merged_h5_path = args.merged_dataset_h5
gt_csv_folder = args.gt_csv_folder
output_folder = args.output_folder
min_overlap_pct = args.min_overlap_pct

# Create output folder
output_folder.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("="*80)
print("STAGE 5a: CREATE FINAL CSV FROM LP RESULTS WITH TIMING AND METRICS")
print("="*80)
print(f"LP results CSV: {lp_results_csv}")
print(f"Merged HDF5: {merged_h5_path}")
print(f"GT CSV folder: {gt_csv_folder}")
print(f"Output folder: {output_folder}")
print(f"Min overlap %: {min_overlap_pct}")
print("="*80)

# Step 1: Load LP results
lp_df = load_lp_results(lp_results_csv)

# Step 2: Load timing information from HDF5
timing_df = load_merged_h5_timing_info(merged_h5_path)

# Step 3: Merge LP results with timing info
print(f"\nMerging LP results with timing information...")
merged_df = pd.merge(lp_df, timing_df, on='merged_unique_id', how='left')
print(f"  ✓ Merged dataframe shape: {merged_df.shape}")

# Step 4: Load GT data
gt_df = load_gt_csvs(gt_csv_folder)

# Step 5: Match predictions to GT
merged_df = match_predictions_to_gt(merged_df, gt_df)

# Step 6: Calculate metrics
metrics = calculate_metrics(merged_df, gt_df=gt_df, min_overlap_pct=min_overlap_pct)

# Step 7: Save detailed results CSV
output_csv_path = output_folder / 'lp_results_with_timing_and_gt.csv'
merged_df.to_csv(output_csv_path, index=False)
print(f"\n✓ Detailed results saved: {output_csv_path}")

# Step 8: Create per-long-wav CSVs
print(f"\nCreating per-long-wav CSV files...")
for long_wav_name, group_df in merged_df.groupby('long_wav_filename'):
    per_wav_csv = output_folder / f'{long_wav_name}_lp_predictions.csv'

    # Select relevant columns
    output_cols = ['merged_unique_id', 'start_time', 'end_time', 'duration',
                   'lp_label', 'lp_confidence', 'matched_gt_speaker',
                   'gt_overlap_pct', 'hdbscan_label', 'hdbscan_prob']

    group_df[output_cols].to_csv(per_wav_csv, index=False)
    print(f"  ✓ {per_wav_csv.name}: {len(group_df)} segments")

# Step 9: Generate plots
if metrics is not None:
    # Confusion matrix
    conf_matrix_path = output_folder / 'confusion_matrix.png'
    plot_confusion_matrix(metrics['confusion_matrix'], metrics['labels'], conf_matrix_path)

    # Metrics summary
    metrics_summary_path = output_folder / 'metrics_summary.png'
    plot_metrics_summary(metrics, metrics_summary_path)

# Step 10: Save metrics report
if metrics is not None:
    metrics_report_path = output_folder / 'metrics_report.txt'
    with open(metrics_report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LABEL PROPAGATION METRICS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Dataset Information:\n")
        f.write(f"  Total samples: {len(merged_df)}\n")
        f.write(f"  Valid samples (>{min_overlap_pct}% overlap): {metrics['n_valid_samples']}\n")
        f.write(f"  Number of classes: {len(metrics['labels'])}\n")
        f.write(f"  Classes: {', '.join(metrics['labels'])}\n\n")

        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']*100:.2f}%\n")
        f.write(f"  Precision: {metrics['precision']*100:.2f}% (macro)\n")
        f.write(f"  Recall:    {metrics['recall']*100:.2f}% (macro)\n")
        f.write(f"  F1 Score:  {metrics['f1']*100:.2f}% (macro)\n\n")

        # DER metrics
        if metrics.get('der_metrics'):
            der = metrics['der_metrics']
            f.write(f"Diarization Error Rate (DER):\n")
            f.write(f"  DER:               {der['der']*100:.2f}%\n")
            f.write(f"  False Alarm Rate:  {der['false_alarm_rate']*100:.2f}%\n")
            f.write(f"  Miss Rate:         {der['miss_rate']*100:.2f}%\n")
            f.write(f"  Speaker Error Rate:{der['speaker_error_rate']*100:.2f}%\n")
            f.write(f"  Total GT Time:     {der['total_gt_time']:.2f}s\n\n")

        # Confidence statistics
        if metrics.get('confidence_stats'):
            conf = metrics['confidence_stats']
            f.write(f"Confidence Statistics:\n")
            f.write(f"  Mean:      {conf['mean_confidence']:.4f}\n")
            f.write(f"  Median:    {conf['median_confidence']:.4f}\n")
            f.write(f"  Std Dev:   {conf['std_confidence']:.4f}\n")
            f.write(f"  Min:       {conf['min_confidence']:.4f}\n")
            f.write(f"  Max:       {conf['max_confidence']:.4f}\n")
            f.write(f"  Q25:       {conf['q25_confidence']:.4f}\n")
            f.write(f"  Q75:       {conf['q75_confidence']:.4f}\n\n")

        # Entropy metrics
        if metrics.get('entropy_metrics'):
            ent = metrics['entropy_metrics']
            f.write(f"Label Distribution Entropy:\n")
            f.write(f"  GT Entropy:        {ent['gt_entropy']:.4f} (normalized: {ent['gt_entropy_normalized']:.4f})\n")
            f.write(f"  Pred Entropy:      {ent['pred_entropy']:.4f} (normalized: {ent['pred_entropy_normalized']:.4f})\n")
            f.write(f"  Matched GT Entropy:{ent['matched_gt_entropy']:.4f}\n")
            f.write(f"  GT Classes:        {ent['n_classes_gt']}\n")
            f.write(f"  Pred Classes:      {ent['n_classes_pred']}\n\n")

        f.write(f"Per-Class Metrics:\n")
        for label, class_metrics in metrics['per_class_metrics'].items():
            f.write(f"  {label}:\n")
            f.write(f"    Accuracy: {class_metrics['accuracy']*100:.2f}%\n")
            f.write(f"    Count: {class_metrics['count']}\n")

        f.write(f"\nLabel Distribution:\n")
        f.write(f"  Ground Truth:\n")
        for label, count in sorted(metrics['gt_counts'].items()):
            f.write(f"    {label}: {count}\n")
        f.write(f"  Predictions:\n")
        for label, count in sorted(metrics['pred_counts'].items()):
            f.write(f"    {label}: {count}\n")

        f.write(f"\n{'='*80}\n")
        f.write(f"CLASSIFICATION REPORT\n")
        f.write(f"{'='*80}\n\n")
        f.write(metrics['classification_report'])

        f.write(f"\n{'='*80}\n")
        f.write(f"CONFUSION MATRIX\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Labels: {metrics['labels']}\n\n")
        for i, gt_label in enumerate(metrics['labels']):
            f.write(f"{gt_label}: {metrics['confusion_matrix'][i]}\n")

    print(f"\n✓ Metrics report saved: {metrics_report_path}")

print(f"\n{'='*80}")
print("STAGE 4F COMPLETED")
print("="*80)
print(f"All results saved to: {output_folder}")
