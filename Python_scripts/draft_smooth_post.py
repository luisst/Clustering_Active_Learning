import numpy as np
from collections import Counter
from typing import List, Tuple, Optional

def filter_quick_interjections(
    speaker_labels: List[str], 
    time_step: float = 0.4,
    min_segment_duration: float = 1.5,
    max_interjection_duration: float = 1.0,
    min_gap_to_consider: float = 0.8
) -> List[str]:
    """
    Remove quick interjections and brief speaker changes, keeping only substantial speech segments.
    
    Args:
        speaker_labels: List of speaker labels for each time chunk
        time_step: Time between consecutive chunks (seconds)
        min_segment_duration: Minimum duration to keep a speaker segment (seconds)
        max_interjection_duration: Maximum duration considered as interjection (seconds)
        min_gap_to_consider: Minimum gap between same speaker to not merge (seconds)
    
    Returns:
        Filtered speaker labels with interjections removed
    """
    
    if len(speaker_labels) == 0:
        return speaker_labels
    
    # Convert durations to chunk counts
    min_segment_chunks = int(min_segment_duration / time_step)
    max_interjection_chunks = int(max_interjection_duration / time_step)
    min_gap_chunks = int(min_gap_to_consider / time_step)
    
    # Step 1: Identify speaker segments
    segments = identify_speaker_segments(speaker_labels)
    
    # Step 2: Filter out brief interjections
    filtered_segments = []
    
    for i, (speaker, start_idx, end_idx, duration_chunks) in enumerate(segments):
        
        # Keep long segments unconditionally
        if duration_chunks >= min_segment_chunks:
            filtered_segments.append((speaker, start_idx, end_idx, duration_chunks))
            continue
        
        # For short segments, check if they're interjections
        is_interjection = is_likely_interjection(
            segments, i, duration_chunks, max_interjection_chunks, min_gap_chunks
        )
        
        if not is_interjection:
            filtered_segments.append((speaker, start_idx, end_idx, duration_chunks))
    
    # Step 3: Reconstruct timeline with filtered segments
    smoothed_labels = reconstruct_timeline(
        speaker_labels, filtered_segments, min_gap_chunks
    )
    
    return smoothed_labels


def identify_speaker_segments(speaker_labels: List[str]) -> List[Tuple[str, int, int, int]]:
    """
    Identify continuous segments of the same speaker.
    
    Returns:
        List of (speaker, start_idx, end_idx, duration_chunks)
    """
    segments = []
    if not speaker_labels:
        return segments
    
    current_speaker = speaker_labels[0]
    start_idx = 0
    
    for i in range(1, len(speaker_labels)):
        if speaker_labels[i] != current_speaker:
            # End of current segment
            segments.append((
                current_speaker, 
                start_idx, 
                i - 1, 
                i - start_idx
            ))
            current_speaker = speaker_labels[i]
            start_idx = i
    
    # Add final segment
    segments.append((
        current_speaker, 
        start_idx, 
        len(speaker_labels) - 1, 
        len(speaker_labels) - start_idx
    ))
    
    return segments


def is_likely_interjection(
    segments: List[Tuple[str, int, int, int]], 
    segment_idx: int, 
    duration_chunks: int, 
    max_interjection_chunks: int,
    min_gap_chunks: int
) -> bool:
    """
    Determine if a short segment is likely an interjection.
    
    Criteria:
    1. Short duration (< max_interjection_chunks)
    2. Surrounded by the same speaker with small gaps
    3. Or isolated brief segment between longer segments of other speakers
    """
    
    if duration_chunks > max_interjection_chunks:
        return False
    
    current_speaker, start_idx, end_idx, _ = segments[segment_idx]
    
    # Check previous segment
    prev_speaker = None
    prev_end_idx = -1
    if segment_idx > 0:
        prev_speaker, _, prev_end_idx, _ = segments[segment_idx - 1]
    
    # Check next segment  
    next_speaker = None
    next_start_idx = len(segments) + 100  # Large number as default
    if segment_idx < len(segments) - 1:
        next_speaker, next_start_idx, _, _ = segments[segment_idx + 1]
    
    # Case 1: Same speaker before and after with small gaps
    if prev_speaker == next_speaker and prev_speaker != current_speaker:
        gap_before = start_idx - prev_end_idx - 1 if prev_end_idx >= 0 else float('inf')
        gap_after = next_start_idx - end_idx - 1 if next_start_idx < 100 else float('inf')
        
        if gap_before <= min_gap_chunks and gap_after <= min_gap_chunks:
            return True
    
    # Case 2: Very brief isolated segment
    if duration_chunks <= 2:  # Less than ~1 second with 0.4s steps
        return True
    
    # Case 3: Brief segment between much longer segments of different speakers
    if prev_speaker and next_speaker and prev_speaker != next_speaker:
        # Get duration of surrounding segments
        prev_duration = segments[segment_idx - 1][3] if segment_idx > 0 else 0
        next_duration = segments[segment_idx + 1][3] if segment_idx < len(segments) - 1 else 0
        
        # If current segment is much shorter than neighbors, it's likely interjection
        if (duration_chunks < prev_duration / 3 and duration_chunks < next_duration / 3 and
            prev_duration > 5 and next_duration > 5):  # Neighbors are substantial
            return True
    
    return False


def reconstruct_timeline(
    original_labels: List[str], 
    filtered_segments: List[Tuple[str, int, int, int]],
    min_gap_chunks: int
) -> List[str]:
    """
    Reconstruct timeline by filling gaps left by removed interjections.
    """
    result = [''] * len(original_labels)
    
    # First, place all kept segments
    for speaker, start_idx, end_idx, _ in filtered_segments:
        for i in range(start_idx, end_idx + 1):
            result[i] = speaker
    
    # Fill gaps by extending neighboring segments or using majority vote
    for i in range(len(result)):
        if result[i] == '':
            result[i] = fill_gap(result, i, original_labels, min_gap_chunks)
    
    return result


def fill_gap(
    result: List[str], 
    gap_idx: int, 
    original_labels: List[str], 
    min_gap_chunks: int
) -> str:
    """
    Fill a gap left by removed interjection.
    """
    # Look for nearest non-empty labels
    left_speaker = None
    right_speaker = None
    
    # Search left
    for i in range(gap_idx - 1, -1, -1):
        if result[i] != '':
            left_speaker = result[i]
            left_distance = gap_idx - i
            break
    
    # Search right  
    for i in range(gap_idx + 1, len(result)):
        if result[i] != '':
            right_speaker = result[i]
            right_distance = i - gap_idx
            break
    
    # Decision logic
    if left_speaker == right_speaker and left_speaker is not None:
        # Same speaker on both sides - extend that speaker
        return left_speaker
    elif left_speaker is not None and right_speaker is not None:
        # Different speakers - choose the closer one, or left if equal
        if 'left_distance' in locals() and 'right_distance' in locals():
            return left_speaker if left_distance <= right_distance else right_speaker
        return left_speaker
    elif left_speaker is not None:
        return left_speaker
    elif right_speaker is not None:
        return right_speaker
    else:
        # Fallback to original label
        return original_labels[gap_idx]


def analyze_filtering_results(
    original_labels: List[str], 
    filtered_labels: List[str],
    time_step: float = 0.4
) -> None:
    """
    Print analysis of what was filtered out.
    """
    original_segments = identify_speaker_segments(original_labels)
    filtered_segments = identify_speaker_segments(filtered_labels)
    
    print(f"Original segments: {len(original_segments)}")
    print(f"Filtered segments: {len(filtered_segments)}")
    print(f"Removed segments: {len(original_segments) - len(filtered_segments)}")
    
    # Find removed segments
    removed_count = 0
    total_removed_time = 0
    
    for orig_seg in original_segments:
        speaker, start, end, duration = orig_seg
        found = False
        for filt_seg in filtered_segments:
            if (filt_seg[0] == speaker and 
                abs(filt_seg[1] - start) <= 2 and 
                abs(filt_seg[2] - end) <= 2):
                found = True
                break
        
        if not found:
            removed_count += 1
            total_removed_time += duration * time_step
            if duration * time_step <= 1.2:  # Likely interjections
                print(f"Removed short segment: {speaker} ({duration * time_step:.1f}s)")
    
    print(f"Total removed time: {total_removed_time:.1f} seconds")


# Example usage
if __name__ == "__main__":
    # Example with typical kids' speech pattern
    sample_labels = [
        'Alice', 'Alice', 'Alice', 'Bob', 'Alice', 'Alice', 'Alice', 'Alice',
        'Charlie', 'Charlie', 'Charlie', 'Charlie', 'Alice', 'Charlie', 'Charlie',
        'Bob', 'Bob', 'Bob', 'Bob', 'Bob', 'Alice', 'Bob', 'Bob', 'Bob'
    ]
    
    print("Original labels:")
    print(sample_labels)
    
    filtered = filter_quick_interjections(
        sample_labels, 
        time_step=0.4,
        min_segment_duration=1.6,  # Keep segments >= 1.6 seconds
        max_interjection_duration=1.0  # Remove interjections <= 1.0 seconds
    )
    
    print("\nFiltered labels:")
    print(filtered)
    
    analyze_filtering_results(sample_labels, filtered)