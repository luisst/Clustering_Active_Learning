from pathlib import Path
import argparse

from pipeline_utilities import valid_path, ffmpeg_split_audio, log_print


base_path_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO','TestAO-Liz')
audio_folder_ex = base_path_ex.joinpath('Testset_stage1','input_wavs')
csv_folder_ex = base_path_ex.joinpath('Testset_stage1','input_csv')
chunks_WAV_ex = base_path_ex.joinpath('Testset_stage2','wav_chunks')
seg_ln_ex = '1.0'
step_size_ex = '0.2'

parser = argparse.ArgumentParser()
parser.add_argument('--stg1_wavs', type=valid_path, default=audio_folder_ex, help='Stg1 WAVs folder path')
parser.add_argument('--stg1_final_csv', type=valid_path, default=csv_folder_ex, help='Stg1 VAD csvs folder path')
parser.add_argument('--stg1_chunks_wavs', type=valid_path, default=chunks_WAV_ex, help='Stg2 chunks wavs folder path')
parser.add_argument('--ln', type=float, default=seg_ln_ex, help='Stg2 chunks length ihn seconds')
parser.add_argument('--st', type=float, default=step_size_ex, help='Stg2 chunks step_size in seconds')
parser.add_argument('--min_overlap_pert',type=float, default=0.7, help='Minimum overlap percentage for the metric calculation')
parser.add_argument('--azure_flag', type=bool, default=False, help='Flag to indicate csv line columns')
parser.add_argument('--GT_folder_path', default=None, help='Ground Truth CSV folder path')

args = parser.parse_args()

audio_folder = args.stg1_wavs 
csv_folder = args.stg1_final_csv
chunks_wav_folder = args.stg1_chunks_wavs
azure_flag = args.azure_flag
GT_folder_path = Path(args.GT_folder_path) if args.GT_folder_path else None
min_overlap_percentage = float(args.min_overlap_pert)

# Setup logging
log_file = chunks_wav_folder / 'stage1c_chunk_creation_log.txt'
chunks_wav_folder.mkdir(parents=True, exist_ok=True)

# Initialize log file with header
with open(log_file, 'w') as f:
    f.write("Stage 1c - Audio Chunk Creation Log\n")
    f.write("="*50 + "\n")
    f.write(f"Timestamp: {Path(__file__).stat().st_mtime}\n")
    f.write(f"Audio folder: {audio_folder}\n")
    f.write(f"CSV folder: {csv_folder}\n")
    f.write(f"Output folder: {chunks_wav_folder}\n")
    f.write(f"Chunk duration: {args.ln}s\n")
    f.write(f"Step size: {args.st}s\n")
    f.write(f"Azure flag: {azure_flag}\n")
    f.write(f"Min overlap percentage: {min_overlap_percentage}\n")
    f.write("-"*50 + "\n\n")

# Verify GT folder path exists in the OS
if GT_folder_path is None or not GT_folder_path.exists():
    GT_flag = False
    GT_folder_path = 'No GT folder provided'
    log_print('No GT folder provided', lp=log_file)
else:
    GT_folder_path = Path(GT_folder_path)
    GT_flag = True
    log_print(f'GT folder path: {GT_folder_path}', lp=log_file)

chunk_duration = float(args.ln)
minimum_chunk_duration = chunk_duration - 0.1 # seconds
step_length = float(args.st) 
verbose = True

# Add threshold for detecting 50-50 splits
ambiguous_threshold = 0.1  # If the difference between top two overlaps is less than 10%, it's ambiguous

log_print(f'chunk_duration: {chunk_duration}', lp=log_file)
log_print(f'step_size: {step_length}', lp=log_file)
log_print(f'Azureflag: {azure_flag}', lp=log_file)
log_print(f'Minimum overlap percentage: {min_overlap_percentage}', lp=log_file)
log_print(f'Ambiguous overlap threshold: {ambiguous_threshold * 100}%', lp=log_file)

# Statistics tracking
total_chunks_created = 0
total_ambiguous_chunks = 0
speaker_chunk_counts = {}

# Iterate through each of the csv files
for csv_file in csv_folder.glob('*.txt'):
    # Get the filename without extension
    csv_filename = csv_file.stem

    GT_segments = []

    if GT_flag:
        GT_wav_file_path = GT_folder_path.joinpath(f'{csv_filename}_GT.csv')
        # Read GT from tab-separated csv file with columns: speaker_id, lang, start_time, stop_time, filename, rnd_idx
        try:
            for gt_line in GT_wav_file_path.open():
                gt_speaker, gt_lang, gt_start, gt_stop, gt_filename, gt_rnd_idx = gt_line.strip().split('\t')
                GT_segments.append((gt_speaker, float(gt_start), float(gt_stop)))
            log_print(f'Loaded {len(GT_segments)} GT segments for {csv_filename}', lp=log_file)
        except FileNotFoundError:
            log_print(f'WARNING: GT file not found: {GT_wav_file_path}', lp=log_file)
        except Exception as e:
            log_print(f'ERROR loading GT file {GT_wav_file_path}: {e}', lp=log_file)

    log_print(f'Processing {csv_filename}...', lp=log_file)
    
    # Find the matching audio file in the audio folder
    audio_file = audio_folder.joinpath(csv_filename + '.wav')

    #Verify that the audio file exists
    if not audio_file.exists():
        log_print(f'WARNING: {audio_file} does not exist. Skipping...', lp=log_file)
        continue

    file_chunk_count = 0
    idx_total = 0
    
    # Iterate each line in the csv file
    for line in csv_file.open():
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        # # Get the start and stop times
        try:
            if azure_flag:
                pred_label, start_time, stop_time, text_pred, prob_pred = line.split('\t')
            else:
                filename, start_time, stop_time = line.split('\t')
        except ValueError as e:
            log_print(f'ERROR parsing line in {csv_filename}: {line} - {e}', lp=log_file)
            continue

        start_time = float(start_time)
        stop_time = float(stop_time)

        # Verify that the chunk duration is at least the minimum chunk duration
        if stop_time - start_time < minimum_chunk_duration:
            continue

        # Iterate through the audio file
        current_time = start_time
        while current_time < stop_time:

            # Compute the stop time
            current_stop_time = min(current_time + chunk_duration, stop_time)
            if current_time + chunk_duration > stop_time:
                if verbose:
                    log_print(f'\tUsed end of audio: {current_time + chunk_duration:.2f} > {stop_time:.2f}\n\n', lp=log_file)
                current_stop_time = stop_time
            else:
                current_stop_time = current_time + chunk_duration

            # Verify that the chunk duration is at least the minimum chunk duration
            if current_stop_time - current_time < minimum_chunk_duration:
                if verbose:
                    log_print(f'\tBREAK\t Last iteration: {current_stop_time - current_time:.2f} < {minimum_chunk_duration:.2f}\n\n\n', lp=log_file)
                break

            # Speaker ID
            speaker_id = f'spkX'

            # If GT folder path is provided, evaluate all overlapping GT segments
            if GT_flag:

                # When GT is provided, non GT chunks will have spkNoise
                speaker_id = "spkNoise"

                overlap_results = []
                chunk_duration_actual = current_stop_time - current_time
                
                # Evaluate all GT segments for overlap
                for gt_speaker, gt_start, gt_stop in GT_segments:
                    overlap = max(0, min(current_stop_time, gt_stop) - max(current_time, gt_start))

                    # defensive: avoid div-by-zero and clamp to [0,1]
                    if chunk_duration_actual > 0.0:
                        overlap_ratio = overlap / chunk_duration_actual
                        # clamp to [0,1] to avoid tiny floating-point >1 or <0
                        overlap_ratio = max(0.0, min(1.0, overlap_ratio))
                    else:
                        overlap_ratio = 0.0
                    
                    if overlap_ratio >= min_overlap_percentage:
                        overlap_results.append((gt_speaker, overlap_ratio, overlap, gt_start, gt_stop))
                
                # Process overlap results
                if overlap_results:
                    # Sort by overlap ratio (descending)
                    overlap_results.sort(key=lambda x: x[1], reverse=True)
                    
                    best_speaker, best_ratio, best_overlap, best_start, best_stop = overlap_results[0]
                    
                    # Check for ambiguous overlaps (roughly 50-50 split)
                    if len(overlap_results) >= 2:
                        second_speaker, second_ratio, second_overlap, second_start, second_stop = overlap_results[1]
                        ratio_difference = best_ratio - second_ratio
                        
                        if ratio_difference <= ambiguous_threshold:
                            total_ambiguous_chunks += 1
                            log_print("="*80, lp=log_file)
                            log_print("WARNING: AMBIGUOUS OVERLAP DETECTED!", lp=log_file)
                            log_print("="*80, lp=log_file)
                            log_print(f"Chunk: {current_time:.2f} - {current_stop_time:.2f} (duration: {chunk_duration_actual:.2f}s)", lp=log_file)
                            log_print(f"File: {csv_filename}", lp=log_file)
                            log_print(f"Top overlaps:", lp=log_file)
                            for i, (spk, ratio, ovlp, start, stop) in enumerate(overlap_results[:3]):
                                log_print(f"  {i+1}. Speaker '{spk}': {ratio:.1%} overlap ({ovlp:.2f}s) - GT segment: {start:.2f}-{stop:.2f}", lp=log_file)
                            log_print(f"Difference between top two: {ratio_difference:.1%}", lp=log_file)
                            log_print("Consider skipping this chunk or reviewing GT segmentation!", lp=log_file)
                            log_print("="*80, lp=log_file)
                    
                    # Assign the best overlapping speaker
                    speaker_id = best_speaker
                    
                    if verbose:
                        overlap_info = f"Best overlap: {best_speaker} ({best_ratio:.1%} | {best_ratio:.2f}, {best_overlap:.2f}s)"
                        if len(overlap_results) > 1:
                            overlap_info += f" | {len(overlap_results)} total overlaps"
                        log_print(f'\t{overlap_info} - Chunk: {current_time:.2f} - {current_stop_time:.2f}', lp=log_file)

            # Track speaker statistics
            if speaker_id not in speaker_chunk_counts:
                speaker_chunk_counts[speaker_id] = 0
            speaker_chunk_counts[speaker_id] += 1

            # Create the output file name
            output_filename = f'{csv_filename}_{speaker_id}_{current_time:.2f}_{current_stop_time:.2f}.wav'
            output_file = chunks_wav_folder.joinpath(output_filename)

            if verbose:
                log_print(f'{output_filename} - New chunk: {current_time:.2f} - {current_stop_time:.2f}', lp=log_file)

            # Split the audio file
            try:
                _, _ = ffmpeg_split_audio(audio_file, output_file, \
                    start_time_csv = str(current_time),
                    stop_time_csv = str(current_stop_time),
                    sr = 16000,
                    verbose = False,
                    formatted = False,
                    output_video_flag = False,
                    times_as_integers = False)
                
                total_chunks_created += 1
                file_chunk_count += 1
                
            except Exception as e:
                log_print(f'ERROR creating chunk {output_filename}: {e}', lp=log_file)

            # Update the current time
            current_time += step_length

    log_print(f'Completed {csv_filename}: {file_chunk_count} chunks created', lp=log_file)

# Log final statistics
log_print("\n" + "="*50, lp=log_file)
log_print("PROCESSING SUMMARY", lp=log_file)
log_print("="*50, lp=log_file)
log_print(f"Total chunks created: {total_chunks_created}", lp=log_file)
log_print(f"Total ambiguous chunks detected: {total_ambiguous_chunks}", lp=log_file)

if speaker_chunk_counts:
    log_print("\nSpeaker chunk distribution:", lp=log_file)
    for speaker, count in sorted(speaker_chunk_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_chunks_created) * 100 if total_chunks_created > 0 else 0
        log_print(f"  {speaker}: {count} chunks ({percentage:.1f}%)", lp=log_file)

log_print(f"\nLog file saved to: {log_file}", lp=log_file)
log_print("Processing completed successfully!", lp=log_file)