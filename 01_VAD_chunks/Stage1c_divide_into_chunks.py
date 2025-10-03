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
parser.add_argument('--min_overlap_pert',type=float, default=0.0, help='Minimum overlap percentage for the metric calculation')
parser.add_argument('--azure_flag', type=bool, default=False, help='Flag to indicate csv line columns')
parser.add_argument('--GT_folder_path', default=None, help='Ground Truth CSV folder path')

args = parser.parse_args()

audio_folder = args.stg1_wavs 
csv_folder = args.stg1_final_csv
chunks_wav_folder = args.stg1_chunks_wavs
azure_flag = args.azure_flag
GT_folder_path = Path(args.GT_folder_path)
min_overlap_percentage = float(args.min_overlap_pert)

# Verify GT folder path exists in the OS
if not GT_folder_path.exists():
    GT_flag = False
    GT_folder_path = 'No GT folder provided'
    print('No GT folder provided')
else:
    GT_folder_path = Path(GT_folder_path)
    GT_flag = True
    print(f'GT folder path: {GT_folder_path}')

chunk_duration = float(args.ln)
minimum_chunk_duration = chunk_duration - 0.1 # seconds
step_length = float(args.st) 
verbose = True

print(f'chunk_duration: {chunk_duration}')
print(f'step_size: {step_length}')
print(f'Azureflag: {azure_flag}')
print(f'Minimum overlap percentage: {min_overlap_percentage}')

# Iterate through each of the csv files
for csv_file in csv_folder.glob('*.txt'):
    # Get the filename without extension
    csv_filename = csv_file.stem

    print(f'Processing {csv_filename}...')
    
    # Find the matching audio file in the audio folder
    audio_file = audio_folder.joinpath(csv_filename + '.wav')

    #Verify that the audio file exists
    if not audio_file.exists():
        print(f'WARNING: {audio_file} does not exist. Skipping...')
        continue

    idx_total = 0
    # Iterate each line in the csv file
    for line in csv_file.open():

        # # Get the start and stop times
        if azure_flag:
            pred_label, start_time, stop_time, text_pred, prob_pred = line.split('\t')
        else:
            filename, start_time, stop_time = line.split('\t')


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
                    print(f'\tUsed end of audio: {current_time + chunk_duration:.2f} > {stop_time:.2f}')
                current_stop_time = stop_time
            else:
                current_stop_time = current_time + chunk_duration

            # Verify that the chunk duration is at least the minimum chunk duration
            if current_stop_time - current_time < minimum_chunk_duration:
                if verbose:
                    print(f'\tBREAK\t Last iteration: {current_stop_time - current_time:.2f} < {minimum_chunk_duration:.2f}\n')
                break

            # Speaker ID
            speaker_id = f'spkX'

            # If GT folder path is provided, check if the current chunk overlaps with any GT segment
            if GT_flag:
                GT_wav_file_path = GT_folder_path.joinpath(f'{csv_filename}_GT.csv')
                # Read GT from tab-separated csv file with colums: speaker_id, lang, start_time, stop_time
                GT_segments = []
                for gt_line in GT_wav_file_path.open():
                    gt_speaker, gt_lang, gt_start, gt_stop = gt_line.split('\t')
                    GT_segments.append((gt_speaker, float(gt_start), float(gt_stop)))
            
                # Check for overlap
                for gt_speaker, gt_start, gt_stop in GT_segments:
                    overlap = max(0, min(current_stop_time, gt_stop) - max(current_time, gt_start))
                    overlap_ratio = overlap / (current_stop_time - current_time)
                    if overlap_ratio >= min_overlap_percentage:
                        log_print(f'\tOverlap with GT: {current_time:.2f} - {current_stop_time:.2f}')
                        speaker_id = gt_speaker
                        break


            # Create the output file name
            output_filename = f'{csv_filename}_{speaker_id}_{current_time:.2f}_{current_stop_time:.2f}.wav'
            output_file = chunks_wav_folder.joinpath(output_filename)

            if verbose:
                print(f'{output_filename} - New chunk: {current_time:.2f} - {current_stop_time:.2f}')

            # Split the audio file
            _, _ = ffmpeg_split_audio(audio_file, output_file, \
                start_time_csv = str(current_time),
                stop_time_csv = str(current_stop_time),
                sr = 16000,
                verbose = False,
                formatted = False,
                output_video_flag = False,
                times_as_integers = False)

            # Update the current time
            current_time += step_length


    

