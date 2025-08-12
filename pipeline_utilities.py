import os
import shutil
import re
import csv
import time
import sys
import subprocess as subp
import json
import pandas as pd
import datetime
from pathlib import Path
import argparse

def log_print(*args, **kwargs):
    """Prints to stdout and also logs to log_path."""

    log_path = kwargs.pop('lp', 'default_log.txt')
    print_to_console = kwargs.pop('print', True)

    message = " ".join(str(a) for a in args)
    if print_to_console:
        print(message)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def get_total_video_length(input_video_path):
    script_out = subp.check_output(["ffprobe", "-v", "quiet", "-show_format", "-print_format", "json", input_video_path])
    ffprobe_data = json.loads(script_out)
    video_duration_seconds = float(ffprobe_data["format"]["duration"])

    return video_duration_seconds


def extract_basename(input_str, suffix_added):
    mymatch = re.search(r'.+(?=_{})'.format(suffix_added), input_str)
    if mymatch != None:
        mystring = mymatch.group()    
    else:
        mystring = ''
    return mystring


def find_audio_duration(current_transcript_pth, audios_folder, suffix_added, verbose=False):
    # Find the path of the audio
    if suffix_added != '':
        current_basename = extract_basename(current_transcript_pth.stem, suffix_added)
        candidate_path = audios_folder.joinpath(current_basename + '.wav')
    else:
        candidate_path = audios_folder.joinpath(current_transcript_pth.stem + '.wav')
    
    if candidate_path.exists ():
        if verbose:
            print (f'File exist: {candidate_path}')
        return get_total_video_length(candidate_path)
    else:
        sys.exit(f'Error! Audio {candidate_path} was not located!')

def get_platform():
    platforms = {
        'linux1' : 'Linux',
        'linux2' : 'Linux',
        'darwin' : 'OS X',
        'win32' : 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform
    
    return platforms[sys.platform]


def ffmpeg_split_audio(input_video, output_pth,
            start_time_csv = '0.00',
            stop_time_csv = 'default',
            sr = 16000,
            verbose = False,
            formatted = False,
            output_video_flag = False,
            times_as_integers = False):

    if times_as_integers:
        start_time_csv = str(start_time_csv)
        stop_time_csv = str(stop_time_csv)

    if formatted:
        (hstart, mstart, sstart) = start_time_csv.split(':')
        start_time_csv = str(float(hstart) * 3600 + float(mstart) * 60 + float(sstart))

        (hstop, mstop, sstop) = stop_time_csv.split(':')
        stop_time_csv = str(float(hstop) * 3600 + float(mstop) * 60 + float(sstop))

    if verbose:
        if stop_time_csv == 'default':
            if get_platform() == 'Linux':
                cmd = f"ffmpeg -i '{input_video}' -acodec pcm_s16le -ac 1 -ar {sr} '{output_pth}'"
            else:
                cmd = f"ffmpeg -i {input_video} -acodec pcm_s16le -ac 1 -ar {sr} {output_pth}"
            subp.run(cmd, shell=True)
            return 'non_valid', 'non_valid'
    else:
        if stop_time_csv == 'default':
            if get_platform() == 'Linux':
                cmd = f"ffmpeg -i '{input_video}' -hide_banner -loglevel error -acodec pcm_s16le -ac 1 -ar {sr} '{output_pth}'"
            else:
                cmd = f"ffmpeg -i {input_video} -hide_banner -loglevel error -acodec pcm_s16le -ac 1 -ar {sr} {output_pth}"
            subp.run(cmd, shell=True)
            return 'non_valid', 'non_valid'

    video_duration_seconds = get_total_video_length(input_video)

    # Check stop time is larger than start time
    if float(start_time_csv) >= float(stop_time_csv):
        sys.exit(f'Error! Start time {start_time_csv} is larger than stop time {stop_time_csv}')

    # Check stop time is less than duration of the video
    if float(stop_time_csv) > video_duration_seconds:
        sys.exit(f'Warning! [changed] Stop time {stop_time_csv} is larger than video duration {video_duration_seconds}')
        stop_time_csv = str(video_duration_seconds)
    
    # convert the starting time/stop time from seconds -> 00:00:00
    start_time_format = time.strftime("%H:%M:%S", time.gmtime(int(start_time_csv.split('.')[0]))) + \
        '.' + start_time_csv.split('.')[-1][0:2]
    stop_time_format = time.strftime("%H:%M:%S", time.gmtime(int(stop_time_csv.split('.')[0]))) + \
        '.' + stop_time_csv.split('.')[-1][0:2]

    if verbose:
        print(f'{start_time_format} - {stop_time_format}')
        if output_video_flag:
            ffmpeg_params = f' -c:v libx264 -crf 30 '
        else:
            ffmpeg_params = f' -acodec pcm_s16le -ac 1 -ar {sr} '
    else:
        if output_video_flag:
            ffmpeg_params = f' -hide_banner -loglevel error -c:v libx264 -crf 30 '
        else:
            ffmpeg_params = f' -hide_banner -loglevel error -acodec pcm_s16le -ac 1 -ar {sr} '

    if get_platform() == 'Linux':
        cmd = f"ffmpeg -i '{input_video}' '{ffmpeg_params}' -ss '{start_time_format}' -to  '{stop_time_format}' '{output_pth}'"
    else:
        cmd = f"ffmpeg -i {input_video}  {ffmpeg_params} -ss {start_time_format} -to  {stop_time_format} {output_pth}"

    # print(cmd)

    subp.run(cmd, shell=True)

    return start_time_csv, stop_time_csv


def matching_basename_pathlib_gt_pred(GT_pth, pred_pth, 
        gt_suffix_added='', pred_suffix_added='',
        gt_ext = 'txt', pred_ext = 'txt', verbose = False):

    if gt_suffix_added == '':
        GT_list = sorted(list(GT_pth.glob(f'*.{gt_ext}')))
    else:
        GT_list = sorted(list(GT_pth.glob(f'*_{gt_suffix_added}.{gt_ext}')))

    if pred_suffix_added == '':
        pred_list = sorted(list(pred_pth.glob(f'*.{pred_ext}')))
    else:
        pred_list = sorted(list(pred_pth.glob(f'*_{pred_suffix_added}.{pred_ext}')))


    if len(GT_list) == 0:
        print(f'ERROR GT list empty. Check suffix')
    
    if len(pred_list) == 0:
        print(f'ERROR!! Pred list is empty. Check suffix')

    # Extract basenames from pathlib

    if gt_suffix_added == '':
        gt_list_basenames = [x.stem for x in GT_list]
    else:
        gt_list_basenames = [extract_basename(x.name, gt_suffix_added) for x in GT_list]

    if pred_suffix_added == '':
        pred_list_basenames = [x.stem for x in pred_list]
    else:
        pred_list_basenames = [extract_basename(x.name, pred_suffix_added) for x in pred_list]

    if verbose:
        print(f'GT: {gt_list_basenames}\nPred: {pred_list_basenames}')

    # Check for duplicates
    if len(gt_list_basenames) != len(list(set(gt_list_basenames))):
        sys.exit(f'Duplicates found at folder {GT_pth}')

    if len(pred_list_basenames) != len(list(set(pred_list_basenames))):
        sys.exit(f'Duplicates found at folder {pred_pth}')

    gt_idxs = []
    for idx, current_gt in enumerate(gt_list_basenames):
        if current_gt in pred_list_basenames:
            gt_idxs.append(idx)

    pred_idxs = []
    for idx, current_pred in enumerate(pred_list_basenames):
        if current_pred in gt_list_basenames:
            pred_idxs.append(idx)

    # Verify same length
    if len(gt_idxs) != len(pred_idxs):
        sys.exit(f'matching indexes are not equal!')

    # Return the tuples
    matching_list = []
    for idx in range(0, len(gt_idxs)):
        matching_list.append((GT_list[gt_idxs[idx]], pred_list[pred_idxs[idx]]))

    if verbose:
        print(matching_list)

    return matching_list