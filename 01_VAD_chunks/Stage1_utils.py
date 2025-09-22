from pathlib import Path
from pyannote.core import Annotation, Segment
import sys
import re

def vad_format(current_transcript_pth):
    # Src	StartTime	EndTime
    # s0s0	3.93	5.86
    with open(current_transcript_pth) as f:
        lines = [line.rstrip() for line in f]
    lines.pop(0)

    reference = Annotation()
    for line in lines:
        Speaker, Lang, start_time, stop_time = line.split('\t')
        reference[Segment(float(start_time), float(stop_time))] = 'speech'

    return reference


def select_method(pred_method):
    if pred_method == 'audiotok' or \
        pred_method == 'shas' or \
        pred_method == 'BAS' or \
        pred_method == 'cobra' or \
        pred_method == 'silero':
        return audiotok_format
    elif pred_method == 'inaSS':
        return inaSS_format
    elif pred_method == 'speechbrain':
        return speechbrain_format
    elif pred_method == 'whisper':
        return whisper_format
    elif pred_method == 'aws':
        return aws_format
    elif pred_method == 'azure':
        return azure_format
    

def whisper_format(current_transcript_pth):
    # Speech	en	38.0	39.0	I can't.	0.012211376801133001

    with open(current_transcript_pth) as f:
        try:
            with open(current_transcript_pth, encoding="utf8") as f:
                lines = [line.rstrip() for line in f]
        except UnicodeDecodeError as e:
            if str(e) == "'charmap' codec can't decode byte 0x90 in position 1100: character maps to <undefined>'":
                import pdb; pdb.set_trace()
            else:
                raise e

    hypothesis = Annotation()
    for line in lines:
        my_source, my_lang, start_time, stop_time, txt_pred, my_prob = line.split('\t')
        
        hypothesis[Segment(float(start_time), float(stop_time))] = 'speech'
    
    return hypothesis


def aws_format(current_transcript_pth):
    # Talking60_easy_rnd-006.wav	spk_0	en-US	1.31	2.14	Okay	0.4618

    with open(current_transcript_pth) as f:
        lines = [line.rstrip() for line in f]

    hypothesis = Annotation()
    for line in lines:
        fname, my_source, my_lang, start_time, stop_time, txt_pred, my_prob = line.split('\t')
        hypothesis[Segment(float(start_time), float(stop_time))] = 'speech'
    
    return hypothesis


def azure_format(current_transcript_pth):
    # 1	0.33	1.12	Try to run it again.	0.7578845

    with open(current_transcript_pth) as f:
        lines = [line.rstrip() for line in f]

    hypothesis = Annotation()
    for line in lines:
        my_source, start_time, stop_time, txt_pred, my_prob = line.split('\t')
        hypothesis[Segment(float(start_time), float(stop_time))] = 'speech'
    
    return hypothesis


def speechbrain_format(current_transcript_pth):
    # 'male/female/music/noise	0.10	20.1'

    with open(current_transcript_pth) as f:
        lines = [line.rstrip() for line in f]

    hypothesis = Annotation()
    for line in lines:
        _ , start_time, stop_time, my_source = line.split('\t')
        if my_source == 'SPEECH' : 
            hypothesis[Segment(float(start_time), float(stop_time))] = 'speech'
    
    return hypothesis


def audiotok_format(current_transcript_pth):
    # 'voice	0.10	20.1'
    with open(current_transcript_pth) as f:
        lines = [line.rstrip() for line in f]

    hypothesis = Annotation()
    for line in lines:
        _ , start_time, stop_time = line.split('\t')
        hypothesis[Segment(float(start_time), float(stop_time))] = 'speech'
    
    return hypothesis


def inaSS_format(current_transcript_pth):
    # 'male/female/music/noise	0.10	20.1'

    with open(current_transcript_pth) as f:
        lines = [line.rstrip() for line in f]

    hypothesis = Annotation()
    for line in lines:
        my_source , start_time, stop_time = line.split('\t')
        if my_source == 'male' or my_source == 'female': 
            hypothesis[Segment(float(start_time), float(stop_time))] = 'speech'
    
    return hypothesis
