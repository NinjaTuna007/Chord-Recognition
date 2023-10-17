"""
Generate training data based on the ground truth files
this process leverages the frontend and the ground truth data
"""
import os

import math
import numpy as np
import librosa
import feature

from .chords import convert_gt, chord_nums_to_inds, chords_nums_to_inds


def get_feature(audiopath, args):
    x, _ = librosa.load(audiopath, sr=args.sr)
    # x = librosa.effects.harmonic(x)
    if args.feature_type == 'CQT':
        X = feature.get_cqt(x, args)
    elif args.feature_type == 'MFCC':
        X = feature.get_mfcc(x, args)
    elif args.feature_type == 'STFT':
        X = feature.get_stft(x, args)
    elif args.feature_type == 'MEL':
        X = feature.get_mel_spectrogram(x, args)
    elif args.feature_type == 'CHROMA_CQT':
        X = feature.get_chroma_cqt(x, args)
    elif args.feature_type == 'CHROMA_STFT':
        X = feature.get_chroma_stft(x, args)
    return X.T

def iter_songs_list(data_list):
    with open(data_list, 'r') as f:
        for song_folder in f:
            song_folder = song_folder.rstrip()
            song_title = song_folder[:-len(song_folder.split('.')[-1]) - 1]
            yield song_title, song_folder


def gen_test_data(data_list, audio_path, params):
    for song_name, audio_path in iter_songs_list(data_list):
        yield (song_name, get_feature(f'{audio_path}/{audio_path}', params, mod_steps=(0,)))


def gen_train_data(args, data_list):
    data = []
    for song_title, song_folder in iter_songs_list(data_list):
        print('collecting training data of ', song_title)
        X = get_feature(f'{args.audio_path}/{song_folder}', args)

        y_nums = convert_gt(f'{args.gt_path}/{song_title}.lab', args.hop_length, args.sr, len(X), args.category)

        y = chords_nums_to_inds(y_nums, args.category)
        y = np.array(y)
        data.append((song_title, X, y))
    
    return data