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
from .params import feature_params


def get_feature(audiopath, feature_type):
    x, sr = librosa.load(audiopath, sr=feature_params[feature_type]['fs'])
    if feature_type == 'CQT':
        X = feature.get_cqt(x, feature_param=feature_params[feature_type])
    elif feature_type == 'MFCC':
        X = feature.get_mfcc(x, feature_param=feature_params[feature_type])
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


def gen_train_data(feature_type, data_list, audio_path, gt_path, category):
    data = []
    for song_title, song_folder in iter_songs_list(data_list):
        print('collecting training data of ', song_title)
        X = get_feature(f'{audio_path}/{song_folder}', feature_type)

        y_nums = convert_gt(f'{gt_path}/{song_title}.lab', feature_params[feature_type]['hop_length'], feature_params[feature_type]['fs'], len(X), category)

        y = chords_nums_to_inds(y_nums, category)
        y = np.array(y)
        data.append((song_title, X, y))
    
    return data