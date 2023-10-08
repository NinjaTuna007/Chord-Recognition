import librosa
import numpy as np

def get_mfcc(x, feature_param):
    mfcc = librosa.feature.mfcc(y=x, sr=feature_param['fs'], n_mfcc=feature_param['n_mfcc'], hop_length=feature_param['hop_length'], n_fft=feature_param['n_fft'], window='hamming')
    return mfcc
    