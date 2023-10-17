import librosa
import numpy as np

def get_mfcc(x, args):
    mfcc = librosa.feature.mfcc(y=x, sr=args.sr, n_mfcc=args.n_mfcc, hop_length=args.hop_length, n_fft=args.n_fft, window='hamming')
    return mfcc
    