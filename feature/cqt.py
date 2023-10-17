import librosa
import numpy as np

def get_cqt(x, args):
    cqt = librosa.core.cqt(x, sr=args.sr, n_bins=args.n_bins, hop_length=args.hop_length, window='hamming', norm=2)
    cqt = np.abs(cqt)
    return cqt

def get_chroma(x, args):
    chroma = librosa.feature.chroma_cqt(y=x, sr=args.sr, hop_length=args.hop_length)
    return chroma