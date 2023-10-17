import librosa
import numpy as np

def get_stft(x, args):
    stft = librosa.core.stft(y=x, n_fft=args.n_fft, hop_length=args.hop_length, window='hamming')
    stft = np.abs(stft)
    return stft

def get_chroma(x, args):
    chroma = librosa.feature.chroma_stft(y=x, sr=args.sr, hop_length=args.hop_length)
    return chroma