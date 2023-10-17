import librosa
import numpy as np

def get_cqt(x, args):
    cqt = librosa.core.cqt(x, sr=args.sr, n_bins=args.n_bins, hop_length=args.hop_length, window='hamming', norm=2)
    cqt = np.abs(cqt)
    return cqt

def get_mfcc(x, args):
    mfcc = librosa.feature.mfcc(y=x, sr=args.sr, n_mfcc=args.n_mfcc, hop_length=args.hop_length, n_fft=args.n_fft, window='hamming')
    return mfcc

def get_stft(x, args):
    stft = librosa.core.stft(y=x, n_fft=args.n_fft, hop_length=args.hop_length, window='hamming')
    stft = np.abs(stft)
    return stft

def get_mel_spectrogram(x, args):
    mel_spectrogram = librosa.feature.melspectrogram(y=x, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length, window='hamming', n_mels=args.n_mels)
    mel_spectrogram = np.abs(mel_spectrogram)
    return mel_spectrogram

# get chroma feature
def get_chroma_cqt(x, args):
    chroma = librosa.feature.chroma_cqt(y=x, sr=args.sr, hop_length=args.hop_length)
    return chroma

def get_chroma_stft(x, args):
    chroma = librosa.feature.chroma_stft(y=x, sr=args.sr, hop_length=args.hop_length, n_fft=args.n_fft)
    return chroma