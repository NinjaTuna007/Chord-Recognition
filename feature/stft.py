import librosa
import numpy as np

def get_stft(x, feature_param):
    stft = librosa.core.stft(x, n_fft=feature_param['n_fft'], hop_length=feature_param['hop_length'], window='hamming')
    stft = np.abs(stft)
    return stft

def get_chroma(x, feature_param):
    chroma = librosa.feature.chroma_stft(y=x, sr=feature_param['fs'], hop_length=feature_param['hop_length'])
    return chroma