import librosa
import numpy as np

def get_cqt(x, feature_param):
    cqt = librosa.core.cqt(x, sr=feature_param['fs'], n_bins=feature_param['n_bins'], hop_length=feature_param['hop_length'], window='hamming', norm=2)
    cqt = np.abs(cqt)
    return cqt
