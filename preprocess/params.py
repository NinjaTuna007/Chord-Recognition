feature_params = {
    'CQT': {
        'n_bins': 84,
        'hop_length': 512,
        'fs': 44100,
    },
    'MFCC': {
        'n_mfcc': 20,
        'hop_length': 512,
        'fs': 44100,
        'n_fft': 2048,
    },
    'STFT': {
        'n_fft': 2048,
        'hop_length': 512,
        'fs': 44100,
    },
    'CHROMA_CQT': {
        'hop_length': 4096,
        'fs': 44100,
    },
    'CHROMA_STFT': {
        'hop_length': 512,
        'fs': 44100,
    },
}

def get_input_size(feature_type):
    if feature_type == 'CQT':
        return feature_params['CQT']['n_bins']
    elif feature_type == 'MFCC':
        return feature_params['MFCC']['n_mfcc']
    elif feature_type == 'STFT':
        return feature_params['STFT']['n_fft'] // 2 + 1
    else:
        raise NotImplementedError