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
    }
}

def get_input_size(feature_type):
    if feature_type == 'CQT':
        return feature_params['CQT']['n_bins']
    elif feature_type == 'MFCC':
        return feature_params['MFCC']['n_mfcc']
    else:
        raise NotImplementedError