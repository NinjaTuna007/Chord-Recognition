import numpy as np
import os
import random
import logging
import logging.config
import torch
import torch.nn as nn

def get_input_size(args):
    if args.feature_type == 'CQT':
        return args.n_bins
    elif args.feature_type == 'MFCC':
        return args.n_mfcc
    elif args.feature_type == 'STFT':
        return args.n_fft // 2 + 1
    elif args.feature_type == 'MEL':
        return args.n_mels
    elif args.feature_type == 'CHROMA_CQT':
        return 12
    elif args.feature_type == 'CHROMA_STFT':
        return 12
    else:
        raise NotImplementedError

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def fix_seed(seed=2023):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_logger(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


def save_checkpoint(model: nn.Module, path: str):
    logging.info('Checkpoint: save to checkpoint %s' % path)
    state_dict = model.state_dict()
    torch.save(state_dict, path)

def load_checkpoint(model: nn.Module, path: str) -> dict:
    state_dict = torch.load(path)
    model_state_dict = model.state_dict()
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                state_dict[k] = model_state_dict[k]
                logging.warning('Ignore module: %s'% k)
    model.load_state_dict(state_dict, strict=False)
    return model