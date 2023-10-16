import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import torch
import utils
from args import get_args
from pprint import pformat
from method import match_template_inference, hmm_inference, lstm_inference


def inference(args):

    print('Arguments:\n' + pformat(args.__dict__))
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.fix_seed()
    
    if args.method == "match_template":
        acc = match_template_inference(args)
    elif args.method == "hmm":
        acc = hmm_inference(args)
    elif args.method == "lstm":
        acc = lstm_inference(args)
    else:
        raise NotImplementedError

    print('Accuracy:', acc)

if __name__ == "__main__":
    args = get_args()
    inference(args)
