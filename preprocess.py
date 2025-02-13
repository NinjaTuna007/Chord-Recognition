import os
import argparse
import numpy as np

from preprocess import gen_train_data
from pprint import pformat
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', default='data/audio', type=str)
    parser.add_argument('--gt_path', default='data/gt', type=str)
    parser.add_argument('--data_snapshot_path', default='data/snapshot', type=str)
    parser.add_argument('--category', default='MirexMajMin', type=str, choices=['MirexMajMin'])
    parser.add_argument('--data_list', type=str, required=True)
    parser.add_argument('--feature_type', type=str, default='CQT', choices=['CQT', 'STFT', 'MFCC'])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print('Arguments:\n' + pformat(args.__dict__))
    data = gen_train_data(args.feature_type, args.data_list, args.audio_path, args.gt_path, args.category)
    os.makedirs(args.data_snapshot_path, exist_ok=True)
    data_list_name = args.data_list.split('/')[-1].split('.')[0]
    data_snapshot_name = '_'.join([data_list_name, args.feature_type, args.category]) + '.pt'
    torch.save(data, os.path.join(args.data_snapshot_path, data_snapshot_name))
    print('Saved to', os.path.join(args.data_snapshot_path, data_snapshot_name))