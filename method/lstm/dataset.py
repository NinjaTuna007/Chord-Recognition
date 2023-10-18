import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

def batch_padding(data_batch, max_len=None):
    if max_len is None:
        max_len = max([len(d[1]) for d in data_batch])
    for i in range(len(data_batch)):
        data_batch[i] = (data_batch[i][0], np.pad(data_batch[i][1], ((0, max_len - len(data_batch[i][1])), (0, 0)), 'constant', constant_values=((-1, -1), (-1, -1))), np.pad(data_batch[i][2], (0, max_len - len(data_batch[i][2])), 'constant', constant_values=(-1, -1)))
    return data_batch

def split_data_to_batch(data, args, padding=True):
    inds_len = int(args.len_sub_audio * (args.sr / args.hop_length))
    data_batch = []
    for d in data:
        audio_name, X, y = d
        # split audio to sub-audios
        X = np.array_split(X, np.ceil(len(X) / inds_len))
        y = np.array_split(y, np.ceil(len(y) / inds_len))
        assert len(X) == len(y)
        for i in range(len(X)):
            data_batch.append((audio_name, X[i], y[i]))
    # padding for X and y
    # max_len = 3441
    if padding:
        data_batch = batch_padding(data_batch)
    return data_batch


class ChordDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        return self.data[index][1], self.data[index][2]
    
    def __len__(self):
        return len(self.data)