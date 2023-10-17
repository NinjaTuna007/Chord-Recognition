import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
from .hmm import find_chords
from preprocess import gen_train_data
from preprocess.chords import chord_to_categories, chords_nums_to_inds

def inference(args):
    data = gen_train_data(args)
    predictions = []
    total, correct = 0, 0
    for i in range(len(data)):
        total += len(data[i][2])
        x = data[i][1].T
        final_chords = find_chords(x, args)
        chord_nums = []
        # print(final_chords)
        for chord in final_chords:
            chord_num = chord_to_categories(chord, args.category)
            chord_nums.append(chord_num)
        chord_nums = chords_nums_to_inds(chord_nums, args.category)
        predictions.append(chord_nums)
        correct += (chord_nums == data[i][2]).sum()
    #     print(chord_nums)
    #     print(list(data[i][2]))
    # print(correct)
    acc = 100 * correct / total
    return acc