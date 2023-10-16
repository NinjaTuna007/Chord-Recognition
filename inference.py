import numpy as np
import os
import matplotlib.pyplot as plt
from method import hmm_find_chords, match_template_find_chords
import librosa
import pydub
import subprocess
from args import get_args
from pprint import pformat
from preprocess import gen_train_data
from method import match_template_inference


def inference(args):

    print('Arguments:\n' + pformat(args.__dict__))
    
    data = gen_train_data(args.feature_type, args.data_list, args.audio_path, args.gt_path, args.category)

    acc = match_template_inference(args, data)

    print('Accuracy:', acc)



    # print(data[0][1].shape, data[0][2].shape)
    # # find the chords
    # if args.method == "match_template":
    #     timestamp, final_chords = match_template_find_chords(x, fs, plot=args.plot)
    # else:
    #     timestamp, final_chords = hmm_find_chords(x, fs, plot=args.plot)

    # # print chords with timestamps
    # print("Time (s)", "Chord")
    # for n in range(len(timestamp) - 1):
    #     # if the chord is same as previous chord, then skip
    #     if final_chords[n] == final_chords[n + 1]:
    #         continue
    #     else:
    #         # print start time of chord, end time of chord, and chord
    #         print(timestamp[n], timestamp[n + 1], final_chords[n])


if __name__ == "__main__":
    args = get_args()
    inference(args)
