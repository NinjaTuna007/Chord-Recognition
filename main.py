import numpy as np
import os, sys, getopt
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import json
from chromagram import compute_chroma
import hmm as hmm
import librosa
import pydub
import subprocess
import argparse


def get_templates(chords):
    """read from JSON file to get chord templates"""
    with open("data/chord_templates.json", "r") as fp:
        templates_json = json.load(fp)
    templates = []

    for chord in chords:
        if chord in templates_json:
            templates.append(templates_json[chord])
        else:
            continue
            # "N": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "N1": [1,1,1,1,1,1,1,1,1,1,1,1]}

    return templates


def get_nested_circle_of_fifths():
    chords = [
        "N",
        "C:maj", 
        "C#:maj",
        "D:maj",
        "D#:maj",
        "E:maj", 
        "F:maj", 
        "F#:maj", 
        "G:maj",
        "G#:maj", 
        "A:maj", 
        "A#:maj",
        "B:maj",
        "C:min", 
        "C#:min", 
        "D:min", 
        "D#:min", 
        "E:min", 
        "F:min", 
        "F#:min", 
        "G:min",
        "G#:min", 
        "A:min", 
        "A#:min",
        "B:min",
    ]
    nested_cof = [
        "C:maj",
        "E:min",
        "G:maj",
        "B:min",
        "D:maj",
        "F#:min",
        "A:maj",
        "C#:min",
        "E:maj",
        "G#:min",
        "B:maj",
        "D#:min",
        "F#:maj",
        "A#:min",
        "C#:maj",
        "F:min",
        "G#:maj",
        "C:min",
        "D#:maj",
        "G:min",
        "A#:maj",
        "D:min",
        "F:maj",
        "A:min",
    ]
    return chords, nested_cof


def find_chords(
    x: np.ndarray,
    fs: int,
    templates: list,
    chords: list,
    nested_cof: list = None,
    method: str = None,
    plot: bool = False,
):
    """
    Given a mono audio signal x, and its sampling frequency, fs,
    find chords in it using 'method'
    Args:
        x : mono audio signal
        fs : sampling frequency (Hz)
        templates: dictionary of chord templates
        chords: list of chords to search over
        nested_cof: nested circle of fifth chords
        method: template matching or HMM
        plot: if results should be plotted
    """

    # framing audio, window length = 8192, hop size = 1024 and computing PCP
    nfft = int(8192 * 0.5)
    hop_size = int(1024 * 0.5)
    nFrames = int(np.round(len(x) / (nfft - hop_size)))
    # zero padding to make signal length long enough to have nFrames
    x = np.append(x, np.zeros(nfft))
    xFrame = np.empty((nfft, nFrames))
    start = 0
    num_chords = len(templates)
    # print(num_chords)
    chroma = np.empty((num_chords // 2, nFrames))
    # print(chroma.shape)
    id_chord = np.zeros(nFrames, dtype="int32")
    timestamp = np.zeros(nFrames)
    max_cor = np.zeros(nFrames)

    # step 1. compute chromagram
    for n in range(nFrames):
        xFrame[:, n] = x[start : start + nfft]
        start = start + nfft - hop_size
        timestamp[n] = n * (nfft - hop_size) / fs
        # chroma[:, n] = compute_chroma(xFrame[:, n], fs)

    # chroma using librosa
    chroma = librosa.feature.chroma_stft(y=x, sr=fs, n_fft=nfft, hop_length=hop_size)

    # visualize the chroma
    # plt.figure()
    # plt.imshow(chroma, cmap='gray_r', origin='lower', aspect='auto')
    # plt.xlabel('Time (frames)')
    # plt.ylabel('Chroma')
    # plt.title('Chromagram')
    # plt.show()
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(chroma, sr=fs, x_axis="frames",  y_axis="chroma")
    plt.title("Chroma Features")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

        # print("Chroma for frame ", n, ": \n", chroma[:, n])

    # for n in range(nFrames):
        # normalize chroma
        # chroma[:, n] /= np.sum(chroma[:, n])
        # print("Chroma for frame ", n, ": \n", chroma[:, n])
        # continue

    if method == "match_template":
        # correlate 12D chroma vector with each of
        # 24 major and minor chords
        for n in range(nFrames):
            cor_vec = np.zeros(num_chords) # added dtype=object
            for ni in range(num_chords):
                #print(np.correlate(chroma[:, n], np.array(templates[ni])))
                cor_vec[ni] = np.correlate(chroma[:, n], np.array(templates[ni]))
            max_cor[n] = np.max(cor_vec)
            id_chord[n] = np.argmax(cor_vec) + 1

        # if max_cor[n] < threshold, then no chord is played
        # might need to change threshold value
        id_chord[np.where(max_cor < 0.1 * np.max(max_cor))] = 0
        final_chords = [chords[cid] for cid in id_chord]

    elif method == "hmm":
        # get max probability path from Viterbi algorithm
        (PI, A, B) = hmm.initialize(chroma, templates, chords, nested_cof, init_method = "theory")
        # print(PI.shape)
        # use baum-welch algorithm to train hmm
        (PI, A, B) = hmm.baum_welch(PI, A, B)
        (path, states, state_seq) = hmm.viterbi_log(PI, A, B)

        # print("Path: ", path)
        # print("States: ", states)

        if False: # original code
            # normalize path
            for i in range(nFrames):
                path[:, i] /= sum(path[:, i])
        
        else: # modified code
            # normalize path - log version [path contains log values]
            for i in range(nFrames):
                path[:, i] -= np.max(path[:, i])
                path[:, i] = np.exp(path[:, i])
                path[:, i] /= sum(path[:, i])

        # choose most likely chord - with max value in 'path'
        final_chords = []
        indices = np.argmax(path, axis=0)
        final_states = np.zeros(nFrames)
        #final_states = []

        # find no chord zone
        set_zero = np.where(np.max(path, axis=0) < 0.3 * np.max(path))[0]
        if np.size(set_zero) > 0:
            indices[set_zero] = -1

        # identify chords
        if True: # original code
            for i in range(nFrames):
                if indices[i] == -1:
                    final_chords.append("NC")
                else:
                    final_states[i] = states[indices[i], i]
                    final_chords.append(chords[int(final_states[i])])
        else: # modified code
            # use state_seq returned by viterbi_log
            for i in range(nFrames):
                if indices[i] == -1:
                    final_chords.append("NC")
                else:
                    final_states[i] = state_seq[i]
                    final_chords.append(chords[int(final_states[i])])

        

    if plot:
        plt.figure()
        if method == "match_template":
            plt.yticks(np.arange(num_chords + 1), chords)
            plt.plot(timestamp, id_chord, marker="o")

        else:
            plt.yticks(np.arange(num_chords), chords)
            plt.plot(timestamp, np.int32(final_states), marker="o")

        plt.xlabel("Time in seconds")
        plt.ylabel("Chords")
        plt.title("Identified chords")
        plt.grid(True)
        plt.show()

    return timestamp, final_chords

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, help="input file")
    parser.add_argument("-m", "--method", type=str, default="match_template", help="method")
    parser.add_argument("-p", "--plot", type=bool, help="plot")
    args = parser.parse_args()
    return args

def main(args):

    print("Input file is ", args.input_file)
    print("Method is ", args.method)
    directory = os.getcwd() + "/data/test_chords/"
    # read the input file
    #(fs, s) = read(directory + args.input_file)
    #audio = pydub.AudioSegment.from_file(f"{directory + args.input_file}")
    x, fs = librosa.load(directory + args.input_file)

    # Suppress percussive elements
    x = librosa.effects.harmonic(x, margin=4)

    # get chords and circle of fifths
    chords, nested_cof = get_nested_circle_of_fifths()
    # get chord templates
    templates = get_templates(chords)

    # find the chords
    if args.method == "match_template":
        timestamp, final_chords = find_chords(
            x, fs, templates=templates, chords=chords, method=args.method, plot=args.plot
        )
    else:
        timestamp, final_chords = find_chords(
            x,
            fs,
            templates=templates,
            chords=chords[1:],
            # chords=chords,
            nested_cof=nested_cof,
            method=args.method,
            plot=args.plot,
        )

    # print chords with timestamps
    print("Time (s)", "Chord")
    start_time = timestamp[0]
    for n in range(len(timestamp) - 1):
        # if the chord is same as previous chord, then skip
        if final_chords[n] == final_chords[n + 1]:
            continue
        else:

            # print start time of chord, end time of chord, and chord
            print(start_time, timestamp[n + 1], final_chords[n])
            start_time = timestamp[n + 1]

    # print last chord if start time of last chord is not same as end time of previous chord
    if start_time != timestamp[-1]:
        print(start_time, timestamp[-1], final_chords[-1])


if __name__ == "__main__":
    args = get_args()
    main(args)
