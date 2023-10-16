import numpy as np
import os
import matplotlib.pyplot as plt
from .get_templates import get_templates, get_nested_circle_of_fifths
from . import _hmm as hmm
import librosa


def find_chords(x, fs, plot):
    """
    Given a mono audio signal x, and its sampling frequency, fs,
    find chords in it using 'method'
    Args:
        x : mono audio signal
        fs : sampling frequency (Hz)
        plot: if results should be plotted
    """

    chords, nested_cof = get_nested_circle_of_fifths()
    templates = get_templates(chords)

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
        chroma[:, n] = compute_chroma(xFrame[:, n], fs)
        #chroma[:, n] = chroma[:, n]/np.sum(chroma[:, n])

        # normalize chroma
        chroma[:, n] /= np.sum(chroma[:, n])

        # print("Chroma for frame ", n, ": \n", chroma[:, n])

        # normalize chroma
        chroma[:, n] /= np.sum(chroma[:, n])

        # print("Chroma for frame ", n, ": \n", chroma[:, n])



    # get max probability path from Viterbi algorithm
    (PI, A, B) = hmm.initialize(chroma, templates, chords, nested_cof, init_method = "theory")
    # print(PI.shape)
    # use baum-welch algorithm to train hmm
    (PI, A, B) = hmm.baum_welch(PI, A, B)
    (path, states) = hmm.viterbi(PI, A, B)

    # print("Path: ", path)
    # print("States: ", states)

    # normalize path
    for i in range(nFrames):
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
    for i in range(nFrames):
        if indices[i] == -1:
            final_chords.append("NC")
        else:
            final_states[i] = states[indices[i], i]
            final_chords.append(chords[int(final_states[i])])

    if plot:
        plt.figure()
        plt.yticks(np.arange(num_chords), chords)
        plt.plot(timestamp, np.int32(final_states), marker="o")

        plt.xlabel("Time in seconds")
        plt.ylabel("Chords")
        plt.title("Identified chords")
        plt.grid(True)
        plt.show()

    return timestamp, final_chords
