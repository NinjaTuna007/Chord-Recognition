import numpy as np
import os
import matplotlib.pyplot as plt
from .get_templates import get_templates, get_nested_circle_of_fifths
from . import _hmm as hmm
import librosa

def find_chords(x, args):
    chords, nested_cof = get_nested_circle_of_fifths()
    templates = get_templates(chords)

    # compute chromagram using librosa
    chroma = x
    nFrames = chroma.shape[1]
    
    # get max probability path from Viterbi algorithm
    (PI, A, B) = hmm.initialize(chroma, templates, chords, nested_cof, init_method = "theory")
    # print(PI.shape)
    # use baum-welch algorithm to train hmm
    (PI, A, B) = hmm.baum_welch(PI, A, B, args.max_iters, args.tol)
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

    return final_chords
