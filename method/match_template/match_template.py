import numpy as np
import os
import matplotlib.pyplot as plt
from .get_templates import get_templates, get_chords
import librosa


def find_chords(chroma, args):
    chords = get_chords()
    templates = get_templates(chords)
    num_chords = len(templates)

    # correlate 12D chroma vector with each of
    # 24 major and minor chords
    nFrames = chroma.shape[1]
    max_cor = np.zeros(nFrames)
    id_chord = np.zeros(nFrames, dtype="int32")

    for n in range(nFrames):
        cor_vec = np.zeros(num_chords)
        for ni in range(num_chords):
            cor_vec[ni] = np.correlate(chroma[:, n], np.array(templates[ni]))
        max_cor[n] = np.max(cor_vec)
        id_chord[n] = np.argmax(cor_vec) + 1

    # if max_cor[n] < threshold, then no chord is played
    # might need to change threshold value
    id_chord[np.where(max_cor < args.threshold * np.max(max_cor))] = 0
    final_chords = [chords[cid] for cid in id_chord]

    return final_chords