"""Automatic chord recogniton with HMM, as suggested by Juan P. Bello in
'A mid level representation for harmonic content in music signals'
@author ORCHISAMA DAS, 2016"""

from __future__ import division
from chromagram import compute_chroma
import os
import numpy as np


"""calculates multivariate gaussian matrix from mean and covariance matrices"""


def multivariate_gaussian(x, meu, cov):

    det = np.linalg.det(cov)
    val = np.exp(-0.5 * np.dot(np.dot((x - meu).T, np.linalg.inv(cov)), (x - meu)))
    try:
        val /= np.sqrt(((2 * np.pi) ** 12) * det)
    except:
        print("Matrix is not positive, semi-definite")
    if np.isnan(val):
        val = np.finfo(float).eps
    return val


"""initialize the emission, transition and initialisation matrices for HMM in chord recognition
PI - initialisation matrix, #A - transition matrix, #B - observation matrix"""


def initialize(chroma, templates, chords, nested_cof):

    """initialising PI with equal probabilities"""
    num_chords = len(chords)
    PI = np.ones(num_chords) / num_chords

    """initialising A based on nested circle of fifths"""
    eps = 0.01
    A = np.empty((num_chords, num_chords))
    for chord in chords:
        ind = nested_cof.index(chord)
        t = ind
        for i in range(num_chords):
            if t >= num_chords:
                t = t % num_chords
            A[ind][t] = (abs(num_chords // 2 - i) + eps) / (
                num_chords**2 + num_chords * eps
            )
            t += 1

    """initialising based on tonic triads - Mean matrix; Tonic with dominant - 0.8,
    tonic with mediant 0.6 and mediant-dominant 0.8, non-triad diagonal elements 
    with 0.2 - covariance matrix"""

    nFrames = np.shape(chroma)[1]
    B = np.zeros((num_chords, nFrames))
    meu_mat = np.zeros((num_chords, num_chords // 2))
    cov_mat = np.zeros((num_chords, num_chords // 2, num_chords // 2))
    meu_mat = np.array(templates)
    # print(meu_mat.shape)
    offset = 0
    # print(num_chords)

    for i in range(num_chords):
        if i == num_chords // 2:
            offset = 0
        tonic = offset
        if i < num_chords // 2:
            mediant = (tonic + 4) % (num_chords // 2)
        else:
            mediant = (tonic + 3) % (num_chords // 2)
        dominant = (tonic + 7) % (num_chords // 2)

        # weighted diagonal
        # print(i)

        cov_mat[i, tonic, tonic] = 1.0
        cov_mat[i, mediant, mediant] = 1.0
        cov_mat[i, dominant, dominant] = 1.0

        cov_mat[i, tonic, dominant] = 0.8
        cov_mat[i, dominant, tonic] = 0.8
        
        cov_mat[i, mediant, dominant] = 0.8
        cov_mat[i, dominant, mediant] = 0.8

        cov_mat[i, tonic, mediant] = 0.6
        cov_mat[i, mediant, tonic] = 0.6

        # off-diagonal - matrix not positive semidefinite, hence determinant is negative
        # for n in [tonic,mediant,dominant]:
        #   for m in [tonic, mediant, dominant]:
        #       if (n is tonic and m is mediant) or (n is mediant and m is tonic):
        #           cov_mat[i,n,m] = 0.6
        #       else:
        #           cov_mat[i,n,m] = 0.8

        # filling non zero diagonals
        for j in range(num_chords // 2):
            if cov_mat[i, j, j] == 0:
                cov_mat[i, j, j] = 0.2
        offset += 1

    """observation matrix B is a multivariate Gaussian calculated from mean vector and 
    covariance matrix"""

    for m in range(nFrames):
        for n in range(num_chords):

            # print("n: ", n)
            # print("m: ", m)
            # print("chroma: ", chroma[:, m])
            # print("meu_mat: ", meu_mat[n, :])
            # print("cov_mat: \n", cov_mat[n, :, :])
            # print(wow_matrix)


            wow_prob = multivariate_gaussian(
                chroma[:, m], meu_mat[n, :], cov_mat[n, :, :]
            )
            # print("wow_matrix: ", wow_prob)
            B[n, m] = wow_prob
    
    # print(bow_wow_matrix)

    return (PI, A, B)


"""Viterbi algorithm to find Path with highest probability - dynamic programming"""


def viterbi(PI, A, B):

    # PI represents initialisation matrix - num_chords x 1
    # A represents transition matrix - num_chords x num_chords
    # B represents observation matrix - num_chords x nFrames
    # nFrames is the number of frames in the chromagram, num_chords is the number of chords in the vocabulary

    # viterbi takes an observation sequence and returns the most likely state sequence
    # here, the observation sequence is the chromagram

    
    (nrow, ncol) = np.shape(B)
    path = np.zeros((nrow, ncol))
    states = np.zeros((nrow, ncol))
    path[:, 0] = PI * B[:, 0]

    # print("PI: ", PI)
    # print("A: ", A)
    # print("B: ", B)

    for i in range(1, ncol):
        for j in range(nrow):
            s = [(path[k, i - 1] * A[k, j] * B[j, i], k) for k in range(nrow)]
            (prob, state) = max(s)
            path[j, i] = prob
            states[j, i - 1] = state

    return (path, states)


"""Baum-Welch to fine-tune A,B, PI based on Emission Sequences"""

def baum_welch(PI, A, B):
    
    # A: transition matrix - num_chords x num_chords
    # B: observation matrix - num_chords x nFrames
    # PI: initialisation matrix - num_chords x 1

    return None