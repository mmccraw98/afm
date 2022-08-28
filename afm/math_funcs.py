import numpy as np
import scipy as sp


def sma(arr, window_size):
    return np.convolve(arr, np.ones(window_size), 'valid') / window_size


def sma_shift(arr, window_size):
    arr_new = sma(arr, window_size)
    return arr_new - arr_new[0]


def row2mat(row, n): #@TODO move to helperfunctions
    '''
    stacks a row vector (numpy (m, )) n times to create a matrix (numpy (m, n)) NOTE: CAN SLOW DOWN COMPUTATION IF DONE MANY TIMES
    :param row: numpy array row vector
    :param n: int number of replications to perform
    :return: numpy matrix (m, n) replicated row vector
    '''
    # do once at the beginning of any calculation to improve performance
    return np.tile(row, (n, 1)).T

def norm_cross_corr(f1, f2, method = 'same'):
    '''
    computes the cross correlationg between two signals f1 and f2 and then normalizes the result
    by dividing by the square root of the product of the max of each auto correlation sqrt(f1*f1 f2*f2)
    :param f1: numpy array (n,)
    :param f2: numpy array (m,) n not necessarily == m
    :return: numpy array (n,)
    '''
    return sp.signal.correlate(f1, f2, mode=method) / np.sqrt(sp.signal.correlate(f1, f1, mode=method)[int(f1.size / 2)] * signal.correlate(f2, f2, mode=method)[int(f2.size / 2)])

def downsampu2v(u, v):
    '''
    downsamples an array, u, to the size of another, smaller array, v
    :param u: numpy array (n,)
    :param v: numpy array (m,) where m<=n
    :return: numpy array, u* (m,)
    '''
    return u[np.round(np.linspace(0, u.size - 1, v.size)).astype(int)]

