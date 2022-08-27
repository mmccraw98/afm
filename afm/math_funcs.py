import numpy as np


def sma(arr, window_size):
    return np.convolve(arr, np.ones(window_size), 'valid') / window_size


def sma_shift(arr, window_size):
    arr_new = sma(arr, window_size)
    return arr_new - arr_new[0]
