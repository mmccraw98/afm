import numpy as np

def get_r(x):
    '''
    get the time decay constant needed for a given signal
    :param x: some signal (listlike)
    :return: optimal time decay constant (float)
    '''
    first_nonzero = np.nonzero(x)[0][0]
    return abs(x[-1] / x[first_nonzero]) ** (1 / (x.size - first_nonzero))


def mdft(x, r, length=None):
    '''
    perform the modified discrete fourier transform of a signal at a given radial distance
    :param x: some signal (listlike)
    :param r: radial distance defining the circle on which the modified fourier transform will be calculated (float)
    (r=1.0 gives a discrete fourier transform)
    :param length: number of elements in the transformed signal (int)
    (default is None, which leaves length=len(x); recommended to set equal to the length of shortest signal in batch)
    :return: modified discrete fourier transform of x at r (numpy array)
    '''
    n = np.arange(0, x.size, 1)
    return np.fft.fftshift(np.fft.fft(x * r ** -n, n=length))
