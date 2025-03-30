import numpy as np

def pi_1(n):
    '''mario
    '''
    return np.sqrt(6*sum(1/np.linspace(1, n, n)**2))


def pi_2(n):
    '''luigi
    '''
    pi = 0
    for i in range(1, n):
        pi += 1/i**2

    pi = np.sqrt(6*pi)
    return pi
