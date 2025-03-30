import numpy as np

def automorfico(arr):
    ''' ifuyut
    '''

    a = []

    for n in arr:
        n2 = n ** 2
        if str(n2).endswith(str(n)):
            a.append(n)

    return np.array(a)

N = np.linspace(1, 100, 100)
print(automorfico(N))

