import numpy as np
def somma_cond(arr, thr):
    '''ifweoà
    '''

    tmp = arr[arr >= thr]
    return sum(tmp)

a = np.array([3, 6, 2, 9, 5, 4, 7, 5, 3, 8, 9])
print(somma_cond(a, 6))