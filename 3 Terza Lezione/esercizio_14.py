import numpy as np

x = np.array([3, 4, np.nan, 8, 1, np.nan, np.nan, 9])
def clean_data(x):
    ''', .khv
    '''
    y = []
    for i, el in enumerate(x):
        if el != el:
            y.append(0)
        else:
            y.append(x[i])
    return y

print(clean_data(x))