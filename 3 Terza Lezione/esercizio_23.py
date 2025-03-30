import numpy as np

def grandi_N(N):
    '''bgwn
    '''
    pro_avg = np.cumsum(np.random.uniform(0, 1, N)) / np.arange(1, N + 1)
    
    return pro_avg[-1]

# Esempio di utilizzo
print(grandi_N(int(5e5))-0.5)
