import numpy as np

def osservabile(x):
    '''
    funzione che dato un set di dati x in input
    restituisce media e deviazione standard
    '''
    N = len(x)
    
    media = sum(x)/N
    varianza = sum( (x - media)**2)/(N*(N-1))
    dev_std = np.sqrt(varianza)
    
    return np.array([media, dev_std])

dati = np.array([0, 3, 5, 1, 5, 667, 2, 4, 9, 3, 33])

m, dm = osservabile(dati)
print(f"media = {m:.3f} +- {dm:.3f}")