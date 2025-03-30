import numpy as np
from scipy.special import ellipe

def p_ellisse(a, b, N=10000):
    t = np.linspace(0, 2 * np.pi, N + 1)  # Suddivisione in N punti
    x = a * np.cos(t)
    y = b * np.sin(t)
    
    # Calcolo delle distanze tra punti successivi
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    
    # Somma totale
    return np.sum(ds)

# Esempio: semiassi a = 5, b = 3
a, b = 5, 3
perimetro = p_ellisse(a, b)
print(f"Il perimetro dell'ellisse è circa {perimetro}")