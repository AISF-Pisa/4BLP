import numpy as np
def EMCD(a, b):
    '''
    algoritmo esteso di eculide, dati due nuemri a e b
    ne calcola  il massimo comun divisore a anche i coefficenti
    X e Y tali che: a*X + b*Y = MCD(a, b) [identità di Bézout].
    Se a e b sono coprimi allora si ha che:
    1)X è l'inverso moltiplicativo di a modulo b;
    2)Y è l'inverso moltiplicativo di b modulo a.

    Parameters
    ----------
    a, b : int
         numeri di cui calcolare MCD

    Returns
    ----------
    R : int
        resto della divisione
    X, Y : int
        coefficenti dll'identità di Bézout
    '''
    x = 0; X = 1
    y = 1; Y = 0
    r = b; R = a
    while r != 0:
        q = R // r
        R, r = r, R - q*r
        X, x = x, X - q*x
        Y, y = y, Y - q*y
    return np.array([R, X, Y])

print(EMCD(4653, 2793))