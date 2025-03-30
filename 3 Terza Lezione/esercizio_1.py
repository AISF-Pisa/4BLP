"""
Code fror exercise one
"""
import numpy as np

def area(l, n):
    '''
    funzione per calcolare l'area di un poligono
    regolare di lato l e numero lati n
    '''
    a = l/2 * 1/np.tan(np.pi/n) # apotema
    p = n*l                     # perimetro
    return p*a/2                # area



def pitagora(list_l, n):
    '''
    Funzione che controlla il teorema di pitagora
    '''
    aree = []
    for l in list_l:
        aree.append(area(l, n))
    print(f"A1+A2={aree[0]+aree[1]}, A3={aree[2]}")
    return aree[0]+aree[1] - aree[2]


lati  = [3, 4, 5]
nlati = [4, 6, 8]
for n in nlati:
    print(pitagora(lati, n))
