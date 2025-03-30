# Primo programma di 4BLP
import math
import numpy as np
"""
fdkfsdkfdskfksdfd
fdfsdfsefsefsfesfds
dsdsfdsfse
"""
print("Ciao a tutti!")
a = 21
b = 2.3

print(a ** b)

lista = [10, "ciao", 3.1415, a]
lista1 = [24, "trentatré trentini", "Firenze"]
tupla = (2, 3, "10", 15, a)
print(tupla)
a = lista
lista[0] = 9
lista[1] = "Fisica1 è bella"
lista1[2] = "Firenze è stunning"
print(lista)
c = float(input("Inserire l'altezza: "))
print(type(c))
# int: trasforma in intero
# float: trasforma in un numero decimale
c = int(c)
print(type(c))
# area = lato * lato
print(c)
# usare la libreria math
print(math.pow(27, 1/3))
print(math.log(25, 5))
x = 10
x *= 2
print(x)