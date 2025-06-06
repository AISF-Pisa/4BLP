import numpy as np

#=============================
# Primo Esercizio
#=============================
print("Primo esercizio:")

a = [1, 2, 3, 4]
print(f"array iniziale: {a}, id: {id(a)}")

b = a
b[0] = 7

print(f"array iniziale: {a}, id: {id(a)}")
print(f"array finale  : {b}, id: {id(b)}")

print()
#=============================
# Secondo Esercizio
#=============================
print("Secondo esercizio:")

I = np.zeros((3, 3))
idx = [0, 1, 2] # lista degli indici

I[idx, idx] = 1

v = np.array([4, 8, 2]) # vettore a caso
# calcolo prodotto scalare
print(v.T @ I @ v)
print(sum(v**2))

print()
#=============================
# Terzo Esercizio
#=============================
print("Terzo esercizio:")

L = np.zeros((4, 4))
I = np.zeros((4, 4))
B = np.zeros((4, 4))
idx = [0, 1, 2, 3] # lista degli indici

L[idx, idx] = [-1, 1, 1, 1]
I           = np.eye(4)

beta  = 0.1
gamma = 1/np.sqrt(1 - beta**2)
B[idx, idx] = [gamma, gamma, 1, 1]
B[0, 1] = B[1, 0] = -beta*gamma

v = np.array([4, 8, 2, 1]) # vettore a caso
print(v.T @ L @ v, v.T @ I @ v)
w = B @ v
print(w.T @ L @ w, w.T @ I @ w)

print()
#=============================
# Quarto Esercizio
#=============================
print("Quarto esercizio:")

A = np.zeros((4, 2))
A[:, 0] = 1
print(A)
# faccio lo swap cambiando l'ordine degli indici
print(A[:, [1, 0]])

print()
#=============================
# Quinto Esercizio
#=============================
print("Quinto esercizio:")

x = np.linspace(0, 20, 21, dtype=int)

pari = x[x%2 == 0] # maschera per i pari
disp = x[x%2 == 1] # maschera per i dispari
print(pari)
print(disp)

print()
#=============================
# Sesto Esercizio
#=============================
print("Sesto esercizio:")

x = np.linspace(0, 20, 21, dtype=int)

# maschera con due condizioni
mask = (x > 5) & (x < 15)
# "&" per and mentre "|" per or
print(x[mask])

print()
#=============================
# Settimo Esercizio
#=============================
print("Settimo esercizio:")

x = np.linspace(0, 10, 11, dtype=int)
# trasformo il vettore in una matrice N x 1
y = x[:, None]
print(x)
print(y)
print(x*y)

print()
#=============================
# Ottavo Esercizio
#=============================
print("Ottavo esercizio:")

decade_iniziale = -3  # da 10^-3
decade_finale   = 3   # fino a 10^3
base            = 10  # base della scala
numero_punti    = 15

x     = np.linspace(decade_iniziale, decade_finale, numero_punti)
x_log = base**x
n_log = np.logspace(decade_iniziale, decade_finale, numero_punti)

print(x_log)
print(n_log)

print()
#=============================
# Nono Esercizio
#=============================
print("Nono esercizio:")

a = np.array([1, 5, 2, 0, 4, 7, 8, 3, 9])
b = np.array([2, 4, 6])

a = np.setdiff1d(a, b)
print(a)

print()
#=============================
# Decimo Esercizio
#=============================
print("Decimo esercizio:")

# Ogni entrata della matrice é una variabile estratta da una distribuzione uniforme
Mat = np.random.random((3, 3))
print(Mat)

np.set_printoptions(precision=2)
print(Mat)

print()
#=============================
# Undicesimo Esercizio
#=============================
print("Undicesimo esercizio:")

x = np.random.random(10)
print(x)
x[x.argmin()], x[x.argmax()] = 1/7, 7
print(x)

print()
#=============================
# Dodicesimo Esercizio
#=============================
print("Dodicesimo esercizio:")

x = np.array([1, 2, 2, 3, 4, 4, 5])
print(x)
unique_x = np.unique(x)
print(unique_x)

print()
#=============================
# Tredicesimo Esercizio
#=============================
print("Tredicesimo esercizio:")

x = np.array([1, 2, 3]) 

# Creo due matrici che sono la copia
# del mio array di partenza ripetuto
# tre volte per righe e per colonne
X, Y = np.meshgrid(x, x)
print(X)
print(Y)
# Con ravel trasformo le matrici in vettori 
# e con stack li attacco secondo l'asse
# specificato (i.e. 1 fa elemento di x elemento di y
# 0 fa tutti elementi x, tutti elmenti di y)
Z = np.stack((X.ravel(), Y.ravel()), axis=1)

print(Z)

print()
#=============================
# Quattordicesimo Esercizio
#=============================
print("Quattordicesimo esercizio:")

x = np.array([1, 2, 1])
v = np.array([0, 1, 0])
L = np.cross(x, v)
print(L)

print()
#=============================
# Quindicesimo Esercizio
#=============================
print("Quindicesimo esercizio:")

x     = np.array([1, 1])
r     = np.linalg.norm(x)
theta = np.arctan2(x[1], x[0])
pol   = np.array([r, theta])
print(pol)

print()
#=============================
# Sedicesimo Esercizio
#=============================
print("Sedicesimo esercizio:")

A = np.random.rand(4, 3)
AT_A = A.T @ A
print("Matrice originale:\n", A)
print("Matrice trasposta per originale:\n", AT_A)
print("È simmetrica?", np.allclose(AT_A, AT_A.T))

print()
#=============================
# Diciasettesimo Esercizio
#=============================
print("Diciasettesimo esercizio:")

theta = np.radians(45)  # angolo di rotazione in radianti
R = np.array([[np.cos(theta), -np.sin(theta)], 
              [np.sin(theta), np.cos(theta)]])

v = np.array([1, 0])  # vettore iniziale
v_rot = R @ v

print("Vettore originale:", v)
print("Vettore ruotato:", v_rot)
print("Lunghezza invariata?", np.isclose(np.linalg.norm(v), np.linalg.norm(v_rot)))

print()
#=============================
# Diciottesimo Esercizio
#=============================
print("Diciottesimo esercizio:")

arr = np.arange(1, 21)
arr[1::2] = arr[1::2] ** 2
print(arr)

print()
#=============================
# Diciannovesimo Esercizio
#=============================
print("Diciannovesimo esercizio:")

pos_iniziale = np.array([0, 0, 0])
vel = np.array([1, 2, 3])  # velocità costante
t = 10  # tempo
pos_finale = pos_iniziale + vel * t
print("Posizione finale:", pos_finale)

print()
#=============================
# Ventesimo Esercizio
#=============================
print("Ventesimo esercizio:")

x = np.linspace(0, 2 * np.pi, 100)

matrice_armoniche = np.column_stack((np.sin(x), np.cos(x)))
somme_colonne = matrice_armoniche.sum(axis=0)

print("Somma della colonna sin(x):", somme_colonne[0])
print("Somma della colonna cos(x):", somme_colonne[1])

print()
#=============================
# Ventunesimo Esercizio
#=============================
print("Ventunesimo esercizio:")

arr = np.array([10, 20, 30, 40, 50])  # Esempio di array
terzo = arr[2]
ultimo = arr[-1]
print("Terzo elemento:", terzo)
print("Ultimo elemento:", ultimo)

print()
#=============================
# Ventiduesimo Esercizio
#=============================
print("Ventiduesimo esercizio:")

array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([6, 7, 8, 9, 10])
array = np.concatenate((array1, array2))
print("Array concatenato:", array)

print()
#=============================
# Ventitreesimo Esercizio
#=============================
print("Ventitreesimo esercizio:")

numeri = np.arange(1, 11)  # Array di numeri da 1 a 10
subarray_primi_5 = numeri[:5]
subarray_ultimi_3 = numeri[-3:]
print("Primi 5 elementi:", subarray_primi_5)
print("Ultimi 3 elementi:", subarray_ultimi_3)