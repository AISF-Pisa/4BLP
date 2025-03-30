import numpy as np
import matplotlib.pyplot as plt


#========================================================================
# Secondo Esercizio
#========================================================================

power = [0.5, 1, 2]         # potenze
x = np.linspace(0, 1, 1000) # range sulle x

plt.figure(1)
for p in power:
    plt.plot(x, x**p) # un plot alla volta sulla stessa figura

# bellurie
plt.grid()
plt.title("Esercizio 2")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

#========================================================================
# Terzo Esercizio
#========================================================================

power = [0.5, 1, 2]                        # potenze
color = ['k', 'r', 'b']                    # colore di ogni curva
lnsty = ['-', '--', '-.']                  # stile di ogni curva
label = [r'$\sqrt{x}$', r'$x$', r'$x^2$']  # nome della curva
x = np.linspace(0, 1, 1000)

plt.figure(1)
for p, c, ls, lb in zip(power, color, lnsty, label):
    # un plot alla volta sulla stessa figura
    plt.plot(x, x**p, c=c, linestyle=ls, label=lb)

#bellurie
plt.grid()
plt.title("Esercizio 3")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend(loc='best')
plt.show()

#========================================================================
# Quarto Esercizio
#========================================================================
print("Quarto Esercizio:")


def read():
    '''
    funzione che legge da input un numero con la condizione che esso
    sia maggiore di zero e che dia la possibilità di inserirlo
    nuovamente finché la condizione non è verificata.
    Volendo si può generalizzare il codice passando la condizione come input
    '''
    
    while True:  # Il codice deve runnare finchè non inserisco un numero buono
        
        try: # provo a leggere il numero e a renderlo intero
            x = int(input("Iserisci un numero: "))
            
        except ValueError: # se non riesco sollevo l'eccezione
            print(f"Fra ti ho chiesto di mettere un numero") # messaggio di errore
            continue # questo comando fa ripartire il ciclo da capo
        
        if x > 0: # se la lettura è andata a buon fine verifico la condizione
            return x # se è verificata ritorno il numero
        else :
            # alrimenti stampo un messaggio di errore
            print("In numero inserito è minore di zero, sceglierne un altro.")
            continue # e faccio ripartire il ciclo da capo


x = read()
print(f"Il numero letto è: {x}")
print()
#========================================================================
# Quinto Esercizio
#========================================================================
print("Quinto Esercizio:")

# Gaussiana
m = 0
s = 1
z = [np.random.normal(m, s) for _ in range(int(1e5))]

# Plot dati
plt.figure(1)
plt.hist(z, bins=50, density=True, histtype='step', label='dati')
plt.grid()
plt.xlabel("x")
plt.ylabel("P(x)")
plt.title("Distribuzione gaussiana")

# Plot curva
x = np.linspace(-5*s + m, 5*s + m, 1000)
plt.plot(x, np.exp(-(x-m)**2 / (2*s**2))/np.sqrt(2*np.pi*s**2), 'b', label=f"N({m}, {s})")
plt.legend(loc='best')
plt.show()

print()
#========================================================================
# Venticinquesimo Esercizio
#========================================================================
print("Venticinquesimo Esercizio:")


def compleanno(n, prove):
    '''
    Funzione che calcola la probabilita' che almeno due persone
    in un gruppo di n persone facciano il compleanno insieme.
    Il calcolo è eseguito generando prove numero di gruppi e 
    verificando se ci sono coincidenze.

    Parameters
    ---------- 
    n : int
        numero di persone nel gruppo
    prove : int
        numero di prove da fare
    
    Returns
    -------
    p : float
        casi favorevoli / casi totali
    '''
    s = 0
    
    for _ in range(prove):
        # Generazione casuale di n compleanni (numeri da 1 a 365)
        c = np.random.randint(1, 365 + 1, n)
        
        # Verifica se ci sono duplicati nei compleanni
        if len(c) != len(set(c)):
            s += 1

    # Calcolo della probabilità stimata
    p = s / prove
    return p

N     = 70
p     = int(3e4)
all_n = range(2, N + 1)
P     = [compleanno(n, p) for n in all_n]

plt.figure(1)
plt.plot(all_n, P, label="Probabilità stimata")
plt.xlabel("Numero di persone nell gruppo")
plt.ylabel("Probabilità di coincidenza di compleanni")
plt.title("Paradosso del Compleanno")
plt.grid()
plt.show()





























