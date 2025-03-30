import math 

def newton_sqrt(a, tol=1e-5):
    '''jjbojb
    '''
    x = a
    while True:
        next_x = 0.5 * (x + a / x)
        if abs(x - next_x) < tol:
            break
        x = next_x
    return x

# Esempio di utilizzo
a = 2864
print(f"Radice quadrata di {a} (Newton-Raphson): {newton_sqrt(a)}")
print(f"Radice quadrata di {a} (math.sqrt):      {math.sqrt(a)}")
