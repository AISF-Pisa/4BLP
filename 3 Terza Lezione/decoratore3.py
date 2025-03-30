import time 

def cache(func):
    '''
    Funzione decoratore

    Parameters
    ----------
    func : callable
        funzione da decorare
    
    Return
    ------
    wrapper : callable
        funzione che incapsula e modifica func
    '''
    risultati = {}
    
    def wrapper(*args):
        '''
        Wrapper del decoratore

        Parameters
        ----------
        args : tuple
            argumets of func
        
        Returns
        -------
        risultato: int
            valore della funzione calcolata in args
        '''
        if args in risultati:
            return risultati[args]
      
        risultato = func(*args)
        risultati[args] = risultato
        return risultato
    
    return wrapper

def callcounter(func):
    '''
    Funzione decoratore

    Parameters
    ----------
    func : callable
        funzione da decorare
    
    Return
    ------
    wrapper_cc : callable
        funzione che incapsula e modifica func
    '''

    def wrapper_cc(*args):
        '''
        Wrapper del decoratore

        Parameters
        ----------
        args : tuple
            argumets of func
        
        Returns
        -------
        valore della funzione calcolata in args
        '''
        wrapper_cc.ncalls += 1  # Conta il numero di chiamate
        return func(*args)
    
    wrapper_cc.ncalls = 0       # Inizializza il contatore per ogni funzione decorata
    return wrapper_cc

@callcounter
@cache
def fibonacci(n):
    ''' Funzione ricosiva di fibonacci
    '''
    if n <= 1:
        return n
    
    return fibonacci(n-1) + fibonacci(n-2)

n = 38
start = time.time()
F = fibonacci(n)
end = time.time() - start
print(f"F({n}) = {F}")
print(f"Tempo impiegato: {end:.2f}")
print(f"Numero di chiamate effettuate: {fibonacci.ncalls}")
print(f"Nome funzione: {fibonacci.__name__}")


exit()
def count_calls(n, k):
    if n < k:
        return 0
    if n == k:
        return 1
    return count_calls(n-1, k) + count_calls(n-2, k)

# Numero di chiamate a fibonacci(5) dentro fibonacci(38)
calls = count_calls(38, 5)
print(f"Fibonacci(5) viene chiamato {calls} volte in Fibonacci(38)")