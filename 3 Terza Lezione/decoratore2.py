def decoratore(func):
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
    def wrapper(*args, **kwargs):
        '''
        Wrapper del decoratore
        Ovviamente il nome è convenzionale

        Parameters
        ----------
        args : tuple
            argumets of func
        kwargs : dict
            keyword argumets of func
        '''
        print(f"Eseguendo: {func.__name__} con argomenti {args} {kwargs}")
        result = func(*args, **kwargs)
        print(f"Risultato: {result}")
        return result
    
    return wrapper


@decoratore
def op(a, b, k=1):
    '''
    Funzione che implementa somma e sottrazione

    Parameters
    ----------
    a, b : float
        valori di cui calcolare somma o sottrazione
    k : int, optional, default 1
        flag per passare da somma a sottrazione
    '''
    return a + k * b

op(3, 5, k=-1)
op(3, 5)

def op(a, b, **kwargs):
    '''
    Funzione che implementa le 4 operazioni
    
    Parameters
    ----------
    a, b : float
        valori di input
    
    Returns
    -------
    float
        risultato dell'operazione richiesta
    
    Other Parameters
    ----------------
    type : str, optional, default 'sum'
        tipo di operazione da eseguire
    '''
    # Assegno valore di default a type
    type = kwargs.get('type', 'sum')
    if type == 'sum':
        return a + b
    elif type == 'sub':
        return a - b
    elif type == 'mul':
        return a * b
    elif type == 'div':
        return a / b
    else:
        return
    
print(op(2, 3, type='sum'))
