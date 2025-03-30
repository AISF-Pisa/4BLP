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
    def wrapper():
        '''
        Wrapper del decoratore
        Ovviamente il nome è convenzionale
        '''
        print("Codice eseguito prima")
        func()
        print("Codice eseguito dopo")
    
    return wrapper

@decoratore
def f():
    ''' Funzione che vogliamo decorare
    '''
    print("Funzione originale che fa cose")

f()
