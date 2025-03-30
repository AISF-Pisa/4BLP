from functools import wraps

def ripeti(n):
    '''
    Funzione per passare argomenti al decoratore

    Parameters
    ----------
    n : int
        numero di chiamate da fare
    
    Return
    ------
    decoratore : callable
        decoratore
    '''
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
        # wrappo il wrapper, così le informazioni
        # originali di func nnon vanno perse
        @wraps(func)
        def wrapper():
            ''' wrapper, esegue n volte func
            '''
            for _ in range(n):
                func()
        return wrapper
    return decoratore

@ripeti(3)
def hello():
    '''
    Importante e complessa docstring
    '''
    print("Hello World!")

hello()
print(hello.__name__)
print(hello.__doc__)