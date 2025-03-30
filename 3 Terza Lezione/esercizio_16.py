def fattori(n):
    '''
    Calcolo brutale dei fattori primi di un numero
    Parameters
    ----------
    n : int
        numero di cui calcolare i fattori primi

    Returns
    ----------
    f : list
        lista dei fattori primi di n
    '''
    i = 2
    f = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            f.append(i)
    if n > 1:
        f.append(n)
    return f

print(fattori(713491741))