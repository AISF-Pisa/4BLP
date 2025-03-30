def persistenza(n):
    '''wbcowub
    '''
    count = 0
    while n >= 10:
        prodotto = 1
        for cifra in str(n):
            prodotto *= int(cifra)
        n = prodotto
        count += 1
    return count


print(persistenza(39))
