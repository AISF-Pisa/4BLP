'''jòouviv
'''

def decimal_to_binary(n):
    '''pup
    '''
    if n == 0:
        return "0"
    binario = ""
    while n > 0:
        binario = str(n % 2) + binario
        n  = n // 2
    return binario

def binary_to_decimal(n_b):
    '''bib
    '''
    n_d = 0
    length = len(n_b)
    for i in range(length):
        bit = int(n_b[length - 1 - i])
        n_d += bit * (2 ** i)
    return n_d

print(decimal_to_binary(42))
print(binary_to_decimal('101010'))
