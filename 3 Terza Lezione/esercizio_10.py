import numpy as np

def is_prime(n):
    '''jlbb
    '''
    if(n==1):
        return False
    if(n%2 == 0 and n!=2):
        return False
    for i in range(3, n//2, 2):
        if(n%i == 0):
            return False
    return True

def trova_primi(N):
    '''jviy
    '''
    x = []
    for i in range(N):
        if(is_prime(i)):
            x.append(i)
        
    return np.array(x)

print(trova_primi(200))
print(is_prime(813491741))