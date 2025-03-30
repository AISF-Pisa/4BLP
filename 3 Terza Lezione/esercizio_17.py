def primo(n):
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

def goldbach(n):
    '''kv
    '''

    a = []
    n1=2
    n2=n-2
    while(n1 <= n2):
        if(primo(n1) and primo(n2)):
            a.append((n1, n2))
        n1+=1
        n2-=1
    return a

print(goldbach(10))
