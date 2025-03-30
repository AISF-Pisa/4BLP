def palindromo(n):
    '''jbkhv
    '''
    
    original = n
    reverse  = 0
    
    while n != 0:
        r       = n%10
        reverse = reverse*10 + r
        n       = n//10
    
    if reverse == original :
        return True
    
    return False

print(palindromo(12521))