import numpy as np

def eq(a, b, c):
    '''bvc gkc 
    '''
    det= b**2-4*a*c

    x=[]

    if det<0:
        return np.array(x)
    else:
        if a==0:
            x.append(-b/c)
            return np.array(x)
        
        else:
            x.append((-b+np.sqrt(det))/(2*a))
            x.append((-b-np.sqrt(det))/(2*a))
            return np.array(x)

print(-np.log10(eq(1, 1.75e-5, -0.173*1.8e-5)[0]))
