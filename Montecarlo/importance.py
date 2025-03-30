import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

np.random.seed(69420)

def mc_integration(f, dist, N, **kwargs):
    '''
    Function for a montecarlo integration

    Parameters
    ----------
    f : callable
        function to integrate
    dist : callable
        distribution from which to
        extract the random variables
    N : int
        total number of points
    
    Return
    ------
    integral : float
        integral of f
    error : float
        error over mean value of f
    
    Other parameters
    ----------------
    args_f : tuple
        extra argumets to pass to f
    args_dist : tuple
        extra argumets to pass to dist
    '''
    # Default values
    args_f    = kwargs.get("args_f" , ())
    args_dist = kwargs.get("args_dist", ())

    points   = dist(N,   *args_dist)
    values   = f(points, *args_f   )
    integral = np.mean(values)
    error    = np.sqrt(np.sum((values - integral)**2)/(N*(N-1)))
    
    return integral, error

def uniform(N, x_min=0, x_max=1):
    return np.random.uniform(x_min, x_max, N)

def exp_quantile(N):
    x = np.random.uniform(0, 1, N)
    return -np.log(1 - x)

def f(x):
    return np.exp(-x**2/0.1)

def weighed_f(x):
    return f(x)/np.exp(-x)

# Numerical value for comparison
I_exact, _ = quad(f, 0, 2)
print(I_exact)

# Valutiamo per diversi N
N_values = np.logspace(1, 7, 20, dtype=int)
errors_simple_cfr = []
errors_import_cfr = []
errors_simple     = []
errors_import     = []

for N in N_values:
    I_simple, e_simple = mc_integration(f, uniform, N, args_dist=(0, 2))
    I_import, e_import = mc_integration(weighed_f, exp_quantile, N)

    errors_simple_cfr.append(abs(I_simple - I_exact))
    errors_import_cfr.append(abs(I_import - I_exact))
    errors_simple.append(e_simple)
    errors_import.append(e_import)
    

# Plot della funzione e delle distribuzioni
plt.figure(1)
plt.loglog(N_values, errors_simple_cfr, 'o-', label="abs err simple", color='b')
plt.loglog(N_values, errors_import_cfr, 's-', label="abs err importance", color='r')
plt.loglog(N_values, errors_simple, 'd--', label="mc err simple", color='g')
plt.loglog(N_values, errors_import ,'*--', label="mc err importance", color='c')
plt.loglog(N_values, 1/np.sqrt(N_values), 'k:', label=r"$1/ \sqrt{N}$")
plt.xlabel("$N$", fontsize=15)
plt.ylabel("Error", fontsize=15)
plt.legend()

plt.tight_layout()
plt.show()
