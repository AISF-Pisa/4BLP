import numpy as np
import matplotlib.pyplot as plt

np.random.seed(69420)

def f(coords):
    return np.sum(coords**2, axis=1)

def mc_integration(f, D, N):
    '''
    Function for a montecarlo integration

    Parameters
    ----------
    f : callable
        function to integrate
    D : int
        number of dimensions
    N : int
        total number of points
    
    Return
    ------
    integral : float
        integral of f
    error : float
        error over mean value of f
    '''

    # N random points in D dimensions
    points   = np.random.rand(N, D)
    values   = f(points)
    integral = np.mean(values)
    error    = np.sqrt(np.sum((values - integral)**2)/(N*(N-1)))
    
    return integral, error

def rectangular_integration(f, D, N):
    '''
    Function for rectangular integration

    Parameters
    ----------
    f : callable
        function to integrate
    D : int
        number of dimensions
    N : int
        total number of points
    
    Return
    ------
    integral : float
        integral of f
    '''

    points_per_dim = np.linspace(0, 1, N)

    grid   = np.meshgrid(*[points_per_dim]*D)
    coords = np.stack(grid, axis=-1).reshape(-1, D)
    
    integral = np.mean(f(coords))
    
    return integral


dimensions = np.array([i for i in range(1, 8 + 1)])
N = int(2.5e7)

errors_rectangular = []
errors_monte_carlo = []
diff_mc            = []

for D in dimensions:
    exact_value = D/3
    
    # rectangular
    rect_integral = rectangular_integration(f, D, int(N**(1/D)))
    errors_rectangular.append(abs(exact_value - rect_integral))
     
    # Montecarlo
    mc_integral , e = mc_integration(f, D, N)
    errors_monte_carlo.append(e)
    diff_mc.append(abs(exact_value - mc_integral))


plt.figure(figsize=(10, 6))
plt.plot(dimensions, errors_rectangular, label='rect', marker='.')
plt.plot(dimensions, diff_mc, label="MC diff", marker='.')
plt.plot(dimensions, errors_monte_carlo, label='MC error', marker='.')
plt.plot(dimensions, np.ones(len(dimensions))/np.sqrt(N), 'b--' , label=r"$1/\sqrt{N}$")
plt.plot(dimensions, 0.2*dimensions/(N**(1/dimensions)), 'k--', label=r'$0.2*D/N^{1/D}$')
plt.xlabel('Dimensions (D)')
plt.ylabel('|analytical - numerical|')
plt.title('Error vs Dimension')
plt.yscale('log')
plt.legend()
plt.grid()
plt.show()

