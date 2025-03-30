"""
Code for diagonalizing a Hamiltonian of the form H = P^2 + V(x).
The code uses the power method using the conjugate gradient as a method for inverting a matrix.
Everything is implemented in the matrixless form, i.e. no matrices are allocated
instead whenever necessary the result of the matrix-vector product is calculated.
"""
import time
import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from scipy.special import eval_hermite


def conj_grad_matrixless(mult, b, tol=1e-6):
    """
    Matrix-free implementation of conjugate gradient method for solving M v = b.
    
    Parameters
    ----------
    mult : callable
        A function that returns the product of the matrix with a vector (matrix-free approach).
    b : 1darray
        Ordinate or "dependent variable" values.
    tol : float, optional
        Required tolerance (default 1e-6).

    Return
    ------
    x : 1darray
        Solution of the system.
    err : float
        Error of the solution.
    """

    N = len(b)
    x = np.zeros(N)  # Initial guess

    r  = b - mult(x) # Residuals
    p  = r           # Descent direction
    r2 = sum(r*r)    # Norm^2 of residuals

    iter = 0

    while True:
        Ap = mult(p)              # Compute matrix-free product M @ p
        alpha = r2 / (p @ Ap)     # Descent step

        x = x + alpha * p         # Update position
        r = r - alpha * Ap        # Update residuals

        r2_new = sum(r*r)         # Norm^2 of new residuals
        beta = r2_new / r2        # Compute step for p

        r2 = r2_new  # Update norm

        if np.sqrt(r2_new) < tol:  # Break condition
            break

        p = r + beta * p  # Update descent direction
        iter += 1

    
    err = np.sqrt(r2_new)
    return x, err


def eig_matrixless(mult, N, k=None, tol=1e-10, magnitude='small', inverse_solver='cg'):
    '''
    Compute the eigenvalue decomposition of a matrix using power iteration
    or inverse iteration in a matrix-free manner, avoiding storing the matrix.

    Parameters
    ----------
    mult : Callable
        A function that returns the product of the matrix with a vector (matrix-free approach).
    k : int or None
        Number of eigenvalues and eigenvectors to find. If None, find all.
    tol : float
        Tolerance for convergence.
    magnitude : string
        'small' to find smallest eigenvalues, 'big' to find largest eigenvalues.
    inverse_solver : string
        Method for solving linear systems. Default is 'cg' (Conjugate Gradient).
    
    Returns
    -------
    eigval : 1darray
        Eigenvalues found.
    eigvec : 2darray
        Corresponding eigenvectors.
    counts : 1darray
        Number of iterations for each eigenvalue.
    '''

    # Handle whether we want the smallest or largest eigenvalues
    if magnitude == 'small':
        # Define a matrix-free inverse operator using your Conjugate Gradient implementation
        def mult_inv(v):
            if inverse_solver == 'cg':
                # Use custom CG solver to solve M*x = v (instead of np.linalg.solve)
                x, _ = conj_grad_matrixless(mult, v)
                return x
            else:
                raise NotImplementedError(f"Inverse solver '{inverse_solver}' not implemented")
        
        mat = mult_inv
    else:
        # Normal power iteration
        mat = mult

    if k is None:
        k = N

    eigvec = []
    eigval = []
    counts = []

    for _ in range(k):

        # Initializzation
        v_p = np.random.randn(N)
        v_p = v_p / np.linalg.norm(v_p)
        l_v = np.random.random()
        Iter = 0

        while True:
            l_o = l_v
            v_o = v_p                   # Update vector
            v_p = mat(v_p)              # Compute new vector
            v_p /= np.linalg.norm(v_p)  # Normalizzation
            
            # Orthogonalization respect
            # all eigenvectors find previously
            for i in range(len(eigvec)):
                v_p = v_p - np.dot(eigvec[i], v_p) * eigvec[i]

            # Eigenvalue of v_p, A @ v_p = l_v * v_p
            # Multiplying by the transposed => (A @ v_p) @ v_p.T = l_v
            # Using v_p @ v_p.T = 1
            l_v = np.dot(v_p, mat(v_p))
            
            R1 = np.linalg.norm(v_p - v_o)
            R2 = np.linalg.norm(v_o + v_p)
            R3 = abs(l_v - l_o)               # In eigenvalues the convergence is quadratic
            
            Iter += 1
            if R1 < tol or R2 < tol or R3 < tol:
                break

        eigvec.append(v_p)
        eigval.append(l_v)
        counts.append(Iter)

    if magnitude == 'small':
        eigval = 1 / np.array(eigval)
    else:
        eigval = np.array(eigval)

    eigvec = np.array(eigvec).T
    return eigval, eigvec, counts



if __name__ == "__main__":

    np.random.seed(69420)
    
#===============================================================================
# Computational parameter
#===============================================================================
         
    k  = 10                    # how many levels compute
    n  = 1000                  # size of matrix
    xr = 10                    # bound
    xl = -10                   # bound
    L  = xr - xl               # dimension of box
    h  = (xr - xl)/(n)         # step size
    tt = np.linspace(0, n, n)  # array form 0 to n
    xp = xl + h*tt             # array of position
    
#===============================================================================
# Hamiltonian
#===============================================================================

    def H(psi, V, x):
        '''
        This function computes the product M @ v without storing M.
        In this case we have a simple tridiagonal matrix of the form:
        
                  [ 2 -1  0  0  0 ]       [ V0 0  0  0  0 ]
                  [-1  2 -1  0  0 ]       [ 0  V1 0  0  0 ]
        1/(2*h^2) [ 0 -1  2 -1  0 ]   +   [ 0  0  V3 0  0 ]
                  [ 0  0 -1  2 -1 ]       [ 0  0  0  V4 0 ]
                  [ 0  0  0 -1  2 ]       [ 0  0  0  0  V5]
        
        Parameters
        ----------
        psi : 1darray
            vector to multiply to the matrix
        V : callable
            potential of the sistem
        x : 1darray
            grid of our discretized sistem
        
        Return
        ------
        H_psi : 1darray
            result of H @ psi
        '''
        N = len(x)
        h = np.diff(x)[0]
        H_psi = np.zeros(N)
        
        # Kinetic term (tridiagonal part)
        factor = - 1/ (2 * h**2)
        
        for i in range(1, N-1):  # Loop through interior points only
            H_psi[i] = psi[i - 1] - 2 * psi[i] + psi[i + 1]
        
        # Boundary conditions
        H_psi[0]   =  -2 * psi[0]   + psi[1]
        H_psi[N-1] =  -2 * psi[N-1] + psi[N-2]
        
        H_psi *= factor

        # Potential term (diagonal part)
        H_psi += V(x) * psi
        
        return H_psi
    
    def Potential(x):
        return 0.5 * x**2

#===============================================================================
# Computation
#===============================================================================
    
    start = time.time()
    eigval, eigvec, Iter = eig_matrixless(lambda v: H(v, Potential, xp), n, k, tol=1e-5, magnitude='small')

    end = time.time() - start
    print(f'Elapsed time       = {end:.5f}\n')
    
    print("Theoretical     Computed          error")
    print("-----------------------------------------")
    for i in range(k):
        print(f'{i+0.5} \t \t {eigval[i]:.5f} \t {eigval[i]-(i+0.5):.2e}')
    
    psi = eigvec/np.sqrt(h)
    
#===============================================================================
# Plot
#===============================================================================
    
    def G(x, m):
        return (1/(np.pi)**(1/4))*(1/np.sqrt((2**m)*factorial(m)))*eval_hermite(m, x)*np.exp(-(x**2)/2)
   
    plt.figure(1)
    plt.title("Oscillatore armonico")#, fontsize=15)
    plt.xlabel('x')#, fontsize=15)
    plt.ylabel(r'$|\psi(x)|^2$')#, fontsize=15)
    plt.grid()
    plt.ylim(0, 3)
    plt.xlim(-5, 5)

    plt.plot(xp, Potential(xp) , color='black', label='V(x)= $ x^2/2 $')
    c = ['b', 'r', 'g']
    
    for L in range(3):
        
        plt.errorbar(xp, abs(psi[:,L])**2 + eigval[L], color=c[L], fmt='.')
        plt.plot(xp, np.ones(len(xp))*eigval[L], color=c[L], linestyle='--', label='$E_{%d}=%f$' %(L, eigval[L]))
        plt.plot(xp, abs(G(xp, L))**2 + eigval[L], color='k')
        
    #plt.legend(loc='best')
    
    plt.figure(2)
    plt.title("$\psi(x)$", fontsize=20)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('$\psi(x)$', fontsize=15)
    plt.grid()

    c = ['b', 'r', 'g']
    
    for L in range(3):
        plt.errorbar(xp, abs(abs(psi[:,L])**2 - abs(G(xp, L))**2), color=c[L], fmt='.')

    plt.show()