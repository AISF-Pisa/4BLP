import numpy as np
import numba
import matplotlib.pyplot as plt
import time

@numba.jit("Tuple([f8[:, :], i8])(f8[:, :], f8, f8)", nopython=True, nogil=True, parallel=True)
def sol_potential(potential, tau, w):
    """
    Compute the solution of 2D Laplacian

    Params
    -----
    potential: matrix
        matrix representing the grid of the point in which the potential is valued
    tau:
        difference between two successive iteration
    w:
        relaxiation parameter for the SOR method
    Output:
    ------
    (potential, index): tuple
        tuple contained as first element a matrix in which the potential is valued and the number of iterations needed
    """
    length = len(potential[0])
    potential_0 = np.zeros((length, length), dtype="float64")
    index = 0
    integ0 = 0
    while True:
        integ = 0
        for i in numba.prange(1, length-1):
            for j in numba.prange(1, length-1):
                potential[j][i] = w * 0.25 * (potential[j+1][i] + potential[j-1][i] + potential[j][i+1] + potential[j][i-1]) + (1-w)*potential[j][i]
                integ += np.abs(potential[j][i])
        if  np.abs(integ - integ0)/((length)*(length)) > tau:
            integ0 = integ
            index += 1
        else:
            break
    return (potential, index)


edge = np.linspace(-1, 1, 300) 
bordo_supy = np.cos(np.pi * edge / 2)
bordo_infy = edge**4
bordo_supx = 1/(np.e**-1 - np.e) * (np.exp(edge)-np.e)
bordo_infx = 0.5 * (edge**2 - edge)
# Creiamo una meshgrid


potenziale = np.zeros((300, 300))
potenziale[0, :] = bordo_infy
potenziale[-1, :] = bordo_supy
potenziale[:, 0] = bordo_infx
potenziale[:, -1] = bordo_supx
xv, yv = np.meshgrid(edge, edge)

start = time.time()
result = sol_potential(potenziale, tau=1e-8, w=1.99)
end = time.time()
fig, ax = plt.subplots(1, 1, figsize=(8,6))
clr_plot = ax.contourf(xv, yv, result[0], 30)
ax.set_xlabel('x/a', fontsize=18)
ax.set_ylabel('y/a', fontsize=18)
cbar = fig.colorbar(clr_plot)
cbar.set_label('V/V0', fontsize=18)
cbar.ax.tick_params(labelsize=10)
ax.set_title('Potenziale', fontsize=18)
plt.savefig("potenziale_senzablocco_GSA.png")
print(f"Sono state necessarie {result[1]} iterazioni e {end-start} secondi")
plt.show()
