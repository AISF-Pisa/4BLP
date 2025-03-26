"""
	Immaginiamo di essere in una scatola (-1, 1)^2
"""
import numpy as np
import numba
import matplotlib.pyplot as plt
import time
edge = np.linspace(-1, 1, 300) 
bordo_supy = np.cos(np.pi * edge / 2)
bordo_infy = edge**4
bordo_supx = 1/(np.e**-1 - np.e) * (np.exp(edge)-np.e)
bordo_infx = 0.5 * (edge**2 - edge)
# Creiamo una meshgrid
xv, yv = np.meshgrid(edge, edge) # Crea un array dove tutte le possibili combinazioni dei vettori edge, edge vengono scissi in due matrici dove (xv[i, j], yv[i, j]) rappresenta un punto della griglia

@numba.jit("f8[:,:](f8[:,:], i8)", nopython=True, nogil=True)
def sol_potential(potential, n_iter):
    length = len(potential[0])
    for n in range(n_iter):
        for i in range(1, length-1):
            for j in range(1, length-1):
                potential[j][i] = 1/4 * (potential[j+1][i] + potential[j-1][i] + potential[j][i+1] + potential[j][i-1])
    return potential

#	Settiamo le condizioni al bordo
potenziale = np.zeros((300, 300))
potenziale[0, :] = bordo_infy
potenziale[-1, :] = bordo_supy
potenziale[:, 0] = bordo_infx
potenziale[:, -1] = bordo_supx

potenziale_numbato = np.copy(potenziale)
start = time.time()
potenziale_numbato = sol_potential(potenziale_numbato, n_iter=10000)
end = time.time()
print(f"La soluzione con numba è: {end-start}")

def no_numbed_sol_potential(potential, n_iter):
    length = len(potential[0])
    for n in range(n_iter):
        for i in range(1, length-1):
            for j in range(1, length-1):
                potential[j][i] = 1/4 * (potential[j+1][i] + potential[j-1][i] + potential[j][i+1] + potential[j][i-1])
    return potential

"""start = time.time()
potenziale_notnumb = no_numbed_sol_potential(potenziale, n_iter=500)
end = time.time()
print(f"La soluzione non numbata è: {end-start}")"""


potenziale = sol_potential(potenziale, n_iter=10000)
fig, ax = plt.subplots(1, 1, figsize=(8,6))
clr_plot = ax.contourf(xv, yv, potenziale_numbato, 30)
ax.set_xlabel('x/a', fontsize=18)
ax.set_ylabel('y/a', fontsize=18)
cbar = fig.colorbar(clr_plot)
cbar.set_label('V/V0', fontsize=18)
cbar.ax.tick_params(labelsize=10)
ax.set_title('Potenziale', fontsize=18)
plt.savefig("_potenziale_senzablocco.png")
plt.show()
fig, ax = plt.subplots(1, 1, figsize=(8,6))
clr_plot = ax.contourf(xv, yv, potenziale, 30)
ax.set_xlabel('x/a', fontsize=18)
ax.set_ylabel('y/a', fontsize=18)
cbar = fig.colorbar(clr_plot)
cbar.set_label('V/V0', fontsize=18)
cbar.ax.tick_params(labelsize=10)
ax.set_title('Potenziale', fontsize=18)
plt.savefig("potenziale_senzablocco.png")
plt.show()

## BLOCCO
def blocco_potenziale(x, y):
    return np.select([(x>0.3)*(x<0.6)*(y > 0.3)*(y<0.6), (x <= 0.3)* (x>= 0.6)*(y<=0.3)*(y>=0.6)], [1., 0])
plt.figure(figsize=(5,4))
plt.contourf(xv, yv, blocco_potenziale(xv, yv))
plt.colorbar()
fixed = blocco_potenziale(xv, yv)
_bool = fixed!= 0
@numba.jit("f8[:, :](f8[:, :], b1[:, :], i8)",nopython=True, nogil=True)
def solve_potential_fixed(potential, fixed_bool, n_iter):
    length = len(potential[0])
    for n in range(n_iter):
        for i in range(1, length-1):
            for j in range(1, length-1):
                if not(fixed_bool[i][j]):
                    potential[i][j] = 1/4 * (potential[i+1][j] + potential[i-1][j] + potential[i][j+1] + potential[i][j-1])
    return potential

potenziale = np.zeros((300, 300))
potenziale[0, :] = bordo_infy
potenziale[-1, :] = bordo_supy
potenziale[:, 0] = bordo_infx
potenziale[:, -1] = bordo_supx
potenziale[_bool] = fixed[_bool]
potenziale = solve_potential_fixed(potenziale, _bool, n_iter=10000)

fig, ax = plt.subplots(1, 1, figsize=(8,6))
clr_plot = ax.contourf(xv, yv, potenziale, 30)
ax.set_xlabel('x/a', fontsize=18)
ax.set_ylabel('y/a', fontsize=18)
cbar = fig.colorbar(clr_plot)
cbar.set_label('V/V0', fontsize=18)
cbar.ax.tick_params(labelsize=10)
ax.set_title('Potenziale', fontsize=18)
plt.savefig("potenziale_conblocco.png")
plt.show()