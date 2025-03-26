import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from scipy.io import wavfile
from IPython.display import Audio
import numba


# Values of the grid
Nx = 101
Nt = 500000
L = 0.7
dx = L/(Nx-1)
f = 440
c = 2*L*f
dt = 5e-6
l=5e-5
gamma=5e-5

# Boundary conditions
ya = np.linspace(0, 0.01, 70)
yb = np.linspace(0.01, 0, 31)
y0 = np.concatenate([ya, yb])


# Creation of the grid
sol = np.zeros((Nt, Nx))
sol[0] = y0
sol[1] = y0


@numba.jit("f8[:,:](f8[:,:], i8, i8, f8, f8, f8, f8)", nopython=True, nogil=True)
def compute_sol(d, times, length, dt, dx, l, gamma):
	"""
	Compute the solution via finite difference method

	Params
	-----
		d: 2d array
			Matrix representing the space-time grid where the solution is evaluated
		times: int
			Height of the grid (number of points the time is discretized)
		length: int
			Basis of the grid (number of points the x is discretized)
		dt: float
			Difference between two consecutive 'time' points
		dx: float
			Difference between two consecutive 'x' points
		l: float
			Damping coefficient
		gamma: float
			Stifness term
	"""
	for t in range(1, times-1):
		for i in range(2, length-2):
			outer_fact = (1/(c**2 * dt**2) + gamma/(2*dt))**(-1)
			p1 = 1/dx**2 * (d[t][i-1] - 2*d[t][i] + d[t][i+1])
			p2 = 1/(c**2 * dt**2) * (d[t-1][i] - 2*d[t][i])
			p3 = gamma/(2*dt) * d[t-1][i]
			p4 = l**2 / dx**4 * (d[t][i+2] - 4*d[t][i+1] + 6*d[t][i] - 4*d[t][i-1] + d[t][i-2])
			d[t+1][i] = outer_fact * (p1 - p2 + p3 - p4)
	return d

# Computing the solution and plotting some frames
sol = compute_sol(sol, Nt, Nx, dt, dx, l, gamma)
plt.plot(sol[500], label="Frame 500")
plt.plot(sol[1000], label="Frame 1000")
plt.legend()
plt.savefig("frame1000.png")
plt.show()

# Generating an animation
def animate(i):
    ax.clear()
    ax.plot(sol[i*10])
    ax.set_ylim(-0.01, 0.01)
    
fig, ax = plt.subplots(1,1)
ax.set_ylim(-0.01, 0.01)
ani = animation.FuncAnimation(fig, animate, frames=500, interval=50)
ani.save('string.gif',writer='pillow',fps=20)
def get_integral_fast(n):
	"""
		Computing the coefficient of the n-th harmonic of the signal in all the point of 'time'

		Params
		-----
			n: int
				number of the harmonics we want to find the coefficient

		Return
		-----
			arr: 1darray
				array cointaing in every cell the value that signal in 
	"""
	sin_arr = np.sin(n*np.pi*np.linspace(0,1,101))
	return np.multiply(sol, sin_arr).sum(axis=1) # Sommiamo sulle x, ovvero sulle 'righe'

# Estraiamo solamente le prime 10 armoniche tramite questa list comprension
hms = [get_integral_fast(n) for n in range(10)]

# Campioniamo l'ampiezzia del segnale campionando temporalmente dei punti a distanza di 10
tot = sum(hms)[::10] # compute the instananeous value of the audio signal
tot = tot.astype(np.float32)
wavfile.write('la.wav',20000,tot)	
### DOPO AVER GENERATO I FILE AUDIO A DIFFERENTI FREQUENZE
c = wavfile.read('do4.wav')[1]
e = wavfile.read('mi4.wav')[1]
g = wavfile.read('sol4.wav')[1]
wavfile.write('c_maj4.wav', 20000, 2 * c + 2 * e+ 2 * g)