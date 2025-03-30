# Introduciamo il parametro di softening per evitare la divergenza del potenziale kepleriano
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import animation

# classe per le particelle
class Particle:
    def __init__(self, x, y, vx, vy, m=1):
        # posizione
        self.x = x
        self.y = y
        # velocità
        self.vx = vx
        self.vy = vy
        # massa
        self.m = m
    
    def n_vel(self, fx, fy, dt):
        self.vx += fx * dt
        self.vy += fy * dt

    def n_pos(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

class Sistema:
    """
    Classe per il sistema di corpi
    Il costruttore prende in input i seguenti parametri:
    -   corpi: list
        lista di oggetti della classe Particle che viene passata al sistema
    -   G: float
        costante di gravitazione universale
    -   sp: float, optional, default=0
        parametro di softening
    """
    def __init__(self, corpi, G, sp=0):
        self.corpi = corpi
        self.G = G
        self.sp = sp
    
    def evolve(self, dt):
        """
        Funzione che descrive l'evoluzione del sistema
        dt: float
            spacing temporale
        """
        for corpo_1 in self.corpi:
            fx = 0
            fy = 0
            for corpo_2 in self.corpi:
                if corpo_1 != corpo_2:
                    dx = corpo_2.x - corpo_1.x
                    dy = corpo_2.y - corpo_1.y
                    d = np.sqrt(dx ** 2 + dy ** 2)
                    fx += self.G * corpo_1.m * corpo_2.m * dx / (d**2 + self.sp)**(3/2)
                    fy += self.G * corpo_1.m * corpo_2.m * dy / (d**2 + self.sp)**(3/2)
        corpo_1.n_vel(fx, fy, dt)
        for corpo in self.corpi:
            corpo.n_pos(dt)

# inizializzazione delle particelle
random.seed(69420)
dt = 1/20000000
T = int(4/dt)
E = np.zeros(T)
L = np.zeros(T)
G = 1

N = 10
C = []
for n in range(N//2):
    v_x = random.uniform(-0.5, 0.5)
    v_y = random.uniform(-0.5, 0.5)
    C.append(Particle(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), v_x, v_y))
    C.append(Particle(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), -v_x, -v_y))

# Tensore posizione
X = np.zeros((2, T, N))

# Creazione del sistema
soft = 0.01
sist = Sistema(C, G, soft)

for t in range(T):
    sist.evolve(dt)
    for n, corpo in enumerate(sist.corpi):
        X[:, t, n] = corpo.x, corpo.y

fig = plt.figure("Nbody")
plt.grid()
plt.xlim(np.min(X[::2, :]) - 0.5, np.max(X[::2, :]) + 0.5)
plt.ylim(np.min(X[1::2, :]) - 0.5, np.max(X[1::2, :]) + 0.5)
colors = ['b'] * N
dot = np.array([])
for c in colors:
    dot = np.append(dot, plt.plot([], [], 'o', c=c))

def animate(i):
    for k in range(N):
        dot[k].set_data((X[0, i, k],) , (X[1, i, k],))
    return dot
print(dot)
anim = animation.FuncAnimation(fig, animate, frames=np.arange(0, T, 100000), interval=1, repeat=True)
plt.title("Nbody problem")
plt.xlabel("X(t)", fontsize=20)
plt.ylabel("Y(t)", fontsize=20)
anim.save('Nbody.gif', writer='imagemagick', fps=60)