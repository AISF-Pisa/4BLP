import numpy as np
import matplotlib.pyplot as plt

np.random.seed(69420)

p = lambda x : np.exp(-x)
q = lambda x : -np.log(1-x)

N   = int(1e6)
x_i =  np.random.rand(N)
y_i = q(x_i)

x = np.linspace(0, 14, 1000)

plt.title("Distribuzione esponenziale")
plt.ylabel("P(x)", fontsize=10)
plt.xlabel("x", fontsize=10)
plt.plot(x, p(x), 'b-', label=r'$e^{-x}$')
plt.hist(y_i, bins=100, histtype='step',
    density=True, color='orange', label='campionamento')
plt.legend()
plt.show()