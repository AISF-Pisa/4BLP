import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

path = r"D:\Dati Windows\Desktop\dati_4_lez.txt"


y, t, sigma_y = np.loadtxt(path, unpack=True, dtype='float64')
init = (300, 6.24, -np.pi/3, 1/9, 400)
def f(x, A, w, phi, b, c):
    return A * np.exp(- b * x) * np.cos(w*x + phi) + c
xx = np.linspace(min(t), max(t), num=1000)
plt.errorbar(t, y, yerr=sigma_y, fmt='.')
plt.plot(xx, f(xx, *init), label='parametri ad occhio')
plt.show()

popt, pcov = curve_fit(f, t, y, sigma=sigma_y, p0=init)
print(popt)
print(pcov)
plt.plot(xx, f(xx, *popt), label='curve_fit')
plt.legend()

print(f"Il chi quadro è: {np.sum( ((y - f(t, *popt))/sigma_y)**2)}")
print(f"I parametri stimati sono: {popt}")
print(f"Gli errori sui parametri sono {np.sqrt(np.diag(pcov))}")