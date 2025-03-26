import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

path = r"D:\Dati Windows\Desktop\dati_.txt"

t, sigma_t, x, sigma_x = np.loadtxt(path, dtype='float64', unpack=True)

def f(x, a):
    return a*x**2

popt, pcov = curve_fit(f, t, x, sigma=sigma_x)
print(f"{popt}")
print(f"{pcov}")

chisq = np.sum( ((x - f(t, *popt))/sigma_x) ** 2)
print(f"{chisq}")

plt.errorbar(t, x, fmt='o', xerr=sigma_t, yerr=sigma_x)
plt.grid()
plt.plot(t, f(t, *popt))
plt.show()


