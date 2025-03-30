import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

# Distribution to be sampled, as if they were experimental data
x = np.linspace(0, 1.5, 50000)
y = x**x

# Compute the integral to normalize it, the normalization will
# ensure that we can always use the uniform generator between 0 and 1
Norm = si.simpson(y, x=x)
y = y/Norm

# Interpolations
s3 = InterpolatedUnivariateSpline(x, y, k=3)

# Compute cumulative ditribution function
cdf = np.array([s3.integral(x[0], i) for i in x])

# Remove possible equal values ​​otherwise the subsequent interpolation would not work
xq, iq = np.unique(cdf, return_index=True)

# Swap x with y to invert the cumulative function and get the quantile function
yq = x[iq]
quantile = InterpolatedUnivariateSpline(xq, yq, k=3)

Y=quantile(np.random.uniform(size=int(1e6)))


plt.figure(1)
plt.grid()
plt.title('Sampling with quantile function')
plt.xlabel('x', fontsize=10)
plt.ylabel('p(x)', fontsize=10)
plt.plot(x, y, '.', label='original data')
plt.plot(x, s3(x), 'k', label='spline')
plt.hist(Y, 100, histtype='step', density=True, label='sampling')
plt.legend(loc='best')
plt.show()