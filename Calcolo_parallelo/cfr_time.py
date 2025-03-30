import time
import numba
import numpy as np

@numba.jit("f8[:,:](f8[:,:])", nopython=True, nogil=True)
def f1(x):
    return np.sin(x) + np.cos(x)

x = np.linspace(0, 10, int(1e4))
x = x * x[:, None]


start = time.time()
f1(x)
end = time.time()
print(f"First call:  {end - start:.4f} s")

start = time.time()
f1(x)
end = time.time()
print(f"Second call: {end - start:.4f} s")

start = time.time()
f1(x)
end = time.time()
print(f"Third call:  {end - start:.4f} s")
