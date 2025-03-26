import numpy as np

a = np.array(range(0,100))
mask = (a > 50) & ((a % 4) == 0)
print(a[mask])

b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
c = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]])
print(b * c)
print(b @ c)
