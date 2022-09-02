import numpy as np

x1 = np.arange(9.0).reshape(3, 3)
x2 = np.arange(3.0)
x3 = np.arange(9.0).reshape(3, 3)
x = np.add(x1, x3)

print(x1)
print(x3)
print(x)
