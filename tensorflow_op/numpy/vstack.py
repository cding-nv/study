import numpy as np
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
z = np.array([666, 666, 666])

r = np.vstack([x,y])
print(x.shape)
print(r.shape)
print(r)
