import numpy as np

a = np.arange(24).reshape(6, 4)
print(a)

np.savetxt("6x4.txt", a)

a1,a2,a3 = np.split(a, 3, axis=0)
print(a1)
np.savetxt("6x4_split.txt", a1, delimiter=',')
np.savetxt("6x4_split.txt", a2, delimiter=',')
np.savetxt("6x4_split.txt", a3, delimiter=',')


