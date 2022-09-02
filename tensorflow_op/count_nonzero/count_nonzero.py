import numpy as np

#a = [ 0  1  2  0  4  5  6  7  8  0  0 11  0 13 14 15 16 17 18  0  0 21  0  0  24  0 26  0 28  0 30 31  0  0  0  0 36 37  0 39  0 41 42 43  0 45 46  0  48  0 50 51 52  0  0 55  0 57 58  0 60  0  0  0 64 65  0 67 68 69 70  0  0  0 74 75  0  0 78  0]

a = [ 0,  1,  2,  0,  4,  5,  6,  7,  8,  0 ] 
e_group = np.split(a, 10, axis=0)
#a = np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]], axis=1)
zero = np.count_nonzero(e_group, axis=1)

print(zero)
