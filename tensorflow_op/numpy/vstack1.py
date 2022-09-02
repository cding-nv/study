import numpy as np

B = 1
H = 16
S = 384

np_input1 = np.full((H, S, S), 1.0)
np_input2 = np.full((H, S, S), 2.0)
np_input = np.vstack([np_input1, np_input2])
print(np_input.shape)
