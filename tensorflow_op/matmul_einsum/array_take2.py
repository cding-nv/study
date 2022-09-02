import numpy as np

aa = np.array([
    [[[0,1,2,3]]],
    [[[4,5,6,7]]],
    [[[8,9,10,11]]],
    [[[12,13,14,15]]]
])

index = range(1,2)
#bb = aa[[1,2,0]]
bb = aa[index]
print(bb)
print(aa.shape)
print(bb.shape)

exit()

cc = np.array([[0, 0, 0],
 [0, 0, 3],
 [0, 0, 4],
 [0, 0, 7],
 [1, 0, 0],
 [1, 0, 1],
 [1, 0, 3],
 [1, 0, 4],
 [1, 0, 5],
 [1, 0, 6],
 [2, 0, 1],
 [2, 0, 5],
 [2, 0, 6],
 [2, 0, 7],
 [3, 0, 4],
 [3, 0, 5],
 [3, 0, 6],
 [3, 0, 7],
 [4, 0, 1],
 [4, 0, 2],
 [4, 0, 4],
 [4, 0, 5],
 [4, 0, 6],
 [4, 0, 7],
 [5, 0, 2],
 [5, 0, 3],
 [5, 0, 4],
 [5, 0, 5],
 [6, 0, 0],
 [6, 0, 2],
 [6, 0, 4],
 [6, 0, 6],
 [6, 0, 7],
 [7, 0, 0],
 [7, 0, 1],
 [7, 0, 2],
 [7, 0, 5],
 [7, 0, 6],
 [7, 0, 7],
 [8, 0, 2],
 [8, 0, 3],
 [8, 0, 5],
 [8, 0, 6],
 [9, 0, 1],
 [9, 0, 3],
 [9, 0, 5],
 [9, 0, 6],
 [9, 0, 7]])

print(cc.shape)

max_num = 6
E = 10
position_v = np.arange(cc.shape[0])
print(position_v)
index_0 =0
pos = 0
for i in range(E * max_num):
    print("i ", i, " index_0 ", index_0, " pos ", pos)
    if (pos >= cc.shape[0]):
        break
    if cc[pos][0] == index_0:
        position_v[pos] = i
        pos += 1
    if (i + 1) % max_num == 0:
        index_0 += 1
        
print("position_v: ", position_v)

G = 1
C = 8
index_0 = 0
pos = 0
position_v_pad = np.arange(cc.shape[0])

for i in range(cc.shape[0]):
    position_v_pad[i] = cc[i][0] * C + cc[i][2]

print("position_v_pad: ", position_v_pad)


