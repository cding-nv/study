import os
import matplotlib.pyplot as plt


#file_name = 'log-bert-large-training-default.txt'
file_name = 'log-bert-large-training-custom_op.txt'

#target = 'eval_loss'
target = 'total_loss'
values = []


f = open(file_name)
lines = f.readlines()
i=0
for line in lines:
    i+=1
    if target in line :
        line = line.split(':')
        print(line[7])
        print(line[7].split(' ')[1])
        value = float(line[7].split(' ')[1])
        values.append(value)

    #if i>1000:break

print(values)

x = [i for i in range(len(values))]
plt.plot(x,values, 'ro-', label=target)
plt.legend(loc='best')
plt.show()
plt.savefig('image.jpg')
