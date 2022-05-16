'''reduce'''
from functools import reduce                # 3.x ??????????
def sum(x,y):
    return x+y
l = [1,2,3,4,5,6]
l = reduce(sum,l)
print(l)

l = [1,2,3,4,5,6]
l = reduce(lambda x,y:x+y,l)                # ????lambda
print(l)
help(reduce)                                # ???? reduce ????
