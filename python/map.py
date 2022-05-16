'''map'''
def sqr(i):
    return i**2
l = [1,2,3]
l = map(sqr,l)                                
print(l)                                     # 3.x ???????????? map object
l = list(l)
print(l)
l = [1,2,3]
l = list(map(lambda x : x**2, l))            # ????lambda
print(l)
