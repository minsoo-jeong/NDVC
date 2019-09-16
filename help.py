import os
import numpy as np
import torch
from sklearn.metrics import *

a= np.arange(0,10)
print(a)
exit()
a=torch.tensor(np.random.rand(15).reshape(5,3))

start=0
end=5
f=a[start:end+1,:]
f2=torch.cat([a[:start,:],a[end+1:,:]])
print(a)
print(f)
print(f2)



exit()

a = [1, 2, 3, 4]
b = ['a', 'b', 'c', 'd']
c = zip(a, b)
for i, cc in enumerate(c):
    print(i, cc)

print(c)
exit()
print(tuple(a.values()))

a = [1, 0, 0, 1]
b = [.6, 0.6, .8, 0.7]
prec, rec, _ = precision_recall_curve(a, b)
print(prec, rec)

a = [1, 0, 0, 1, 0]
b = [.6, 0.6, .8, 0.7, 0.8]
prec, rec, _ = precision_recall_curve(a, b)
print(prec, rec)

a = [1, 0, 0, 1, 1, 0]
b = [.6, 0.6, .8, 0.7, 0.8, 0]
prec, rec, _ = precision_recall_curve(a, b)
print(prec, rec)

exit()
c = [{'1': 1, '2': [11]}, {'1': 1, '2': [12]}, {'1': 1, '2': [13]}]
a = {'1': 4, '2': [11]}
print(a, c)
print(a in c)
exit()

a = np.random.rand(5 * 5).reshape(5, 5)
print(a)

exit()
a = torch.rand(55, 5)
print(a)
l = a.split(10, dim=0)
print(l)
c = torch.cat(l, dim=0)
print(c.shape)

exit()
for i in range(10, -1, -1):
    print(i)
l = np.array(list(range(5 * 5))).reshape(5, 5)
print(l)
l[2, :] = 0
print(l)
print(l[:1])
l[:, 2] = 0
print(l)
l.remove(2)
print(l)
a = l.pop(2)
print(l, a)
exit()
p = '/DB/CC_WEB_VIDEO/frame_1_per_sec/frames'
for root, dirs, files in os.walk(p):
    if root == p: continue

    if len(files) == 0: print(root)
    # print(root, dirs,len(files))
