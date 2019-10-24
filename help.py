import os
import numpy as np
import torch
import logging
import logging.config
import logging.handlers
import json
from datetime import datetime,timedelta

from collections import deque


l=[1,2,3]
print(l,l[:0])
exit()
def nn(l):
    for i, gt in enumerate(l):
       if gt==4:
            l.pop(i)
            break
    return l

a=[1,2,3,4,5,6]
print(nn(a))
print(a)
exit()

a='11112'
print(a[::-1],set(a))



exit()

s="pPoooyY"
s=str.lower(s)
print(s.count('p'),s.count('y'))
exit()
x = torch.randn(3, 4)
print(x)
a=torch.index_select(x,0,torch.tensor([1,2]))
print(a)

exit()


t=(datetime.now()+timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%s")
print('[{} {}]'.format(t,os.path.basename(__file__)))
'[2019-09-17 05:25:17 INFO train.py]'
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
