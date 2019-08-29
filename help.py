import os

p='/DB/CC_WEB_VIDEO/frame_1_per_sec/frames'
for root, dirs, files in os.walk(p):
    if root==p: continue

    if len(files)==0: print(root)
    #print(root, dirs,len(files))