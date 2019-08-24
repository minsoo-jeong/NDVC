import os
import wget

import threading

def download(qid, name, dir):
    try:
        filename = wget.download(url='http://vireo.cs.cityu.edu.hk/webvideo/videos/{}/{}'.format(qid, name), out=dir)
        return 0, filename
    except:
        print('fail to download ... {}'.format(os.path.join(dir,name)))
        return -1, name

p = '/media/mmlab/hdd/ms/NearDuplicateVideo/CC_WEB_VIDEO/Video_List.txt'
out = '/media/mmlab/hdd/ms/NearDuplicateVideo/CC_WEB_VIDEO/videos'

cmd = 'wget http://vireo.cs.cityu.edu.hk/webvideo/videos/'  # http://vireo.cs.cityu.edu.hk/webvideo/videos/1/1_38_Y.flv

of = open('log.txt', 'w')
ff = open('fail.txt', 'w')

with open(p) as file:
    for line in file.readlines():
        vid, qid, src, name, url = line.strip().split('\t')
        dir = os.path.join(out, qid)
        if not os.path.exists(dir):
            os.makedirs(dir)
            print('make dir ... {}'.format(dir))

        if os.path.exists(os.path.join(dir,name)):
            print('alread exist ... {}'.format(os.path.join(dir,name)))
            continue


        status, filename = download(qid, name, dir)
        if status==0:
            of.write(os.path.join(dir, filename))
        elif status==-1:
            ff.write(os.path.join(dir, filename))
        print(vid, os.path.join(dir, name))

of.close()
ff.close()

