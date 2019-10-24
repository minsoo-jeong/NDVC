meta=open('/DB/videoCopyDetection/background_dataset/bg_meta.txt')
#re_meta=open('/DB/videoCopyDetection/background_dataset/bg_meta2.txt','w')

for line in meta.readlines():
    name, cls, fps, dur, nf = line.rstrip().split(',')
    cls='bg_'+cls
    l = ','.join(map(str, [name, cls, fps, dur, nf])) + '\n'
    #re_meta.write(l)
