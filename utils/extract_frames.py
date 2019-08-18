import os
import subprocess
from dataset.cc_web_video import CC_WEB_VIDEO


def execute_ffmpeg(argv):
    cmd = 'ffmpeg {}'.format(argv)
    p = subprocess.Popen(args=cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    return p.returncode, out.decode('utf8'), err.decode('utf8')


def extract_1_frame_per_second(video, dst, fmt='%d.jpg'):
    if not os.path.exists(dst):
        os.makedirs(dst)
    ret, out, err = execute_ffmpeg('-i {} -r 1 -f image2 {}/{}'.format(video, dst, fmt))
    return ret, out, err


if __name__ == '__main__':
    db = CC_WEB_VIDEO()

    for l in db.get_VideoList():
        p = l['VideoPath']
        vid = l['VideoID']
        d = '/DB/CC_WEB_VIDEO/frame_1_per_sec/frames/{}'.format(os.path.splitext(l['VideoName'])[0])
        dst = '/DB/CC_WEB_VIDEO/frame_1_per_sec/frames/{}'.format(vid)
        os.system('mv {} {}'.format(d, dst))
        print(d, vid)
        # r, o, e = extract_1_frame_per_second(p, d)
        # print(r,d)

    '''
    v = l[0]['VideoPath']
    d='/DB/CC_WEB_VIDEO/frame_1_per_sec/{}'.format(os.path.splitext(l[0]['VideoName'])[0])
    print(d)
    r, o, e = extract_1_frame_per_second(v, d)
    #print(r, o, e)
    '''
