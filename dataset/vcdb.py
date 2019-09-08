import os
import re
import numpy as np
import cv2


class VCDB:
    def __init__(self, root='/DB/VCDB'):
        # managed with VideoID order
        self.root = root
        self.core_vlist, self.title, self.qids = self.__read_core_annotation()
        self.gt, self.reference_video = self.__read_core_GT()
        self.gt_video_list = self.make_GT_list()

    def __read_core_annotation(self):
        core_annotation = os.path.join(self.root, 'core_annotation.txt')
        title = os.path.join(self.root, 'title.txt')
        if not os.path.exists(core_annotation) or not os.path.exists(title):
            print('make core annotation...')
            self.__make_core_annotation()
        with open(title, 'r') as f:
            title = []
            for line in f.readlines():
                title.append(line.rstrip())

        with open(core_annotation, 'r') as f:
            corelist = []
            qids = set()
            for line in f.readlines():
                idx, vid, qid, videoename, nframes, duration, fps, videopath = line.rstrip().split(',')
                idx, vid, qid = list(map(int, [idx, vid, qid]))
                qids.add(qid)
                info = {"index": idx, "VideoID": vid, "QueryID": qid, "VideoName": videoename, 'frames': nframes,
                        "Duration": float(duration), 'FPS': fps, "VideoPath": videopath}
                corelist.append(info)
        qids = list(qids)
        qids.sort(key=int)
        return corelist, title, qids

    def __make_core_annotation(self):

        f = open(os.path.join(self.root, 'core_annotation.txt'), 'w')
        t = open(os.path.join(self.root, 'title.txt'), 'w')
        core_path = os.path.join(self.root, 'core_dataset')
        title = os.listdir(core_path)
        title.sort()
        for ti in title:
            t.write(ti + '\n')

        vlist = []
        vid = 1
        idx = 0
        for qid, t in enumerate(title, 1):
            videos = os.listdir(os.path.join(core_path, t))
            videos.sort()
            for v in videos:
                path = os.path.join(core_path, t, v)
                cv_vid = cv2.VideoCapture(path)
                nframes = int(cv_vid.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cv_vid.get(cv2.CAP_PROP_FPS)
                dur = nframes / fps

                info = {'index': idx, 'QueryID': qid, 'VideoID': vid, 'VideoName': v,
                        'VideoPath': os.path.join(core_path, t, v), 'Duration': dur, 'frames': nframes, 'fps': fps}
                out = '{},{},{},{},{},{},{},{}\n'.format(info['index'], info['VideoID'], info['QueryID'],
                                                         info['VideoName'], nframes, dur, fps, info['VideoPath'])
                f.write(out)
                vlist.append(info)
                vid += 1
                idx += 1
                cv_vid.release()

    def __read_core_GT(self):
        core_gt = os.path.join(self.root, 'core_gt.txt')
        if not os.path.exists(core_gt):
            self.__make_core_GT()
        core_gt = open(core_gt, 'r')
        gt = []
        reference_video = [set() for i in self.core_vlist] + [set()]

        for l in core_gt.readlines():
            line = l.rstrip().split('\t')
            line[0], line[2] = map(int, [line[0], line[2]])
            reference_video[line[0]].add(line[2])
            reference_video[line[2]].add(line[0])

            info = {'A': line[0], 'A_VideoName': line[1],
                    'B': line[2], 'B_VideoName': line[3],
                    'A_start': line[4], 'A_end': line[5],
                    'B_start': line[6], 'B_end': line[7], }
            gt.append(info)
        return gt, reference_video

    def __make_core_GT(self):
        core_gt = open(os.path.join(self.root, 'core_gt.txt'), 'w')
        gt_path = os.path.join(self.root, 'annotation')
        for p in self.title:
            f = open(os.path.join(gt_path, '{}.txt'.format(p)), 'r')
            for l in f.readlines():
                a, b, start_a, end_a, start_b, end_b = l.rstrip().split(',')
                vid_a = list(filter(lambda x: x['VideoName'] == a, self.core_vlist))[0]
                vid_b = list(filter(lambda x: x['VideoName'] == b, self.core_vlist))[0]
                output = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(vid_a['VideoID'], vid_a['VideoName'],
                                                                   vid_b['VideoID'], vid_b['VideoName'],
                                                                   start_a, end_a, start_b, end_b)
                core_gt.write(output)

    def __validate_vid(self, vid):
        if isinstance(vid, int):
            vid = [vid]
        elif not isinstance(vid, list):
            vid = list(vid)
        if not len(vid): vid = [i for i, _ in enumerate(self.core_vlist, 1)]
        return vid

    # return reference video id about query video id
    def get_reference_video_index(self, vid=[]):
        vid = self.__validate_vid(vid)
        l = []
        for v in vid:
            l.append([i['index'] for i in self.core_vlist if i['VideoID'] in self.reference_video[v]])
        return l

    def __time2sec(self, t):
        h, m, s = t.split(':')
        s = int(s) + 60 * int(m) + 3600 * int(h)
        return s

    def get_GT(self, vid=[]):
        vid = self.__validate_vid(vid)
        l = []
        for v in vid:
            for i in self.gt:
                if i['A'] == v:
                    l.append({'vid': i['A'], 'start': self.__time2sec(i['A_start']),
                              'end': self.__time2sec(i['A_end']), 'ref_vid': i['B'],
                              'ref_start': self.__time2sec(i['B_start']),
                              'ref_end': self.__time2sec(i['B_end'])})
                elif i['B'] == v:
                    l.append({'vid': i['B'], 'start': self.__time2sec(i['B_start']),
                              'end': self.__time2sec(i['B_end']), 'ref_vid': i['A'],
                              'ref_start': self.__time2sec(i['A_start']),
                              'ref_end': self.__time2sec(i['A_end'])})
                '''
                if i['A'] == v:
                    l.append([i['A'], self.time2sec(i['A_start']), self.time2sec(i['A_end']),
                              i['B'], self.time2sec(i['B_start']), self.time2sec(i['B_end'])])
                elif i['B'] == v:
                    l.append([i['B'], self.time2sec(i['B_start']), self.time2sec(i['B_end']),
                              i['A'], self.time2sec(i['A_start']), self.time2sec(i['A_end'])])
                '''
        return l

    def get_VideoList(self, vid=[]):
        vid = self.__validate_vid(vid)
        l = [v for v in self.core_vlist if v['VideoID'] in vid]
        return l

    def time_to_idx(self, gt):
        ref_vid = gt['ref_vid']
        q_vid = gt['vid']
        q, ref = self.get_VideoList(vid=[q_vid, ref_vid])
        q_d = q['Duration']
        ref_d = ref['Duration']

    def make_GT_list(self):
        gt_videos = []
        for gt in self.gt:
            A = {'vid': gt['A'], 'name': gt['A_VideoName'], 'start': self.__time2sec(gt['A_start']),
                 'end': self.__time2sec(gt['A_end'])}
            B = {'vid': gt['B'], 'name': gt['B_VideoName'], 'start': self.__time2sec(gt['B_start']),
                 'end': self.__time2sec(gt['B_end'])}
            if A not in gt_videos:
                gt_videos.append(A)
            if B not in gt_videos:
                gt_videos.append(B)
        return gt_videos

    def get_GT_list(self, vid=[]):
        vid = self.__validate_vid(vid)
        l = [v for v in self.gt_video_list if v['vid'] in vid]
        return l


if __name__ == '__main__':
    db = VCDB()
    # print(db.core_vlist)
    l = db.get_GT_list(vid=1)
    print(l)
    b=db.get_GT(vid=1)
    print(b)
    exit()
    print(db.title)
    l = db.get_GT(vid=5)
    for i in l:
        print(i)

    db.time_to_idx({'vid': 3, 'start': 5, 'end': 12, 'ref_vid': 1, 'ref_start': 0, 'ref_end': 7})
    exit()

    print(db.core_vlist[220:225])
    l = db.get_reference_video_index()
    print(len(l))
    l = db.get_GT()
    print(len(l))
    print(l[:5])
    print(db.get_VideoList(vid=[3]))
    # print(l)
