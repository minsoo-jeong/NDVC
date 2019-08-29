import os
import re
import numpy as np


class VCDB:
    def __init__(self, root='/DB/VCDB'):
        # managed with VideoID order
        self.root = root
        self.core_vlist, self.title, self.qids = self.__read_core_annotation()
        self.gt, self.gt_idx_per_core = self.__read_core_GT()

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
                idx, vid, qid, videoename, videopath = line.rstrip().split('\t')
                idx, vid, qid = list(map(int, [idx, vid, qid]))
                qids.add(qid)
                info = {"index": idx, "VideoID": vid, "QueryID": qid, "VideoName": videoename, "VideoPath": videopath}
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
        for qid, t in enumerate(title, 1):
            videos = os.listdir(os.path.join(core_path, t))
            videos.sort()
            for v in videos:
                idx = vid - 1
                info = {'index': idx, 'QueryID': qid, 'VideoID': vid, 'VideoName': v,
                        'VideoPath': os.path.join(core_path, t, v)}
                out = '{}\t{}\t{}\t{}\t{}\n'.format(info['index'], info['VideoID'], info['QueryID'],
                                                    info['VideoName'], info['VideoPath'])
                f.write(out)
                vlist.append(info)

                vid += 1

    def __read_core_GT(self):
        core_gt = os.path.join(self.root, 'core_gt.txt')
        if not os.path.exists(core_gt):
            self.__make_core_GT()
        core_gt = open(core_gt, 'r')
        gt = []
        gt_idx_per_core = [set() for i in self.core_vlist] + [set()]

        for l in core_gt.readlines():
            line = l.rstrip().split('\t')
            line[0], line[2] = map(int, [line[0], line[2]])
            gt_idx_per_core[line[0]].add(line[2])
            gt_idx_per_core[line[2]].add(line[0])

            info = {'A': line[0], 'A_VideoName': line[1],
                    'B': line[2], 'B_VideoName': line[3],
                    'A_start': line[4], 'A_end': line[5],
                    'B_start': line[6], 'B_end': line[7], }
            gt.append(info)
        return gt, gt_idx_per_core

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

    def get_GT(self, vid=[]):
        vid = self.__validate_vid(vid)
        l = []
        for v in vid:
            l.append([i['index'] for i in self.core_vlist if i['VideoID'] in self.gt_idx_per_core[v]])
        return l

    def time2sec(self, t):
        h, m, s = t.split(':')
        s = int(s) + 60 * int(m) + 3600 * int(h)
        return s

    def get_GT_time(self, vid=[]):
        vid = self.__validate_vid(vid)
        l = []
        for v in vid:
            for i in self.gt:
                if i['A'] == v:
                    l.append([i['A'], self.time2sec(i['A_start']), self.time2sec(i['A_end']),
                              i['B'], self.time2sec(i['B_start']), self.time2sec(i['B_end'])])
                elif i['B'] == v:
                    l.append([i['B'], self.time2sec(i['B_start']), self.time2sec(i['B_end']),
                              i['A'], self.time2sec(i['A_start']), self.time2sec(i['A_end'])])
        return l


if __name__ == '__main__':
    db = VCDB()
    # print(db.core_vlist)
    # print(db.gt)
    l = db.get_GT()
    print(l)
    print(len(l))
    print([len(i) for i in l if len(i) == 0])

    l = db.get_GT_time()
    print(l)
