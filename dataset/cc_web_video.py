import os
import re
import numpy as np


class CC_WEB_VIDEO:
    def __init__(self, root='/DB/CC_WEB_VIDEO'):
        # managed with VideoID order
        self.root = root
        self.filelist = []
        self.seed = self.__read_seed()
        self.gt = self.__read_GT()

        self.filelist, self.blacklist, self.qids = self.__read_annotation()
        self.num_query = len(self.qids)

    def __repr__(self):
        out = '>> NAME : CC_WEB_VIDEO\n'
        out += '>> PATH : {}\n'.format(self.root)
        out += '>> {} videos ({} videos are deprecated)'.format(len(self.filelist), len(self.blacklist))

        return out

    def __read_seed(self):
        f = open(os.path.join(self.root, 'Seed.txt'), 'r', encoding='utf8')
        seed = []
        for line in f.readlines():
            line = line.replace('\r', '').rstrip().split('\t')
            seed.append({'QueryID': int(re.findall("\d+", line[0])[0]), 'VideoID': int(line[1])})
        f.close()

        return seed

    def __read_annotation(self):
        if not os.path.exists(os.path.join(self.root, 'annotation.txt')):
            print('make annotation...')
            self.__make_annotation()

        f = open(os.path.join(self.root, 'annotation.txt'), 'r')
        files = []
        blacklist = []
        qids = set()
        for line in f.readlines():
            idx, vid, qid, status, filename, filepath = line.rstrip().split('\t')
            idx, vid, qid = list(map(int, [idx, vid, qid]))
            qids.add(qid)
            info = {"index": idx, "VideoID": vid, "QueryID": qid,
                    "status": status, "VideoName": filename, "VideoPath": filepath}
            if idx < 0:
                blacklist.append(info)
            else:
                files.append(info)
        f.close()
        qids = list(qids)
        qids.sort(key=int)
        return files, blacklist, qids

    def __read_GT(self):
        gt = []
        for q in range(1, 25):
            f = open(os.path.join(self.root, 'GT', 'GT_{}.rst'.format(q)))
            for line in f.readlines():
                line = line.replace('\r', '').rstrip().split('\t')
                gt.append({'QueryID': q, 'VideoID': int(line[0]), "status": line[1]})
            f.close()
        return gt

    def __make_annotation(self):
        video_annotation = open(os.path.join(self.root, 'annotation.txt'), 'w')
        f = open(os.path.join(self.root, 'Video_List.txt'), 'r', encoding='utf8')
        filelist = []
        idx = 0
        for line in f.readlines():
            line = line.rstrip().split('\t')
            path = os.path.join(self.root, 'videos', line[1], line[3])
            gt = list(filter(lambda x: x['VideoID'] == int(line[0]), self.gt))
            status = '0' if len(gt) == 0 else gt[0]['status']
            seed = [s['VideoID'] for s in self.seed]
            if int(line[0]) in seed: status = 'Q'
            info = {"VideoID": int(line[0]), "QueryID": int(line[1]), "VideoName": line[3],
                    "VideoPath": path, "index": idx, 'status': status}
            if os.path.exists(path):
                filelist.append(info)
                idx += 1
            else:
                info['index'] = -1
                filelist.append(info)
            out = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(info['index'], info['VideoID'], info['QueryID'],
                                                    info['status'], info['VideoName'], info['VideoPath'])
            video_annotation.write(out)
        f.close()

    def __validate_fmt(self, fmt):
        format = ["QueryID", "VideoID", "VideoName", 'VideoPath', 'index', 'status']
        if not len(fmt): fmt = format
        if isinstance(fmt, str):
            fmt = [fmt]
        elif not isinstance(fmt, list):
            fmt = list(fmt)
        fmt = [format[[f.lower() for f in format].index(fm.lower())] for fm in fmt]
        return fmt

    def __validate_qid(self, qid):
        if isinstance(qid, int):
            qid = [qid]
        elif not isinstance(qid, list):
            qid = list(qid)
        if not len(qid): qid = self.qids
        return qid

    def get_VideoList(self, qid=[], fmt=[]):
        qid = self.__validate_qid(qid)
        fmt = self.__validate_fmt(fmt)
        l = [{f: info[f] for f in fmt} for info in self.filelist if int(info['QueryID']) in qid]
        return l

    def get_VideoByVid(self, vid=[]):
        return list(filter(lambda x: x['VideoID'] in vid, self.filelist))

    def get_BlackList(self, qid=[], fmt=[]):
        qid = self.__validate_qid(qid)
        fmt = self.__validate_fmt(fmt)
        l = [{f: info[f] for f in fmt} for info in self.blacklist if int(info['QueryID']) in qid]
        return l

    def get_Query(self):
        return list(filter(lambda x: x['status'] == 'Q', self.filelist))

    def get_reference_video_index(self, qid=[], status=''):
        qid = self.__validate_qid(qid)

        if status.find('-1') != -1:
            a, b = status.split('-1')
            status = list(a) + list(b) + ['-1']
        else:
            status = list(status)
        l = []
        for q in qid:
            l.append([info['index'] for info in self.filelist if info['QueryID'] == q and info['status'] in status])

        # l = [{f: info[f] for f in fmt} for info in self.filelist if info['QueryID'] in qid and info['status'] in status]
        return l


if __name__ == '__main__':
    db = CC_WEB_VIDEO()
    print(db)
    print('=====')

    print(db.blacklist)
    print(db.get_Query())
    print(db.filelist[:5])
    l = db.get_reference_video_index(qid=[1, 2], status='-1')
    print(len(l), l)
