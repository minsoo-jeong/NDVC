import os
import re


class CC_WEB_VIDEO:
    def __init__(self, root='/DB/CC_WEB_VIDEO'):
        self.root = root
        self.seed = self._read_seed()
        self.filelist, self.blacklist, self.qids = self._read_filelist()
        self.gt = self._read_GT()
        self.num_query = len(self.qids)

    def __repr__(self):
        out = '>> NAME : CC_WEB_VIDEO\n' \
              '>> PATH : {}\n' \
              '>> {} videos ({} videos are deprecated)' \
            .format(self.root, len(self.filelist), len(self.blacklist))

        return out

    def _read_seed(self):
        f = open(os.path.join(self.root, 'Seed.txt'), 'r', encoding='utf8')
        seed = []
        for line in f.readlines():
            line = line.replace('\r', '').rstrip().split('\t')
            seed.append({'QueryID': re.findall("\d+", line[0])[0], 'VideoID': line[1]})
        f.close()

        return seed

    def _read_filelist(self):
        f = open(os.path.join(self.root, 'Video_List.txt'), 'r', encoding='utf8')
        files = []
        blacklist = []
        qids = set()
        for line in f.readlines():
            line = line.rstrip().split('\t')
            path = os.path.join(self.root, 'videos', line[1], line[3])
            qids.add(int(line[1]))
            if os.path.exists(path):
                files.append({"VideoID": line[0], "QueryID": line[1], "VideoName": line[3], "VideoPath": path})
            else:
                blacklist.append({"VideoID": line[0], "QueryID": line[1], "VideoName": line[3], "VideoPath": path})
        f.close()
        qids = list(qids)
        qids.sort(key=int)
        return files, blacklist, qids

    def _read_GT(self):
        gt = []
        for q in range(1, 24):
            f = open(os.path.join(self.root, 'GT', 'GT_{}.rst'.format(q)))
            for line in f.readlines():
                line = line.replace('\r', '').rstrip().split('\t')
                gt.append({'QueryID': q, 'VideoID': line[0], "status": line[1]})
            f.close()
        return gt

    def get_VideoList(self, qid=[], fmt=[]):
        if isinstance(qid, int):
            qid = [qid]
        elif not isinstance(qid, list):
            qid = list(qid)

        if not len(qid): qid = self.qids
        format = ["QueryID", "VideoID", "VideoName", 'VideoPath']

        if not len(fmt): fmt = format
        if isinstance(fmt, str):
            fmt = [fmt]
        elif not isinstance(fmt, list):
            fmt = list(fmt)
        fmt = [format[[f.lower() for f in format].index(fm.lower())] for fm in fmt]
        l = [{f: info[f] for f in fmt} for info in self.filelist if int(info['QueryID']) in qid]

        '''
        for info in self.filelist:
            if int(info['QueryID']) in qid:
                a = dict()
                for f in fmt:
                    a[f] = info.get(f)
                l.append(a)
        '''
        return l

    def get_AllVideoPath(self):
        return [fl['VideoPath'] for fl in self.filelist]

    def get_AllVideoName(self):
        return [fl['VideoName'] for fl in self.filelist]

    def get_BlackList(self):
        return self.blacklist

    def get_BlackListName(self):
        return [fl['VideoName'] for fl in self.blacklist]


if __name__ == '__main__':
    db = CC_WEB_VIDEO()
    print(db)
    print('=====')
    print(db.get_BlackList())

    l=db.get_VideoList(fmt=['QueryID'])

    print(l[:10])
