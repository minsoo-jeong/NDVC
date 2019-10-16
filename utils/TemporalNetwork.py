import numpy as np
from queue import Queue
import copy
import sys
from utils.Period import Period


class TN(object):
    def __init__(self, score, idx, delimiter_idx=[0], TOP_K=20, SCORE_THR=0.9, TEMP_WND=1, MIN_PATH=3, MIN_MATCH=1):
        sys.setrecursionlimit(5000)

        # thresholds
        self.SCORE_THR = SCORE_THR
        self.TOP_K = TOP_K
        self.TEMP_WND = TEMP_WND
        self.MIN_PATH = MIN_PATH
        self.MIN_MATCH = MIN_MATCH

        self.query_length = score.shape[0]

        self.topk_score = score
        self.topk_idx = idx
        self.delimiter_idx = delimiter_idx

        if not TOP_K < 1:
            self.topk_score = self.topk_score[:, :self.TOP_K]
            self.topk_idx = self.topk_idx[:, :self.TOP_K]

        self.topk_frame_idx = []
        self.k = [len(s[s > SCORE_THR]) for n, s in enumerate(self.topk_score)]
        for n, k in enumerate(self.k):
            self.topk_score[n, k:] = -1
            self.topk_idx[n, k:] = -1
            self.topk_frame_idx.append(list(map(lambda x: self.__idx_to_frame_idx_per_video(x), self.topk_idx[n, :k])))

        # self.topk_score = [s[:self.k[n]] for n, s in enumerate(self.topk_score)]
        # self.topk_idx = [s[:self.k[n]] for n, s in enumerate(self.topk_idx)]
        '''
        for n, i in enumerate(self.topk_idx):
            self.topk_frame_idx.append(
                list(map(lambda x: self.__idx_to_frame_idx_per_video(x), i[:self.k[n]])))
        '''

        self.maximum_ref_table = np.ones((self.query_length, self.TOP_K, 5), dtype=np.int) * -1
        self.maximum_score_table = np.zeros((self.query_length, self.TOP_K), dtype=np.float)

        self.sink = []

        # self.__redundant_path()

    def fit(self):
        for t, vid in enumerate(self.topk_frame_idx):
            for rank, (vidx, fidx) in enumerate(vid):
                (next_q, next_d, max_q, max_d, max_match), score = self.__maximum_ref_path(t, rank)
                # print(self.topk_idx[t][rank],max_d)

                if max_d - self.topk_idx[t][rank] >= self.MIN_PATH and max_match > self.MIN_MATCH:

                    detect = {'query': Period(t, max_q),
                              'ref': Period(self.topk_idx[t][rank], max_d),
                              'ref_vid_idx':self.topk_frame_idx[t][rank][0],
                              'score': score}
                    sinkable = True

                    for n, s in enumerate(self.sink):
                        if detect['ref_vid_idx'] == s['ref_vid_idx'] and detect['query'].is_overlap(s['query']) and \
                                detect['ref'].is_overlap(s['ref']):
                            '''
                            self.sink[n] = {'query': detect['query'].union(s['query']),
                                            'ref': detect['ref'].union(s['ref']),
                                            'score': max(s['score'], detect['score'])}
                            '''
                            self.sink[n] = detect if detect['score'] > s['score'] else s
                            sinkable = False
                            break

                    if sinkable:
                        self.sink.append(detect)
        # print(self.maximum_ref_table)
        # self.sink.sort(key=lambda x: (x[0].start, x[1].start))
        # print(self.sink)
        # print(len(self.sink),self.sink)
        self.sink.sort(key=lambda x:x['score'],reverse=True)
        return self.sink

    def __maximum_ref_path(self, t, rank):
        if self.maximum_ref_table[t][rank][0] == -1:
            neighbor = self.__get_neighbor(t, rank)
            # neighbor = self.__get_neighbor2(t, rank)
            max_d = []
            for (n_t, n_rank, n_idx) in neighbor:
                (_, _, q, d, c), s = self.__maximum_ref_path(n_t, n_rank)
                max_d.append([n_t, n_rank, q, d, c + 1, s])
            if not len(neighbor):
                self.maximum_ref_table[t][rank] = [0, 0, t, self.topk_idx[t][rank], 1]
                self.maximum_score_table[t][rank] = self.topk_score[t][rank]
                # self.maximum_ref_table[t][rank] = [0, 0, t, self.topk_frame_idx[t][rank][1], 1]
            else:
                prev_max = max(max_d, key=lambda x: x[5])
                self.maximum_ref_table[t][rank] = prev_max[:5]
                self.maximum_score_table[t][rank] = prev_max[5] + self.topk_score[t][rank]
        # print(t,rank,self.max_link[t][rank])

        return self.maximum_ref_table[t][rank], self.maximum_score_table[t][rank]

    def __get_neighbor(self, t, rank):
        vid = self.topk_frame_idx[t][rank][0]
        last_ref_fidx = self.topk_frame_idx[t][rank][1]
        pos = []
        l_pos = []
        for time in range(t + 1, min(t + self.TEMP_WND + 1, len(self.topk_frame_idx))):
            m_pos = -1 if not len(l_pos) else min(l_pos)
            for r, (vidx, fidx) in enumerate(self.topk_frame_idx[time]):
                if vidx == vid and (last_ref_fidx < fidx <= last_ref_fidx + self.TEMP_WND):
                    if m_pos != -1:
                        if fidx < m_pos:
                            pos.append((time, r, self.topk_idx[time][r]))
                            l_pos.append(fidx)
                    else:
                        pos.append((time, r, self.topk_idx[time][r]))
                        l_pos.append(fidx)
        return pos

    def __get_neighbor3(self, t, rank):
        vid = self.topk_frame_idx[t][rank][0]
        last_ref_fidx = self.topk_frame_idx[t][rank][1]
        pos = []
        l_pos = []
        for time in range(t + 1, min(t + self.TEMP_WND + 1, len(self.topk_frame_idx))):
            for r, (vidx, fidx) in enumerate(self.topk_frame_idx[time]):
                if vidx == vid and (last_ref_fidx < fidx <= last_ref_fidx + self.TEMP_WND):
                    pos.append((time, r, self.topk_idx[time][r]))

        return pos

    def __get_neighbor2(self, t, rank):
        vid = self.topk_frame_idx[t][rank][0]
        last_ref_fidx = self.topk_frame_idx[t][rank][1]
        pos = []
        l_pos = []
        for time in range(min(t + self.TEMP_WND, len(self.topk_frame_idx) - 1), t, -1):
            m_pos = -1 if not len(l_pos) else max(l_pos)
            for r, (vidx, fidx) in enumerate(self.topk_frame_idx[time]):
                if (vidx == vid) and (last_ref_fidx < fidx <= last_ref_fidx + self.TEMP_WND):
                    if m_pos != -1:
                        if fidx > m_pos:
                            pos.append((time, r, self.topk_idx[time][r]))
                            l_pos.append(fidx)
                    else:
                        pos.append((time, r, self.topk_idx[time][r]))
                        l_pos.append(fidx)
        return pos

    def follow_link(self, t, rank):
        while self.maximum_ref_table[t][rank] != [0, 0]:
            t, rank = self.maximum_ref_table[t][rank]
        return t, rank

    def __idx_to_frame_idx_per_video(self, idx):
        video_idx = np.where(self.delimiter_idx > idx)[0]
        frame_idx = -1
        if len(video_idx):
            video_idx = video_idx[0] - 1
            frame_idx = idx - self.delimiter_idx[video_idx]

        return (video_idx, frame_idx)

    def __redundant_path(self):
        for n, p in enumerate(self.sink):
            query = p['q']
            ref = p['r']
            for m, pp in enumerate(self.sink[:n]):
                q = pp['q']
                r = pp['r']
                if q[0] <= query[0] and query[1] <= q[1] and r[0] <= ref[0] and ref[1] <= r[1]:
                    p['l'] = -1
                    break
        self.sink = list(filter(lambda x: x['l'] != -1, self.sink))


if __name__ == '__main__':
    from dataset.vcdb import VCDB
    import torch
    import os
    from utils import cosine_similarity, cosine_similarity_split

    db = VCDB()
    SCORE_THR = 0.5
    TEMP_WND = 5
    MIN_PATH = 3
    TOP_K = 30

    n_hit = 0
    n_det = 0
    n_ground = 0

    feature_path = '/DB/VCDB/frame_1_per_sec/resnet50-rmac/f-features'
    features = []
    vid_l = db.get_VideoList()
    delimiter_idx = [0]

    for vid in vid_l:
        videoid = vid['VideoID']
        f = torch.load(os.path.join(feature_path, '{}.pt'.format(videoid)))
        features.append(f)
        delimiter_idx.append(delimiter_idx[-1] + f.shape[0])
    features = torch.cat(features)
    delimiter_idx = np.array(delimiter_idx)
    print(features.shape)

    for qq in range(len(delimiter_idx) - 1):
        q_v_idx = qq
        # print(delimiter_idx[q_v_idx], delimiter_idx[q_v_idx + 1])
        q = features[delimiter_idx[q_v_idx]:delimiter_idx[q_v_idx + 1], :]
        print('{} {}'.format(q_v_idx, q.shape[0]))
        if q.shape[0] > 200:
            score, idx, cos = cosine_similarity_split(q, features, cuda=False, numpy=True)
        else:
            score, idx, cos = cosine_similarity(q, features, cuda=False, numpy=True)

        tn = TN(score, idx, delimiter_idx, TOP_K=TOP_K, SCORE_THR=SCORE_THR, TEMP_WND=TEMP_WND,
                MIN_PATH=MIN_PATH)

        tn.fit()

        # det = tn.get_sink()

        # print(det)
