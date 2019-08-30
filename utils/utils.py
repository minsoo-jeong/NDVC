import numpy as np
import torch
import os
from torch.nn import functional as F
from datetime import datetime
from dataset.cc_web_video import CC_WEB_VIDEO
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from queue import Queue


def format_bytes(size):
    # 2**10 = 1024
    power = 2 ** 10
    n = 0
    power_labels = {0: '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n] + 'bytes'


# multiple query
def cosine_similarity(query, features, cuda=True, numpy=True):
    if cuda:
        query = query.cuda()
        features = features.cuda()
    query = F.normalize(query, 2, 1)
    features = F.normalize(features, 2, 1)

    cos = torch.mm(features, query.t()).t()
    score, idx = torch.sort(cos, descending=True)

    post = lambda x: x.cpu().numpy() if numpy else x.cpu()

    score, idx, cos = map(post, [score, idx, cos])

    return score, idx, cos


def evaluate_mAP(indice, gt, k=[1, 500, 1000], pr_curve=False):
    aps = []
    ap_ks = []
    prs = []
    for i in range(len(gt)):
        ap, ap_k, pr = evaluate_AP(indice[i], gt[i], k, pr_curve)
        print(ap, ap_k)
        aps.append(ap)
        ap_ks.append(ap_k)
        prs.append(pr)

    aps = np.array(aps)
    ap_ks = np.array(ap_ks)
    if pr_curve:
        print(len(prs))
        for pr in prs[1:]:
            for id, r in enumerate(pr):
                prs[0][id][0] += r[0]
                prs[0][id][1] += r[1]
        prs = [p[0] / (p[1] + 1e-12) for p in prs[0]]

    else:
        prs = None
    mAP = aps.mean(axis=0)
    mAP_K = ap_ks.mean(axis=0)
    return mAP, mAP_K, prs


def evaluate_AP(indice, gt, k=[1, 500, 1000], pr_curve=False):
    indice = np.array(indice)
    gt = np.array(gt)
    # print(indice,len(gt))
    c = 0
    ap = 0.0
    ap_k = []
    re_c = [int(i * len(gt) * 0.04) for i in range(0, 26)]
    pr = []
    for i in range(26):
        pr.append([])
    for rank, idx in enumerate(indice, 1):
        # print(rank,c,len(gt))
        if c > len(gt): break
        if idx in gt:
            c += 1
            ap += (c / rank)
        if rank in k:
            ap_k.append(ap / c)
        if pr_curve and c in re_c:
            pr[re_c.index(c)].append(round(c / rank, 8))
    ap /= c
    while len(ap_k) != len(k):  ap_k.append(ap)
    if pr_curve:
        pr = [[sum(p), len(p)] for p in pr]

    return ap, ap_k, pr


def draw_pr_curve(pr):
    p = pr
    r = [i * 0.04 for i in range(0, 26)]
    plt.plot(r, p, 'b', label='Model')
    plt.legend(loc='upper right')
    plt.savefig('show/pr.jpg')


"""
temporal_network
# Parameters
   score,idx : (q,db) array sorted by cosine similarity score per each query segment
   SCORE_THR : Threshold about score
   TEMP_WND : Threshold about timestamp 
        - if segments path is like [ 1.54, (Q1,DB3),(Q2,DB4) ...]
         Q2-Q1 < TEMP_WND and  DB4-DB3 < TEMP_WND
# Return    
    path : paths of TN ... [score,(Q,DB),(Q,DB)......]

"""


def temporal_network(score, idx, TOP_K=20, SCORE_THR=0.9, TEMP_WND=1, MIN_PATH=3):
    path = []
    active_path = Queue()
    n_qseg = score.shape[0]
    for q_seg in range(n_qseg):
        top_score = score[q_seg][score[q_seg] > SCORE_THR]
        if TOP_K!=0:
            top_score=top_score[:TOP_K]
        top_idx = idx[q_seg, :len(top_score)]
        active_rank = []
        # connect active path
        for pi in range(active_path.qsize()):
            p = active_path.get()
            added = False
            prev_q = p[-1][0]
            prev_db = p[-1][1]
            if q_seg <= prev_q + TEMP_WND:
                for rank, ti in enumerate(top_idx):
                    if ti > prev_db and ti <= prev_db + TEMP_WND:
                        np = p.copy()
                        np.append((q_seg, ti))
                        np[0] += top_score[rank]
                        active_path.put(np)
                        active_rank.append(rank)
                        added = True
                if not added:
                    active_path.put(p)
            else:
                path.append(p)
        [active_path.put([top_score[rank], (q_seg, ti)]) for rank, ti in enumerate(top_idx) if not rank in active_rank]
        # print(active_path.queue,path)

    path += list(active_path.queue)
    path = list(filter(lambda x: len(x) >= MIN_PATH + 1, path))
    path.sort(key=lambda x: x[0], reverse=True)

    # involve
    for n, p in enumerate(path):
        query = (p[1][0], p[-1][0])
        ref = (p[1][1], p[-1][1])
        for m, pp in enumerate(path[:n]):
            q = (pp[1][0], pp[-1][0])
            r = (pp[1][1], pp[-1][1])
            if q[0]<=query[0] and query[1]<=q[1] and r[0]<=ref[0] and ref[1]<=r[1]:
                p[0]=-1
                break
    path = list(filter(lambda x: x[0]!=-1, path))
    return path


if __name__ == '__main__':
    db = CC_WEB_VIDEO()
    dbl = db.get_VideoList(fmt=['videoid', 'queryid', 'status'])
    dbnp = np.array(dbl)
    l = np.array([v['VideoID'] for v in dbl])
    print(len(l))
    querys = db.get_Query()
    queryVids = np.array([q['VideoID'] for q in querys])
    print(l[:10])
    print(queryVids)

    print([np.where(l == q)[0][0] for q in queryVids])
    print(dbl[11458])

    feature = torch.load('/DB/CC_WEB_VIDEO/frame_1_per_sec/resnet50/v-feature.pt')
    print(feature.shape)

    fpath = '/DB/CC_WEB_VIDEO/frame_1_per_sec/resnet50/v-feature'
    l = os.listdir(fpath)
    l.sort(key=lambda x: int(x.split('.')[0]))
    vv = [torch.load(os.path.join(fpath, vf)) for vf in l]
    vv = torch.cat(vv)
    print(vv.shape)

    q = db.get_Query()
    qidx = [qi['index'] for qi in q]
    print(qidx)
    qv = vv[qidx].cuda()
    vv = vv.cuda()
    sc, idx = cosine_similarity(qv, vv)
    # print(db.get_VideoList(qid=1))

    # print(dbnp[idx[:, :100]])
    print(idx[:, :100])
    gt = db.get_reference_video_index(status='QESVML')
    evaluate_mAP(idx, gt)
