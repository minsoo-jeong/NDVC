import numpy as np
import torch
import os
from torch.nn import functional as F
from datetime import datetime
from dataset.cc_web_video import CC_WEB_VIDEO


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
def cosine_similarity(query, features, cuda=True):
    if cuda:
        query = query.cuda()
        features = features.cuda()
    query = F.normalize(query, 2, 1)
    features = F.normalize(features, 2, 1)

    cos = torch.mm(features, query.t()).t()
    score, idx = torch.sort(cos, descending=True)
    score = score.cpu()
    idx = idx.cpu()
    cos=cos.cpu()
    return score, idx, cos


def evaluate_mAP(indice, gt, k=[1, 500, 1000]):
    aps = []
    ap_ks = []
    for i in range(len(gt)):
        ap, ap_k = evaluate_AP(indice[i], gt[i], k)
        print(ap, ap_k)
        aps.append(ap)
        ap_ks.append(ap_k)

    aps = np.array(aps)
    ap_ks = np.array(ap_ks)

    mAP = aps.mean(axis=0)
    mAP_K = ap_ks.mean(axis=0)
    return mAP, mAP_K


def evaluate_AP(indice, gt, k=[1, 500, 1000]):
    indice = np.array(indice)
    gt = np.array(gt)
    # print(indice,len(gt))
    c = 0
    ap = 0.0
    ap_k = []
    for rank, idx in enumerate(indice, 1):
        # print(rank,c,len(gt))
        if c >= len(gt): break
        if idx in gt:
            c += 1
            ap += (c / rank)
        if rank in k:
            ap_k.append(ap / c)
    ap /= c
    while len(ap_k) != len(k):  ap_k.append(ap)
    return ap, ap_k


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
    gt = db.get_GT(status='QESVML')
    evaluate_mAP(idx, gt)
