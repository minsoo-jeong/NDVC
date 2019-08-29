import numpy as np
import torch
import os
from torch.nn import functional as F
from datetime import datetime
from dataset.cc_web_video import CC_WEB_VIDEO
from dataset.vcdb import VCDB
from utils.utils import *
from PIL import Image
import cv2
from queue import Queue


def evaluate_cc_web():
    db = CC_WEB_VIDEO()
    gt = db.get_GT(status='QESVML')

    query = db.get_Query()
    query_idx = [q['index'] for q in query]
    vfeatures = torch.load('/DB/CC_WEB_VIDEO/frame_1_per_sec/resnet50/v-feature.pt')

    query_features = vfeatures[query_idx]

    score, idx, _ = cosine_similarity(query_features, vfeatures, cuda=False)

    mAP, mAP_K, pr = evaluate_mAP(idx, gt, k=[10, 50, 100], pr_curve=False)
    # draw_pr_curve(pr)
    print(mAP, mAP_K)


def evaluate_vc():
    db = VCDB()
    gt = db.get_GT()

    # vfeatures = torch.load('/DB/VCDB/frame_1_per_sec/resnet50/v-feature.pt')
    vfeatures = torch.load('/DB/VCDB/frame_1_per_sec/resnet50/v-feature.pt')
    vfeatures_sub = torch.load('/DB/CC_WEB_VIDEO/frame_1_per_sec/resnet50/v-feature.pt')
    ff = torch.cat([vfeatures, vfeatures_sub])

    score, idx, _ = cosine_similarity(vfeatures, vfeatures, cuda=False)

    mAP, mAP_K, pr = evaluate_mAP(idx, gt, k=[10, 50, 100], pr_curve=False)
    # draw_pr_curve(pr)
    print(mAP, mAP_K, )


def index_to_file(start, idx):
    out = -1
    for i, st in enumerate(start):
        if idx >= st and idx < start[i + 1]:
            out = i
            break;
    return out


if __name__ == '__main__':
    # evaluate_vc()
    # exit()
    p = '/DB/VCDB/frame_1_per_sec/resnet50-rmac/f-feature'
    l = os.listdir(p)
    l.sort(key=lambda x: int(os.path.splitext(x)[0]))
    ff = []
    shapes = []
    start = [0]
    videoid = []
    du = []
    for i in l:
        feature = torch.load(os.path.join(p, i))
        ff.append(feature)
        videoid.append(os.path.splitext(i)[0])
        shapes.append(feature.shape[0])
        du.append([start[-1], start[-1] + feature.shape[0] - 1])
        start.append(start[-1] + feature.shape[0])

    ff = torch.cat(ff)

    max_idx = np.argmax(shapes)
    print(max_idx)

    ans = 0
    prec = 0
    rec = 0
    db = VCDB()

    SCORE_THR = 0.95
    TEMP_WND = 1

    for i, _ in enumerate(start[:-1]):
        q = ff[start[i]:start[i + 1], :]
        score, idx, cos = cosine_similarity(q, ff, cuda=False, numpy=True)
        paths = []  # path: [weight,(q_seq,idx),(q_seq,idx)....]
        act_path = Queue()
        for q_seq in range(q.shape[0]):
            top_score = score[q_seq][score[q_seq] > SCORE_THR]
            top_idx = idx[q_seq, :len(top_score)]
            append_idx = []
            for ii in range(act_path.qsize()):
                p = act_path.get()
                added = False
                prev_q_idx = p[-1][0]
                prev_db_idx = p[-1][1]
                if q_seq <= prev_q_idx + TEMP_WND:
                    for rank, t_idx in enumerate(top_idx):
                        if t_idx > prev_db_idx and t_idx <= prev_db_idx + TEMP_WND:
                            np = p.copy()
                            np.append((q_seq, t_idx))
                            np[0] += top_score[rank]
                            act_path.put(np)
                            append_idx.append(rank)
                            added = True
                    if not added:
                        act_path.put(p)

                else:
                    paths.append(p)

            for rank, t_idx in enumerate(top_idx):
                if not rank in append_idx:
                    act_path.put([score[q_seq][rank], (q_seq, t_idx)])

        paths += list(act_path.queue)
        paths = list(filter(lambda x: len(x) >= 6, paths))
        paths.sort(key=lambda x: x[0], reverse=True)

        # print(len(paths), paths)
        det = set()
        for p in paths:
            for d in p[1:]:
                det.add(d)
        det = list(det)
        det.sort(key=lambda x: x[0])
        if i == 1:
            for p in paths:
                print((p[1][0], p[-1][0]), (p[1][1], p[-1][1]), p[0])
            print(det)
            print(paths)
            exit()
        det = set(det)
        # print([(p[0], p[1], p[-1]) for p in paths])
        gt = db.get_GT_time(vid=i + 1)
        gt_det = set()
        for g in gt:
            qid = g[0]
            rid = g[3]
            d = du[rid - 1]
            if (g[2] - g[1] >= 1):
                for t in range(g[2] - g[1]):
                    gt_det.add((g[1] + t, d[0] + g[4] + t))

        inter = det.intersection(gt_det)
        pr_sub = len(inter) / len(det)
        re_sub = len(inter) / len(gt_det)
        print(i, len(inter), len(det), len(gt_det), pr_sub, re_sub,
              ((2 * pr_sub * re_sub) / ((pr_sub) + re_sub)))
        ans += len(inter)
        prec += len(det)
        rec += len(gt_det)

    prec = ans / prec
    rec = ans / rec
    print(prec, rec, (2 * prec * rec) / (prec + rec))

    # print(paths)

    cos = (cos.numpy() * 255).astype(np.uint8)
    cos[cos < 230] = 0
    cos = cv2.cvtColor(cos, cv2.COLOR_GRAY2BGR)
    cos[:, start[:-1], 2] = 255
    print(cos.shape, cos.dtype)
    cv2.imwrite('show/cos.png', cos)
