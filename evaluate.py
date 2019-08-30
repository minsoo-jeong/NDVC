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
    gt = db.get_reference_video_index(status='QESVML')

    query = db.get_Query()
    query_idx = [q['index'] for q in query]
    vfeatures = torch.load('/DB/CC_WEB_VIDEO/frame_1_per_sec/resnet50/v-features.pt')

    query_features = vfeatures[query_idx]

    score, idx, _ = cosine_similarity(query_features, vfeatures, cuda=False)

    mAP, mAP_K, pr = evaluate_mAP(idx, gt, k=[10, 50, 100], pr_curve=False)
    # draw_pr_curve(pr)
    print(mAP, mAP_K)


def evaluate_vc():
    db = VCDB()
    gt = db.get_reference_video_index()

    # vfeatures = torch.load('/DB/VCDB/frame_1_per_sec/resnet50/v-feature.pt')
    vfeatures = torch.load('/DB/VCDB/frame_1_per_sec/resnet50-r/v-features.pt')
    # vfeatures_sub = torch.load('/DB/CC_WEB_VIDEO/frame_1_per_sec/resnet50-rmac/v-features.pt')
    # ff = torch.cat([vfeatures, vfeatures_sub])

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
    p = '/DB/VCDB/frame_1_per_sec/resnet50-rmac/f-features'
    l = os.listdir(p)
    l.sort(key=lambda x: int(os.path.splitext(x)[0]))
    ff = []
    shapes = []
    start = [0]
    videoid = []
    du = []

    st = [0]
    end = []

    # /DB/VCDB/{frame_per_sec}/{resnet50-rmac}/f-features/{vid}.pt
    for o, i in enumerate(l):
        feature = torch.load(os.path.join(p, i))
        ff.append(feature)
        n = feature.shape[0]

        end.append(st[-1] + n - 1)
        st.append(st[-1] + n)

        videoid.append(os.path.splitext(i)[0])
        shapes.append(feature.shape[0])
        du.append([start[-1], start[-1] + feature.shape[0] - 1])
        start.append(start[-1] + feature.shape[0])

    ff = torch.cat(ff)

    max_idx = np.argmax(shapes)
    print(max_idx)
    print(st)
    print(end)


    p_a=0
    r_a=0
    a_d=0
    a_g=0

    db = VCDB()

    SCORE_THR = 0.90
    TEMP_WND = 1
    MIN_PATH = 3
    TOP_K=10

    for i, _ in enumerate(start[:-1]):
        q = ff[start[i]:start[i + 1], :]

        score, idx, cos = cosine_similarity(q, ff, cuda=False, numpy=True)
        paths = temporal_network(score, idx, TOP_K=TOP_K, SCORE_THR=SCORE_THR, TEMP_WND=TEMP_WND, MIN_PATH=MIN_PATH)
        # print(len(paths), paths)
        print()
        det = []
        for p in paths:
            ss = p[0]
            q_start_f_idx = p[1][0]
            q_end_f_idx = p[-1][0]
            db_start_f_idx = p[1][1]
            db_end_f_idx = p[-1][1]

            vid_idx = 0
            for n, start_idx in enumerate(st):
                if start_idx > db_start_f_idx:
                    vid_idx = n - 1
                    break
            # print(vid_idx, (st[i]+q_start_f_idx,st[i]+q_end_f_idx),(db_start_f_idx,db_end_f_idx),ss)
            det.append([(st[i] + q_start_f_idx, st[i] + q_end_f_idx), (db_start_f_idx, db_end_f_idx)])
        paths = [[p[0], (p[1][0], p[-1][0]), (p[1][1], p[-1][1])] for p in paths]
        # print(len(paths), paths)

        '''
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
            print(act_path.queue, paths)
        
        paths += list(act_path.queue)
        paths = list(filter(lambda x: len(x) >= 3, paths))
        paths.sort(key=lambda x: x[0], reverse=True)
        '''

        gt = db.get_GT(vid=i + 1)
        q_info = db.get_VideoList(vid=i + 1)[0]
        q_frame_idx = [st[q_info['index']], end[q_info['index']]]
        ground = []
        for g in gt:
            info = db.get_VideoList(g['ref_vid'])[0]
            ref_frame_idx = [st[info['index']], end[info['index']]]

            idx_cnt = ref_frame_idx[1] - ref_frame_idx[0] + 1
            d = float(info['Duration'])
            idx1 = int(int(g['ref_start']) * idx_cnt / d) + ref_frame_idx[0]
            idx2 = int(int(g['ref_end']) * idx_cnt / d) + ref_frame_idx[0]

            qidx1 = int(int(g['start']) * idx_cnt / d) + q_frame_idx[0]
            qidx2 = int(int(g['end']) * idx_cnt / d) + q_frame_idx[0]

            duration = int(float(info['Duration']))
            # print((qidx1, qidx2), (idx1, idx2), duration,ref_frame_idx, g)
            ground.append([(qidx1, qidx2), (idx1, idx2)])
        gt = [[(g['start'], g['end']), (g['ref_start'], g['ref_end'])] for g in gt]
        # print(len(gt), gt)

        #print(len(det), det)
        #print(len(ground), ground)

        dh = []
        gh = []
        for di, d in enumerate(det):
            for gi, g in enumerate(ground):
                if (not (d[0][1] < g[0][0] or d[0][0] > g[0][1])) and (not (d[1][1] < g[1][0] or d[1][0] > g[1][1])):
                    dh.append(di)
                    gh.append(gi)
        dhs = list(set(dh))
        ghs = list(set(gh))
        print(len(dh), len(dhs), dh, dhs)
        print(len(gh), len(ghs), gh, ghs)

        pp = round(len(dhs) / len(det),4)
        rr = round(len(ghs) / len(ground),4)
        print(len(dhs), len(det), len(ghs), len(ground))

        p_a += len(dhs)
        a_d += len(det)
        r_a +=len(ghs)
        a_g +=len(ground)
        prec=p_a/a_d
        rec=r_a/a_g

        print(i, pp, rr, round(prec,4), round(rec,4), round((2 * prec * rec) / (prec + rec),4))

        '''
        inter = det.intersection(gt_det)
        pr_sub = len(inter) / len(det)
        re_sub = len(inter) / len(gt_det)
        print(i, len(inter), len(det), len(gt_det), pr_sub, re_sub,
              ((2 * pr_sub * re_sub) / ((pr_sub) + re_sub)))
        
        ans += len(inter)
        prec += len(det)
        rec += len(gt_det)
        '''
    prec = p_a / a_d
    rec = r_a / a_g
    print(prec, rec, (2 * prec * rec) / (prec + rec))

    # print(paths)
    '''

    cos = (cos.numpy() * 255).astype(np.uint8)
    cos[cos < 230] = 0
    cos = cv2.cvtColor(cos, cv2.COLOR_GRAY2BGR)
    cos[:, start[:-1], 2] = 255
    print(cos.shape, cos.dtype)
    cv2.imwrite('show/cos.png', cos)
    '''
