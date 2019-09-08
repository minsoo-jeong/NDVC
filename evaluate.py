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
from datetime import datetime
from utils.Period import Period
from utils.TemporalNetwork import TN
from utils.utils import int_round
from dataset.n_vcdb import VCDB
from math import ceil, floor


def prev_evaluate_cc_web():
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


def prev_evaluate_vc():
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


def prev_index_to_file(start, idx):
    out = -1
    for i, st in enumerate(start):
        if idx >= st and idx < start[i + 1]:
            out = i
            break;
    return out


def time2idx(time, intv_per_sec):
    return int(time * intv_per_sec)


if __name__ == '__main__':
    db = VCDB()
    print(db.frame_cnt)

    SCORE_THR = 0.92
    TEMP_WND = 10
    TOP_K = 100
    MIN_PATH = 3
    MIN_MATCH = 2

    query_videos = db.query_list
    query_names = list(set([v['name'] for v in query_videos]))
    query_features = []
    query_features_origin_length = []
    for qv in query_videos:
        f, l = db.get_feature(qv, entire=False)
        query_features.append(f)
        query_features_origin_length.append(l)

    ref_videos = db.video_list
    ref_features = []
    ref_v = []
    delimiter_idx = [0]
    for rv in ref_videos:
        if rv['name'] in query_names:
            f, _ = db.get_feature(rv, True)
            ref_features.append(f)
            ref_v.append(rv['name'])
            delimiter_idx.append(delimiter_idx[-1] + f.shape[0])

    print(len(query_features), len(ref_features))
    print(delimiter_idx)
    entire_ref_features = torch.cat(ref_features)
    delimiter_idx = np.array(delimiter_idx)

    tps = 0
    fps = 0
    fns = 0
    for q_idx, (qv, q, ql) in enumerate(zip(query_videos, query_features, query_features_origin_length)):
        q_intv = ql / qv['duration']
        q_start_idx = time2idx(qv['start'], q_intv)
        q_end_idx = time2idx(qv['end'], q_intv)
        # print(q_idx, qv, q.shape, q_start_idx, q_end_idx)

        # compare with each videos
        '''
        for r_idx, (rv, r) in enumerate(zip(ref_videos, ref_features)):
            r_intv = r.shape[0] / rv['duration']

            pair = [[(p[0]['name'], Period(time2idx(p[0]['start'], q_intv), time2idx(p[0]['end'], q_intv))),
                     (p[1]['name'], Period(time2idx(p[1]['start'], r_intv), time2idx(p[1]['end'], r_intv)))]
                    for p in db.get_relative_pair2(qv, rv)]

            pair = [(p[0][1], p[1][1]) for p in pair if not (q_end_idx < p[0][1].start or p[0][1].end < q_start_idx)]

            score, idx, cos = cosine_similarity(q, r, cuda=True, numpy=True)
            tn = TN(score, idx, [0, r.shape[0]], TOP_K=TOP_K, SCORE_THR=SCORE_THR,
                    TEMP_WND=TEMP_WND, MIN_PATH=MIN_PATH, MIN_MATCH=MIN_MATCH)
            path = tn.fit()
            for p in path:
                # print(path)
                p[0].start += q_start_idx
                p[0].end += q_start_idx

            if len(path) or len(pair):
                #print('Query- ', qv, Period(q_start_idx, q_end_idx))
                #print('Ref- ', rv)
                #print('DETECT')
                #print(path)
                #print('GT')
                #print(pair)

                # calc TP, FP, FN
                tp, fp, fn = matching(path, pair)
                print('TP: ', tp)
                print('FP: ', fp)
                print('FN: ', fn)
                ntp = len(tp)
                nfp = len(fp)
                nfn = len(fn)
                prec = ntp / (ntp + nfp + 1e-6)
                rec = ntp / (ntp + nfn + 1e-6)
                f1 = (2 * prec * rec) / (prec + rec + 1e-6)
                tps += ntp
                fps += nfp
                fns += nfn
                tot_prec = tps / (tps + fps + 1e-6)
                tot_rec = tps / (tps + fns + 1e-6)
                tot_f1 = (2 * tot_prec * tot_rec) / (tot_prec + tot_rec + 1e-6)
                print('{}/{}: prec: {}({})   recall: {}({})   f1: {}({})'.format(q_idx, r_idx
                                                                                 , round(prec,4), round(tot_prec,4)
                                                                                 , round(rec,4), round(tot_rec,4)
                                                                                 , round(f1,4), round(tot_f1,4)))
        '''

        # compare with entire videos
        if q.shape[0] < 200:
            score, idx, cos = cosine_similarity(q, entire_ref_features, cuda=True, numpy=True)
        else:
            score, idx, cos = cosine_similarity_split(q, entire_ref_features, cuda=True, numpy=True)

        tn = TN(score, idx, delimiter_idx, TOP_K=TOP_K, SCORE_THR=SCORE_THR,
                TEMP_WND=TEMP_WND, MIN_PATH=MIN_PATH, MIN_MATCH=MIN_MATCH)
        path = tn.fit()
        for p in path:
            p[0].start += q_start_idx
            p[0].end += q_start_idx
        print('Query- ', q_idx, qv, Period(q_start_idx, q_end_idx))
        print('DETECT- ', path)

        pair = [((p[0]['name'], Period(time2idx(p[0]['start'], q_intv), time2idx(p[0]['end'], q_intv))), (p[1]))
                for p in db.get_relative_pair(qv) if not (qv['end'] < p[0]['start'] or qv['start'] > p[0]['end'])]
        for n, p in enumerate(pair):
            ref = p[1]
            idx = ref_v.index(ref['name'])
            ref_feature = ref_features[idx]
            ref_intv = ref_feature.shape[0] / ref['duration']
            ref_start_idx = delimiter_idx[idx] + time2idx(ref['start'], ref_intv)
            ref_end_idx = delimiter_idx[idx] + time2idx(ref['end'], ref_intv)
            pair[n] = (p[0][1], Period(ref_start_idx, ref_end_idx))
        print('GT- ', pair)
        tp, fp, fn = matching(path, pair)
        print('TP: ', tp)
        print('FP: ', fp)
        print('FN: ', fn)
        ntp = len(tp)
        nfp = len(fp)
        nfn = len(fn)
        prec = ntp / (ntp + nfp + 1e-6)
        rec = ntp / (ntp + nfn + 1e-6)
        f1 = (2 * prec * rec) / (prec + rec + 1e-6)
        tps += ntp
        fps += nfp
        fns += nfn
        tot_prec = tps / (tps + fps + 1e-6)
        tot_rec = tps / (tps + fns + 1e-6)
        tot_f1 = (2 * tot_prec * tot_rec) / (tot_prec + tot_rec + 1e-6)
        print('{} : prec: {}({})   recall: {}({})   f1: {}({})'.format(q_idx, round(prec, 4), round(tot_prec, 4)
                                                                       , round(rec, 4), round(tot_rec, 4)
                                                                       , round(f1, 4), round(tot_f1, 4)))
        print('==================\n')
    exit()
    db = VCDB()
    SCORE_THR = 0.9
    TEMP_WND = 10
    MIN_PATH = 2
    TOP_K = 50

    n_hit = 0
    n_det = 0
    n_ground = 0

    feature_path = '/DB/VCDB/frame_1_per_sec/resnet50-rmac/f-features'
    features = []
    delimiter_idx = [0]
    for vid in db.get_VideoList():
        videoid = vid['VideoID']
        f = torch.load(os.path.join(feature_path, '{}.pt'.format(videoid)))
        features.append(f)
        delimiter_idx.append(delimiter_idx[-1] + f.shape[0])
    features = torch.cat(features)
    delimiter_idx = np.array(delimiter_idx)
    print(features.shape)
    print(delimiter_idx)

    querys = db.get_GT_list()
    query_feature = []
    for q in querys:
        q_vid = q['vid']
        start = q['start']
        end = q['end']
        q_v_idx = q_vid - 1
        q_feature = features[delimiter_idx[q_v_idx] + start:delimiter_idx[q_v_idx] + end, :]
        print(q_feature.shape, start, end, delimiter_idx[q_v_idx], delimiter_idx[q_v_idx] + end)

    print(querys)
    exit()

    for qq in range(len(delimiter_idx) - 1):
        q_v_idx = 3

        q = features[delimiter_idx[q_v_idx]:delimiter_idx[q_v_idx + 1], :]
        gt = db.get_GT(vid=q_v_idx + 1)
        q = q[3:6]
        if q.shape[0] > 200:
            score, idx, cos = cosine_similarity_split(q, features, cuda=True, numpy=True)
        else:
            score, idx, cos = cosine_similarity(q, features, cuda=True, numpy=True)

        tn = TN(score, idx, delimiter_idx, TOP_K=TOP_K, SCORE_THR=SCORE_THR, TEMP_WND=TEMP_WND,
                MIN_PATH=MIN_PATH)
        det = tn.fit()
        print([[(v['vid'], v['start'], v['end']), (
            v['ref_vid'], delimiter_idx[v['ref_vid'] - 1] + v['ref_start'],
            delimiter_idx[v['ref_vid'] - 1] + v['ref_end'])]
               for v in gt])
        print(det)
        exit()
        gt = db.get_GT(vid=q_v_idx + 1)
        q_d = db.get_VideoList(vid=gt[0]['vid'])[0]['Duration']
        q_cnt = q.shape[0]
        ground = []
        for g in gt:
            r = db.get_VideoList(vid=g['ref_vid'])[0]

            st = delimiter_idx[r['index']]
            r_d = r['Duration']
            cnt = delimiter_idx[r['index'] + 1] - delimiter_idx[r['index']]
            r_start_idx = int((g['ref_start'] / r_d) * cnt) + delimiter_idx[r['index']]
            r_end_idx = int((g['ref_end'] / r_d) * cnt) + delimiter_idx[r['index']]

            q_start_idx = int((g['start'] / q_d) * q_cnt)
            q_end_idx = int((g['end'] / q_d) * q_cnt)
            ground.append([Period(q_start_idx, q_end_idx), Period(r_start_idx, r_end_idx)])
        # print(det)
        # print(ground)

        # cos_im = cos_to_cv(cos,delimiter_idx,SCORE_THR,MIN_PATH)
        # cv2.imwrite('show/aa.png',cos_im)

        hit = matching_gt(det, ground) if len(det) else 0

        n_hit += hit
        n_det += len(det)
        n_ground += len(ground)

        prec = hit / (len(det) + 1e-6)
        recall = hit / len(ground)
        fscore = (2 * prec * recall) / (prec + recall + 1e-6)

        a_prec = n_hit / (n_det + 1e-6)
        a_recall = n_hit / n_ground
        a_fscore = (2 * a_prec * a_recall) / (a_prec + a_recall + 1e-6)

        out = '{} prec: {}({})'.format(qq, round(prec, 4), round(a_prec, 4))
        out += ' recall: {}({})'.format(round(recall, 4), round(a_recall, 4))
        out += ' fscore: {}({})'.format(round(fscore, 4), round(a_fscore, 4))
        out += ' - hit: {} det: {} gt: {}'.format(hit, len(det), len(ground))
        out += ' - len: {} wnd: {}'.format(q.shape[0], TEMP_WND)

        print(out)
