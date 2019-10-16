from dataset.vcdb import VCDB
from utils.utils import *
from utils.Period import Period
from utils.TemporalNetwork import TN
from dataset.vcdb3 import VCDB

import gc


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


def evaluate(db, SCORE_THR, TOP_K, TEMP_WND, MIN_PATH, MIN_MATCH, phase):
    query_videos = db.query_list_val
    query_names = list(set([v['name'] for v in query_videos]))

    query_features = []
    for n, qv in enumerate(query_videos):
        f = db.get_fingerprint(qv, split=True)
        query_features.append(f)

    ref_videos = db.video_list_val
    ref_features = []
    ref_v = []
    delimiter_idx = [0]

    tot_tp = 0
    tot_fn = 0
    tot_fp = 0

    for rv in ref_videos:
        # if rv['name'] in reference_videos:
        f = db.get_fingerprint(rv, split=False)
        ref_features.append(f)
        ref_v.append(rv['name'])
        delimiter_idx.append(delimiter_idx[-1] + rv['fingerprint'])
    entire_ref_features = torch.cat(ref_features)
    delimiter_idx = np.array(delimiter_idx)

    print('',end='')
    for q_idx, (qv, q) in enumerate(zip(query_videos, query_features)):
        if q_idx % 100 == 0:
            gc.collect()

        if q.shape[0] < 200:
            score, idx, cos = cosine_similarity(q, entire_ref_features, cuda=True, numpy=True)
        else:
            score, idx, cos = cosine_similarity_split(q, entire_ref_features, cuda=True, numpy=True)

        tn = TN(score, idx, delimiter_idx, TOP_K=TOP_K, SCORE_THR=SCORE_THR,
                TEMP_WND=TEMP_WND, MIN_PATH=MIN_PATH, MIN_MATCH=MIN_MATCH)
        path = tn.fit()

        for p in path:
            p['query'].start += qv['start_idx']
            p['query'].end += qv['start_idx']
        pair = [{'query': (op[0]['name'], Period(op[0]['start_idx'], op[0]['end_idx'])),
                 'ref': (op[1])}
                for op in db.get_relative_pair2(qv) if
                op[1]['name'] in query_names and not (
                        qv['end_idx'] < op[0]['start_idx'] or qv['start_idx'] > op[0]['end_idx'])]

        for n, op in enumerate(pair):
            ref = op['ref']
            idx = ref_v.index(ref['name'])
            ref_start_idx = delimiter_idx[idx] + ref['start_idx']
            ref_end_idx = delimiter_idx[idx] + ref['end_idx']
            pair[n] = {'query': op['query'][1], 'ref': Period(ref_start_idx, ref_end_idx)}


        _, tp, _, fp, _, fn = matching(path, pair)
        prec, rec, f1 = calc_precision_recall_f1(tp, fp, fn)
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        tot_prec, tot_rec, tot_f1 = calc_precision_recall_f1(tot_tp, tot_fp, tot_fn)

        print('\r{}: prec: {}({})   recall: {}({})   f1: {}({}) / TP: {}({}) FP: {}({}) FN: {}({}) {} {}'.format(
            q_idx, round(prec, 4), round(tot_prec, 4), round(rec, 4), round(tot_rec, 4)
            , round(f1, 4), round(tot_f1, 4), tp, tot_tp, fp, tot_fp, fn, tot_fn, qv, q.shape),end='')
    print('SCORE THR: {}  TOPK: {} TEMP_WND: {} MIN_PATH: {} MIN_MATCH {}'.format(SCORE_THR, TOP_K, TEMP_WND, MIN_PATH,
                                                                                  MIN_MATCH))
if __name__ == '__main__':
    db = VCDB(n_fold=5, fold_idx=1)

    print(len(db.query_list))

    SCORE_THR = 0.9
    TEMP_WND = 10
    TOP_K = 50
    MIN_PATH = 3
    MIN_MATCH = 2
    evaluate(db, SCORE_THR, TOP_K, TEMP_WND, MIN_PATH, MIN_MATCH, 'aa')
    exit()
    query_videos = db.query_list
    # query_videos = db.query_list_val
    query_names = list(set([v['name'] for v in query_videos]))

    query_features = []
    query_features_origin_length = []
    for qv in query_videos:
        f, l = db.get_feature(qv, entire=False)
        query_features.append(f)
        query_features_origin_length.append(l)

    ref_videos = db.video_list
    # ref_videos = db.video_list_val
    ref_features = []
    ref_v = []
    delimiter_idx = [0]
    for rv in ref_videos:
        # if rv['name'] in reference_videos:
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
    probs = []
    labels = []
    for q_idx, (qv, q, ql) in enumerate(zip(query_videos, query_features, query_features_origin_length)):
        if q_idx % 100 == 0:
            gc.collect()
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
        del tn, score, idx, cos
        for p in path:
            p['query'].start += q_start_idx
            p['query'].end += q_start_idx

        # print('Query- ', q_idx, qv, Period(q_start_idx, q_end_idx))
        # print('DETECT- ', path)
        '''
        pair = [{'query': (op[0]['name'], Period(time2idx(op[0]['start'], q_intv), time2idx(op[0]['end'], q_intv))),
                 'ref': (op[1])}
                #for op in db.get_relative_pair(qv) if
                for op in db.get_relative_pair(qv,phase='all') if
                op[1]['name'] in query_names and not (qv['end'] < op[0]['start'] or qv['start'] > op[0]['end'])]
        '''
        pair = [{'query': (op[0]['name'], Period(time2idx(op[0]['start'], q_intv), time2idx(op[0]['end'], q_intv))),
                 'ref': (op[1])} for op in db.get_gt(qv)]

        for n, op in enumerate(pair):
            ref = op['ref']
            idx = ref_v.index(ref['name'])
            ref_feature = ref_features[idx]
            ref_intv = ref_feature.shape[0] / ref['duration']
            ref_start_idx = delimiter_idx[idx] + time2idx(ref['start'], ref_intv)
            ref_end_idx = delimiter_idx[idx] + time2idx(ref['end'], ref_intv)
            pair[n] = {'query': op['query'][1], 'ref': Period(ref_start_idx, ref_end_idx)}
        # print('GT- ', pair)

        _, tp, _, fp, _, fn = matching(path, pair)
        # print('TP: ', tp)
        # print('FP: ', fp)
        # print('FN: ', fn)
        labels += ([1] * len(tp))
        for t in tp:
            probs.append(t[1]['score'])
        labels += ([0] * len(fp))
        for t in fp:
            probs.append(t['score'])
        probs += ([0] * len(fn))
        labels += ([1] * len(fn))
        precision, recall, _ = precision_recall_curve(labels, probs)
        max_f1 = np.max(2 * (precision * recall) / (precision + recall + 10e-8))

        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = (2 * prec * rec) / (prec + rec + 1e-6)
        tps += tp
        fps += fp
        fns += fn
        tot_prec = tps / (tps + fps + 1e-6)
        tot_rec = tps / (tps + fns + 1e-6)
        tot_f1 = (2 * tot_prec * tot_rec) / (tot_prec + tot_rec + 1e-6)
        print('{}: {} prec: {}({})   recall: {}({})   f1: {}({}) / TP: {}({}) FP: {}({}) FN: {}({}) {} {}'.format(
            q_idx, round(max_f1, 4), round(prec, 4), round(tot_prec, 4), round(rec, 4), round(tot_rec, 4)
            , round(f1, 4), round(tot_f1, 4), tp, tps, fp, fps, fn, fns, qv, q.shape))
        # print('==================\n')

    print('SCORE THR: {}  TOPK: {} TEMP_WND: {} MIN_PATH: {} MIN_MATCH {}'.format(SCORE_THR, TOP_K, TEMP_WND, MIN_PATH,
                                                                                  MIN_MATCH))
    print(precision)
    print(recall)
