from torchvision.models import resnet18, resnet50, alexnet
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch

import numpy as np
import os

from utils.Period import Period
from dataset.vcdb3 import VCDB_dataset, OnlineTripletFramePairDataset, VideoFrameDataset, ListDataset
from models.lossses import *
from models.nets import *
from models.utils import *
from utils.utils import *
import logging
import gc
from utils.TemporalNetwork import TN
import traceback


def get_time():
    return (datetime.now() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S")


def train(model, criterion, optim, loader, epoch, verbose=True):
    losses = AverageMeter()
    triplets = AverageMeter()
    max_iter = loader.__len__()
    model.train()

    f_log = logging.getLogger("file-log")
    print('train', end='')
    for iter, ((p0, f0), (p1, f1), cls) in enumerate(loader):
        optim.zero_grad()
        frames = torch.cat([f0, f1])
        target = torch.cat([cls, cls])
        out = model(frames.cuda())
        loss, n_triplet = criterion(out, target)
        losses.update(loss.item())
        triplets.update(n_triplet)
        loss.backward()
        optim.step()
        out_str = '[Train - epoch: {}, iter: {}/{}] loss: {}({}) triplet: {}({})' \
            .format(epoch, iter + 1, max_iter, round(losses.val, 4), round(losses.avg, 4),
                    triplets.val, round(triplets.avg))
        # f_log.info(out_str)
        if iter % 10 == 0:
            f_log.info(out_str)
            print('\r[{} {}]'.format(get_time(), os.path.basename(__file__)) + out_str, end='')
    print("")
    return losses, triplets


def validate(model, criterion, loader, epoch, verbose=True):
    f_log = logging.getLogger("file-log")
    # calc loss
    losses = AverageMeter()
    triplets = AverageMeter()
    max_iter = loader.__len__()
    model.eval()
    print('valid', end='')
    with torch.no_grad():
        for iter, ((p0, f0), (p1, f1), cls) in enumerate(loader):
            frames = torch.cat([f0, f1])
            target = torch.cat([cls, cls])
            out = model(frames.cuda())
            loss, n_triplet = criterion(out, target)
            losses.update(loss.item())
            triplets.update(n_triplet)

            out_str = '[Valid - epoch: {}, iter: {}/{}] loss: {}({}) triplet: {}({})' \
                .format(epoch, iter + 1, max_iter, round(losses.val, 4), round(losses.avg, 4),
                        triplets.val, round(triplets.avg))
            # f_log.info(out_str)
            if iter % 10 == 0:
                f_log.info(out_str)
                print('\r[{} {}]'.format(get_time(), os.path.basename(__file__)) + out_str, end='')
    print("")
    # extract frames
    # extract fingerprint
    # eval TN
    return losses, triplets


def eval(model, db, epoch):
    try:
        f_log = logging.getLogger("file-log")
        fps = 10
        ref_videos = VideoFrameDataset(db.core_video_val, fps, 1, query=False)
        query_videos = VideoFrameDataset(db.query_val, fps, 1, query=True)
        print('Number of Query : {} / Number of Reference Video : {} '.format(len(query_videos),len(ref_videos)))

        SCORE_THR = 0.9
        TEMP_WND = 1
        TOP_K = 50
        MIN_PATH = 3
        MIN_MATCH = 2
        model.eval()
        query_features = []
        query_names = []
        ref_features = []
        ref_names = []
        delimiter_idx = [0]

        all_tp = 0
        all_fp = 0
        all_fn = 0

        q_len = len(query_videos)
        model.eval()
        with torch.no_grad():
            # Extract Reference Video's segment features
            for iter, (frames, v, _) in enumerate(ref_videos):
                vf = []
                for f, p in DataLoader(ListDataset(frames), batch_size=128, shuffle=False, num_workers=4):
                    out = model(f.cuda())
                    vf.append(out)
                vf = torch.cat(vf).cpu()
                print("[{}/{}] Extract Reference Video Features ... shape: {}, video: {}"
                      .format(iter,ref_videos.__len__(), vf.shape, v))
                ref_features.append(vf)
                ref_names.append(v['name'])
                delimiter_idx.append(delimiter_idx[-1] + vf.shape[0])
            entire_ref_features = torch.cat(ref_features)
            delimiter_idx = np.array(delimiter_idx)

            # For each Query
            v_feature = None
            vid = ''
            for iter, (frames, qv, i) in enumerate(query_videos):
                if vid != qv['name']:
                    vf = []
                    for f, p in DataLoader(ListDataset(frames), batch_size=128, shuffle=False, num_workers=4):
                        out = model(f.cuda())
                        vf.append(out)
                    v_feature = torch.cat(vf).cpu()
                    vid = qv['name']
                q = torch.index_select(v_feature, 0, torch.tensor(i))

                print("[{}/{}] Matching Query with Reference Video Features ... shape: {}, video: {}"
                      .format(iter, len(query_videos), q.shape, qv))

                gt_pair = db.get_relative_pair(qv)
                gt_targets=[]
                for l in gt_pair:
                    gt=dict()
                    gt['name']=l[1]['name']
                    gt['start']=l[1]['start']*fps
                    gt['end'] = l[1]['end'] * fps
                    gt_targets.append(gt)

                score, idx, cos = cosine_similarity(q, entire_ref_features, cuda=True, numpy=True) if q.shape[0] < 200 \
                    else cosine_similarity_split(q, entire_ref_features, cuda=True, numpy=True)

                '''
                query - ref
                if cos.shape[0]>50:
                    cos_im = (cos * 255).astype(np.uint8)
                    #cos_im[cos_im < 0.7 * 255] = 0
                    cos_im[cos_im < 0.85*255] = 0
    
                    #lines = cv2.HoughLinesP(cos_im, 1, np.pi / 180, 1, MIN_PATH, 1)
                    cos_im = cv2.cvtColor(cos_im, cv2.COLOR_GRAY2BGR)
                    cos_im[:, delimiter_idx[:-1], 2] = 255
                    cv2.imwrite('show/{}.png'.format(iter),cos_im)
                    #exit()
                '''

                tn = TN(score, idx, delimiter_idx, TOP_K=TOP_K, SCORE_THR=SCORE_THR,
                        TEMP_WND=TEMP_WND, MIN_PATH=MIN_PATH, MIN_MATCH=MIN_MATCH)
                path = tn.fit()

                # align Detect result

                print('\t',len(path) ,len(gt_targets))
                TP = []
                FP = []
                TP_gt = []
                for p in path:
                    query_idx = p['query']
                    ref_idx = p['ref']
                    delimiter = delimiter_idx[p['ref_vid_idx']]
                    p.update({'query': Period(query_idx.start + i[0], query_idx.end + i[0]),
                              'ref': Period(ref_idx.start - delimiter, ref_idx.end - delimiter),
                              'ref_vid': ref_names[p['ref_vid_idx']]})

                    isHit, matched = matching_with_gt(p, gt_targets)
                    if isHit is True:
                        TP.append(p)
                        TP_gt.append(matched)
                    else:
                        FP.append(p)
                FN = gt_targets
                # print(path)
                print('\t TP :',len(TP), TP)
                print('\t TP-gt: ',TP_gt)
                print('\t FP: ',len(FP), FP)
                print('\t FN: ',len(FN), FN)
                all_tp += len(TP)
                all_fp += len(FP)
                all_fn += len(FN)

                prec, rec, f1 = calc_precision_recall_f1(len(TP), len(FP), len(FN))
                tprec, trec, tf1 = calc_precision_recall_f1(all_tp, all_fp, all_fn)
                out_str = '[Eval - epoch: {}, iter: {}/{}] prec: {}({})   recall: {}({})   f1: {}({}) / TP: {}({}) FP: {}({}) FN: {}({}) {} {}'.format(
                    int(epoch), iter + 1, q_len, round(prec, 4), round(tprec, 4), round(rec, 4), round(trec, 4)
                    , round(f1, 4), round(tf1, 4), len(TP), all_tp, len(FP), all_fp, len(FN), all_fn, qv, q.shape)
                print(out_str)
    except Exception as e:
        traceback.print_exc()

    print(
        '\nSCORE THR: {}  TOPK: {} TEMP_WND: {} MIN_PATH: {} MIN_MATCH {}'.format(SCORE_THR, TOP_K, TEMP_WND, MIN_PATH,
                                                                                  MIN_MATCH))
    exit()
    return tprec, trec, tf1


if __name__ == '__main__':
    desc = 'RMAC'
    init_logger(desc)

    epoch = 50
    n_fold = 5
    fold_idx = 0
    batch_size = 64
    triplet_margin = 0.3

    ckpt = None
    date = (datetime.now() + timedelta(hours=9)).strftime("%Y%m%d")
    if not os.path.exists('ckpts/{}'.format(date)):
        os.makedirs('ckpts/{}'.format(date))

    db = VCDB_dataset(n_folds=n_fold, fold_idx=fold_idx)
    tr_dataset = OnlineTripletFramePairDataset(db.core_video_train, db.pairs_train, 10, 1)
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    va_dataset = OnlineTripletFramePairDataset(db.core_video_val, db.pairs_val, 10, 1)
    va_loader = DataLoader(va_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = Resnet50_RMAC()
    model.cuda()
    model = torch.nn.DataParallel(model)

    # for n, p in model.named_parameters():
    #    print(n, p.requires_grad)

    optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9,
                          weight_decay=0.005)

    criterion = OnlineTripletLoss(margin=triplet_margin,
                                  triplet_selector=SemihardNegativeTripletSelector(margin=triplet_margin))
    start_epoch = 0
    if ckpt is not None:
        ckpt = torch.load(ckpt)
        model.module.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch'] + 1
        (prec, rec, f1) = ckpt['perform']

    if start_epoch == 0:
        eval(model, db, False)
    for ep in range(start_epoch, epoch + 1):
        losses_train, triplet_train = train(model, criterion, optimizer, tr_loader, ep)
        losses_val, triplet_val = validate(model, criterion, va_loader, ep)
        # prec, rec, f1 = eval(model, db,ep)

        torch.save({'epoch': ep,
                    'state_dict': model.module.state_dict(),
                    'train': (losses_train.avg, triplet_train.avg),
                    'valid': (losses_val.avg, triplet_val.avg),
                    'perform': (prec, rec, f1)
                    }, 'ckpts/{}/{}-ep{}.pt'.format(date, desc, ep))
