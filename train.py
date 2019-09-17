from torchvision.models import resnet18, resnet50, alexnet
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch

import numpy as np
import os

from utils.Period import Period
from dataset.vcdb3 import VCDB, FingerPrintDataset, OnlineTripletFingerprintPairDataset
from models.lossses import *
from models.nets import *
from models.utils import *
from utils.utils import *
import logging
import gc
from utils.TemporalNetwork import TN


def train(model, criterion, optim, loader, epoch, verbose=True):
    losses = AverageMeter()
    triplets = AverageMeter()
    max_iter = loader.__len__()
    model.train()

    f_log = logging.getLogger("file-log")
    print('train', end='')
    for iter, ((fp0, v0), (fp1, v1), cls) in enumerate(loader):
        optim.zero_grad()
        feature = torch.cat([fp0, fp1])
        target = torch.cat([cls, cls])
        out = model(feature.cuda())
        loss, n_triplet = criterion(out, target)
        losses.update(loss.item())
        triplets.update(n_triplet)
        loss.backward()
        optim.step()
        out_str = '[Train - epoch: {}, iter: {}/{}] loss: {}({}) triplet: {}({})' \
            .format(epoch, iter, max_iter, round(losses.val, 4), round(losses.avg, 4),
                    triplets.val, round(triplets.avg))
        f_log.info(out_str)
        if iter % 10 == 0:
            print('\r' + out_str, end='')
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
        for iter, ((fp0, v0), (fp1, v1), cls) in enumerate(loader):
            feature = torch.cat([fp0, fp1])
            target = torch.cat([cls, cls])
            out = model(feature.cuda())
            loss, n_triplet = criterion(out, target)
            losses.update(loss.item())
            triplets.update(n_triplet)

            out_str = '[Valid - epoch: {}, iter: {}/{}] loss: {}({}) triplet: {}({})' \
                .format(epoch, iter, max_iter, round(losses.val, 4), round(losses.avg, 4),
                        triplets.val, round(triplets.avg))
            f_log.info(out_str)
            if iter % 10 == 0:
                print('\r' + out_str, end='')
    print("")
    # extract frames
    # extract fingerprint
    # eval TN
    return losses, triplets


def eval(model, db):
    f_log = logging.getLogger("file-log")
    ref_video_dataset = FingerPrintDataset(db.video_list_val, db.fingerprint_root, query=False)
    query_dataset = FingerPrintDataset(db.query_list_val, db.fingerprint_root, query=True)
    SCORE_THR = 0.9
    TEMP_WND = 10
    TOP_K = 50
    MIN_PATH = 3
    MIN_MATCH = 2
    model.eval()
    query_features = []
    query_names = []
    ref_features = []
    ref_names = []
    delimiter_idx = [0]

    q_len = len(query_dataset)

    with torch.no_grad():
        for iter, (feature, v) in enumerate(query_dataset):
            out = model(feature.cuda())
            query_features.append(out.cpu())
            query_names.append(v['name'])

        for iter, (feature, v) in enumerate(ref_video_dataset):
            out = model(feature.cuda())
            ref_features.append(out.cpu())
            ref_names.append(v['name'])
            delimiter_idx.append(delimiter_idx[-1] + v['fingerprint'])

    entire_ref_features = torch.cat(ref_features)
    delimiter_idx = np.array(delimiter_idx)

    tot_prec = tot_rec = tot_f1 = tot_tp = tot_fn = tot_fp = 0
    for q_idx, (qv, q) in enumerate(zip(query_dataset.videos, query_features)):
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
                 'ref': (op[1])} for op in db.get_relative_pair(qv, phase='val') if
                op[1]['name'] in query_names and not (
                        qv['end_idx'] < op[0]['start_idx'] or qv['start_idx'] > op[0]['end_idx'])]

        for n, op in enumerate(pair):
            ref = op['ref']
            idx = ref_names.index(ref['name'])
            ref_start_idx = delimiter_idx[idx] + ref['start_idx']
            ref_end_idx = delimiter_idx[idx] + ref['end_idx']
            pair[n] = {'query': op['query'][1], 'ref': Period(ref_start_idx, ref_end_idx)}

        _, tp, _, fp, _, fn = matching(path, pair)
        prec, rec, f1 = calc_precision_recall_f1(tp, fp, fn)
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        tot_prec, tot_rec, tot_f1 = calc_precision_recall_f1(tot_tp, tot_fp, tot_fn)

        out_str = '[{}/{}] prec: {}({})   recall: {}({})   f1: {}({}) / TP: {}({}) FP: {}({}) FN: {}({}) {} {}'.format(
            q_idx, q_len, round(prec, 4), round(tot_prec, 4), round(rec, 4), round(tot_rec, 4)
            , round(f1, 4), round(tot_f1, 4), tp, tot_tp, fp, tot_fp, fn, tot_fn, qv, q.shape)
        f_log.info(out_str)
        print('\r' + out_str, end='')

    print(
        '\nSCORE THR: {}  TOPK: {} TEMP_WND: {} MIN_PATH: {} MIN_MATCH {}'.format(SCORE_THR, TOP_K, TEMP_WND, MIN_PATH,
                                                                                  MIN_MATCH))
    return tot_prec, tot_rec, tot_f1


if __name__ == '__main__':
    init_logger('simpleFC')

    epoch = 50
    n_fold = 5
    fold_idx = 0
    batch_size = 512
    triplet_margin = 0.3

    ckpt = 'ckpts/10.pt'

    db = VCDB(n_fold=n_fold, fold_idx=fold_idx)
    tr_dataset = OnlineTripletFingerprintPairDataset(db.pairs_train, db.fingerprint_root)
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    va_dataset = OnlineTripletFingerprintPairDataset(db.pairs_val, db.fingerprint_root)
    va_loader = DataLoader(va_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = SimpleFC(normalize=True)
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
        (prec,rec,f1)=ckpt['perform']


    if start_epoch == 0:
        eval(model, db)
    for ep in range(start_epoch, epoch + 1):
        losses_train, triplet_train = train(model, criterion, optimizer, tr_loader, ep)
        losses_val, triplet_val = validate(model, criterion, va_loader, ep)
        prec, rec, f1 = eval(model, db)

        torch.save({'epoch': ep,
                    'state_dict': model.module.state_dict(),
                    'train': (losses_train.avg, triplet_train.avg),
                    'valid': (losses_val.avg, triplet_val.avg),
                    'perform': (prec, rec, f1)
                    }, 'ckpts/{}.pt'.format(ep))
