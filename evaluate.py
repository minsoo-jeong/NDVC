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


def evaluate_cc_web():
    db = CC_WEB_VIDEO()
    gt = db.get_GT(status='QESVML')

    query = db.get_Query()
    query_idx = [q['index'] for q in query]
    vfeatures = torch.load('/DB/CC_WEB_VIDEO/frame_1_per_sec/resnet50/v-feature.pt')

    query_features = vfeatures[query_idx]

    score, idx, _ = cosine_similarity(query_features, vfeatures)

    mAP, mAP_K = evaluate_mAP(idx, gt, k=[1, 10, 100])

    print(mAP, mAP_K)


def evaluate_vc():
    db = VCDB()
    gt = db.get_GT()

    vfeatures = torch.load('/DB/VCDB/frame_1_per_sec/resnet50/v-feature.pt')
    vfeatures_sub = torch.load('/DB/CC_WEB_VIDEO/frame_1_per_sec/resnet50/v-feature.pt')
    ff = torch.cat([vfeatures, vfeatures_sub])

    score, idx, _ = cosine_similarity(vfeatures, ff, cuda=True)

    mAP, mAP_K = evaluate_mAP(idx, gt, k=[10, 30, 50])
    print(mAP, mAP_K)


if __name__ == '__main__':
    evaluate_vc()
    exit()
    p = '/DB/VCDB/frame_1_per_sec/resnet50/f-feature'
    l = os.listdir(p)
    l.sort(key=lambda x: int(os.path.splitext(x)[0]))
    ff = []
    length = []
    ii = []
    prev = 0
    for i in l:
        feature = torch.load(os.path.join(p, i))
        ff.append(feature)
        prev += feature.shape[0]
        length.append(feature.shape[0])
        ii.append(prev)

    ff = torch.cat(ff)
    print(ff.shape)

    print(length)
    print(ii)

    max_idx = np.argmax(length)
    print(max_idx)

    q = ff[ii[2]:ii[3], :]

    vfeatures = torch.load('/DB/VCDB/frame_1_per_sec/resnet50/v-feature.pt')

    score, idx, cos = cosine_similarity(q, ff, cuda=True)

    # idx to rank

    print(score)
    print(idx)
    print(cos)

    cos = (cos.numpy() * 255).astype(np.uint8)
    cos[cos < 150] = 0
    cos = cv2.cvtColor(cos, cv2.COLOR_GRAY2BGR)
    cos[:, ii[:-1], 2] = 255
    print(cos.shape, cos.dtype)
    # cv2.imwrite('show/cos.png', cos)
