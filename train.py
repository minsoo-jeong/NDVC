from torchvision.models import resnet18, resnet50, alexnet
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch

import numpy as np
import os

from utils.Period import Period
from dataset.vcdb3 import *
from models.lossses import *
from models.nets import *
from models.utils import *
from utils.utils import *


def train(model, criterion, optim, loader, epoch, verbose=True):
    losses = AverageMeter()
    triplets = AverageMeter()
    max_iter = loader.__len__()

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
        print(
            'Train [ epoch: {}, iter: {}/{}] loss: {}({}) triplet: {}({})'.format(epoch, iter, max_iter,
                                                                                  round(losses.val, 4),
                                                                                  round(losses.avg, 4),
                                                                                  triplets.val, round(triplets.avg)))


def validate(model, criterion, loader, epoch, verbose=True):
    print('val')
    # calc loss
    losses = AverageMeter()
    triplets = AverageMeter()
    max_iter = loader.__len__()
    with torch.no_grad():
        for iter, ((fp0, v0), (fp1, v1), cls) in enumerate(loader):
            feature = torch.cat([fp0, fp1])
            target = torch.cat([cls, cls])
            out = model(feature.cuda())
            loss, n_triplet = criterion(out, target)
            losses.update(loss.item())
            triplets.update(n_triplet)
            print(
                'Valid [ epoch: {}, iter: {}/{}] loss: {}({}) triplet: {}({})'.format(epoch, iter, max_iter,
                                                                                      round(losses.val, 4),
                                                                                      round(losses.avg, 4),
                                                                                      triplets.val,
                                                                                      round(triplets.avg)))

    # extract frames
    # extract fingerprint
    # eval TN


if __name__ == '__main__':

    epoch = 5
    n_fold = 2
    fold_idx = 0
    batch_size = 512
    triplet_margin = 0.3

    db = VCDB(n_fold=3, fold_idx=0)
    tr_dataset = OnlineTripletFingerprintPairDataset(db.pairs_train, db.fingerprint_root)
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    va_dataset = OnlineTripletFingerprintPairDataset(db.pairs_val, db.fingerprint_root)
    va_loader = DataLoader(va_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = SimpleFC(normalize=True)
    model.cuda()
    model = torch.nn.DataParallel(model)

    for n, p in model.named_parameters():
        print(n, p.requires_grad)

    optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9,
                          weight_decay=0.005)

    criterion = OnlineTripletLoss(margin=triplet_margin,
                                  triplet_selector=SemihardNegativeTripletSelector(margin=triplet_margin))

    start_epoch = 0
    for ep in range(start_epoch, epoch + 1):
        train(model, criterion, optimizer, tr_loader, ep)
        validate(model, criterion, va_loader, ep)