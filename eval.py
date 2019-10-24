from utils.utils import cosine_similarity_auto, match, MeasureMeter
from utils.TemporalNetwork import TN
from utils.Period import Period

from dataset.vcdb3 import VCDB_dataset
from models.nets import DummyEmb

from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms as trn
from torch.utils.data import DataLoader, Dataset
import torch

import numpy as np
import argparse
import os


class FrameDataset(Dataset):
    def __init__(self, videos, fps=1, isQuery=False, root="/DB/videoCopyDetection"):
        self.root = root
        self.videos = videos
        self.fps = fps
        self.isQuery = isQuery

    def __getitem__(self, idx):
        """
        :return: list, sampled frame paths
        """
        v = self.videos[idx]
        title = v['title']
        video_fps = v['fps']
        interval = video_fps / self.fps if video_fps >= self.fps else 1
        name = os.path.splitext(v['name'])[0]
        frame_root = os.path.join(self.root, 'background_dataset', 'frames', title, name) if title.find('bg_') != -1 \
            else os.path.join(self.root, 'core_dataset', 'frames', title, name)

        frames = sorted(os.listdir(frame_root))
        start = 0
        end = len(frames) - 1
        if self.isQuery:
            start = max(0, round(v['start'] * video_fps))
            end = min(end, round(v['end'] * video_fps))

        frame_indices = [int(np.round(i)) for i in np.arange(start + (interval / 2), end, interval)]
        l = [os.path.join(frame_root, frames[i]) for i in frame_indices]

        return l, v

    def __len__(self):
        return len(self.videos)


class ListDataset(Dataset):
    def __init__(self, l):
        self.l = l
        self.loader = default_loader
        self.default_trn = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        path = self.l[idx]
        frame = self.default_trn(self.loader(path))

        return path, frame

    def __len__(self):
        return len(self.l)


def eval(model, ref_videos, query, relative_pair, fps, batch_sz, num_workers):
    measure = MeasureMeter()
    SCORE_THR = 0.95
    TEMP_WND = 1
    TOP_K = 0
    MIN_PATH = 3
    MIN_MATCH = 2
    model.eval()
    ref_videos_fingerprint = []
    ref_videos_names = []
    ref_videos_delimter = [0]
    with torch.no_grad():
        listLoader = DataLoader(ListDataset([]), batch_size=batch_sz, shuffle=False, num_workers=num_workers)
        print("Extract Reference Video Fingerprint")
        for video_idx, (frame_list, video) in enumerate(ref_videos):
            listLoader.dataset.l = frame_list
            video_fingerprint = []
            for chunk_idx, (paths, frames) in enumerate(listLoader):
                out = model(frames.cuda())
                video_fingerprint.append(out)
            video_fingerprint = torch.cat(video_fingerprint).cpu()
            print("[{}/{}] Extract Reference Video FingerPrint .... shape: {}, video: {}"
                  .format(video_idx, len(ref_videos), video_fingerprint.shape, video))
            ref_videos_fingerprint.append(video_fingerprint)
            ref_videos_names.append(video['name'])
            ref_videos_delimter.append(ref_videos_delimter[-1] + video_fingerprint.shape[0])

        ref_videos_fingerprint = torch.cat(ref_videos_fingerprint)
        ref_videos_delimter = np.array(ref_videos_delimter)
        print("Total Reference Video: {} FingerPrint shape: {}".format(len(ref_videos), ref_videos_fingerprint.shape))

        last_query_video = ''
        for query_idx, (frame_list, query_video) in enumerate(query):
            listLoader.dataset.l = frame_list
            query_fingerprint = []
            for chunk_idx, (paths, frames) in enumerate(listLoader):
                out = model(frames.cuda())
                query_fingerprint.append(out)
            query_fingerprint = torch.cat(query_fingerprint).cpu()
            print("[{}/{}] Matching Query with Reference Video ... shape: {}, video: {}"
                  .format(query_idx, len(query), query_fingerprint.shape, query_video))

            score, idx, cos = cosine_similarity_auto(query_fingerprint, ref_videos_fingerprint, cuda=False, numpy=True)

            tn = TN(score, idx, ref_videos_delimter, TOP_K=TOP_K, SCORE_THR=SCORE_THR,
                    TEMP_WND=TEMP_WND, MIN_PATH=MIN_PATH, MIN_MATCH=MIN_MATCH)
            detect = tn.fit()

            detect = [{'query_name': query_video['name'],
                       'ref_name': ref_videos_names[p['ref_vid_idx']],
                       'query': Period(query_video['start'] + (p['query'].start / fps),
                                       query_video['start'] + (p['query'].end / fps)),
                       'ref': Period((p['ref'].start - ref_videos_delimter[p['ref_vid_idx']]) / fps,
                                     (p['ref'].end - ref_videos_delimter[p['ref_vid_idx']]) / fps),
                       'score': p['score']} for p in detect]

            ground = [{'query_name': l[0]['name'],
                       'ref_name': l[1]['name'],
                       'query': Period(l[0]['start'], l[0]['end']),
                       'ref': Period(l[1]['start'], l[1]['end'])} for l in relative_pair(query=query_video)]

            ground = sorted(ground, key=lambda x: x['ref'].end - x['ref'].start, reverse=True)

            # print(len(detect), len(ground))
            TP, TP_gt, FP, FN = match(detect, ground)
            measure.update(len(TP), len(FP), len(FN))
            out_str = '\t{}\n'.format(measure.__str__())
            out_str += '\tDetected: {}, Ground: {}\n'.format(len(detect), len(ground))
            out_str += '\tTP: {}\n'.format(TP)
            out_str += '\tTP-ground: {}\n'.format(TP_gt)
            out_str += '\tFP: {}\n'.format(FP)
            out_str += '\tFN: {}'.format(FN)
            print(out_str)


def main():
    parser = argparse.ArgumentParser(description="Video Copy Detection Evaluation")
    parser.add_argument('--vcdb_root', type=str, default="/DB/videoCopyDetection")
    parser.add_argument('--nfold', type=int, default=1,
                        help="Number of split for train or valid datasets (default(1) means no split)")
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='Validation dataset idx, it should be smaller than nsplit (valid_idx < nsplit)')
    parser.add_argument('--fps', type=int, default=1, help="Sampling FPS")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers")
    args = parser.parse_args()

    vcdb_root = args.vcdb_root
    nfold = args.nfold
    fold_idx = args.fold_idx
    fps = args.fps
    batch_sz = args.batch_size
    num_workers = args.num_workers

    nfold = 5
    fps = 5

    assert os.path.exists(vcdb_root)
    assert 0 < nfold
    assert 0 <= fold_idx < nfold

    vcdb = VCDB_dataset(root=vcdb_root, n_folds=nfold, fold_idx=fold_idx)
    ref_videos = FrameDataset(vcdb.get_reference_videos(valid=True, train=False), fps=fps, isQuery=False)
    query = FrameDataset(vcdb.get_query(valid=True, train=False), fps=fps, isQuery=True)

    print(len(vcdb.core_video_val))  # , vcdb.core_video_val)
    print(len(vcdb.query_val))  # , vcdb.query_val)
    # print(vcdb.query_val)

    model = DummyEmb()
    model.cuda()
    model = torch.nn.DataParallel(model)
    '''
    load model ckpts
    '''
    eval(model, ref_videos=ref_videos, query=query, relative_pair=vcdb.get_relative_pair, fps=fps, batch_sz=batch_sz,
         num_workers=num_workers)


if __name__ == '__main__':
    main()
