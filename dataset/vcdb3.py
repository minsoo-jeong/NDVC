from torchvision.datasets.folder import make_dataset, DatasetFolder
from torchvision.transforms import transforms as trn
from torchvision.models import resnet18, resnet50, alexnet
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torchsummary import summary
from math import ceil, floor
import imageio
import warnings
import cv2
import sys
import os
import torch
import numpy as np

from utils.extract_frames import execute_ffmpeg
from utils.utils import int_round
from dataset.ListDataset import ListDataset
from models.nets import Resnet50_RMAC

import copy


class VCDB(object):
    def __init__(self, root='/DB/VCDB', fingerprint='fps_1_resnet50_rmac', fingerprint_intv=1, fps=1, n_fold=1,
                 fold_idx=0):
        print('Init VCDB')
        self.core_video_root = os.path.join(root, 'core_dataset')
        self.video_root = os.path.join(root, 'core_dataset')
        self.frame_root = os.path.join(root, 'frames')
        self.fingerprint_root = os.path.join(root, 'features', fingerprint)
        annotation_dir = os.path.join(root, 'annotation')

        self.__validate(root, fingerprint_intv, fps, n_fold, fold_idx)

        # Video
        print('read video list ...', end='')
        self.titles = sorted(os.listdir(self.video_root))
        self.video_list = self.__read_meta_file()
        print('\rread video list ...ok')

        # GT
        print('Parse annotation ... ', end='')
        self.query_list, self.pairs = self.__parse_annotation(annotation_dir)
        print('\rParse annotation ... ok')

        # split train-valid
        self.video_list_train = self.video_list_val = self.video_list
        self.pairs_train = self.pairs_val = self.pairs
        self.query_list_train = self.query_list_val = self.query_list
        if self.n_folds > 1:
            print('Split annotation ... ', end='')
            self.video_list_train, self.pairs_train, self.video_list_val, self.pairs_val, self.query_list_train, self.query_list_val = self.split_train_val()
            print('\rSplit annotation ... ok ')

    def __validate(self, root, fingerprint_intv, fps, n_folds, fold_idx):
        assert fold_idx < n_folds
        self.root = root
        self.fingerprint_intv = fingerprint_intv
        self.fps = fps
        self.n_folds = n_folds
        self.fold_idx = fold_idx

    def __make_meta_file(self, meta_file):
        meta = []
        with open(meta_file, 'w') as f:
            for t in self.titles:
                for v in sorted(os.listdir(os.path.join(self.video_root, t))):
                    metadata = self.__get_video_metadata(os.path.join(self.video_root, t, v))
                    fps = metadata['fps']
                    duration = metadata['duration']
                    nframes = ceil(fps * duration)
                    meta.append((fps, duration))
                    v = os.path.basename(v)
                    c = self.classes[c]
                    line = ','.join(list(map(str, [v, c, fps, nframes, duration]))) + '\n'
                    f.write(line)
        return meta

    def __read_meta_file(self):
        meta_file = os.path.join(self.root, 'core_video_meta.txt')
        if not os.path.exists(meta_file):
            self.__make_meta_file(meta_file)
        # meta = []
        video_list = []
        with open(meta_file, 'r') as f:
            for line in f.readlines():
                name, cls, fps, nf, dur = line.rstrip().split(',')
                fps = float(fps)
                dur = float(dur)
                nf = len(os.listdir(os.path.join(self.frame_root, cls, name.split('.')[0])))  # int(nf)
                fp = torch.load(os.path.join(self.fingerprint_root, cls, name.split('.')[0] + '.pt')).shape[0]
                video_list.append({'title': cls, 'name': name,
                                   'fps': fps, 'nframes': nf,
                                   'duration': dur, 'fingerprint': fp})

        video_list.sort(key=lambda x: (x['title'], x['name']))

        return video_list

    def __get_video_metadata(self, path):
        try:
            vid = imageio.get_reader(path, "ffmpeg")
            meta = vid.get_meta_data()  # ['fps']
            # self.fps_cache[self.video_path(v)] = vid.get_meta_data()["fps"]
        except Exception as e:
            print(e)
            warnings.warn("Unable to get metadata for video {}".format(path))
        return meta

    def __parse_annotation(self, annotation_dir):
        query_list = []
        pairs = []
        l = sorted(os.listdir(annotation_dir))
        for i in l:
            with open(os.path.join(annotation_dir, i), 'r') as f:
                for line in f.readlines():
                    name_a, name_b, start_a, end_a, start_b, end_b = line.rstrip().split(',')
                    video_a = copy.deepcopy(self.find_video_with_name(name_a)[0])
                    video_b = copy.deepcopy(self.find_video_with_name(name_b)[0])
                    intv_a = video_a['fingerprint'] / video_a['duration']
                    start_idx_a = int(self.__time2sec(start_a) * intv_a)
                    end_idx_a = min(int(self.__time2sec(end_a) * intv_a), video_a['fingerprint'] - 1)
                    intv_b = video_b['fingerprint'] / video_b['duration']
                    start_idx_b = int(self.__time2sec(start_b) * intv_b)
                    end_idx_b = min(int(self.__time2sec(end_b) * intv_b), video_b['fingerprint'] - 1)
                    video_a.update({'start': self.__time2sec(start_a), 'end': self.__time2sec(end_a),
                                    'start_idx': start_idx_a, 'end_idx': end_idx_a})
                    video_b.update({'start': self.__time2sec(start_b), 'end': self.__time2sec(end_b),
                                    'start_idx': start_idx_b, 'end_idx': end_idx_b})

                    if video_a not in query_list:
                        query_list.append(video_a)
                    if video_b not in query_list:
                        query_list.append(video_b)
                    pairs.append((video_a, video_b))
        return query_list, pairs

    def __time2sec(self, time):
        t = list(map(int, time.split(':')))
        return t[0] * 3600 + t[1] * 60 + t[2]

    def __sampling_frames(self, videos, fps=1):
        fr = []
        cnt = []
        for i, v in enumerate(videos):
            title = v['title']
            name = v['name'].split('.')[0]
            frames = os.listdir(os.path.join(self.frame_root, title, name))
            frames.sort()
            intv = int(round(v['fps']) / fps)
            if fps > 0:
                sample_frames = [os.path.join(self.frame_root, title, name, frames[f]) for f in
                                 range(int(intv / 2), len(frames), intv)]
            else:
                sample_frames = [os.path.join(self.frame_root, title, name, frames[f]) for f in range(0, len(frames))]
            fr.append(sample_frames)
            cnt.append(len(sample_frames))
        return fr, cnt

    def __get_entire_fingerprint(self, video):
        path = os.path.join(self.fingerprint_root, video['title'], video['name'].split('.')[0] + '.pt')
        f = torch.load(path)
        return f

    def get_fingerprint(self, video, split=False):
        f = self.__get_entire_fingerprint(video)
        if split:
            f = f[video['start_idx']:video['end_idx'] + 1]

        return f

    def get_relative_pair(self, video, phase='all'):
        l = []
        if phase == 'train':
            sp = self.pairs_train
        elif phase == 'val':
            sp = self.pairs_val
        else:
            sp = self.pairs
        for p in sp:
            if p[0]['name'] == video['name']:
                l.append([p[0], p[1]])
            elif p[1]['name'] == video['name']:
                l.append([p[1], p[0]])
        return l

    def split_train_val(self):
        cc = self.titles
        ncc = len(cc)

        val_cc = cc[(ncc // self.n_folds) * self.fold_idx: (ncc // self.n_folds) * (self.fold_idx + 1)]
        train_cc = (cc[: (ncc // self.n_folds) * self.fold_idx]
                    + cc[(ncc // self.n_folds) * (self.fold_idx + 1):])
        video_list_train = [v for v in self.video_list if v["title"] in train_cc]
        video_list_val = [v for v in self.video_list if v["title"] in val_cc]

        videos_train_name = [v['name'] for v in video_list_train]
        videos_val_name = [v['name'] for v in video_list_val]

        pairs_train = [p for p in self.pairs if p[0]["name"] in videos_train_name]
        pairs_val = [p for p in self.pairs if p[0]["name"] in videos_val_name]

        query_list_train = [p for p in self.query_list if p['name'] in videos_train_name]
        query_list_val = [p for p in self.query_list if p['name'] in videos_val_name]

        return video_list_train, pairs_train, video_list_val, pairs_val, query_list_train, query_list_val

    def find_video_with_name(self, name):
        return [v for v in self.video_list if v['name'] == name]

    def extract_frames_ffmpeg(self, videos, target):
        for i, v in enumerate(videos):
            dst = os.path.join(target, v['title'], v['name'].split('.')[0])
            src = os.path.join(self.video_root, v['title'], v['name'])
            if not os.path.exists(dst):
                os.makedirs(dst)
            args = '-i {} -f image2 {}/%5d.jpg'.format(src, dst)
            ret, out, err = execute_ffmpeg(args)
            print(i, ret, v['title'], v['name'])

    def extract_cnn_fingerprint(self, videos, target='/DB/VCDB/features/fps_1_alexnet'):
        # model=Resnet50_RMAC()
        # model = resnet50(pretrained=True)
        # model = torch.nn.Sequential(*list(model.children())[:-1])
        model = alexnet(pretrained=True)
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])
        model.cuda()
        model = torch.nn.DataParallel(model)
        summary(model, (3, 224, 224))
        model.eval()

        normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        video_trn = trn.Compose([
            trn.Resize(224),
            trn.ToTensor(),
            normalize
        ])
        frame, frame_cnt = self.__sampling_frames(videos=videos, fps=self.fps)
        with torch.no_grad():
            for i, v in enumerate(videos):
                dl = DataLoader(ListDataset(frame[i], transform=video_trn), batch_size=64, num_workers=4)
                dst = os.path.join(target, v['title'], v['name'].split('.')[0] + '.pt')
                print(dst)
                if not os.path.exists(os.path.join(target, v['title'])):
                    os.makedirs(os.path.join(target, v['title']))
                frame_feature = []
                for n, (im, path) in enumerate(dl):
                    out = model(im.cuda()).squeeze(-1).squeeze(-1)
                    frame_feature.append(out)
                    print('{:3d}: extract vid: {}/{} duration: {} shape: {}'.format(i, v['title'], v['name'],
                                                                                    v['duration'], out.shape))
                frame_feature = torch.cat(frame_feature)
                # video_feature = torch.mean(frame_feature, dim=0, keepdim=True)

                frame_feature = frame_feature.cpu()
                # video_feature = video_feature.cpu()

                torch.save(frame_feature, dst)


class OnlineTripletFingerprintPairDataset(Dataset):
    def __init__(self, pairs, fingerprint_root):
        self.pairs = pairs
        self.feature_root = fingerprint_root

        self.samples = self.generate_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        (v0, fp_idx0), (v1, fp_idx1), cls = self.samples[index]
        fp0 = self.get_fingerprint_with_idx(v0, fp_idx0)
        fp1 = self.get_fingerprint_with_idx(v1, fp_idx1)
        v0['fingerprint_idx'] = fp_idx0
        v1['fingerprint_idx'] = fp_idx1

        return (fp0, v0), (fp1, v1), cls

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '\tNumber of pair: {}\n'.format(len(self.pairs))
        fmt_str += '\tNumber of samples: {}\n'.format(len(self.samples))
        return fmt_str

    def generate_samples(self):
        samples = []
        for n, p in enumerate(self.pairs):
            intv = (p[1]['end_idx'] + 1 - p[1]['start_idx']) / (p[0]['end_idx'] + 1 - p[0]['start_idx'])
            for tc, t0 in enumerate(range(p[0]['start_idx'], p[0]['end_idx'] + 1)):
                offset = intv * tc
                t1 = min(int(p[1]['start_idx'] + offset), p[1]['end_idx'])
                samples.append([(p[0], t0), (p[1], t1), n])
        return samples

    def get_fingerprint_with_idx(self, video, fp_idx):
        path = os.path.join(self.feature_root, video['title'], video['name'].split('.')[0] + '.pt')
        f = torch.load(path)[fp_idx]
        return f


class FingerPrintDataset(Dataset):
    def __init__(self, videos, fp_root, query=False):
        self.videos = videos
        self.fingerprint_root = fp_root
        self.query = query

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        v = self.videos[index]
        f = self.get_fingerprint(v, split=True) if self.query else self.get_fingerprint(v, split=False)

        return f, v

    def get_fingerprint(self, video, split=False):
        path = os.path.join(self.fingerprint_root, video['title'], video['name'].split('.')[0] + '.pt')
        f = torch.load(path)
        if split:
            f = f[video['start_idx']:video['end_idx'] + 1]
        return f


if __name__ == '__main__':

    db = VCDB(n_fold=3, fold_idx=2)
    print(len(db.query_list_train))
    print(len(db.query_list_val))
    print(len(db.query_list))

