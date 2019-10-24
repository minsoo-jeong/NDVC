from torchvision.datasets.folder import make_dataset, DatasetFolder, default_loader
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
import traceback


class VCDB(object):
    def __init__(self, root='/DB/VCDB', fingerprint='fps_1_resnet50_rmac', fingerprint_intv=1, fps=1, n_fold=1,
                 fold_idx=0):
        print('Init VCDB')
        self.core_video_root = os.path.join(root, 'core_dataset')
        self.frame_root = os.path.join(root, 'frames')
        self.fingerprint_root = os.path.join(root, 'features', fingerprint)
        annotation_dir = os.path.join(root, 'annotation')

        self.__validate(root, fingerprint_intv, fps, n_fold, fold_idx)

        # Video
        print('read video list ...', end='')
        self.titles = sorted(os.listdir(self.core_video_root))
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
                for v in sorted(os.listdir(os.path.join(self.core_video_root, t))):
                    metadata = self.__get_video_metadata(os.path.join(self.core_video_root, t, v))
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

    def get_relative_pair2(self, video, phase='all'):
        l = []
        if phase == 'train':
            sp = self.pairs_train
        elif phase == 'val':
            sp = self.pairs_val
        else:
            sp = self.pairs
        for p in sp:
            if p[0] == video:
                l.append([p[0], p[1]])
            elif p[1] == video:
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
            src = os.path.join(self.core_video_root, v['title'], v['name'])
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


class VCDB_video(object):
    def __init__(self, root='/DB/VCDB/'):
        self.root = root
        self.core_video_root = os.path.join(root, 'core_dataset')
        self.core_frame_root = os.path.join(root, 'frames')
        self.bg_video_root = os.path.join(root, 'background_dataset', 'videos')
        self.bg_frames_root = os.path.join(root, 'background_dataset', 'frames')
        self.core_video_list, self.bg_video_list = self.__scan_video()

    def __scan_video(self):
        core_videos = [{'title': t, 'name': v} for t in sorted(os.listdir(self.core_video_root))
                       for v in sorted(os.listdir(os.path.join(self.core_video_root, t)))]
        bg_videos = [{'title': t, 'name': v} for t in sorted(os.listdir(self.bg_video_root)) if
                     os.path.isdir(os.path.join(self.bg_video_root, t))
                     for v in sorted(os.listdir(os.path.join(self.bg_video_root, t)))]
        return core_videos, bg_videos

    def extract_core_frames(self):
        root = self.core_video_root
        self.__extract_frames_ffmpeg(self.core_video_list, self.core_video_root, os.path.join(self.root, 'frames'))

    def extract_bg_frames(self, title=None):
        if title is None:
            title = list(
                filter(lambda x: os.path.isdir(os.path.join(self.bg_video_root, x)), os.listdir(self.bg_video_root)))
            print(title)
        if isinstance(title, int):
            title = [str(title)]
        v = [v for v in self.bg_video_list if v['title'] in title]
        self.__extract_frames_ffmpeg(v, self.bg_video_root, os.path.join(self.root, 'background_dataset', 'frames'))

        # self.__extract_frames_ffmpeg(self.core_video_list,os.path.join(self.root,'frames'))

    def __extract_frames_ffmpeg(self, videos, src_root, dst_root):
        for i, v in enumerate(videos):
            dst = os.path.join(dst_root, v['title'], v['name'].split('.')[0])
            src = os.path.join(src_root, v['title'], v['name'])
            if not os.path.exists(dst):
                os.makedirs(dst)
            args = '-i {} -f image2 {}/%5d.jpg'.format(src, dst)
            ret, out, err = execute_ffmpeg(args)
            print(i, ret, v['title'], v['name'], dst)


class VCDB_annotation(object):
    def __init__(self, dir='/DB/VCDB/annotation'):
        self.dir = dir
        self.pairs, self.query_list, self.video_list = self.__scan_annotation()

    def __scan_annotation(self):
        def time2sec(time):
            t = list(map(int, time.split(':')))
            return t[0] * 3600 + t[1] * 60 + t[2]

        pair = []
        query_list = set()
        vid = set()
        l = sorted(os.listdir(self.dir))
        keys = ['title', 'name', 'start', 'end']
        for t in l:
            with open(os.path.join(self.dir, t), 'r') as f:
                for line in f.readlines():
                    name_a, name_b, start_a, end_a, start_b, end_b = line.rstrip().split(',')
                    videoA = (t, name_a, time2sec(start_a), time2sec(end_a))
                    videoB = (t, name_b, time2sec(start_b), time2sec(end_b))
                    pair.append((dict(zip(keys, videoA)), dict(zip(keys, videoB))))
                    query_list.add(videoA)
                    query_list.add(videoB)
                    vid.add((t, name_a))
                    vid.add((t, name_b))

        query_list = [dict(zip(keys, q)) for q in query_list]
        query_list.sort(key=lambda x: x['end'], reverse=True)
        query_list.sort(key=lambda x: (x['title'], x['name'], x['start']))

        vid = [{'title': v[0], 'name': v[1]} for v in sorted(vid, key=lambda x: (x[0], x[1]))]

        return pair, query_list, vid

    def split_train_val(self, n_folds=1, fold_idx=0):
        cc = sorted(os.listdir(self.dir))
        ncc = len(cc)

        val_cc = cc[(ncc // n_folds) * fold_idx: (ncc // n_folds) * (fold_idx + 1)]
        train_cc = (cc[: (ncc // n_folds) * fold_idx]
                    + cc[(ncc // n_folds) * (fold_idx + 1):])
        video_list_train = [v for v in self.video_list if v["title"] in train_cc]
        video_list_val = [v for v in self.video_list if v["title"] in val_cc]

        videos_train_name = [v['name'] for v in video_list_train]
        videos_val_name = [v['name'] for v in video_list_val]

        pairs_train = [p for p in self.pairs if p[0]["name"] in videos_train_name]
        pairs_val = [p for p in self.pairs if p[0]["name"] in videos_val_name]

        query_list_train = [p for p in self.query_list if p['name'] in videos_train_name]
        query_list_val = [p for p in self.query_list if p['name'] in videos_val_name]

        return (video_list_train, pairs_train, query_list_train), (video_list_val, pairs_val, query_list_val)


class VCDB_dataset(object):
    def __init__(self, root='/DB/VCDB/', n_folds=1, fold_idx=0):
        self.root = root
        self.annotation_dir = os.path.join(root, 'annotation')
        self.core_video_root = os.path.join(root, 'core_dataset', 'videos')
        self.core_frame_root = os.path.join(root, 'core_dataset', 'frames')
        self.core_meta = os.path.join(self.root, 'core_dataset', 'core_meta.txt')

        self.bg_video_root = os.path.join(root, 'background_dataset', 'videos')
        self.bg_frames_root = os.path.join(root, 'background_dataset', 'frames')
        self.bg_meta = os.path.join(self.root, 'background_dataset', 'bg_meta.txt')

        self.core_video, self.bg_video = self.__scan_video()
        self.pairs, self.query_list = self.__scan_annotation()
        self.core_video_train, self.pairs_train, self.query_train, self.core_video_val, self.pairs_val, self.query_val = self.__split_train_val(
            n_folds, fold_idx)

    def __scan_video(self):
        def get_fps_duration(path):
            try:
                vid = imageio.get_reader(path, "ffmpeg")
                meta = vid.get_meta_data()  # ['fps']
                # self.fps_cache[self.video_path(v)] = vid.get_meta_data()["fps"]
            except Exception as e:
                warnings.warn("Unable to get metadata for video {}".format(path))
            return meta['fps'], meta['duration']

        def write_video_meta(meta, video_root, frame_root):
            with open(meta, 'w') as f:
                for t in sorted(os.listdir(video_root)):
                    for v in sorted(os.listdir(os.path.join(video_root, t))):
                        fps, duration = get_fps_duration(os.path.join(video_root, t, v))
                        nframes = os.listdir(os.path.join(frame_root, t, v.split('.')[0])).__len__()
                        line = ','.join(map(str, [v, t, fps, duration, nframes])) + '\n'
                        f.write(line)

        def read_video_meta(meta):
            videos = []

            with open(meta, 'r') as f:
                for line in f.readlines():
                    name, cls, fps, dur, nf = line.rstrip().split(',')
                    videos.append(
                        {'title': cls, 'name': name, 'fps': float(fps), 'duration': float(dur), 'nframes': int(nf)})

            return videos

        if not os.path.exists(self.core_meta):
            write_video_meta(self.core_meta, self.core_video_root, self.core_frame_root)
        core_videos = read_video_meta(self.core_meta)
        if not os.path.exists(self.bg_meta):
            write_video_meta(self.bg_meta, self.bg_video_root, self.bg_frames_root)
        bg_videos = read_video_meta(self.bg_meta)

        return core_videos, bg_videos

    def __scan_annotation(self):
        def time2sec(time):
            t = list(map(int, time.split(':')))
            return t[0] * 3600 + t[1] * 60 + t[2]

        pair = []
        query_list = set()
        # vid = set()
        l = sorted(os.listdir(self.core_video_root))
        keys = ['title', 'name', 'start', 'end', 'fps', 'duration', 'nframes']
        for t in l:
            with open(os.path.join(self.annotation_dir, t + '.txt'), 'r') as f:
                for line in f.readlines():
                    name_a, name_b, start_a, end_a, start_b, end_b = line.rstrip().split(',')
                    vidA = list(filter(lambda x: x['name'] == name_a, self.core_video))[0]
                    vidB = list(filter(lambda x: x['name'] == name_b, self.core_video))[0]
                    videoA = (
                        t, name_a, time2sec(start_a), time2sec(end_a), vidA['fps'], vidA['duration'], vidA['nframes'])
                    videoB = (
                        t, name_b, time2sec(start_b), time2sec(end_b), vidB['fps'], vidB['duration'], vidB['nframes'])
                    pair.append((dict(zip(keys, videoA)), dict(zip(keys, videoB))))
                    query_list.add(videoA)
                    query_list.add(videoB)
                    # vid.add((t, name_a))
                    # vid.add((t, name_b))

        query_list = [dict(zip(keys, q)) for q in query_list]
        query_list.sort(key=lambda x: x['end'], reverse=True)
        query_list.sort(key=lambda x: (x['title'], x['name'], x['start']))
        # vid = [{'title': v[0], 'name': v[1]} for v in sorted(vid, key=lambda x: (x[0], x[1]))]

        return pair, query_list

    def __split_train_val(self, n_folds=1, fold_idx=0):
        cc = sorted(os.listdir(self.core_video_root))
        ncc = len(cc)

        val_cc = cc[(ncc // n_folds) * fold_idx: (ncc // n_folds) * (fold_idx + 1)]
        train_cc = (cc[: (ncc // n_folds) * fold_idx]
                    + cc[(ncc // n_folds) * (fold_idx + 1):])

        video_train = [v for v in self.core_video if v["title"] in train_cc]
        video_val = [v for v in self.core_video if v["title"] in val_cc]

        video_name_train = [v['name'] for v in video_train]
        video_name_val = [v['name'] for v in video_val]

        pairs_train = [p for p in self.pairs if p[0]["name"] in video_name_train]
        pairs_val = [p for p in self.pairs if p[0]["name"] in video_name_val]

        query_train = [p for p in self.query_list if p['name'] in video_name_train]
        query_val = [p for p in self.query_list if p['name'] in video_name_val]

        return video_train, pairs_train, query_train, video_val, pairs_val, query_val

    def get_relative_pair(self, query):
        l = []
        for p in self.pairs:
            if p[0]['name'] == query['name'] and not (p[0]['end'] < query['start'] or query['end'] < p[0]['start']):
                l.append((p[0], p[1]))
            elif p[1]['name'] == query['name'] and not (p[1]['end'] < query['start'] or query['end'] < p[1]['start']):
                l.append((p[1], p[0]))
        return l

    def get_reference_videos(self, valid=True, train=False, background=False, total=528):
        assert valid or train or background
        l = []
        if valid and train:
            l = self.core_video
        elif valid:
            l = self.core_video_val
        elif train:
            l = self.core_video_train

        if background:
            c = total - len(l)
            bi = np.random.choice(len(self.bg_video), size=c,replace=False)
            for i in bi:
                l.append(self.bg_video[i])

        return l

    def get_query(self, valid=True, train=False):
        assert valid or train
        l = []
        if valid and train:
            l = self.query_list
        elif valid:
            l = self.query_val
        elif train:
            l = self.query_train
        return l


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


"""
Train / valid
pair 에 해당하는 frame load
"""


class OnlineTripletFramePairDataset(Dataset):
    def __init__(self, videos, pairs, fps=10, sec_per_group=1, video_root='/DB/VCDB/core_dataset/',
                 frame_root='/DB/VCDB/frames/'):
        self.video_root = video_root
        self.frame_root = frame_root
        self.videos = videos
        self.pairs = pairs
        self.samples = self.generate_samples(fps, sec_per_group)
        normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.default_trn = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            normalize
        ])
        self.loader = default_loader

    def __getitem__(self, index):
        (v0, v0_frames), (v1, v1_frames), cls = self.samples[index]
        p0 = self.get_random_frame(v0, v0_frames)
        p1 = self.get_random_frame(v1, v1_frames)
        f0 = self.default_trn(self.loader(p0))
        f1 = self.default_trn(self.loader(p1))

        return (p0, f0), (p1, f1), cls

    def __len__(self):
        return len(self.samples)

    def get_random_frame(self, video, frame_idx):
        p = os.path.join(self.frame_root, video['title'], video['name'].split('.')[0])
        l = sorted(os.listdir(p))
        if len(frame_idx) == 0:
            print(video)

        c = np.random.choice(frame_idx, size=1)
        choice = os.path.join(p, l[c[0]])

        return choice

    def generate_samples(self, fps, spg):
        def divide_chunks(l, n):
            # looping till length l
            for i in range(0, len(l), n):
                yield l[i:i + n]

        samples = []
        group = int(fps * spg)
        cls = 0
        for p in self.pairs:
            start_idx_a = max(0, round(p[0]['start'] * p[0]['fps']))
            end_idx_a = min(p[0]['nframes'], round(p[0]['end'] * p[0]['fps']))
            intv_a = p[0]['fps'] / fps
            frame_idx_a = [int(round(i)) for i in np.arange(start_idx_a + intv_a / 2, end_idx_a, intv_a)]
            frame_idx_a = list(divide_chunks(frame_idx_a, group))
            # frame_idx_a = [frame_idx_a[i * group:(i + 1) * group] for i in range(0, int(len(frame_idx_a) / group)+1)]

            start_idx_b = max(0, round(p[1]['start'] * p[1]['fps']))
            end_idx_b = min(p[1]['nframes'], round(p[1]['end'] * p[1]['fps']))
            intv_b = p[1]['fps'] / fps
            frame_idx_b = [int(round(i)) for i in np.arange(start_idx_b + intv_b / 2, end_idx_b, intv_b)]
            frame_idx_b = list(divide_chunks(frame_idx_b, group))
            # print(p[0],p[1])
            # print(frame_idx_a,len(frame_idx_a))
            # print(frame_idx_b, len(frame_idx_b))

            intv = len(frame_idx_b) / len(frame_idx_a)
            for tc, t0 in enumerate(frame_idx_a):
                offset = min(int(round(intv * tc)), len(frame_idx_b) - 1)
                samples.append([(p[0], t0), (p[1], frame_idx_b[offset]), cls])
                # print(t0, '=', frame_idx_b[offset], tc, offset,cls)
            cls += 1

        return samples


"""
eval
비디오에 대한 frame dir list
self.samples=비디오 1개에 대한 샘플링된 프레임 list 
"""


class VideoFrameDataset(Dataset):
    def __init__(self, videos, fps=10, sec_per_group=1, video_root='/DB/VCDB/core_dataset/',
                 frame_root='/DB/VCDB/frames/', query=False):
        self.videos = videos
        self.fps = fps
        self.spg = sec_per_group
        self.video_root = video_root
        self.frame_root = frame_root
        self.query = query

    def __getitem__(self, idx):
        v = self.videos[idx]
        p = os.path.join(self.frame_root, v['title'], v['name'].split('.')[0])
        l = sorted(os.listdir(p))
        intv = v['fps'] / self.fps
        group = int(self.fps * self.spg)
        start_idx = 0
        end_idx = len(l) - 1
        if self.query:
            start_idx = max(0, round(v['start'] * v['fps']))
            end_idx = min(end_idx, round(v['end'] * v['fps']))

        f = []
        index = []
        for n, i in enumerate(np.arange(0 + intv / 2, len(l), intv)):
            id = min(int(round(i)), end_idx - 1)
            if start_idx < i < end_idx:
                index.append(n)
            f.append(os.path.join(p, l[id]))
        # f = [os.path.join(p, l[int(round(i))]) for i in np.arange(start_idx + intv / 2, len(l), intv)]

        return f, v, index

    def __len__(self):
        return len(self.videos)


class ListDataset(Dataset):
    def __init__(self, l):
        self.samples = l
        self.loader = default_loader
        self.default_trn = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __getitem__(self, idx):
        path = self.samples[idx]
        frame = self.default_trn(self.loader(path))
        return frame, path

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    v = VCDB_dataset(n_folds=5, fold_idx=0)

    va_dataset = OnlineTripletFramePairDataset(v.core_video_val, v.pairs_val, 1, 1)
    va_loader = DataLoader(va_dataset, batch_size=64, shuffle=True, num_workers=4)
    for v in va_loader:
        pass

    vf = VideoFrameDataset(v.core_video_val, 2, 1, query=False)
    print(vf.__getitem__(9))
    print(len(v.core_video_val))
    for n, (f, v, i) in enumerate(vf):
        print(n, len(f), v, )
        # print(i)

        dl = DataLoader(ListDataset(f), batch_size=4, shuffle=False, num_workers=4)
        for f, p in dl:
            pass
            print(p)

    exit()

    print(v.query_train[:3])
    print(v.bg_video[:3])
    print(len(v.core_video_train), v.core_video[:3])
    print(v.core_video_val[:3])

    fp = OnlineTripletFramePairDataset(v.core_video_train, v.pairs_train, 1, 1)
    dl = DataLoader(fp, batch_size=32, shuffle=True, num_workers=4)
    print(fp.__len__())
    print(dl)
    try:
        for i in range(100):
            for n, ((p0, f0), (p1, f1), cls) in enumerate(dl):
                f = torch.cat([f0, f1])
                print(i, n, f.shape)
    except Exception as e:
        traceback.print_exc()
        input('Wait for key: ')

    exit()
    v = VCDB_annotation()
    train_ann, valid_ann = v.split_train_val()
    print(len(valid_ann[0]), len(valid_ann[1]), len(valid_ann[2]))
    print(valid_ann[0][:3])
    print(valid_ann[1][:3])
    print(valid_ann[2][:3])
    vv = VCDB_video()

    '''
    db = VCDB(n_fold=3, fold_idx=2)
    print(len(db.query_list_train))
    print(len(db.query_list_val))
    print(len(db.query_list))
    '''
