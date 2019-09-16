from torchvision.datasets.folder import make_dataset, DatasetFolder
from torchvision.transforms import transforms as trn
from torchvision.models import resnet18, resnet50, alexnet
from torch.utils.data import DataLoader, Dataset

from torchsummary import summary
from math import ceil, floor
import imageio
import warnings
import cv2
import sys
import os
import torch

from utils.extract_frames import execute_ffmpeg
from utils.utils import int_round
from dataset.ListDataset import ListDataset
from models.nets import Resnet50_RMAC

import copy


class VCDB(object):
    def __init__(self, root='/DB/VCDB', feature_root='fps_1_resnet50_rmac', fps=1, n_folds=1, fold_idx=0):
        print('Init VCDB')
        self.__validate(root, fps, n_folds, fold_idx)
        self.video_root = os.path.join(root, 'core_dataset')
        self.frame_root = os.path.join(root, 'frames')
        self.feature_root = os.path.join(root, 'features', feature_root)
        # self.feature_root = os.path.join(root, 'features', 'fps_1_alexnet')
        # '/DB/VCDB/features/fps_1_alexnet'
        # Video
        self.titles = sorted(os.listdir(self.video_root))
        print('read video list ...', end='')
        self.video_list = self.__read_meta_file()
        print('\rread video list ...ok')
        print('Parse annotation ... ', end='')
        # GT
        self.query_list, self.pairs = self.__parse_annotation(os.path.join(root, 'annotation'))
        print('\rParse annotation ... ok')
        # features
        self.video_list_train = self.pairs_train = self.video_list_val = self.pairs_val = self.query_list_train = self.query_list_val = None
        if self.n_folds > 1:
            print('Split annotation ... ', end='')
            self.video_list_train, self.pairs_train, self.video_list_val, self.pairs_val, self.query_list_train, self.query_list_val = self.split_train_val()
            print('\rSplit annotation ... ok ')

    def __validate(self, root, fps, n_folds, fold_idx):
        if fold_idx > 1:
            assert fold_idx < n_folds
        self.root = root
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.fps = fps

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
                    video_a.update({'start': self.__time2sec(start_a), 'end': self.__time2sec(end_a)})
                    video_b.update({'start': self.__time2sec(start_b), 'end': self.__time2sec(end_b)})

                    if video_a not in query_list:
                        query_list.append(video_a)
                    if video_b not in query_list:
                        query_list.append(video_b)
                    pairs.append((video_a, video_b))
        return query_list, pairs

    def find_video_with_name(self, name):
        return [v for v in self.video_list if v['name'] == name]

    def __time2sec(self, time):
        t = list(map(int, time.split(':')))
        return t[0] * 3600 + t[1] * 60 + t[2]

    def __make_meta_file(self, meta_file):
        meta = []
        with open(meta_file, 'w') as f:
            for t in self.titles:
                for v in sorted(os.listdir(os.path.join(self.video_root, t))):
                    metadata = self.__get_meta_data(os.path.join(self.video_root, t, v))
                    fps = metadata['fps']
                    duration = metadata['duration']
                    nframes = ceil(fps * duration)
                    meta.append((fps, duration))
                    v = os.path.basename(v)
                    c = self.classes[c]
                    line = ','.join(list(map(str, [v, c, fps, nframes, duration]))) + '\n'
                    f.write(line)
        return meta

    def __get_meta_data(self, path):
        try:
            vid = imageio.get_reader(path, "ffmpeg")
            meta = vid.get_meta_data()  # ['fps']
            # self.fps_cache[self.video_path(v)] = vid.get_meta_data()["fps"]
        except Exception as e:
            print(e)
            warnings.warn("Unable to get metadata for video {}".format(path))
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
                nf = len(os.listdir(os.path.join('/DB/VCDB/frames/', cls, name.split('.')[0])))  # int(nf)
                # meta.append((fps, nf, dur))
                # meta.append((fps, dur))
                video_list.append({'title': cls, 'name': name, 'fps': fps, 'nframes': nf, 'duration': dur})

        video_list.sort(key=lambda x: (x['title'], x['name']))

        return video_list

    def __sampling_frames(self, fps=1):
        fr = []
        cnt = []
        for i, v in enumerate(self.video_list):
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

    def __get_entire_feature(self, video):
        path = os.path.join(self.feature_root, video['title'], video['name'].split('.')[0] + '.pt')
        f = torch.load(path)
        return f

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

    def get_feature(self, video, entire=True):
        f = self.__get_entire_feature(video)
        f_len = f.shape[0]
        if not entire:
            feature_len = f.shape[0]
            intv = feature_len / video['duration']
            start = int(video['start'] * intv)
            end = int(video['end'] * intv)
            # print(f.shape,start,end)
            f = f[start:(end + 1), :]

        return f, f_len

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

    def get_gt(self,query):
        l=[]
        for p in self.pairs:
            if p[0]['name'] == query['name'] and p[0]['start']==query['start']and p[0]['end']==query['end']:
                l.append([p[0], p[1]])
            elif p[1]['name'] == query['name']and p[1]['start']==query['start']and p[1]['end']==query['end']:
                l.append([p[1], p[0]])
        return l

    def get_query_relative_pair(self, query, video, phase='all'):
        l = []
        if phase == 'train':
            sp = self.pairs_train
        elif phase == 'val':
            sp = self.pairs_val
        else:
            sp = self.pairs
        for p in sp:
            if p[0]['name'] == query['name'] and p[1]['name'] == video['name'] and not (
                    p[0]['start'] > query['end'] or p[0]['end'] < query['start']):
                l.append([p[0], p[1]])
            elif p[1]['name'] == query['name'] and p[0]['name'] == video['name'] and not (
                    p[1]['start'] > query['end'] or p[1]['end'] < query['start']):
                l.append([p[1], p[0]])
        return l

    def get_relative_pair2(self, video1, video2, phase='all'):
        l = []
        if phase == 'train':
            sp = self.pairs_train
        elif phase == 'val':
            sp = self.pairs_val
        else:
            sp = self.pairs
        for p in sp:
            if p[0]['name'] == video1['name'] and p[1]['name'] == video2['name']:
                l.append([p[0], p[1]])
            elif p[1]['name'] == video1['name'] and p[0]['name'] == video2['name']:
                l.append([p[1], p[0]])
        return l

    def extract_frames_ffmpeg(self, target):
        for i, v in enumerate(self.video_list):
            dst = os.path.join(target, v['title'], v['name'].split('.')[0])
            src = os.path.join(self.video_root, v['title'], v['name'])
            if not os.path.exists(dst):
                os.makedirs(dst)
            args = '-i {} -f image2 {}/%5d.jpg'.format(src, dst)
            ret, out, err = execute_ffmpeg(args)
            print(i, ret, v['title'], v['name'])

    def extract_cnn_fingerprint(self, target='/DB/VCDB/features/fps_1_alexnet'):
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
        frame, frame_cnt = self.__sampling_frames(fps=self.fps)
        with torch.no_grad():
            for i, v in enumerate(self.video_list):
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


if __name__ == '__main__':
    db = VCDB()
    print(db.video_list)
    # db.extract_cnn_features()
    # model=alexnet(pretrained=True)
    # model.classifier=torch.nn.Sequential(*list(model.classifier.children())[:-1])
    # model.cuda()
    # model = torch.nn.DataParallel(model)
    # summary(model, (3, 227, 227))
    # print(model)
