from torchvision.datasets.folder import make_dataset, DatasetFolder
from torchvision.transforms import transforms as trn
from torchvision.models import resnet18, resnet50
from torch.utils.data import DataLoader, Dataset

import torch

from torchsummary import summary
from math import ceil, floor
import imageio
import warnings
import cv2
import sys
import os

from utils.extract_frames import execute_ffmpeg
from utils.utils import int_round
from dataset.ListDataset import ListDataset
from models.nets import Resnet50_RMAC

import copy


# features and ground truth
class VCDB(object):
    def __init__(self, root='/DB/VCDB', fps=1):
        self.root = root
        self.feature_root = os.path.join(root, 'features', 'fps_1_resnet50_rmac')

        self.core_videos_db = VCDB_CORE_VIDEOS(video_root=os.path.join(root, 'core_dataset'), fps=fps)
        self.video_list = self.core_videos_db.video_list
        self.frame_cnt = self.core_videos_db.frames_cnt

        annotation_dir = os.path.join(root, 'annotation')
        self.query_list, self.pairs = self.__parse_annotation(annotation_dir)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '\t{}\n'.format(self.core_videos_db.__repr__())
        fmt_str += '\tNumber of query: {}\n'.format(len(self.query_list))
        fmt_str += '\tNumber of query pair: {}\n'.format(len(self.pairs))
        fmt_str += '\tRoot Location: {}\n'.format(self.root)
        fmt_str += '\tfeature Root Location: {}\n'.format(self.feature_root)
        return fmt_str

    def __parse_annotation(self, annotation_dir):
        query_list = []
        pairs = []
        l = os.listdir(annotation_dir)
        l.sort()
        for i in l:
            with open(os.path.join(annotation_dir, i), 'r') as f:
                for line in f.readlines():
                    name_a, name_b, start_a, end_a, start_b, end_b = line.rstrip().split(',')
                    video_a = copy.deepcopy(self.core_videos_db.find_video_with_name(name_a)[0])
                    video_b = copy.deepcopy(self.core_videos_db.find_video_with_name(name_b)[0])
                    video_a['start'] = self.__time2sec(start_a)
                    video_a['end'] = self.__time2sec(end_a)
                    video_b['start'] = self.__time2sec(start_b)
                    video_b['end'] = self.__time2sec(end_b)

                    if video_a not in query_list:
                        query_list.append(video_a)
                    if video_b not in query_list:
                        query_list.append(video_b)
                    pairs.append([video_a, video_b])

        return query_list, pairs

    def __time2sec(self, time):
        t = list(map(int, time.split(':')))
        return t[0] * 3600 + t[1] * 60 + t[2]

    def get_relative_pair(self, video):
        l = []
        for p in self.pairs:
            if p[0]['name'] == video['name']:
                l.append([p[0], p[1]])
            elif p[1]['name'] == video['name']:
                l.append([p[1], p[0]])
        return l

    def get_relative_pair2(self, video1, video2):
        l = []
        for p in self.pairs:
            if p[0]['name'] == video1['name'] and p[1]['name'] == video2['name']:
                l.append([p[0], p[1]])
            elif p[1]['name'] == video1['name'] and p[0]['name'] == video2['name']:
                l.append([p[1], p[0]])
        return l

    def get_query_relative_pair(self, query, video):
        l = []
        for p in self.pairs:
            if p[0]['name'] == query['name'] and p[1]['name'] == video['name'] and not (
                    p[0]['start'] > p[1]['end'] or p[0]['end'] < p[1]['start']):
                l.append([p[0], p[1]])
            elif p[1]['name'] == query['name'] and p[0]['name'] == video['name'] and not (
                    p[1]['start'] > p[0]['end'] or p[1]['end'] < p[0]['start']):
                l.append([p[1], p[0]])
        return l

    def __get_entire_feature(self, video):
        path = os.path.join(self.feature_root, video['class'], video['name'].split('.')[0] + '.pt')
        f = torch.load(path)
        return f

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

    # only videos and frames


class VCDB_CORE_VIDEOS(Dataset):
    def __init__(self, video_root='/DB/VCDB/core_dataset', fps=1, extensions=['mp4', 'flv']):
        self.video_root = video_root
        self.classes, self.class_to_idx = self._find_classes(self.video_root)

        self.videos = make_dataset(self.video_root, self.class_to_idx, extensions)
        self.video_list, self.meta = self.__read_video_meta()
        self.frames, self.frames_cnt = self.__sampling_frames(fps=fps)

    def __getitem__(self, item):
        ret = self.video_list[item]
        return (ret['class'], ret['name'], ret['fps'], ret['nframes'], ret['duration'])

    def __len__(self):
        return len(self.videos)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of videos: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.video_root)
        fmt_str += '    Sampling Frame cnt: {}\n'.format(sum(self.frames_cnt))
        return fmt_str

    def find_video_with_name(self, name):
        return [v for v in self.video_list if v['name'] == name]

    def _find_classes(self, dir):

        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __read_video_meta(self):
        meta_file = os.path.join('/DB/VCDB/core_video_meta.txt')
        if not os.path.exists(meta_file):
            self.__make_meta_file(meta_file)
        meta = []
        video_list = []
        with open(meta_file, 'r') as f:
            for line in f.readlines():
                name, cls, fps, nf, dur = line.rstrip().split(',')
                fps = float(fps)
                dur = float(dur)
                nf = len(os.listdir(os.path.join('/DB/VCDB/frames/', cls, name.split('.')[0])))  # int(nf)
                # meta.append((fps, nf, dur))
                meta.append((fps, dur))
                video_list.append({'class': cls, 'name': name, 'fps': fps, 'nframes': nf, 'duration': dur})

        video_list.sort(key=lambda x: (x['class'], x['name']))

        return video_list, meta

    def __read_meta_data(self, path):
        try:
            vid = imageio.get_reader(path, "ffmpeg")
            meta = vid.get_meta_data()  # ['fps']
            # self.fps_cache[self.video_path(v)] = vid.get_meta_data()["fps"]
        except Exception as e:
            print(e)
            warnings.warn("Unable to get metadata for video {}".format(path))
        return meta

    def __make_meta_file(self, meta_file):
        meta = []
        with open(meta_file, 'w') as f:
            for v, c in self.videos:
                metadata = self.__read_meta_data(v)
                '''
                cv2
                cv_vid = cv2.VideoCapture(v)
                fps = cv_vid.get(cv2.CAP_PROP_FPS)
                frame_cnt = int(cv_vid.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_cnt / fps
                '''
                fps = metadata['fps']
                duration = metadata['duration']
                nframes = ceil(fps * duration)
                meta.append((fps, duration))
                v = os.path.basename(v)
                c = self.classes[c]
                line = ','.join(list(map(str, [v, c, fps, nframes, duration]))) + '\n'
                f.write(line)

        return meta

    def __sampling_frames(self, fps=1):
        frame_root = '/DB/VCDB/frames'
        fr = []
        cnt = []
        for i in range(len(self)):
            cls, name, origin_fps, frame_cnt, duration = self.__getitem__(i)
            name = name.split('.')[0]
            frames = os.listdir(os.path.join(frame_root, cls, name))
            frames.sort()
            intv = int(round(origin_fps) / fps)
            if fps > 0:
                sample_frames = [os.path.join(frame_root, cls, name, frames[f]) for f in
                                 range(int(intv / 2), len(frames), intv)]
            else:
                sample_frames = [os.path.join(frame_root, cls, name, frames[f]) for f in range(0, len(frames))]
            fr.append(sample_frames)
            cnt.append(len(sample_frames))
        return fr, cnt

    def extract_frames_ffmpeg(self, target):
        for i in range(self.__len__()):
            cls, name, _, _, _ = self.__getitem__(i)
            dst = os.path.join(target, cls, name.split('.')[0])
            src = os.path.join(self.video_root, cls, name)
            if not os.path.exists(dst):
                os.makedirs(dst)
            args = '-i {} -f image2 {}/%5d.jpg'.format(src, dst)
            ret, out, err = execute_ffmpeg(args)
            print(i, ret, cls, name)

    # make minimum segment feature
    def extract_cnn_features(self, target='/DB/VCDB/features/fps_1_resnet50'):
        # model=Resnet50_RMAC()
        model = resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
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

        with torch.no_grad():
            for i in range(self.__len__()):
                cls, name, origin_fps, frame_cnt, duration = self.__getitem__(i)
                frames = self.frames[i]
                frames_cnt = self.frames_cnt[i]
                dl = DataLoader(ListDataset(frames, transform=video_trn), batch_size=256, num_workers=4)
                dst = os.path.join(target, cls, name.split('.')[0] + '.pt')
                print(dst)
                if not os.path.exists(os.path.join(target, cls)):
                    os.makedirs(os.path.join(target, cls))
                frame_feature = []
                for n, (im, path) in enumerate(dl):
                    out = model(im.cuda()).squeeze(-1).squeeze(-1)
                    frame_feature.append(out)
                    print('{:3d}: extract vid: {}/{} duration: {} shape: {}'.format(i, cls, name, duration, out.shape))
                frame_feature = torch.cat(frame_feature)
                # video_feature = torch.mean(frame_feature, dim=0, keepdim=True)

                frame_feature = frame_feature.cpu()
                # video_feature = video_feature.cpu()

                torch.save(frame_feature, dst)


if __name__ == '__main__':
    # db=VCDB_CORE_VIDEOS(fps=1)
    # db.extract_cnn_features()
    db = VCDB()

    exit()

    # print(db.core_videos_db.video_list)

    # db = VCDB_CORE_VIDEOS()
    # db.extract_frames_ffmpeg(target='/DB/VCDB/frames')
    # db = VCDB()
    # video_db = db.core_videos_db
    # print(video_db.frames)
    # video_db.extract_cnn_features()
    # print(video_db.frames_cnt)
