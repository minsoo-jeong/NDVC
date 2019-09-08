import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms as trn
import torch
import os

from dataset.cc_web_video import CC_WEB_VIDEO


class ListDataset(data.Dataset):
    def __init__(self, data_list, transform=None):
        self.samples = data_list

        self.loader = default_loader

        self.transform = transform

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample,path

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return str(self.__len__())


class DirDataset(data.Dataset):
    def __init__(self, dir, transform=None):
        self.dir = os.path.abspath(dir)
        self.samples = self._get_imlist()
        self.loader = default_loader
        self.transform = transform

    def _get_imlist(self):
        l = [os.path.join(self.dir, d) for d in os.listdir(self.dir)]
        l.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        return l

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = '>> PATH : {}\n'.format(self.dir)
        fmt_str += '>> LEN : {}\n'.format(self.__len__())
        return fmt_str

    def get_vidname(self):
        return os.path.dirname(self.dir).split('/')[-1]


if __name__ == '__main__':
    base = '/DB/VCDB/frame_1_per_sec/frames'
    db = CC_WEB_VIDEO()
    db.get_VideoList()
    videos = os.listdir(base)
    videos.sort(key=lambda x: int(x))
    normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    video_trn = trn.Compose([
        trn.Resize(224),
        trn.ToTensor(),
        normalize
    ])

    for vid in videos:
        dt = DirDataset(os.path.join(base, vid),video_trn)
        dl = DataLoader(dt, batch_size=4, num_workers=2)
        for i, (im, path) in enumerate(dl):
            print(i,path)
