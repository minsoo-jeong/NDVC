from torch.utils.data import DataLoader
from torchvision.transforms import transforms as trn
from torchvision.models import resnet18, resnet50
from torchsummary import summary
import torch
import os

from models.nets import Resnet50_RMAC
from dataset.ListDataset import DirDataset
"""
base =/DB/{db_name}/{segments}/frames/
save =/DB/{db_name}/{segments}/{model_name}/{f-features or v-features}/{VideoID}
"""
# base = '/DB/CC_WEB_VIDEO/frame_1_per_sec/frames'
#save = "/DB/CC_WEB_VIDEO/all_frames/resnet50-rmac"
base = '/DB/VCDB/all_frames/frames'
save = "/DB/VCDB/all_frames/resnet50"

if not os.path.exists(base):
    print("base '{}' is not exist.".format(base))
    exit()
if not os.path.exists(save):
    os.makedirs(os.path.join(save,'v-features'))
    os.makedirs(os.path.join(save, 'f-features'))

videos = os.listdir(base)
videos.sort(key=int)
videos=videos[200:]
normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
video_trn = trn.Compose([
    trn.Resize(224),
    trn.ToTensor(),
    normalize
])

model = resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # 2048 - dimension
#model = Resnet50_RMAC()
model.cuda()
model = torch.nn.DataParallel(model)
summary(model, (3, 224, 224))
model.eval()

with torch.no_grad():
    vfeatures = []

    for idx, vid in enumerate(videos):
        dt = DirDataset(os.path.join(base, vid), video_trn)
        dl = DataLoader(dt, batch_size=64, num_workers=0)
        frame_feature = []
        for i, (im, path) in enumerate(dl):
            out = model(im.cuda())
            frame_feature.append(out.squeeze(-1).squeeze(-1))
            print('{} : extract vid: {}/ shape: {}'.format(idx, vid, out.shape))
        frame_feature = torch.cat(frame_feature)
        video_feature = torch.mean(frame_feature, dim=0, keepdim=True)
        # save frame feature
        frame_feature = frame_feature.cpu()
        video_feature = video_feature.cpu()
        torch.save(frame_feature, os.path.join(save, 'f-features', '{}.pt'.format(vid)))
        # save video features
        torch.save(video_feature, os.path.join(save, 'v-features', '{}.pt'.format(vid)))
        # accumulate video features
        vfeatures.append(video_feature)
    vfeatures = torch.cat(vfeatures)
    torch.save(vfeatures, os.path.join(save, 'v-features.pt'))
