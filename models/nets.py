from torchvision import models
from models.pooling import RMAC, L2N
import torch
from torchsummary import summary


class Resnet50_RMAC(torch.nn.Module):
    def __init__(self, ):
        super(Resnet50_RMAC, self).__init__()
        self.base = torch.nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        self.pool = RMAC()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x


if __name__ == '__main__':
    model = Resnet50_RMAC().cuda()
    summary(model, (3, 224, 224))
