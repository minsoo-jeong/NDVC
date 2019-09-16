from torchvision import models
from models.pooling import RMAC, L2N
import torch
from torch import nn
from torchsummary import summary


class Resnet50_RMAC(torch.nn.Module):
    def __init__(self):
        super(Resnet50_RMAC, self).__init__()
        self.base = torch.nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        self.pool = RMAC()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x

class SimpleFC(torch.nn.Module):
    def __init__(self,normalize=True):
        super(SimpleFC, self).__init__()
        self.normalize=normalize
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(2048, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.norm = L2N()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        if self.normalize:
            x=self.norm(x)

        return x


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


if __name__ == '__main__':
    model=TripletNet(SimpleFC(normalize=True))
    print(model)