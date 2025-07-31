import torch
import torchvision
from tensorboard.program import TensorBoard
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from test_tb import writer

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transforms.ToTensor())
dataloader  = DataLoader(dataset, batch_size=64)
class test(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
    def forward(self, x):
        x = self.conv1(x)
        return x
test = test()
print(test)

writer = SummaryWriter("./logs")
step = 0
for data in dataloader:
    imgs,targets = data
    outputs = test(imgs)
    # print(imgs.shape)
    # print(outputs.shape)
    writer.add_images("input",imgs,step)
    outputs= torch.reshape(outputs,(-1,3,30,30))
    writer.add_images("output",outputs,step)
    step += 1

writer.close()