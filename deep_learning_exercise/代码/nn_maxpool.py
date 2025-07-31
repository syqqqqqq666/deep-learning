import torch
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torchvision import transforms


input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]],dtype=torch.float32)
input  = torch.reshape(input,(-1,1,5,5))

class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpoll1 = MaxPool2d(kernel_size = 3,ceil_mode = True)
    def forward(self, input):
        output = self.maxpoll1(input)
        return output


test = Test()
output = test(input)
print(output)