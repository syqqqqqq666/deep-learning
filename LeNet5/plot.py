import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np

train_data=FashionMNIST(root='./data',train=True,
                        transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=224)]),
                        download=True)

train_loader=torch.utils.data.DataLoader(dataset=train_data,
                                         batch_size=64,
                                         shuffle=True,
                                         num_workers=0)
for step, (input, label) in enumerate(train_loader):
    if step>0:
        break
batch_x = input.squeeze().numpy()
batch_y = label.numpy()
print(batch_y)
class_label = train_data.classes
print(class_label)