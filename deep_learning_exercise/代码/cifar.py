import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_data,batch_size = 64 , shuffle = True , num_workers=0,drop_last=True)

#测试数据集中第一张图片及target
img,target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("logs")
step = 0
#
for data in test_loader:
    imgs, targets = data
    writer.add_images("img", imgs, step)
    '''
    你的张量形状是[64, 3, 32, 32]（64 张 3 通道 32×32 的图像）
    add_image默认期望的格式是CHW（通道、高度、宽度），适合单张图像
    批量图像（4D 张量）需要特殊处理
    add_image方法期望 3D 张量（单张图像），但你传入了 4D 张量（批次图像）。解决方案有：使用add_images处理 4D 张量
    '''
    step = step + 1
writer.close()