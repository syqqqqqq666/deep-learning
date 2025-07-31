import  torchvision
from torch.utils.tensorboard import SummaryWriter
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),

])
'''
将 PIL 图像（H×W×C，范围 [0, 255]）或 NumPy 数组（dtype=np.uint8）转换为 PyTorch 张量（C×H×W，范围 [0.0, 1.0]）
使用torchvision.transforms.Compose将多个图像变换操作组合在一起。在这个例子中，只包含了一个变换：
ToTensor()，它的作用是将 PIL 图像或 NumPy 数组转换为 PyTorch 张量（Tensor）。
'''
train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transform ,download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=dataset_transform, download=True)
# print(test_set[0])
# print(test_set.classes)
# img,target = test_set[0]
# print(img)
# print(target)
# img.show()
writer = SummaryWriter("test_cifar")
for i  in range(10):
    img, target = train_set[i]
    writer.add_image("img", img, i)

