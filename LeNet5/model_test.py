import torch
import  torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from model import LeNet


def test_data_process():
    test_data=FashionMNIST(root='./data',
                            train=False,
                            transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=28)]),
                             download=True)
    test_loader = data.DataLoader(dataset=test_data,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=0)

    return test_loader

def test_model(model,test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_acc=0.0
    test_num = 0
    with torch.no_grad():
        for test_data_batch, test_label_batch in test_loader:
            test_data_batch = test_data_batch.to(device)
            test_label_batch = test_label_batch.to(device)
            model.eval()
            output = model(test_data_batch)
            test_acc += torch.sum(torch.argmax(output, dim=1) == test_label_batch.data)
            test_num += test_label_batch.size(0)
    test_acc=test_acc.double().item() /test_num
    print("测试准确率:",test_acc)

if __name__ == '__main__':
    model=LeNet()
    model.load_state_dict(torch.load('LeNet/best_model.pth'))
    test_loader=test_data_process()
    test_model(model,test_loader)