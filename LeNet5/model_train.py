import copy
import time

import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as data
import pandas as pd
from model import LeNet
import os
#数据加载
def train_val_data_process():
    train_data=FashionMNIST(root='./data',
                            train=True,
                            transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=28)]),
                             download=True)
    train_data,val_data = data.random_split(train_data,[round(len(train_data)*0.8),round(len(train_data)*0.2)])
    train_loader = data.DataLoader(dataset=train_data,
                                   batch_size=32,
                                   shuffle=True,
                                   num_workers=0)
    val_loader=data.DataLoader(dataset=val_data,
                               batch_size=32,
                               shuffle=True,
                               num_workers=0)
    return train_loader,val_loader

def train_model_process(model,train_loader,val_loader,num_epochs):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #交叉熵损失，多用于分类问题
    criterion = nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)#梯度下降算法
    model = model.to(device)
    #复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())
    #最高精度
    best_acc = 0.0

    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch{epoch+1}/{num_epochs}')
        print('-'*10)
        #训练集损失函数
        train_loss=0.0
        #训练集准确度
        #训练集样本数量
        train_num=0
        train_acc=0.0
        #训练集损失列表
        val_loss=0.0
        val_acc=0.0
        #验证集样本数量
        val_num=0

        #data 为128*28*28*1
        #target 128*label
        #按批次去取
        for batch_idx, (data, target) in enumerate(train_loader):
            #数据放到设备上
            data, target = data.to(device), target.to(device)
            #模型为训练模式
            model.train()
            #前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            #输出十个值
            output = model(data)
            #查找十个值中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()#更新
            #loss.item是平均值
            train_loss += loss.item()*data.size(0)
            train_acc += torch.sum(pre_lab==target.data)
            train_num += data.size(0)
        for step, (data, target) in enumerate(val_loader):
            data,target = data.to(device),target.to(device)
            model.eval()
            output = model(data)
            #查找每一行最大值对应的坐标
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, target)
            val_loss += loss.item()*data.size(0)
            val_acc += torch.sum(pre_lab==target.data)
            val_num += data.size(0)
        train_loss_all.append(train_loss/train_num)#该轮次的训练集loss值
        val_loss_all.append(val_loss/val_num)#一轮次的验证集损失
        train_acc_all.append(train_acc.double().item() /train_num)#一轮次的训练精度值
        val_acc_all.append(val_acc.double().item() /val_num)#一轮次的验证精度值
        print('{} Train Loss:{:.4f} Train Acc:{:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print('{} Val Loss:{:.4f} Val Acc:{:.4f}'.format(epoch,val_loss_all[-1],val_acc_all[-1]))

        #寻找最高准确度的权重参数
        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]
            #保存当前参数
            best_model_wts = copy.deepcopy(model.state_dict())
        #计算训练耗时
        time_used=time.time()-since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_used//60,time_used%60))

    #选择最优参数
    #加载最高准确率下的模型参数
    save_dir = "./LeNet"
    if not os.path.exists(save_dir):
        # 如果目录不存在，则创建它（包括可能的父目录）
        os.makedirs(save_dir, exist_ok=True)

    # 然后再保存模型
    torch.save(best_model_wts, os.path.join(save_dir, "best_model.pth"))

    train_process=pd.DataFrame(data={"epoch":range(num_epochs),
                                    "train_loss_all":train_loss_all,
                                     "val_loss_all":val_loss_all,
                                     "train_acc_all":train_acc_all,
                                     "val_acc_all":val_acc_all})
    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process["train_loss_all"],'bs-',label='train_loss')
    plt.plot(train_process["epoch"],train_process["val_loss_all"],'bs-',label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc_all"], 'ro-', label='train_acc')
    plt.plot(train_process["epoch"], train_process["val_acc_all"], 'ro-', label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show()

if __name__=='__main__':

    #模型实例化
    LeNet =LeNet()
    train_dataloader,val_dataloader = train_val_data_process()
    train_process= train_model_process(LeNet,train_dataloader,val_dataloader,20)
    matplot_acc_loss(train_process)

