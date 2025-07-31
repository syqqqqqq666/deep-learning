from torch.utils.data import Dataset
from  PIL import Image
import os
class MyData(Dataset):
    #初始化
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_path = os.path.join(self.path, img_name)
        img = Image.open(img_path) #获取图片
        label = self.label_dir
        return img, label

    #返回数据集长度
    def __len__(self):
        return len(self.img_path)

root_dir="hymenoptera_data/train"
ants_label_dir ="ants"
bees_label_dir ="bees"
ants_data = MyData(root_dir,ants_label_dir)
