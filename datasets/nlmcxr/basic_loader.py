import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from datasets.dsutils.data_reader import DataReader


# 读取NIMCXR(IU-X-Ray)数据集，只加载影像与标签（可用于多标签分类）
class BasicLoader(Dataset):
    def __init__(self, root_dir, train=True, debug=False, transform=None):
        super(BasicLoader, self).__init__()
        self.root_dir = root_dir
        # 将图像名和图像标签对应存储起来
        if train:
            self.data_file = 'reports/train_data.txt'
        elif debug:
            self.data_file = 'reports/debug_data.txt'
        else:
            self.data_file = 'reports/val_data.txt'

        self.images, self.labels = DataReader(self.data_file).load_data()
        self.transform = transform

    # 重写这个函数用来进行图像数据的读取
    def __getitem__(self, item):
        # 获取图像名和标签
        img_name = self.images[item]
        label = self.labels[item]
        # 读入图像信息，图像存储路径
        img_dir = os.path.join(self.root_dir, "images")
        # 获取图像的路径
        img_name = '{}.pngs'.format(img_name)
        img_file = os.path.join(img_dir, img_name)
        image = Image.open(img_file).convert('RGB')
        # 处理图像数据
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    # 定义数据的处理方式
    data_transforms = {
        'train': transforms.Compose([
            # 将图像进行缩放，缩放为256*256
            transforms.Resize(256),
            # 在256*256的图像上随机裁剪出224*224大小的图像用于训练
            transforms.RandomResizedCrop(224),
            # 图像用于翻转
            transforms.RandomHorizontalFlip(),
            # 转换成tensor向量
            transforms.ToTensor(),
            # 对图像进行归一化操作
            # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # 生成Pytorch所需的DataLoader数据输入格式
    train_data = BasicLoader('../nlmcxr', debug=True, transform=data_transforms['train'])
    # print(train_Data)
    train_DataLoader = DataLoader(train_data, batch_size=10, shuffle=True,drop_last=True)
    # 验证是否生成DataLoader格式数据
    for data in train_DataLoader:
        inputs, labels = data
        print(inputs.shape)
        print(labels)