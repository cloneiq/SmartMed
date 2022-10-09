import os


class DataReader(object):
    def __init__(self, data_file):
        self.data_file = data_file
        self.images = []
        self.labels = []

    def load_data(self):
        img_names = []
        img_labels = []
        # 将图像名和图像标签对应存储起来
        with open(self.data_file, "r") as f:
            for line in f:
                items = line.split()
                # 获取图像名称
                img_names.append(items[0])
                # 获取图像对应的标签
                label = items[1:]
                label = [int(i) for i in label]
                img_labels.append(label)
        self.images = img_names
        self.labels = img_labels
        return self.images, self.labels
