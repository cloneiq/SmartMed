import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import json
from datasets.nlmcxr.utils.build_vocab import Vocabulary, JsonReader
import numpy as np
import pickle
from datasets.dsutils.data_reader import DataReader


# 自定义的类必须继承Dataset类
class AdvLoader(Dataset):
    # 图像文件目录
    # json文件路径，图像编号-报告内容
    # 文件列表
    # 词典数据
    # 报告中最大句子数
    # 句子中最大单词数
    def __init__(self, image_dir, caption_json, data_file, vocabulary, s_max=10, n_max=50, transforms=None):
        super(AdvLoader, self).__init__()
        self.data_file = data_file
        self.image_dir = image_dir
        # 图像编号-报告原文（findings and impressention）
        self.captions = JsonReader(caption_json)
        # 图像编号-报告0-1标签（已提取好的）
        self.images, self.labels = DataReader(self.data_file).load_data()
        self.vocab = vocabulary
        self.transform = transforms
        self.s_max = s_max  # 报告中最大句子数量
        self.n_max = n_max  # 最大单词数

    def __getitem__(self, index):
        # 获取一个图像名称（带扩展名）
        image_name = '{}.pngs'.format(self.images[index])
        # 将图像转换为RGB，并加载到image中
        org_image = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')
        # 获取该图像对应的标签值（Int类型）
        label = self.labels[index]
        if self.transform is not None:
            org_image = self.transform(org_image)
        # 获取图像对应的报告全文
        try:
            caption = self.captions.data[image_name]
        except Exception as err:
            caption = 'normal. '
        # 当前报告的所有句子对应的单词列表
        target = list()
        # 报告中所有句子最大的单词数量
        max_word_num = 0
        # 以.号分割句子
        for i, sentence in enumerate(caption.split('. ')):
            # 如果到达最大句子数量，则退出
            if i >= self.s_max:
                break
            # 将句子切片，分成单词（对中文要进行分词）
            sentence = sentence.split()
            # 若单词数满足下列条件，则该条语句不处理，继续处理吓一条语句
            if len(sentence) == 0 or len(sentence) == 1 or len(sentence) > self.n_max:
                continue
            tokens = list()
            # print("vocab:",self.vocab)
            # 每一个句子以'<start>'开始
            tokens.append(self.vocab('<start>'))
            # 将句子中单词加入列表，在列表末尾一次性追加另一个序列中的多个值
            # for token in sentence:
            #     print("tokens", token)
            #     print("tokens-id", self.vocab(token))
            # 从句中取出每一个词，并根据词典找到词对应字典编号
            tokens.extend([self.vocab(token) for token in sentence])
            # 每一个句子以'<end>'结束
            tokens.append(self.vocab('<end>'))
            # tokens表示一个句子对应的所有词
            if max_word_num < len(tokens):
                max_word_num = len(tokens)
            # 将当前句子的词列加入列表，tokens存放词对应的编号
            target.append(tokens)
        # 计算当前报告的句子数
        sentence_num = len(target)
        # np.sum(label)统计label为1个数
        # label / np.sum(label)，每个标签值都除以标签为1个数
        # list(label / np.sum(label))的目的是什么？
        # print("image_name:",image_name)
        # print("targets:",target)
        # return：图像内容，图像名称，？，句子列表（每个句子对应一个单词列表）,句子数量，最大的单词数量
        # print("Data loader-------------------------------------")
        # print("Image Name:{}".format(image_name))
        # print("Labels:{}".format(label))
        # print("Target:{}".format(target))

        return org_image, image_name, list(label / np.sum(label)), target, sentence_num, max_word_num

    # 所有图像的格式
    def __len__(self):
        return len(self.images)


def collate_fn(data):
    # 从元组列表（图像，标题，no_of_sent，max_sent_len）的列表创建小批量张量。
    # 自定义的collat​​e_fn,默认的collat​​e_fn不支持合并字幕（包括填充）。
    # 参数
    # -数据：元组列表（图像，标题，no_of_sent，max_sent_len）。
    # -图片：形状为（3，crop_size，crop_size）的张量。
    # -标题：形状为张量（no_of_sent，max_sent_len）； 可变长度。
    # -no_of_sent：字幕中的句子数
    # -max_sent_len：字幕中句子的最大长度
    # 按标题长度（降序）对数据列表进行排序。
    # data.sort（键 = lambda x：len（x[1]），reverse = True）
    images, image_id, label, captions, sentence_num, max_word_num = zip(*data)
    # 合并图像（从3D张量元组到4D张量元组）
    images = torch.stack(images, 0)

    # print(captions)，captions每个报告中的词列表（字典编号）
    # print("image_id:",image_id)
    # print("label:", label)
    # images的形状为:[batch_size,3,244,244],即batch_size个图像（报告），
    # batch_size中最大的句子数
    max_sentence_num = max(sentence_num)
    # batch_size中最大的单词数
    max_word_num = max(max_word_num)
    # 初始化一个targe列表，形状为[captions,max_sentence_num + 1,max_word_num],
    # 即每一个报告是一个（max_sentence_num + 1）*max_word_num的矩阵
    targets = np.zeros((len(captions), max_sentence_num + 1, max_word_num))
    # prob size:(4, 6)
    prob = np.zeros((len(captions), max_sentence_num + 1))
    # print("prob size:{}".format(prob.shape))
    for i, caption in enumerate(captions):
        # print("caption:",caption)
        # print("------------------------------")
        for j, sentence in enumerate(caption):
            # print("sentence:",sentence[:])
            # 行表示句子数，列表示该句子对应的单词
            # print("Sentences:", sentence[:])
            targets[i, j, :len(sentence)] = sentence[:]
            # print("Targets:".format(targets))
            # 控制句子是否结束，如果句子长度为零，则说明后面没有句子了
            # prob size:(4, 6)，表示：batch_size:4,样本最多有6个句子，
            # 其中，可能有些样本没有6个句子，有几个句子，记录几个1，遇到零表示句子结束
            prob[i][j] = len(sentence) > 0
            # print("prob:", prob)
        # print("-----------------------------")
    # Targets size:(4, 6, 18) (batch_size,sentences_num,words)
    # print("Targets size:{}".format(targets.shape))
    # 返回值：
    # 图像：形状为（batch_size，3，crop_size，crop_size）的张量。
    # 目标：形状的割炬张量（batch_size，max_no_of_sent，padded_max_sent_len）。
    # prob：形状的割炬张量（batch_size，max_no_of_sent）
    return images, image_id, torch.Tensor(label), targets, prob


def get_loader(image_dir, caption_json, data_file, vocabulary, transform, batch_size,
               s_max=10,
               n_max=50,
               shuffle=False):
    dataset = AdvLoader(image_dir=image_dir,
                        caption_json=caption_json,
                        data_file=data_file,
                        vocabulary=vocabulary,
                        s_max=s_max,
                        n_max=n_max,
                        transforms=transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader


def get_vocab(vocab_name):
    with open(vocab_name, 'rb') as f:
        vocab_dic = pickle.load(f)
    return vocab_dic


if __name__ == '__main__':

    vocab_path = 'reports/debug_vocab.pkl'
    image_dir = 'images'
    caption_json = 'reports/debug_captions.json'
    data_file = 'reports/debug_data.txt'
    batch_size = 6
    resize = 256
    crop_size = 224

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        # print(vocab.word2idx)

    data_loader = get_loader(image_dir=image_dir,
                             caption_json=caption_json,
                             data_file=data_file,
                             vocabulary=vocab,
                             transform=transform,
                             batch_size=batch_size,
                             shuffle=False)

    for i, (image, image_id, label, target, prob) in enumerate(data_loader):
        print(len(label[0]))
        print(image.shape)
        # print(image_id)
        # print(label)
        # print("target:----", target)
        # print(prob)
        break
