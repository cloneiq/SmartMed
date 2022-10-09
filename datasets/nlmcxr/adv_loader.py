from datasets.nlmcxr.utils.build_vocab import Vocabulary, JsonReader
from datasets.dsutils.data_reader import DataReader
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os
import json
import numpy as np
import pickle


# 自定义的类必须继承Dataset类
class AdvLoader(Dataset):
    # json文件路径，图像编号-报告内容
    # 报告中最大句子数
    # 句子中最大单词数
    def __init__(self, mode='train', media_type='jpg', s_max=10, n_max=50, transform=None):
        super(AdvLoader, self).__init__()
        self.root_dir = os.path.split(os.path.realpath(__file__))[0]
        self.media_type = media_type
        self.image_dir = os.path.join(self.root_dir, '{}s'.format(media_type))
        self.report_dir = os.path.join(self.root_dir, 'reports')
        if mode == 'debug':
            self.caption_json = os.path.join(self.report_dir, 'debug_captions.json')
            self.vocab_file = os.path.join(self.report_dir, 'debug_vocab.pkl')
        else:
            self.caption_json = os.path.join(self.report_dir, 'captions.json')
            self.vocab_file = os.path.join(self.report_dir, 'vocab.pkl')
        self.data_file = os.path.join(self.report_dir, '{}_data.txt'.format(mode))
        # 图像编号-报告原文（findings and impressention）
        self.captions = JsonReader(self.caption_json)
        # 图像编号-报告0-1标签（已提取好的）
        self.images, self.labels = DataReader(self.data_file).load_data()
        with open(self.vocab_file, 'rb') as f:
            self.vocab = pickle.load(f)
        self.transform = transform
        self.s_max = s_max  # 报告中最大句子数量
        self.n_max = n_max  # 最大单词数

    def __getitem__(self, index):
        # 获取一个图像名称（带扩展名）
        image_name = self.images[index]
        # image_name = '{}.{}'.format(self.images[index], self.media_type)
        # 将图像转换为RGB，并加载到image中，原始图像
        org_image = Image.open(os.path.join(self.image_dir, '{}.{}'.format(image_name, self.media_type))).convert('RGB')
        # 获取该图像对应的标签值（Int类型）
        label = self.labels[index]
        if self.transform is not None:
            # 图像尺寸发生变化（）
            trans_images = self.transform(org_image)
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
        # print("Data loader-------------------------------------")
        # print("Image Name:{}".format(image_name))
        # print("Labels:{}".format(label))
        # print("Target:{}".format(target))
        # return：图像内容，图像名称，？，句子列表（每个句子对应一个单词列表）,批量中的句子数量，句子最大长度
        return trans_images, image_name, list(label / np.sum(label)), target, sentence_num, max_word_num

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
    images, image_name, label, captions, sentence_num, max_word_num = zip(*data)

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
    return images, image_name, torch.Tensor(label), targets, prob


def dsLoader(mode='train', media_type='jpg', transform=None, batch_size=64, s_max=10, n_max=50,
             shuffle=False):
    ds = AdvLoader(mode=mode, media_type=media_type,
                   s_max=s_max, n_max=n_max, transform=transform)
    dt_loader = DataLoader(dataset=ds, batch_size=batch_size,
                           shuffle=shuffle, collate_fn=collate_fn, drop_last=True)
    return dt_loader


def ds_vocab(mode='train'):
    root_dir = os.path.split(os.path.realpath(__file__))[0]
    report_dir = os.path.join(root_dir, 'reports')
    if mode == 'debug':
        vocab_file = os.path.join(report_dir, 'debug_vocab.pkl')
    else:
        vocab_file = os.path.join(report_dir, 'vocab.pkl')
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


if __name__ == '__main__':

    b_size = 6
    resize = 256
    crop_size = 224
    tfs = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    # loader = SimpleLoader(data_dir, run_model='train', s_max=10, n_max=50, transforms=None)
    data_loader = dsLoader(mode='train',
                           transform=tfs,
                           batch_size=b_size,
                           shuffle=False)
    vocab = ds_vocab(mode='train')

    print(vocab.idx2word[13])
    print(vocab.word2idx['disease'])

    for i, (image, image_name, label, target, prob) in enumerate(data_loader):
        print(len(label[0]))
        print(image.shape)
        # print(image_name)
        # print(label)
        print("target:----", target)
        # print(prob)
        break
