import pickle
from collections import Counter
import json


# 读取JSON数据，key:图像名称，value:报告内容
class JsonReader(object):
    def __init__(self, json_file):
        # data：{'image_name':'report text'}
        self.data = self.__read_json(json_file)
        self.keys = list(self.data.keys())
        # print(self.keys)

    def __read_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    def __getitem__(self, item):
        # t = self.keys[item]
        # print(self.data[item])
        # return self.data[item]
        return self.data[self.keys[item]]

    def __len__(self):
        return len(self.data)


# 词典类
class Vocabulary(object):
    def __init__(self):
        # 字典类型变量{key,value}形式,{word,id}
        self.word2idx = {}
        # 字典类型变量{key,value}形式,{id,word}
        self.idx2word = {}
        self.idx = 0
        self.add_word('<pad>')
        self.add_word('<end>')
        self.add_word('<start>')
        self.add_word('<unk>')

    # 向字典中添加一个词
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    # 根据词的ID取得词
    def get_word_by_id(self, id):
        return self.idx2word[id]

    # 将类实例转变成一个可调用对象
    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
        # 将类实例转变成一个可调用对象

    # 字典的大小
    def __len__(self):
        return len(self.word2idx)


def build_vocab(json_file, threshold):
    caption_reader = JsonReader(json_file)
    # print("captions",caption_reader[0])
    # 通过Counter进行词频统计
    counter = Counter()
    # 对所有的报告做词频统计
    for items in caption_reader:
        # 将.号和，号有''代替
        text = items.replace('.', '').replace(',', '')
        # 将文本用空格' '分割，并统计词频，如{'xxxx': 3, 'the': 3, 'airspace': 2, 'opacity': 2,...}
        counter.update(text.lower().split(' '))
    # counter.items()以字典的方式返回如[('xxxx', 3), ('airspace', 2),..]
    # 取出词计数大于阈值的词（threshold）
    words = [word for word, cnt in counter.items() if cnt > threshold and word != '']
    # 构建词典
    vocab = Vocabulary()
    # 遍历计数大于阈值的词
    for word in words:
        # 将词添加到词典中
        vocab.add_word(word)
    return vocab


def main(json_file, threshold, vocab_path):
    vocab = build_vocab(json_file, threshold)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Total vocabulary size:{}".format(len(vocab)))
    print("Saved path in {}".format(vocab_path))
    with open(vocab_path, 'rb') as f:
        info = pickle.load(f)

    print(info.idx2word[10])


if __name__ == '__main__':
    main(json_file='../reports/debug_captions.json',
         threshold=0,
         vocab_path='../reports/debug_vocab.pkl')
