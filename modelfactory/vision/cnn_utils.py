from torch.hub import download_url_to_file
from torch.hub import urlparse
import torch.utils.model_zoo as model_zoo
import os
import re
from collections import OrderedDict
model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'model_files')
# 预训练模型下载地址
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'mnasnet0_5': 'https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth',
    'mnasnet1_0': 'https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth',
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth'
}


#
def gen_model_file_name(model_name):
    model_url = model_urls[model_name]
    parts = urlparse(model_url)
    return os.path.basename(parts.path)


def download_model(model_name):
    model_url = model_urls[model_name]
    filename = gen_model_file_name(model_name)
    hash_regex = re.compile(r'-([a-f0-9]*)\.')
    hash_prefix = hash_regex.search(filename).group(1)
    filename = os.path.join(model_dir, filename)
    download_url_to_file(model_url, filename, hash_prefix, True)
    return filename


def trsf_state_dict(src_model_dict, target_model_dict):
    # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    state_dict = {}
    for k, v in src_model_dict.items():
        if k in target_model_dict.keys():
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict


def load_state_dict(model_name):
    return model_zoo.load_url(model_urls[model_name], model_dir=model_dir)


def transfer_model(src_model, target_model):
    pre_model_dict = src_model.state_dict()
    target_model_dict = target_model.state_dict()  # get model dict
    # 在合并前(update),需要去除pretrained_dict一些不需要的参数
    state_dict = {}
    for k, v in pre_model_dict.items():
        if k in target_model_dict.keys():
            # state_dict.setdefault(k, v)
            # print("v:-------", v)
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    target_model_dict.update(state_dict)  # 更新(合并)模型的参数
    target_model.load_state_dict(target_model_dict)
    return target_model


def remove_module(state_dict):
    model_state_dict = OrderedDict()
    for pname in state_dict:
        name = pname
        if name[:7] == 'module.':
            name = pname[7:]  # remove 'module.'
        model_state_dict[name] = state_dict[pname]
    return model_state_dict


def add_module(state_dict):
    model_state_dict = OrderedDict()
    for pname in state_dict:
        name = pname
        if name[:7] != 'module.':
            name = 'module.' + name  # add 'module.'
        model_state_dict[name] = state_dict[pname]
    return model_state_dict


# dict = load_state_dict('resnet18')
# print(dict)
# file_name = gen_model_file_name('resnet18')
# # file_name = download_model('resnet18')
# print(file_name)
# print(dict)
# src_model = cnn_models.load_model("densenet121")
# 显示模型结构
# print(src_model)
# 查看模型参数
# for i, param in enumerate(src_model.parameters()):
#     print("model param:",param)
# 查看预训练参数
# for k in src_model.state_dict().keys():
#     print("预训练参数：", k)
# 修改模型结构
# out_features = src_model.fc.in_features
# print("out_features:", out_features)
# target_model = OrderedDict(src_model.named_children())
# 删除模型对应的层
# target_model.pop("avgpool")
# target_model.pop("fc")
# 增加一层
# target_model["func"] = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
# target_model["class"] = nn.Linear(out_features, 14)
# target_model = torch.nn.Sequential(target_model)
# print(target_model)
# for k in target_model.state_dict().keys():
#     print("修改后：", k)
# new_model = CnnModels.transfer_model(src_model, target_model)

# print(new_model)
