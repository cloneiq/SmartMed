import torch
from collections import OrderedDict
import os


class ModelUtils:
    def __init__(self, model_path, model_name=None):
        self.model_path = model_path
        self.model_name = model_name

    def load_state_dict(self, load_from='gpus-cpu', from_pu=None, to_pu=None):
        model_state_dict = OrderedDict()
        if load_from == 'gpus-cpu':
            # gpu -> cpu
            state_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage)
            if self.model_name:
                temp_dict = state_dict[self.model_name]
            else:
                temp_dict = state_dict
            for pname in temp_dict:
                name = pname
                if name[:7] == 'module.':
                    name = pname[7:]  # remove 'module.'
                model_state_dict[name] = temp_dict[pname]

        elif load_from == 'cpu-gpu':
            # cpu -> gpu(to_pu)
            state_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage.cuda(to_pu))
            if self.model_name:
                temp_dict = state_dict[self.model_name]
            else:
                temp_dict = state_dict
            for pname in temp_dict:
                name = pname
                if name[:7] != 'module.':
                    name = 'module.' + name  # add 'module.'
                model_state_dict[name] = temp_dict[pname]

        elif load_from == 'gpu-gpu':
            # gpu 1 -> gpu 0
            state_dict = torch.load(self.model_path, map_location={from_pu: to_pu})
            if self.model_name:
                model_state_dict = state_dict[self.model_name]
            else:
                model_state_dict = state_dict
        else:
            state_dict = torch.load(self.model_path)
            if self.model_name:
                model_state_dict = state_dict[self.model_name]
            else:
                model_state_dict = state_dict
        return model_state_dict

    @staticmethod
    def remove_module(state_dict):
        model_state_dict = OrderedDict()
        for pname in state_dict:
            name = pname
            if name[:7] == 'module.':
                name = pname[7:]  # remove 'module.'
            model_state_dict[name] = state_dict[pname]
        return model_state_dict

    @staticmethod
    def add_module(state_dict):
        model_state_dict = OrderedDict()
        for pname in state_dict:
            name = pname
            if name[:7] != 'module.':
                name = 'module.' + name  # add 'module.'
            model_state_dict[name] = state_dict[pname]
        return model_state_dict

    # 拷贝元模型参数数到目标模型
    @staticmethod
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


if __name__ == '__main__':
    m_path = "./train_best_loss.pth.tar"
    l_from = 'gpus-cpu'
    f_pu = ''
    t_pu = ''
    mu = ModelUtils(m_path)
    mod_name = 'cnn_model'
    statedict = mu.load_state_dict(l_from)
    # print(statedict[model_name])
    temp = mu.remove_module(statedict[mod_name])
    print(temp)
    # print(statedict)
    # for k in statedict.keys():
    #     print(k)
    # print(statedict.keys())
