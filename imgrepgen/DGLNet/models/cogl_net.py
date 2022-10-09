import torch
import torch.nn as nn
from imgrepgen.DGLNet.models.glb_net import GlbNet
from imgrepgen.DGLNet.models.smpgl_net import SmpGLNet
from imgrepgen.DGLNet.models.cbamgl_net import CbamGLNet
from imgrepgen.DGLNet.models.cog2l_net import CoG2LNet
from imgrepgen.DGLNet.models.col2g_net import CoL2GNet
from imgrepgen.DGLNet.models.attgl_net import AttGLNet
from imgrepgen.DGLNet.models.coglg_net import CoGLGNet
from imgrepgen.DGLNet.models.res_nets import ResNets


class CoGLNet(nn.Module):
    def __init__(self, backbone_name='resnet50', chan_attn=False, spat_attn=False, mode_name='glb', patch_size=384):
        super(CoGLNet, self).__init__()
        self.mode_name = mode_name
        self.patch_size = patch_size
        self.cogl_net, self.enc_dim = self._init_cogl_model(backbone_name, chan_attn, spat_attn)
        # glb:全局模式，传统模式
        # loc:局部模式，传统模式
        # smpgl:简单全局与局部模式(cat)
        # cbamgl:简单全局与局部模式(CBAM注意力之后再cat)
        # cog2l:全局优化局部模式，创新模式
        # col2g:局部优化全局模式，创新模式
        # coglg:全局与局部特征协同融合模式， 创新模式
        # self.mode = mode

    def _init_cogl_model(self, backbone_name, chan_attn, spat_att):
        print('The model name is {}'.format(self.mode_name))
        colg_net = None
        if self.mode_name.lower() == 'glb' or self.mode_name == 'loc':
            colg_net = GlbNet(backbone_name, chan_attn, spat_att)
        elif self.mode_name.lower() == 'smpgl':
            colg_net = SmpGLNet(backbone_name, chan_attn, spat_att)
        elif self.mode_name.lower() == 'attgl':
            colg_net = AttGLNet(backbone_name, chan_attn, spat_att)
        elif self.mode_name.lower() == 'cbamgl':
            colg_net = CbamGLNet(backbone_name, chan_attn, spat_att)
        elif self.mode_name.lower() == 'cog2l':
            colg_net = CoG2LNet(backbone_name, chan_attn, spat_att)
        elif self.mode_name.lower() == 'col2g':
            colg_net = CoL2GNet(backbone_name, chan_attn, spat_att)
        elif self.mode_name.lower() == 'coglg':
            colg_net = CoGLGNet(backbone_name, chan_attn, spat_att)
        enc_dim = colg_net.enc_dim
        return colg_net, enc_dim

    def forward(self, images, csys=None, ratios=None):
        return self.cogl_net.forward(images=images, csys=csys, ratios=ratios)
