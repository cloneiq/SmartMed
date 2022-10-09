import torch
import torch.nn as nn
import torch.nn.functional as F
from imgrepgen.DGLNet.models.res_nets import ResNets
from modelfactory.attention.CBAM import ChanAttn, SpatAttn
from torchvision import transforms


# 简单（全局局部）融合网络
class SmpGLNet(nn.Module):
    def __init__(self, backbone_name='resnet50', chan_attn=False, spat_attn=False):
        super(SmpGLNet, self).__init__()
        self.glb_net = ResNets.create_model(model_name=backbone_name)
        self.loc_net = ResNets.create_model(model_name=backbone_name)
        self.enc_dim = 256 * 4 * 2

    def forward(self, images):
        # 从组合的图像中分离出全局图和局部图
        img_len = images.shape[1]
        # 全局图，局部图块
        glb_imgs, loc_imgs = images.split([1, img_len - 1], dim=1)
        # 全局图
        g_b, g_pcs, g_c, g_w, g_h = glb_imgs.size()
        glb_imgs = glb_imgs.reshape(-1, g_c, g_w, g_h)
        g2, g3, g4, glb_vis = self.glb_net(glb_imgs)
        _, c, h, w = glb_vis.size()
        glb_vis = glb_vis.view(g_b, -1, c, h, w)
        # 局部图块
        l_b, l_pcs, l_c, l_w, l_h = loc_imgs.size()
        loc_imgs = loc_imgs.reshape(-1, l_c, l_w, l_h)
        l2, l3, l4, loc_vis = self.loc_net(loc_imgs)
        _, c, h, w = loc_vis.size()
        loc_vis = loc_vis.view(l_b, -1, c, h, w)
        # 简单融合-拼接
        vis_out = torch.cat((glb_vis, loc_vis), dim=1)
        _, _, c, h, w = vis_out.size()
        vis_out = vis_out.view(-1, c, h, w)
        vis_out = vis_out.permute(0, 3, 2, 1)
        return vis_out
