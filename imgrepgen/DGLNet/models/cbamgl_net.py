import torch
import torch.nn as nn
import torch.nn.functional as F
from imgrepgen.DGLNet.models.res_nets import ResNets
from modelfactory.attention.CBAM import ChanAttn, SpatAttn
from torchvision import transforms


# 简单（全局与局部）融合网络（基于CBAM注意力机制）
class CbamGLNet(nn.Module):
    def __init__(self, backbone_name='resnet50', chan_attn=False, spat_attn=False):
        super(CbamGLNet, self).__init__()
        self.glb_net = ResNets.create_model(model_name=backbone_name)
        self.loc_net = ResNets.create_model(model_name=backbone_name)
        self.enc_dim = 256 * 4 * 2
        self.chan_attn = ChanAttn(self.enc_dim)
        self.spat_attn = SpatAttn()

    def forward(self, images, csys=None, ratios=None):
        # 可以分割出全局图和局部分割图
        # images size:[batch_size,patches,c,w,h]
        # 从组合的图像中分离出全局图和局部图
        img_len = images.shape[1]
        glb_imgs, loc_imgs = images.split([1, img_len - 1], dim=1)
        # 全局图
        g_b, g_pcs, g_c, g_w, g_h = glb_imgs.size()
        glb_imgs = glb_imgs.reshape(-1, g_c, g_w, g_h)
        g2, g3, g4, glb_vis = self.glb_net(glb_imgs)
        # 全局图只考虑空间注意力，关注在哪里的问题
        # glb_att = self.chan_attn(glb_vis) * glb_vis
        glb_att = self.spat_attn(glb_vis) * glb_vis
        glb_vis = glb_vis + glb_att
        _, c, h, w = glb_vis.size()
        glb_vis = glb_vis.view(g_b, -1, c, h, w)
        # 局部图
        l_b, l_pcs, l_c, l_w, l_h = loc_imgs.size()
        loc_imgs = loc_imgs.reshape(-1, l_c, l_w, l_h)
        l2, l3, l4, loc_vis = self.loc_net(loc_imgs)
        # 局部图只考虑通道注意力，关注是什么的问题
        loc_att = self.chan_attn(loc_vis) * loc_vis
        # loc_att = self.spat_attn(loc_vis) * loc_vis
        loc_vis = loc_vis + loc_att
        _, c, h, w = loc_vis.size()
        loc_vis = loc_vis.view(l_b, -1, c, h, w)
        # 简单融合-拼接
        vis_out = torch.cat((glb_vis, loc_vis), dim=1)
        _, _, c, h, w = vis_out.size()
        vis_out = vis_out.view(-1, c, h, w)
        vis_out = vis_out.permute(0, 3, 2, 1)
        return vis_out
