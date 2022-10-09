import torch.nn as nn
from imgrepgen.DGLNet.models.res_nets import ResNets
from imgrepgen.DGLNet.models.coglg_utils import CoGLGUtils
import torch
import torch.nn.functional as F
import numpy as np
import math


# 全局->局部特征融合网络
class CoG2LNet(nn.Module):
    def __init__(self, backbone_name='resnet50', chan_attn=False, spat_attn=False):
        super(CoG2LNet, self).__init__()
        self._up_kwargs = {'mode': 'bilinear'}
        self.glb_net = ResNets.create_model(model_name=backbone_name)
        self.loc_net = ResNets.create_model(model_name=backbone_name)
        self.enc_dim = 256 * 4 * 2
        # self.linear = nn.Linear(self.enc_dim * 2, self.enc_dim)
        self.conv = nn.Conv2d(self.enc_dim * 2, self.enc_dim, kernel_size=1, stride=1, padding=0)
        # self.fusion = CoGLGUtils()

    # 1. 全局特征->局部特征融合----------------------------------------------------------------------------
    # 将每个图像的全局特征图分割成特征块儿（1->n）
    # 输入：glb_vis [2,1,2048,12,12]，坐标、比例数据
    # 输出：out_vis [2,16,2048,3,3]
    def g2l_fusion(self, glb_patch, csys, ratio, loc_patch):
        # 转换成[2,1,2048,12,12]
        glb_fcs, _, c, w, h = glb_patch.size()
        out_vis = []
        for i in range(glb_fcs):
            cur_img = glb_patch[i]
            cur_csys = csys[i]
            cur_ratio = ratio[i]
            cur_out_feats = self.crop_patches_from_glb(cur_img, cur_csys, cur_ratio)
            out_vis.append(cur_out_feats)
        # out_vis [2, 16, 2048, 12, 12]
        out_vis = torch.stack(out_vis, dim=0)
        p, _, c, w, h = out_vis.size()
        out_vis = out_vis.view(-1, c, w, h)
        out_vis = F.interpolate(out_vis, size=loc_patch.size()[2:], **self._up_kwargs, align_corners=False)
        out_vis = torch.cat((out_vis, loc_patch), dim=1)
        out_vis = self.conv(out_vis)
        # out_vis = out_vis.permute(0, 3, 2, 1)
        # out_vis = self.linear(out_vis)
        return out_vis

    #   将一个全局特征图分割成k个特征图块儿
    def crop_patches_from_glb(self, cur_img_vis, csys, ratio):
        # cur_img_vis size:
        _, c, H, W = cur_img_vis.size()
        b = len(csys)
        h, w = int(np.round(H * float(ratio[0]))), int(np.round(W * float(ratio[1])))
        crop_vis = []
        for i in range(b):
            top, left = int(np.round(float(csys[i][0]) * H)), int(np.round(float(csys[i][1]) * W))
            glb_patch = cur_img_vis[0:1, :, top:top + h, left:left + w]
            crop_vis.append(glb_patch[0])
        crop_vis = torch.stack(crop_vis, dim=0)
        # crop_vis size:
        return crop_vis

    # 图像（第一个：全局，其与：局部），比例，坐标（top,left）
    def forward(self, images, csys=None, ratios=None):
        # 从组合的图像中分离出全局图和局部图
        img_len = images.shape[1]
        patch_num = img_len - 1
        glb_imgs, loc_imgs = images.split([1, patch_num], dim=1)
        # 生成全局特征图
        g_b, g_pcs, g_c, g_w, g_h = glb_imgs.size()
        glb_imgs = glb_imgs.view(-1, g_c, g_w, g_h)
        g2, g3, g4, glb_vis = self.glb_net(glb_imgs)
        print("Glb_vis Size:", glb_vis.size())
        # glb_vis size:[2,2048,12,12]
        # 将全局特征图分割成Patch特征
        _, hidden_size, w, h = glb_vis.size()
        glb_vis = glb_vis.view(-1, g_pcs, hidden_size, w, h)
        # glb_vis [2,1,2048,12,12]
        # 生成局部特征图像
        l_b, l_pcs, l_c, l_w, l_h = loc_imgs.size()
        loc_imgs = loc_imgs.reshape(-1, l_c, l_w, l_h)
        l2, l3, l4, loc_vis = self.loc_net(loc_imgs)
        print("Local Vis Size:", loc_vis.size())
        # _, l_hidden_size, l_w, l_h = loc_vis.size()
        # loc_vis = loc_vis.view(-1, l_pcs, l_hidden_size, l_w, l_h)
        # 全局->局部融合(G->L)
        print("glb_vis size:{},loc_vis size:{}".format(glb_vis.size(), loc_vis.size()))
        vis_out = self.g2l_fusion(glb_vis, csys, ratios, loc_vis)
        print("Fusion size:", vis_out.size())
        return vis_out
