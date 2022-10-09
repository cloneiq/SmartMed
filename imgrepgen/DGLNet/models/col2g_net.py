import torch
import torch.nn as nn
from imgrepgen.DGLNet.models.res_nets import ResNets
from imgrepgen.DGLNet.models.coglg_utils import CoGLGUtils
import torch
import torch.nn.functional as F
import numpy as np


# 全局->局部特征融合网络
class CoL2GNet(nn.Module):
    def __init__(self, backbone_name='resnet50', chan_attn=False, spat_attn=False):
        super(CoL2GNet, self).__init__()
        self._up_kwargs = {'mode': 'bilinear'}
        self.glb_net = ResNets.create_model(model_name=backbone_name)
        self.loc_net = ResNets.create_model(model_name=backbone_name)
        self.enc_dim = 256 * 4 * 2
        self.conv = nn.Conv2d(self.enc_dim * 2, self.enc_dim, kernel_size=1, stride=1, padding=0)

    # 2. 局部特征->全局特征融合-------------------------------------------------------------------------------
    # 输入：loc_patch 块特征图[2,16,2048,12,12]
    # 输入：csys [] 图块坐标
    # 输入：ratio [] 放缩比例
    # 功能：将每个图像块儿的特征图，下采用后，根据坐标及房宿比例合并成全局特征图
    # 输出：out_dis [2,4096,12,12]
    def l2g_fusion(self, loc_patch, csys, ratio, glb_vis):
        b_size = loc_patch.shape[0]
        out_vis = []
        # 循环图像
        for i in range(b_size):
            # 获得第一幅图像的patches
            cur_patches = loc_patch[i]
            cur_csys = csys[i]
            cur_ratio = ratio[i]
            merg_vis = self.merg_to_one_glb(cur_patches, cur_csys, cur_ratio)
            out_vis.append(merg_vis)
        out_vis = torch.stack(out_vis, dim=0)
        # out _vis [2, 1, 2048, 12, 12]
        m_b, _, m_c, m_w, m_h = out_vis.size()
        out_vis = out_vis.view(-1, m_c, m_w, m_h)
        # 拼接
        out_vis = out_vis.to(glb_vis.device)
        out_vis = torch.cat((out_vis, glb_vis), dim=1)
        out_vis = self.conv(out_vis)
        # out_vis [2, 4096, 12, 12]
        return out_vis

    def merg_to_one_glb(self, cur_patches, cur_csys, cur_ratio):
        _, c, H, W = cur_patches.size()
        # 要放入device
        merge_img_vis = torch.zeros((1, c, H, W))
        b_patches = len(cur_csys)
        h, w = int(np.round(H * float(cur_ratio[0]))), int(np.round(W * float(cur_ratio[1])))
        # 下采样
        patch_out = F.interpolate(cur_patches, size=(h, w), **self._up_kwargs, align_corners=False)
        # 循环将patches组合成以全局特征图
        for i in range(b_patches):
            sys = cur_csys[i]
            top, left = int(np.round(H * float(sys[0]))), int(np.round(W * float(sys[1])))
            merge_img_vis[:, :, top:top + h, left:left + w] = patch_out[i]
        return merge_img_vis

    # 图像（第一个：全局，其与：局部），比例，坐标（top,left）
    def forward(self, images, csys=None, ratios=None):
        img_len = images.shape[1]
        patch_num = img_len - 1
        # 全局图像,图像块儿
        glb_imgs, loc_imgs = images.split([1, patch_num], dim=1)
        # 生成全局特征图
        g_b, g_pcs, g_c, g_w, g_h = glb_imgs.size()
        glb_imgs = glb_imgs.view(-1, g_c, g_w, g_h)
        g2, g3, g4, glb_vis = self.glb_net(glb_imgs)
        print("Glb_vis Size:", glb_vis.size())
        # 生成局部特征图像
        l_b, l_pcs, l_c, l_w, l_h = loc_imgs.size()
        loc_imgs = loc_imgs.reshape(-1, l_c, l_w, l_h)
        l2, l3, l4, loc_vis = self.loc_net(loc_imgs)
        print("Local Vis Size:", loc_vis.size())
        # loc_vis size:[32,2048,12,12]
        _, hidden_size, w, h = loc_vis.size()
        loc_vis = loc_vis.view(-1, l_pcs, hidden_size, w, h)
        # loc_vis [2,16,2048,12,12]
        # 局部->全局融合L->G (patch_vis_out,patch_vis_out)
        print("glb_vis size:{},loc_vis size:{}".format(glb_vis.size(), loc_vis.size()))
        out_vis = self.l2g_fusion(loc_vis, csys, ratios, glb_vis)
        print("Fusion size:", out_vis.size())
        return out_vis
