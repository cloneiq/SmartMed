import torch
import torch.nn as nn
from imgrepgen.DGLNet.models.res_nets import ResNets
from imgrepgen.DGLNet.models.coglg_utils import CoGLGUtils
from modelfactory.attention.CBAM import ChanAttn, SpatAttn
import torch
import torch.nn.functional as F
import numpy as np


# 全局<->局部特征融合网络
class CoGLGNet(nn.Module):
    def __init__(self, backbone_name='resnet50', chan_attn=False, spat_attn=False):
        super(CoGLGNet, self).__init__()
        self._up_kwargs = {'mode': 'bilinear'}
        self.glb_net = ResNets.create_model(model_name=backbone_name)
        self.loc_net = ResNets.create_model(model_name=backbone_name)
        self.enc_dim = 256 * 4 * 2
        self.chan_attn = ChanAttn(self.enc_dim)
        self.spat_attn = SpatAttn()
        # self.linear = nn.Linear(self.enc_dim * 2, self.enc_dim)
        # glb_vis:[1,2048,12,12]->[1,256,12,12]
        self.glb_conv = nn.Sequential(
            nn.Conv2d(self.enc_dim, 1024, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        )
        # loc_vis:[k,2048,12,12]->[k,256,12,12]
        self.loc_conv = nn.Sequential(
            nn.Conv2d(self.enc_dim, 1024, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        )
        # g2l_vis:[k,2048,12,12]->[k,256,12,12]
        self.g2l_conv = nn.Sequential(
            nn.Conv2d(self.enc_dim * 2, 2048, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        )
        # l2g_vis:[1,4096,12,12]->[1,256,12,12]
        self.l2g_conv = nn.Sequential(
            nn.Conv2d(self.enc_dim * 2, 2048, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        )

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
        # out_vis = self.g2l_conv(out_vis)
        # out_vis = self.conv(out_vis)
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

    # 2. 局部特征->全局特征融合-------------------------------------------------------------------------------
    # 输入：loc_patch 块特征图[2,16,2048,12,12]
    # 输入：csys [] 图块坐标
    # 输入：ratio [] 放缩比例
    # 功能：将每个图像块儿的特征图，下采用后，根据坐标及房宿比例合并成全局特征图
    # 输出：out_dis [2,4096,12,12]
    def l2g_fusion(self, loc_patch, csys, ratio, glb_vis):
        b_size = loc_patch.shape[0]
        out_vis = []
        # 循环特征图像
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
        # out_vis = self.l2g_conv(out_vis)
        # out_vis = out_vis.permute(0, 3, 2, 1)
        # out_vis = self.linear(out_vis)
        # out_vis [2, 4096, 12, 12]
        return out_vis

    def merg_to_one_glb(self, cur_patches, cur_csys, cur_ratio):
        _, c, H, W = cur_patches.size()
        # 要放入device
        merge_img_vis = torch.zeros((1, c, H, W))
        b_patches = len(cur_csys)
        h, w = int(np.round(H * float(cur_ratio[0]))), int(np.round(W * float(cur_ratio[1])))
        # 下采用
        patch_out = F.interpolate(cur_patches, size=(h, w), **self._up_kwargs, align_corners=False)
        # 循环将patches组合成以全局特征图
        for i in range(b_patches):
            sys = cur_csys[i]
            top, left = int(np.round(H * float(sys[0]))), int(np.round(W * float(sys[1])))
            merge_img_vis[:, :, top:top + h, left:left + w] = patch_out[i]
        return merge_img_vis

    # 全局与局部的双向融合
    def glg_fusion(self, glb_vis, l2g_vis, loc_vis, g2l_vis):
        # 共有四类特征
        # 全局特征:glb_vis[1,2048,12,12]->[1,256,12,12]
        # 局部全局特征：l2g_vis[16,2048,12,12]->[k,256,12,12],求均值，携带局部特征的全局特征
        # 局部特征：loc_vis[16,2048,12,12]->[k,256,12,12]求均值
        # 局部全局特征：g2l_vis[1,2048,12,12]->[1,256,12,12]，携带全局信息的局部特征
        # 然后按256纬度进行拼接
        # 如何将其融合成glg_vis[1,2048,12,12]
        b, c, h, w = glb_vis.size()
        glb_vis = self.glb_conv(glb_vis)
        l2g_vis = self.l2g_conv(l2g_vis)
        # loc_vis [k,2048,12,12]->[1,2048,12,12]
        loc_vis = loc_vis.view(b, -1, c, w, h)
        loc_vis_t = []
        for i in range(b):
            cur_loc = loc_vis[i]
            cur_loc = torch.mean(cur_loc, dim=0)
            cur_loc = cur_loc.unsqueeze(0)
            loc_vis_t.append(cur_loc)
        loc_vis = torch.stack(loc_vis_t, dim=0)
        loc_vis = loc_vis.view(-1, c, w, h)
        loc_vis = self.loc_conv(loc_vis)
        # g2l_vis [k,2048,12,12]->[1,2048,12,12]
        g2l_vis = g2l_vis.view(b, -1, c, w, h)
        g2l_vis_t = []
        for i in range(b):
            cur_g2l = g2l_vis[i]
            cur_g2l = torch.mean(cur_g2l, dim=0)
            cur_g2l = cur_g2l.unsqueeze(0)
            g2l_vis_t.append(cur_g2l)
        g2l_vis = torch.stack(g2l_vis_t, dim=0)
        g2l_vis = g2l_vis.view(-1, c, w, h)
        g2l_vis = self.loc_conv(g2l_vis)
        out_vis = torch.cat((glb_vis, l2g_vis, loc_vis, g2l_vis), dim=1)
        return out_vis

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
        _, hidden_size, w, h = glb_vis.size()
        f_glb_vis = glb_vis.view(-1, g_pcs, hidden_size, w, h)
        # 生成局部特征图像
        l_b, l_pcs, l_c, l_w, l_h = loc_imgs.size()
        loc_imgs = loc_imgs.reshape(-1, l_c, l_w, l_h)
        l2, l3, l4, loc_vis = self.loc_net(loc_imgs)
        print("Local Vis Size:", loc_vis.size())
        # loc_vis size:[32,2048,12,12]
        _, hidden_size, w, h = loc_vis.size()
        f_loc_vis = loc_vis.view(-1, l_pcs, hidden_size, w, h)
        # loc_vis [2,16,2048,12,12]
        # 局部->全局融合L->G
        # glb_vis:[2, 1, 2048, 12, 12]
        # loc_vis:[32, 2048, 12, 12]
        g2l_vis = self.g2l_fusion(f_glb_vis, csys, ratios, loc_vis)
        # 全局->局部融合L->G
        # glb_vis[2, 2048, 12, 12]
        # loc_vis[2, 16, 2048, 12, 12]
        l2g_vis = self.l2g_fusion(f_loc_vis, csys, ratios, glb_vis)
        # 共有四类特征
        # 全局特征:glb_vis[1,2048,12,12]->[1,256,12,12]
        # 局部全局特征：l2g_vis[16,2048,12,12]->[k,256,12,12],求均值，携带局部特征的全局特征
        # 局部特征：loc_vis[16,2048,12,12]->[k,256,12,12]求均值
        # 局部全局特征：g2l_vis[1,2048,12,12]->[1,256,12,12]，携带全局信息的局部特征
        # 然后按256纬度进行拼接
        # 如何将其融合成glg_vis[1,2048,12,12]
        print("glb size:{}".format(glb_vis.size()))
        # 全局特征:glb_vis [2,2048,12,12]
        # 局部特征：loc_vis [32,2048,12,12]
        # 局部全局特征：l2g_vis [2,2048,12,12]
        # 局部全局特征：g2l_vis [32,2048,12,12]
        vis_out = self.glg_fusion(glb_vis, l2g_vis, loc_vis, g2l_vis)

        # vis_out [2,2048,12,12]
        return vis_out
