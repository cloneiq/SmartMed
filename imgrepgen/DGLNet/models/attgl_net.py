import torch
import torch.nn as nn
from imgrepgen.DGLNet.models.res_nets import ResNets


# 简单全局与局部注意力融合网络
class AttGLNet(nn.Module):
    def __init__(self, backbone_name='resnet50', chan_attn=False, spat_attn=False):
        super(AttGLNet, self).__init__()
        self.glb_net = ResNets.create_model(model_name=backbone_name)
        self.loc_net = ResNets.create_model(model_name=backbone_name)
        self.enc_dim = 256 * 4 * 2
        self.full_att = nn.Linear(self.enc_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(self.enc_dim * 2, self.enc_dim)

    def forward(self, images, csys=None, ratios=None):
        # 从组合的图像中分离出全局图和局部图
        img_len = images.shape[1]
        # 全局图，局部图块
        glb_imgs, loc_imgs = images.split([1, img_len - 1], dim=1)
        # 全局图
        g_b, g_pcs, g_c, g_w, g_h = glb_imgs.size()
        glb_imgs = glb_imgs.reshape(-1, g_c, g_w, g_h)
        g2, g3, g4, glb_vis_out = self.glb_net(glb_imgs)

        glb_vis_out = glb_vis_out.permute(0, 2, 3, 1)
        glb_vis_out = glb_vis_out.reshape(glb_vis_out.size(0), -1, glb_vis_out.size(-1))
        g_att_score = self.softmax(self.full_att(glb_vis_out))
        # size[2,144,1]:表示每个特征图包含144个特征，每个特征有一个值（权重值）
        glb_vis_out = (glb_vis_out * g_att_score).sum(1)
        # size[2,2048]
        # 局部图块
        l_b, l_pcs, l_c, l_w, l_h = loc_imgs.size()
        loc_imgs = loc_imgs.reshape(-1, l_c, l_w, l_h)
        l2, l3, l4, loc_vis_out = self.loc_net(loc_imgs)
        glb_vis_out = glb_vis_out.unsqueeze(1).unsqueeze(3).unsqueeze(4).repeat(1,
                                                                                loc_vis_out.size(0) // glb_vis_out.size(
                                                                                    0), 1, loc_vis_out.size(2),
                                                                                loc_vis_out.size(3))
        glb_vis_out = glb_vis_out.view(-1, glb_vis_out.size(2), glb_vis_out.size(3), glb_vis_out.size(4))
        # 简单融合-拼接
        vis_out = torch.cat((glb_vis_out, loc_vis_out), dim=1)
        vis_out = vis_out.permute(0, 3, 2, 1)
        vis_out = self.linear(vis_out)
        # vis_out = vis_out.permute(0, 3, 2, 1)
        return vis_out
