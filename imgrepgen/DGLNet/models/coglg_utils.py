import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import math


# 全局->局部特征融合网络
class DGLNetUtils:
    def __init__(self):
        self._up_kwargs = {'mode': 'bilinear'}

    # 输入：glb_imgs 全局图像（不可以GPU并行，glb_imgs不是tensor）
    # 输入：p_size: 图块儿大小
    # 输出：
    @staticmethod
    def glb_imgs_to_patches(glb_imgs, p_size):
        patches = []
        sizes = []
        csys = []
        ratios = [[0, 0]] * len(glb_imgs)
        max_patches = 0
        for i in range(len(glb_imgs)):
            w, h = glb_imgs[i].size
            # 从原始大图上切割一个指定的图像
            # 方法：
            p_w = p_size[0]
            p_h = p_size[1]
            p_num_w = math.floor(w / p_w)
            p_num_h = math.floor(h / p_h)
            # print('patch nums:-----', p_num_w, p_num_h)
            # top, left, height, width
            p_tg_w = p_num_w * p_w
            p_tg_h = p_num_h * p_h
            left = math.floor((w - p_tg_w) / 2)
            top = math.floor((h - p_tg_h) / 2)
            img_crop = transforms.functional.crop(glb_imgs[i], top, left, p_tg_h, p_tg_w)
            # print(p_tg_w, p_tg_h)
            size = (p_tg_w, p_tg_h)
            sizes.append(size)
            ratios[i] = [float(p_size[0]) / size[0], float(p_size[1]) / size[1]]
            # img_crop.save('../samples/test{}.png'.format('glb'))
            # print('img crop size: ', img_crop.size)
            patches.append([img_crop] * (p_num_w * p_num_h))
            csys.append([[0, 0]] * (p_num_w * p_num_h))
            p_top = 0
            p_left = 0
            for j in range(p_num_w):
                for k in range(p_num_h):
                    p_crop = transforms.functional.crop(img_crop, p_top, p_left, p_h, p_w)
                    # p_crop.save('../samples/test{}.png'.format(j * p_num_h + k))
                    patches[i][j * p_num_h + k] = p_crop
                    csys[i][j * p_num_h + k] = [1.0 * p_top / size[0], 1.0 * p_left / size[1]]

                    p_top = p_top + p_h
                p_top = 0
                p_left = p_left + p_w
            pcs_num = p_num_w * p_num_h
            if pcs_num > max_patches:
                max_patches = pcs_num
        # 分割的图像块，最大块儿数
        return patches, size, ratios, csys, max_patches

    # 1. 全局特征->局部特征融合----------------------------------------------------------------------------
    # 将每个图像的全局特征图分割成特征块儿（1->n）
    # 输入：glb_vis [2,1,2048,12,12]，坐标、比例数据
    # 输出：out_vis [2,16,2048,3,3]
    def crop_glb_to_patches(self, glb_patch, csys, ratio, loc_patch):
        # 转换成[2,1,2048,12,12]
        glb_fcs, _, c, w, h = glb_patch.size()
        out_vis = []
        for i in range(glb_fcs):
            cur_img = glb_patch[i]
            cur_csys = csys[i]
            cur_ratio = ratio[i]
            cur_out_feats = self._crop_patches_from_glb(cur_img, cur_csys, cur_ratio)
            out_vis.append(cur_out_feats)
        # out_vis [2, 16, 2048, 12, 12]
        out_vis = torch.stack(out_vis, dim=0)
        p, _, c, w, h = out_vis.size()
        out_vis = out_vis.view(-1, c, w, h)
        out_vis = F.interpolate(out_vis, size=loc_patch.size()[2:], **self._up_kwargs, align_corners=False)
        out_vis = torch.cat((out_vis, loc_patch), dim=1)
        # out_vis = self._glb_to_loc_fusion(out_vis, loc_patch)
        #  size: [32,4096, 12, 12]
        return out_vis

    @staticmethod
    def _crop_patches_from_glb(cur_img_vis, csys, ratio):
        # cur_img_vis size:
        _, c, H, W = cur_img_vis.size()
        b = len(csys)
        h, w = int(np.round(H * float(ratio[0].cpu()))), int(np.round(W * float(ratio[1].cpu())))
        crop_vis = []
        for i in range(b):
            top, left = int(np.round(float(csys[i][0].cpu()) * H)), int(np.round(float(csys[i][1].cpu()) * W))
            glb_patch = cur_img_vis[0:1, :, top:top + h, left:left + w]
            crop_vis.append(glb_patch[0])
        crop_vis = torch.stack(crop_vis, dim=0)
        # crop_vis size:
        return crop_vis

    # 对从全局特征图中分割的局部特征块儿进行上采样，与局部特征图进行融合
    # 输入：glb_patch [32, 2048, 3, 3]
    # 输入：lo_patch [32,2048, 12, 12]
    # 输出: out_vis [32,4096, 12, 12]
    def _glb_to_loc_fusion(self, glb_patch, loc_patch):
        # glb_patch size: [32, 2048, 3, 3]
        # loc_patch size: [32,2048, 12, 12]
        # 对每个块儿进行上采样
        out_vis = F.interpolate(glb_patch, size=loc_patch.size()[2:], **self._up_kwargs, align_corners=False)
        # 拼接
        out_vis = torch.cat((out_vis, loc_patch), dim=1)
        return out_vis

    # 2. 局部特征->全局特征融合-------------------------------------------------------------------------------
    # 输入：loc_patch 块特征图[2,16,2048,12,12]
    # 输入：csys [] 图块坐标
    # 输入：ratio [] 放缩比例
    # 功能：将每个图像块儿的特征图，下采用后，根据坐标及房宿比例合并成全局特征图
    # 输出：out_dis [2,4096,12,12]
    def merg_patchs_to_glb(self, loc_patch, csys, ratio, glb_vis):
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
        out_vis = F.interpolate(out_vis, size=loc_patch.size()[2:], **self._up_kwargs, align_corners=False)
        # 拼接
        out_vis = torch.cat((out_vis, glb_vis), dim=1)
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

    @staticmethod
    def _loc_to_glb_fusion(merge_vis, glb_vis):
        merge_vis = merge_vis.to(glb_vis.device)
        out_fusions = torch.cat((merge_vis, glb_vis), dim=1)
        return out_fusions


# -----------------暂时没有-------------------------
def get_patch_info(shape, p_size):
    # shape: origin image size, (x, y)
    # p_size: patch  size(square)
    # return: n_x, n_y, step_x, step_y
    x = shape[0]
    y = shape[1]
    n = m = 1
    while x > n * p_size:
        n += 1
    while p_size - 1.0 * (x - p_size) / (n - 1) < 50:
        n += 1
    while y > m * p_size:
        m += 1
    while p_size - 1.0 * (y - p_size) / (m - 1) < 50:
        m += 1
    return n, m, (x - p_size) * 1.0 / (n - 1), (y - p_size) * 1.0 / (m - 1)


def images_to_patches(images, p_size):
    # image/label => patches
    # p_size: patch size
    # return: list of PIL patch images; coordinates: images->patches; ratios: (h, w)
    patches = []
    coordinates = []
    templates = []
    sizes = []
    ratios = [(0, 0)] * len(images)
    patch_ones = np.ones(p_size)
    max_patches = 0
    for i in range(len(images)):
        w, h = images[i].size
        size = (w, h)
        sizes.append(size)
        ratios[i] = (float(p_size[0]) / size[0], float(p_size[1]) / size[1])
        template = np.zeros(size)
        n_x, n_y, step_x, step_y = get_patch_info(size, p_size[0])
        # print('n_x={}, n_y={}, step_x={}, step_y={}'.format(n_x, n_y, step_x, step_y))
        patches.append([images[i]] * (n_x * n_y))
        coordinates.append([(0, 0)] * (n_x * n_y))
        for x in range(n_x):
            if x < n_x - 1:
                top = int(np.round(x * step_x))
            else:
                top = size[0] - p_size[0]
            for y in range(n_y):
                if y < n_y - 1:
                    left = int(np.round(y * step_y))
                else:
                    left = size[1] - p_size[1]
                template[top:top + p_size[0], left:left + p_size[1]] += patch_ones
                coordinates[i][x * n_y + y] = (1.0 * top / size[0], 1.0 * left / size[1])
                img_crop = transforms.functional.crop(images[i], top, left, p_size[0], p_size[1])
                # img_crop.save('../samples/test{}.png'.format(x * n_y + y))
                #                 # print('img crop size: ', img_crop.size)
                patches[i][x * n_y + y] = img_crop
        pcs_num = x * n_y + y + 1
        if pcs_num > max_patches:
            max_patches = pcs_num
        templates.append(Variable(torch.Tensor(template).expand(1, 1, -1, -1)))
    return patches, coordinates, templates, sizes, ratios, max_patches


transformer = transforms.Compose([
    transforms.ToTensor()
])


def images_transform(images):
    inputs = []
    for img in images:
        inputs.append(transformer(img))
    inputs = torch.stack(inputs, dim=0)
    return inputs


def resize(images, shape):
    resized = list(images)
    for i in range(len(images)):
        resized[i] = images[i].resize(shape)
    return resized


if __name__ == "__main__":
    utils = DGLNetUtils()
    img1 = Image.open('../samples/CXR1423_IM-0270-1001.jpg').convert('RGB')
    imgs = [img1]
    # images_glb = resize(imgs_in, (256, 256))  # list of resized PIL images
    # images_glb = images_transform(images_glb)  # 转换成tensor
    p_size = (384, 384)
    utils.glb_imgs_to_patches(imgs, p_size)
    # patches, coordinates, templates, sizes, ratios = images_to_patches(imgs_in, p_size)
    # sub_batch_size = 6
    # for i in range(len(imgs_in)):
    #     j = 0
    #     print(len(coordinates[i]))
    #     while j < len(coordinates[i]):
    #         patches_var = images_transform(patches[i][j: j + sub_batch_size])  # b, c, h, w
    #         print(patches_var.size())
    #         j += sub_batch_size
    #         images_glb[i:i + 1]
