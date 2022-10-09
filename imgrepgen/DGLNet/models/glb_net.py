import torch
import torch.nn as nn
import torch.nn.functional as F
from imgrepgen.DGLNet.models.res_nets import ResNets
from torchvision import transforms


# 全局网络
class GlbNet(nn.Module):
    def __init__(self, backbone_name='resnet50', chan_attn=False, spat_attn=False):
        super(GlbNet, self).__init__()
        self._up_kwargs = {'mode': 'bilinear'}
        self.resnet = ResNets.create_model(model_name=backbone_name, chan_attn=chan_attn, spat_attn=spat_attn)
        # # 降低通道数
        # self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        # # Lateral layers
        # self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.enc_dim = 256 * 4 * 2

    # def _upsample_add(self, x, y):
    #     _, _, H, W = y.size()
    #     return F.interpolate(x, size=(H, W), **self._up_kwargs, align_corners=False) + y
    #
    # def _concatenate(self, p5, p4, p3, p2):
    #     _, _, H, W = p2.size()
    #     p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs, align_corners=False)
    #     p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs, align_corners=False)
    #     p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs, align_corners=False)
    #     return torch.cat([p5, p4, p3, p2], dim=1)

    def forward(self, images, csys=None, ratios=None):
        b, num, c, w, h = images.size()
        images = images.view(-1, c, w, h)
        c2, c3, c4, vis_out = self.resnet(images)
        # vis_out = vis_out.permute(0, 3, 2, 1)
        return vis_out


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
    # 初始化两个图像
    img1 = torch.randn(3, 2048, 2048)
    img1 = transforms.ToPILImage()(img1)
    img2 = torch.randn(3, 2048, 2048)
    img2 = transforms.ToPILImage()(img2)
    imgs = [img1, img2]
    glb_imgs = resize(imgs, (512, 512))  # list of resized PIL images
    glb_imgs = images_transform(glb_imgs)  # 转换成tensor
    # 全局网络
    glb_net = GlbNet()
    out = glb_net.forward(glb_imgs)
