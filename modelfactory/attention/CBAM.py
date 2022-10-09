# CBAM: Convolutional Block Attention Module
# Year: 2018
#  ECCV2018
# https://blog.csdn.net/u013738531/article/details/82731257
# https://blog.csdn.net/weixin_45612763/article/details/109501498?utm_medium=distribute.pc_relevant_download.none-task-blog-baidujs-1.nonecase&depth_1-utm_source=distribute.pc_relevant_download.none-task-blog-baidujs-1.nonecase
# source code:https://github.com/luuuyi/CBAM.PyTorch
# https://blog.csdn.net/DD_PP_JJ/article/details/103318617?utm_medium=distribute.pc_relevant_t0.none-task-blog-OPENSEARCH-1.channel_param&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-OPENSEARCH-1.channel_param
import torch
import torch.nn as nn


# CBAM: Convolutional Block Attention Module
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.chan_attn = ChanAttn(channel)
        self.spat_attn = SpatAttn()

    def forward(self, x):
        out = self.chan_attn(x) * x
        out = self.spat_attn(out) * out
        return out


# Channel Attention Module
class ChanAttn(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChanAttn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu1(avg_out)
        avg_out = self.fc2(avg_out)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# Spatial Attention Module
class SpatAttn(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatAttn, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResBlock_CBAM(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )
        self.cbam = CBAM(channel=places * self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


if __name__ == "__main__":
    input = torch.randn(1, 32, 7, 7)
    print(input)
    cbam = CBAM(32)
    out = cbam.forward(input)
    print(out)
    print(out.size())
