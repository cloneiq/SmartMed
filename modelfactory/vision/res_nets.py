import torch
import torch.nn as nn
import math
from modelfactory.attention.CBAM import ChanAttn, SpatAttn
import modelfactory.vision.cnn_utils as cnn_utils


# ResNet with CBAM (Convolutional Block Attention Module)
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 chan_attn=False, spat_attn=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # Channel Apptention
        self.chan_attn = chan_attn
        if self.chan_attn:
            self.ca_attn = ChanAttn(planes)
            print('Channel Apptention has been started...')
        # Spatial Attention
        self.spat_attn = spat_attn
        if self.spat_attn:
            self.sa_attn = SpatAttn()
            print('Spatial Apptention has been started...')
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.chan_attn:
            out = self.ca_attn(out) * out
        if self.spat_attn:
            out = self.sa_attn(out) * out
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 chan_attn=False, spat_attn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # 注意力机制
        # Channel Apptention
        self.chan_attn = chan_attn
        if self.chan_attn:
            self.ca_attn = ChanAttn(planes * 4)
            print('Channel Apptention has been started...')
        # Spatial Attention
        self.spat_attn = spat_attn
        if self.spat_attn:
            self.sa_attn = SpatAttn()
            print('Spatial Apptention has been started...')

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.chan_attn:
            out = self.ca_attn(out) * out
        if self.spat_attn:
            out = self.sa_attn(out) * out
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# 注意：不能改变ResNet的网络结构，故CBAM不能加在block里面，
# 加进去网络结构发生了变化，不能用预训练参数。
# 加在最后一层卷积和第一层卷积不改变网络，可以用预训练参数
class ResNet(nn.Module):
    def __init__(self, block, layers, chan_attn=False, spat_attn=False, num_classes=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.chan_attn = chan_attn
        self.spat_attn = spat_attn
        # 网络的第一层加入注意力机制
        # self.ca_attn0 = ChanAttn(self.inplanes)
        # self.sa_attn0 = SpatAttn()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # # 网络的卷积层的最后一层加入注意力机制
        # self.ca_attn1 = ChanAttn(self.inplanes)
        # self.sa_attn1 = SpatAttn()
        self.num_classes = num_classes
        self.fc_in_features = 512 * block.expansion
        if self.num_classes is not None:
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(self.fc_in_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.chan_attn, self.spat_attn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.num_classes:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class ResNets:
    @staticmethod
    def resnet18(pretrained=False, chan_attn=False, spat_attn=False, num_classes=None):
        model = ResNet(BasicBlock, [2, 2, 2, 2],
                       chan_attn=chan_attn, spat_attn=spat_attn, num_classes=num_classes)
        if pretrained:
            model = ResNets.load_model_dict(model, 'resnet18')
        return model

    @staticmethod
    def resnet34(pretrained=False, chan_attn=False, spat_attn=False, num_classes=None):
        model = ResNet(BasicBlock, [3, 4, 6, 3],
                       chan_attn=chan_attn, spat_attn=spat_attn, num_classes=num_classes)
        if pretrained:
            model = ResNets.load_model_dict(model, 'resnet34')
        return model

    @staticmethod
    def resnet50(pretrained=False, chan_attn=False, spat_attn=False, num_classes=None):
        model = ResNet(Bottleneck, [3, 4, 6, 3],
                       chan_attn=chan_attn, spat_attn=spat_attn, num_classes=num_classes)
        if pretrained:
            model = ResNets.load_model_dict(model, 'resnet50')
        return model

    @staticmethod
    def resnet101(pretrained=False, chan_attn=False, spat_attn=False, num_classes=None):
        model = ResNet(Bottleneck, [3, 4, 23, 3],
                       chan_attn=chan_attn, spat_attn=spat_attn, num_classes=num_classes)
        if pretrained:
            model = ResNets.load_model_dict(model, 'resnet101')
        return model

    @staticmethod
    def resnet152(pretrained=False, chan_attn=False, spat_attn=False, num_classes=None):
        model = ResNet(Bottleneck, [3, 8, 36, 3],
                       chan_attn=chan_attn, spat_attn=spat_attn, num_classes=num_classes)
        if pretrained:
            model = ResNets.load_model_dict(model, 'resnet152')
        return model

    @staticmethod
    def load_model_dict(model, model_name):
        pre_dict = cnn_utils.load_state_dict(model_name)
        # 新的模型的参数
        now_dict = model.state_dict()
        # 将预训练的模型参数拷贝进来
        trsf_dict = cnn_utils.trsf_state_dict(pre_dict, now_dict)
        now_dict.update(trsf_dict)
        model.load_state_dict(now_dict)
        return model


if __name__ == "__main__":
    imgs = torch.randn(1, 3, 224, 224)
    cnn_model = ResNets.resnet18(pretrained=True, num_classes=14,chan_attn=True, spat_attn=True)
    imgs = cnn_model.forward(imgs)
    print(imgs.size())
    # 查看预训练参数(Key)
    for k in cnn_model.state_dict().keys():
        print("预训练参数(Key)：", k)
    # 查看模型参数（Value)
    for i, param in enumerate(cnn_model.parameters()):
        print("预训练参数值(Value)", param)
    # for i, param in enumerate(cnn_model.parameters()):
    #     print("model param:", param)
    # print(cnn_model)
    # imgs_in = cnn_model(imgs_in)
    # print(imgs_in.size())
