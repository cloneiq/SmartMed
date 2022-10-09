import torch
import torch.nn as nn
import torchvision.models as models
from modelfactory.attention.CBAM import CBAM
from modelfactory.vision.res_nets import ResNets


class VGG19(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG19, self).__init__()
        net = models.vgg19(pretrained=pretrained)
        modules = list(net.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.enc_dim = list(net.features.children())[-3].weight.shape[0]

    def forward(self, x):
        # torch.Size([4, 3, 224, 224])
        x = self.model(x)  # (batch_size, enc_dim, enc_img_size, enc_img_size)
        # torch.Size([4, 512, 7, 7])
        # print("VGG19: out feature size is {}".format(x.size()))
        return x


class ResNet(nn.Module):
    def __init__(self, model_name, pretrained=True,
                 chan_attn=False, spat_attn=False, num_classes=None):
        super(ResNet, self).__init__()
        self.model_name = model_name
        mod_str = 'ResNets.' + self.model_name.lower() + \
                  '(pretrained={},chan_attn={},spat_attn={},' \
                  'num_classes={})'.format(pretrained, chan_attn, spat_attn, num_classes)
        print(mod_str)
        self.model = eval(mod_str)
        self.enc_dim = self.model.fc_in_features
        print(self.enc_dim)

    def forward(self, x):
        # torch.Size([4, 3, 224, 224])
        x = self.model(x)  # (batch_size, enc_dim, enc_img_size, enc_img_size)
        # torch.Size([4, 512,7, 7])
        return x


class DenseNet161(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet161, self).__init__()
        net = models.densenet161(pretrained=pretrained)
        modules = list(list(net.children())[0])[:-1]
        self.model = nn.Sequential(*modules)
        self.enc_dim = net.classifier.in_features

    def forward(self, x):
        x = self.model(x)
        # torch.Size([4, 2208, 7, 7])
        print("DenseNet161: out feature size {}".format(x.size()))
        return x


class DenseNet169(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet169, self).__init__()
        net = models.densenet169(pretrained=pretrained)
        modules = list(list(net.children())[0])[:-1]
        self.model = nn.Sequential(*modules)
        self.enc_dim = net.classifier.in_features

    def forward(self, x):
        x = self.model(x)
        # x = x.permute(0, 2, 3, 1)
        # torch.Size([4, 2208, 7, 7])
        print("DenseNet169: out feature size {}".format(x.size()))
        return x


class DenseNet201(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet201, self).__init__()
        net = models.densenet201(pretrained=pretrained)
        modules = list(net.features)
        self.model = nn.Sequential(*modules)
        self.enc_dim = net.classifier.in_features

    def forward(self, x):
        x = self.model(x)
        # x = x.permute(0, 2, 3, 1)
        # torch.Size([4, 1920, 7, 7])
        # print("DenseNet201: out feature size {}".format(x.size()))
        return x


class FeaturesFactory(object):
    def __init__(self, model_name, pretrained=False, chan_attn=False, spat_attn=False, num_classes=None):
        self.model_name = model_name
        self.pretrained = pretrained
        self.chan_attn = chan_attn
        self.spat_attn = spat_attn
        self.num_classes = num_classes

    def create_model(self):
        if self.model_name.lower() == 'vgg19':
            _model = VGG19(pretrained=self.pretrained)
        # ResNet
        elif self.model_name.lower() == 'resnet18':
            _model = ResNet(self.model_name, pretrained=self.pretrained,
                            chan_attn=self.chan_attn, spat_attn=self.spat_attn,
                            num_classes=self.num_classes)
        elif self.model_name.lower() == 'resnet34':
            _model = ResNet(self.model_name, pretrained=self.pretrained,
                            chan_attn=self.chan_attn, spat_attn=self.spat_attn,
                            num_classes=self.num_classes)
        elif self.model_name.lower() == 'resnet50':
            _model = ResNet(self.model_name, pretrained=self.pretrained,
                            chan_attn=self.chan_attn, spat_attn=self.spat_attn,
                            num_classes=self.num_classes)
        elif self.model_name.lower() == 'resnet101':
            _model = ResNet(self.model_name, pretrained=self.pretrained,
                            chan_attn=self.chan_attn, spat_attn=self.spat_attn,
                            num_classes=self.num_classes)
        elif self.model_name.lower() == 'resnet152':
            _model = ResNet(self.model_name, pretrained=self.pretrained,
                            chan_attn=self.chan_attn, spat_attn=self.spat_attn,
                            num_classes=self.num_classes)
        # DenseNet
        elif self.model_name.lower() == 'densenet161':
            _model = DenseNet161(pretrained=self.pretrained)
        elif self.model_name.lower() == 'densenet169':
            _model = DenseNet169(pretrained=self.pretrained)
        elif self.model_name.lower() == 'densenet201':
            _model = DenseNet201(pretrained=self.pretrained)
        return _model


if __name__ == "__main__":
    cnn_model = FeaturesFactory('resnet152', pretrained=False,
                                chan_attn=True, spat_attn=True,num_classes=14).create_model()

    # cnn_model = FeaturesFactory('resnet152', pretrained=False,
    #                             chan_attn=True, spat_attn=True,
    #                             num_classes=120).create_model()
    print(cnn_model)
    imgs = torch.randn(1, 3, 224, 224)
    imgs = cnn_model(imgs)
    print(imgs.size())
