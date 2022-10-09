import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
from torchvision.models.vgg import model_urls as vgg_model_urls
import torchvision.models as models


class DenseNet121(nn.Module):
    def __init__(self, classes=14, pretrained=True):
        super(DenseNet121, self).__init__()
        self.model = torchvision.models.densenet121(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=num_in_features, out_features=classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x) -> object:
        x = self.densenet121(x)
        return x


class DenseNet161(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(DenseNet161, self).__init__()
        self.model = torchvision.models.densenet161(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class DenseNet169(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(DenseNet169, self).__init__()
        self.model = torchvision.models.densenet169(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        x = self.model(x)
        return x


class DenseNet201(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(DenseNet201, self).__init__()
        self.model = torchvision.models.densenet201(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        x = self.model(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        x = self.model(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(ResNet34, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        x = self.model(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(ResNet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        x = self.model(x)
        return x


class ResNet101(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(ResNet101, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        x = self.model(x)
        return x


class ResNet152(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(ResNet152, self).__init__()
        self.model = torchvision.models.resnet152(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        x = self.model(x)
        return x


class VGG19(nn.Module):
    def __init__(self, classes=14, pretrained=True):
        super(VGG19, self).__init__()
        self.model = torchvision.models.vgg19_bn(pretrained=pretrained)

        self.model.classifier = nn.Sequential(
            self.__init_linear(in_features=25088, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            self.__init_linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            self.__init_linear(in_features=4096, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        x = self.model(x)
        return x


class VGG(nn.Module):
    def __init__(self, tags_num):
        super(VGG, self).__init__()
        vgg_model_urls['vgg19'] = vgg_model_urls['vgg19'].replace('https://', 'http://')
        self.vgg19 = models.vgg19(pretrained=True)
        vgg19_classifier = list(self.vgg19.classifier.children())[:-1]
        self.classifier = nn.Sequential(*vgg19_classifier)
        self.fc = nn.Linear(4096, tags_num)
        self.fc.apply(self.init_weights)
        self.bn = nn.BatchNorm1d(tags_num, momentum=0.1)

    #        self.init_weights()

    def init_weights(self, m):
        if type(m) == nn.Linear:
            self.fc.weight.data.normal_(0, 0.1)
            self.fc.bias.data.fill_(0)

    def forward(self, images) -> object:
        visual_feats = self.vgg19.features(images)
        tags_classifier = visual_feats.view(visual_feats.size(0), -1)
        tags_classifier = self.bn(self.fc(self.classifier(tags_classifier)))
        return tags_classifier


class InceptionV3(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(InceptionV3, self).__init__()
        self.model = torchvision.models.inception_v3(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        x = self.model(x)
        return x


class CheXNetDenseNet121(nn.Module):
    def __init__(self, classes=14, pretrained=True):
        super(CheXNetDenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=pretrained)
        num_in_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(in_features=num_in_features, out_features=classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x) -> object:
        """

        :rtype: object
        """
        x = self.densenet121(x)
        return x


class CheXNet(nn.Module):
    def __init__(self, classes=156):
        super(CheXNet, self).__init__()
        self.densenet121 = CheXNetDenseNet121(classes=14)
        self.densenet121 = torch.nn.DataParallel(self.densenet121).cuda()
        self.densenet121.load_state_dict(torch.load('./models/CheXNet.pth.tar')['state_dict'])
        self.densenet121.module.densenet121.classifier = nn.Sequential(
            self.__init_linear(1024, classes),
            nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        """

        :rtype: object
        """
        x = self.densenet121(x)
        return x


class ClassifierFactory(object):
    def __init__(self, model_name, pretrained, classes):
        self.model_name = model_name
        self.pretrained = pretrained
        self.classes = classes

    def create_model(self):
        if self.model_name == 'VGG19':
            _model = VGG19(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'DenseNet121':
            _model = DenseNet121(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'DenseNet161':
            _model = DenseNet161(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'DenseNet169':
            _model = DenseNet169(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'DenseNet201':
            _model = DenseNet201(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'CheXNet':
            _model = CheXNet(classes=self.classes)
        elif self.model_name == 'ResNet18':
            _model = ResNet18(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'ResNet34':
            _model = ResNet34(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'ResNet50':
            _model = ResNet50(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'ResNet101':
            _model = ResNet101(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'ResNet152':
            _model = ResNet152(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'VGG':
            _model = VGG(tags_num=self.classes)
        else:
            _model = CheXNet(classes=self.classes)

        return _model


if __name__ == "__main__":
    classifier = ClassifierFactory('VGG19', False, 20).create_model()
    print(classifier)

