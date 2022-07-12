from abc import ABC

import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module, ABC):
    def __init__(self, _model_):
        super(ResNet, self).__init__()
        self.model = _model_
        self.cnn = nn.Sequential(*list(self.model.children())[:-1], nn.Flatten())
        self.fc = nn.Linear(512, 4)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


class Vgg16bn(nn.Module, ABC):
    def __init__(self, _model_):
        super(Vgg16bn, self).__init__()
        self.model = _model_
        self.cnn = nn.Sequential(*list(self.model.children())[:-1], nn.Flatten())
        self.fc = nn.Linear(25088, 4)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

class ResNeXt(nn.Module, ABC):
    def __init__(self, _model_):
        super(ResNeXt, self).__init__()
        self.model = _model_
        self.cnn = nn.Sequential(*list(self.model.children())[:-1], nn.Flatten())
        self.fc = nn.Linear(in_features=2048, out_features=4)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


## NOTE: This is resnext tweaked for binary classification
class ResNeXt2(nn.Module,ABC):
    def __init__(self, _model_):
        super(ResNeXt2, self).__init__()
        self.model = _model_
        self.cnn = nn.Sequential(*list(self.model.children())[:-1], nn.Flatten())
        self.fc = nn.Linear(in_features=2048, out_features=2)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


def baseline(name, pretrained=True):
    """
    :param name: name of the model
    :param pretrained: use pretrained model or not
    :return: model
    """
    try:
        if name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            return ResNet(model)
        elif name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
            return ResNet(model)
        elif name == 'vgg16':
            model = models.vgg16_bn(pretrained=pretrained)
            return Vgg16bn(model)
        elif name == "resnext50":
            model = models.resnext50_32x4d(pretrained=pretrained)
            return ResNeXt(model)
        elif name == "resnext50_2":
            model = models.resnext50_32x4d(pretrained=pretrained)
            return ResNeXt2(model)
        elif name == "resnext101":
            model = models.resnext101_32x8d(pretrained=pretrained)
            return ResNeXt(model)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    pass
    # m = baseline(name="baseline_resnext50")
    # print(list(m.cnn.children())[-3][-1])
